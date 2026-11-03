# Python ↔ C++ 强化学习：让布娃娃站起来

## 1. 整体思路

```
┌────────────────────────┐        ctypes        ┌──────────────────────────┐
│  py/train.py           │  ──────────────────► │  rl_env.dll (C++/Box2D)  │
│  PyTorch PPO 策略       │  ◄────────────────── │  env_step() 返回观测/奖励 │
└────────────────────────┘                      └──────────────────────────┘
            │                                              ▲
            │ 导出 policy.bin                              │ 同一份 Ragdoll 物理
            ▼                                              │
┌────────────────────────┐                      ┌──────────────────────────┐
│  policy.bin            │  ──────────────────► │  window.exe (OpenGL 渲染) │
└────────────────────────┘                      └──────────────────────────┘
```

关键点：**训练和回放共用同一个 [`Ragdoll`](../src/ragdoll.cpp) 物理类**，
这样策略在窗口里看到的动力学和训练时完全一致。

Python 与 C++ 之间的桥接没有用 pybind11，而是**纯 C ABI + ctypes**：
只要把 [`env_api.cpp`](../src/env_api.cpp) 编译成动态库，Python 用 `ctypes.CDLL`
直接加载即可，不需要额外的编译期依赖（少一层工具链麻烦）。

## 2. 接口约定

| 侧 | 内容 |
| --- | --- |
| 观测 | 66 维：躯干位姿/速度 7 + 10 个部件相对位姿 40 + 9 个关节角 + 10 个部件角速度 |
| 动作 | 8 维：两条手臂（肩、肘）+ 两条腿（髋、膝）的目标角速度，范围 `[-1, 1]`，内部乘以 `kMotorSpeedScale` |
| 奖励 | `0.25 × [ 1.0 存活 + 1.0·躯干直立度 + 1.0·高度塑形 + 0.5·双腿伸展度 − 0.05·动作能量 − 0.02·动作突变 − 0.05·横向漂移 ]` |
| 结束 | 倾斜过度 (`up.y < 0.35`) 或躯干比站姿低 1.2 m（`kFallHeight`，跪姿会直接判失败），或达到 400 步 |

### 奖励为什么这样设计（踩过的坑）

第一版奖励是 `1.0 + 1.0·up.y + 0.5·高度 − …`，结果策略学会**跪着不动**：

* 跪下时躯干依然是竖直的 → `up.y` 照样接近 1；
* 高度项权重只有 0.5 且用了绝对值归一化，跪姿几乎不掉分；
* `kFallHeight = -2.4` 太宽容，跪姿根本没触发终止，能一直跪满 400 步。

于是「跪着」≈ 2.2 分/步、「站着」≈ 2.5 分/步，只差 12% 却省掉了全部控制代价，成了稳定的局部最优。

现在三处针对性修正：

1. **高度塑形按站姿归一化**（[`kStandHeight`](../include/ragdoll.hpp:48)）：站直得 1 分，比站姿低 1.1 m 得 −1 分；
2. **新增「双腿伸展度」**：髋、膝关节越接近伸直得分越高——跪/蹲会把它们弯到接近限位，直接扣分；
3. **跪着直接判失败**：`kFallHeight = kStandHeight - 1.2`，让退化解无法继续刷存活奖励。

另外整体乘 `kRewardScale = 0.25`，价值网络的回归目标从 ~1000 降到 ~250，critic 更容易学。

## 2.1 关于 GPU 与并行加速（重要）

必须澄清一个事实：**Box2D 物理只能在 CPU 上串行步进，搬不到 GPU**。
每个角色每一步都依赖上一步状态，无法向量化到 CUDA。它占了 95% 以上的耗时。

所以「多角色同时训练更快」要分两层实现：

| 层 | 手段 | 作用 |
| --- | --- | --- |
| 物理采样 | `--workers N`（多进程） | **真正提速**：N 个进程跑满多核 CPU，物理真并行 |
| 物理采样 | `--envs M`（每进程多角色） | 摊薄调度开销，一次前向喂 M 个样本 |
| 网络更新 | `--device cuda` | batch 越大越划算，PPO 更新阶段用 GPU |

总并行角色数 = `workers × envs`。RTX 4060 Laptop（8 GB）+ 8 核 CPU 的参考配置：

```bash
# 1) 确认 torch 是否为 CUDA 版（PyPI 的 Windows wheel 是 CPU-only）
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"

# 2) 装 CUDA 版。驱动 560.x 对应 CUDA 12.6，只能用 cu126（cu128 需要 570+ 驱动）
#    注意包体约 2.4 GB，请在自己终端里执行
uv pip install --reinstall torch --index-url https://download.pytorch.org/whl/cu126

# 3) 8 进程 × 4 角色 = 32 个角色同时训练，网络更新走 GPU
python py/train.py --device cuda --workers 8 --envs 4 --steps 2000000
```

几点经验：
* 采样阶段的前向非常轻（2.5 万参数），留在 worker 的 CPU 上更快，
  否则每个 worker 都要建独立 CUDA context 抢显存；
* `--workers` 建议不要超过物理核心数，超了反而变慢；
* 想看 GPU 是否真的在工作：任务管理器 → 性能 → GPU，看 `Compute` 占用。

## 3. 运行步骤

```bash
# 1) 安装 Python 依赖（环境由 uv 管理）
uv pip install -r py/requirements.txt

# 2) 编译 C++：生成 rl_env.dll 和 window.exe
cmake --workflow --preset release

# 3) 先确认桥接正常（只用标准库，不需要 numpy/torch）
python py/smoke_test.py

# 4) 训练：8 进程 × 4 角色 = 32 个角色并行，网络更新走 GPU
python py/train.py --workers 8 --envs 4 --device cuda

# 5) 回放：加载训练好的策略，看角色站起来
build/msvc/bin/Release/window.exe py/policy.bin

# 6) 不加载策略，只看物理自然倒下的原始行为
build/msvc/bin/Release/window.exe

# 7) 校验策略文件能否被 C++ 正确加载（不开窗口）
build/msvc/bin/Release/window.exe --selftest py/policy.bin
```

训练脚本会自动搜索 `build/**/rl_env.dll`，找不到时用 `--lib` 手动指定。

## 3.1 如何观察训练过程

训练是 headless 的（不建窗口，比带渲染快一个数量级），有三种看法：

**① 终端日志**（最直接）每轮打印一行：

```
[  12] 步数  172032 | 平均回报   512.40 | 平均存活  214.3 步 | 策略损失  -0.021 | 价值损失  120.551 | batch 14336
```

**② CSV 曲线**：默认写入 `py/runs/metrics.csv`（`--log` 可改，传空串关闭），
画 `mean_return` 对 `total_steps` 的曲线即可，也能用 Excel 直接打开。

**③ 边训练边看角色**（推荐）：另开一个终端

```powershell
build\msvc\bin\Release\window.exe --watch py/policy.bin
```

窗口每 0.5 秒检查 `policy.bin` 是否更新并**热重载**（加载失败会保留旧策略），
角色摔倒 1.5 秒后**自动复位**，所以训练途中就能看到它从「一碰就倒」到「能站住」。

判断学没学会，看两个数：
* **平均回报**：站直时每步约 0.875 分，站稳 400 步约 350 分（半跪只有约 0.4 分/步且很快终止）；
* **平均存活步数**：上限 400（`kMaxEpisodeSteps`）。

## 4. 常用调参

| 现象 | 调整 |
| --- | --- |
| 一直站不起来 / 回报不涨 | `--steps 1500000` 多训；或把 [`ragdoll.cpp`](../src/ragdoll.cpp) 里 `kMotorTorque` 调大 |
| 学习太慢（采样慢） | 加大 `--workers`（物理真并行），每进程内物理仍是单线程 |
| 学习太慢（更新慢） | `--lr 5e-4`；`--hidden 256` |
| 抖动、动作很猛 | 提高奖励里的能量惩罚系数 `0.03`；或降低 `kMotorSpeedScale` |
| 摔倒判定太严 | 放宽 `kFallUpY` / `kFallHeight` |
| 想看训练中间的效果 | 脚本每次刷新最佳成绩都会覆盖 `py/policy.bin`，随时 `window.exe py/policy.bin` |

## 5. 扩展方向

* **加课程学习**：`env_reset` 里先让身体从半蹲状态开始，逐步增加难度。
* **增加地形随机化**：在 [`env_api.cpp`](../src/env_api.cpp) 的 `createGround` 里随机摩擦系数。
* **换算法**：同一套 C ABI 可以直接喂给 SAC/TD3，或包一层 Gymnasium 的 `Env` 用 `stable-baselines3`。
* **多进程并行**：`env_create()` 可以创建任意多个实例，`--workers` 已经按进程并行采样。
