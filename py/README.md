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
| 奖励 | `1.0（存活） + 1.0·躯干直立度 + 0.5·高度 − 0.03·动作能量 − 0.02·横向漂移` |
| 结束 | 倾斜过度 (`up.y < 0.35`) 或躯干过低 (`y < -2.4`)，或者达到 400 步 |

## 3. 运行步骤

```bash
# 1) 安装 Python 依赖（项目里已有 .venv 可直接用）
python -m pip install -r py/requirements.txt

# 2) 编译 C++：生成 rl_env.dll 和 window.exe
cmake --workflow --preset release

# 3) 训练（默认 60 万步，RTX/CPU 都能跑，约 5~15 分钟）
python py/train.py

# 4) 回放：加载训练好的策略，看角色站起来
build/msvc/bin/Release/window.exe py/policy.bin

# 5) 不加载策略，只看物理自然倒下的原始行为
build/msvc/bin/Release/window.exe
```

训练脚本会自动搜索 `build/**/rl_env.dll`，找不到时用 `--lib` 手动指定。

## 4. 常用调参

| 现象 | 调整 |
| --- | --- |
| 一直站不起来 / 回报不涨 | `--steps 1500000` 多训；或把 [`ragdoll.cpp`](../src/ragdoll.cpp) 里 `kMotorTorque` 调大 |
| 学习太慢 | `--lr 5e-4`；`--hidden 256` |
| 抖动、动作很猛 | 提高奖励里的能量惩罚系数 `0.03`；或降低 `kMotorSpeedScale` |
| 摔倒判定太严 | 放宽 `kFallUpY` / `kFallHeight` |

## 5. 扩展方向

* **加课程学习**：`env_reset` 里先让身体从半蹲状态开始，逐步增加难度。
* **增加地形随机化**：在 [`env_api.cpp`](../src/env_api.cpp) 的 `createGround` 里随机摩擦系数。
* **换算法**：同一套 C ABI 可以直接喂给 SAC/TD3，或包一层 Gymnasium 的 `Env` 用 `stable-baselines3`。
* **多环境并行**：`env_create()` 可以创建任意多个实例，配合多进程就得到向量化环境。
