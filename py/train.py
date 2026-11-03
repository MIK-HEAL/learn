"""用 PPO 训练 Box2D 布娃娃「站起来」。

C++ 侧只保留物理（无渲染）并编译成 rl_env 动态库，Python 通过 ctypes 调用。

性能结构（重要）：
    * Box2D 物理只能在 CPU 上**串行**执行，它是真正的瓶颈；
    * 因此采样用「多进程 × 每进程多角色」并行：--workers N --envs M，
      总共 N*M 个角色同时训练（物理跑满多核 CPU）；
    * 网络更新放在 GPU（--device cuda），batch 越大越划算；
    * 采样阶段的前向很轻（2.5 万参数的小 MLP），留在 CPU 反而更快，
      避免 worker 进程各自建 CUDA context 抢显存。

安装 CUDA 版 PyTorch（RTX 40 系 + 驱动 560.x 对应 cu126）：
    uv pip install --reinstall torch --index-url https://download.pytorch.org/whl/cu126
然后：
    python py/train.py --device cuda --workers 8 --envs 4

训练结束后（以及每次刷新最佳成绩时）会导出 py/policy.bin，
用 OpenGL 窗口回放：build/msvc/bin/Release/window.exe py/policy.bin
"""

from __future__ import annotations

import argparse
import ctypes
import io
import multiprocessing as mp
import struct
from concurrent.futures import ProcessPoolExecutor
from glob import glob
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
MAGIC = 0x504C4D31  # 'P','L','M','1'，必须与 src/policy.cpp 一致
LOG_STD_TAG = 0x4C535444  # 'LSTD'：策略文件尾部的探索噪声参数标记

# worker 进程内的缓存（进程池会复用进程，避免重复建 env / 网络）
_WORKER_ENVS: dict[int, "VectorRagdollEnv"] = {}
_WORKER_NETS: dict[int, "ActorCritic"] = {}


# --------------------------------------------------------------------------- 环境
def find_library() -> Path:
    """在 build 目录里找 rl_env 动态库（优先 Release）。"""
    patterns = ["build/**/rl_env.dll", "build/**/librl_env.so", "build/**/librl_env.dylib"]
    for pattern in patterns:
        found = sorted(glob(str(ROOT / pattern), recursive=True))
        if found:
            found.sort(key=lambda path: "elease" not in path)  # Release 排在前面
            return Path(found[0])
    raise FileNotFoundError(
        "找不到 rl_env 动态库，请先构建：\n"
        "  cmake --workflow --preset release"
    )


class RagdollEnv:
    """对 C ABI 的薄封装，接口风格类似 gym。"""

    def __init__(self, library: Path | str, seed: int = 0) -> None:
        self.lib = ctypes.CDLL(str(library))

        float_p = ctypes.POINTER(ctypes.c_float)
        int_p = ctypes.POINTER(ctypes.c_int)

        self.lib.env_obs_size.restype = ctypes.c_int
        self.lib.env_act_size.restype = ctypes.c_int
        self.lib.env_max_episode_steps.restype = ctypes.c_int
        self.lib.env_create.restype = ctypes.c_void_p

        self.lib.env_destroy.argtypes = [ctypes.c_void_p]
        self.lib.env_set_seed.argtypes = [ctypes.c_void_p, ctypes.c_uint]
        self.lib.env_reset.argtypes = [ctypes.c_void_p, float_p]
        self.lib.env_step.argtypes = [ctypes.c_void_p, float_p, float_p, float_p, int_p]

        self.obs_size = int(self.lib.env_obs_size())
        self.act_size = int(self.lib.env_act_size())
        self.max_episode_steps = int(self.lib.env_max_episode_steps())

        self.handle = self.lib.env_create()
        if not self.handle:
            raise RuntimeError("env_create 返回空指针")
        self.lib.env_set_seed(self.handle, ctypes.c_uint(seed))

        self._obs = np.zeros(self.obs_size, dtype=np.float32)
        self._act = np.zeros(self.act_size, dtype=np.float32)
        self._reward = ctypes.c_float(0.0)
        self._done = ctypes.c_int(1)
        self._obs_p = self._obs.ctypes.data_as(float_p)
        self._act_p = self._act.ctypes.data_as(float_p)

    def reset(self) -> np.ndarray:
        self.lib.env_reset(self.handle, self._obs_p)
        return self._obs.copy()

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool]:
        self._act[:] = action
        self.lib.env_step(
            self.handle,
            self._act_p,
            self._obs_p,
            ctypes.byref(self._reward),
            ctypes.byref(self._done),
        )
        return self._obs.copy(), float(self._reward.value), bool(self._done.value)

    def close(self) -> None:
        if self.handle:
            self.lib.env_destroy(self.handle)
            self.handle = None


class VectorRagdollEnv:
    """把 M 个独立实例打包成向量化环境（同一进程内顺序推进物理）。"""

    def __init__(self, library: Path | str, count: int, seed: int = 0) -> None:
        self.envs = [RagdollEnv(library, seed=seed + index) for index in range(count)]
        self.count = count
        self.obs_size = self.envs[0].obs_size
        self.act_size = self.envs[0].act_size
        self.max_episode_steps = self.envs[0].max_episode_steps
        self.observations = np.stack([env.reset() for env in self.envs])

    def step(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rewards = np.zeros(self.count, dtype=np.float32)
        dones = np.zeros(self.count, dtype=np.float32)
        for index, env in enumerate(self.envs):
            observation, reward, done = env.step(actions[index])
            rewards[index] = reward
            dones[index] = 1.0 if done else 0.0
            # 回合结束后立刻自动重置（auto-reset）
            self.observations[index] = env.reset() if done else observation
        return self.observations, rewards, dones

    def close(self) -> None:
        for env in self.envs:
            env.close()


class RunningNorm:
    """观测归一化（Welford 增量式），导出时把 mean/std 一起写进策略文件。"""

    def __init__(self, size: int, clip: float = 5.0) -> None:
        self.mean = np.zeros(size, dtype=np.float64)
        self.var = np.ones(size, dtype=np.float64)
        self.count = 1e-4
        self.clip = clip

    def update(self, batch: np.ndarray) -> None:
        batch = batch.astype(np.float64)
        batch_count = batch.shape[0]
        if batch_count == 0:
            return
        batch_mean = batch.mean(axis=0)
        batch_var = batch.var(axis=0)
        delta = batch_mean - self.mean
        total = self.count + batch_count
        self.mean = self.mean + delta * batch_count / total
        m2 = self.var * self.count + batch_var * batch_count + delta**2 * self.count * batch_count / total
        self.var = m2 / total
        self.count = total

    @property
    def std(self) -> np.ndarray:
        return np.sqrt(self.var + 1e-8)

    def __call__(self, observation: np.ndarray) -> np.ndarray:
        scaled = (observation - self.mean) / self.std
        return np.clip(scaled, -self.clip, self.clip).astype(np.float32)


# --------------------------------------------------------------------------- 网络
class ActorCritic(nn.Module):
    """actor 输出均值（未 squash），采样时用 tanh 压缩到 [-1, 1]。"""

    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 128) -> None:
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, act_dim),
        )
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        # 初值取 -0.8（std≈0.45），比原来的 0.3 探索更强，避免过早收敛到退化解
        self.log_std = nn.Parameter(torch.full((act_dim,), -0.8))

    def distribution(self, obs: torch.Tensor) -> torch.distributions.Normal:
        mean = self.actor(obs)
        std = self.log_std.clamp(-3.0, 0.5).exp().expand_as(mean)
        return torch.distributions.Normal(mean, std)

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(obs).squeeze(-1)


def squashed_log_prob(
    dist: torch.distributions.Normal, u: torch.Tensor, action: torch.Tensor
) -> torch.Tensor:
    """tanh 压缩后的 log 概率密度（含雅可比修正）。"""
    log_prob = dist.log_prob(u)
    log_prob -= torch.log(1.0 - action.pow(2) + 1e-6)
    return log_prob.sum(dim=-1)


# --------------------------------------------------------------------------- 导出
def export_policy(net: ActorCritic, mean: np.ndarray, std: np.ndarray, path: Path) -> None:
    """把 actor 权重量化成 C++ 可直接读取的二进制（每层 Linear + tanh）。"""
    linears = [module for module in net.actor if isinstance(module, nn.Linear)]
    mean = np.asarray(mean, dtype=np.float32)
    std = np.asarray(std, dtype=np.float32)
    obs_dim = int(mean.shape[0])
    act_dim = int(linears[-1].out_features)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as stream:
        stream.write(struct.pack("<iiii", MAGIC, obs_dim, act_dim, len(linears)))
        stream.write(mean.tobytes())
        stream.write(std.tobytes())
        for linear in linears:
            weight = linear.weight.detach().cpu().numpy().astype(np.float32)
            bias = linear.bias.detach().cpu().numpy().astype(np.float32)
            stream.write(struct.pack("<ii", linear.in_features, linear.out_features))
            stream.write(weight.tobytes())  # 行优先 [out, in]
            stream.write(bias.tobytes())
        # 追加探索噪声参数，供 --resume 恢复训练后期的噪声水平。
        # C++ 侧读完层参数就返回，尾部字节会被忽略，格式保持兼容。
        stream.write(struct.pack("<i", LOG_STD_TAG))
        stream.write(net.log_std.detach().cpu().numpy().astype(np.float32).tobytes())


def load_policy_into(net: "ActorCritic", path: Path) -> tuple[np.ndarray, np.ndarray]:
    """把导出的 policy.bin 读回 actor 权重，用于 --resume 续训。

    返回 (mean, std) 以便恢复观测归一化参数；value 网络信息不在文件里，
    会重新训练（通常几十轮就能追上）。
    """
    with open(path, "rb") as stream:
        header = stream.read(16)
        if len(header) != 16:
            raise ValueError(f"策略文件过短: {path}")
        magic, obs_dim, act_dim, layer_count = struct.unpack("<iiii", header)
        if magic != MAGIC:
            raise ValueError(f"策略文件格式不匹配: {path}")

        mean = np.frombuffer(stream.read(4 * obs_dim), dtype=np.float32).copy()
        std = np.frombuffer(stream.read(4 * obs_dim), dtype=np.float32).copy()

        linears = [module for module in net.actor if isinstance(module, nn.Linear)]
        if len(linears) != layer_count:
            raise ValueError("策略文件层数与网络结构不一致")
        for linear in linears:
            in_dim, out_dim = struct.unpack("<ii", stream.read(8))
            if (in_dim, out_dim) != (linear.in_features, linear.out_features):
                raise ValueError("策略文件层维度与网络不一致")
            weight = np.frombuffer(stream.read(4 * in_dim * out_dim), dtype=np.float32).copy()
            bias = np.frombuffer(stream.read(4 * out_dim), dtype=np.float32).copy()
            with torch.no_grad():
                linear.weight.copy_(torch.as_tensor(weight.reshape(out_dim, in_dim)))
                linear.bias.copy_(torch.as_tensor(bias))

        # 可选的尾部：探索噪声参数（老文件没有这一段）
        log_std: np.ndarray | None = None
        tag = stream.read(4)
        if len(tag) == 4 and struct.unpack("<i", tag)[0] == LOG_STD_TAG:
            raw = stream.read(4 * act_dim)
            if len(raw) == 4 * act_dim:
                log_std = np.frombuffer(raw, dtype=np.float32).copy()

    return mean.astype(np.float64), std.astype(np.float64), log_std


def evaluate_policy(env: "RagdollEnv", net: "ActorCritic", mean: np.ndarray,
                    std: np.ndarray, episodes: int) -> tuple[float, float]:
    """用确定性策略（tanh(mean)）跑若干回合，返回 (平均回报, 平均存活步数)。"""
    device = next(net.parameters()).device
    was_training = net.training
    net.eval()

    returns: list[float] = []
    lengths: list[float] = []
    observation = env.reset()
    for _ in range(max(1, episodes)):
        episode_return = 0.0
        steps = 0
        done = False
        while not done:
            obs_t = torch.as_tensor(_normalize(observation, mean, std), device=device).unsqueeze(0)
            with torch.no_grad():
                action = torch.tanh(net.actor(obs_t)).squeeze(0).cpu().numpy().astype(np.float32)
            observation, reward, done = env.step(action)
            episode_return += reward
            steps += 1
            if done:
                observation = env.reset()
        returns.append(episode_return)
        lengths.append(float(steps))

    net.train(was_training)
    return float(np.mean(returns)), float(np.mean(lengths))


# --------------------------------------------------------------------------- worker
def _normalize(observations: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    scaled = (observations - mean) / std
    return np.clip(scaled, -5.0, 5.0).astype(np.float32)


def worker_rollout(payload: dict) -> dict:
    """在 worker 进程里用给定权重采样 horizon 步。

    单进程模式也复用这个函数（workers=1 时不启进程池）。
    """
    torch.set_num_threads(1)

    worker_id = payload["worker_id"]
    env_count = payload["envs"]
    horizon = payload["horizon"]
    obs_size = payload["obs_size"]
    act_size = payload["act_size"]

    env = _WORKER_ENVS.get(worker_id)
    if env is None:
        env = VectorRagdollEnv(payload["library"], env_count, seed=payload["seed"])
        _WORKER_ENVS[worker_id] = env

    net = _WORKER_NETS.get(worker_id)
    if net is None:
        net = ActorCritic(obs_size, act_size, payload["hidden"])
        net.eval()
        _WORKER_NETS[worker_id] = net
    net.load_state_dict(torch.load(io.BytesIO(payload["state"]), map_location="cpu", weights_only=True))

    mean = payload["mean"]
    std = payload["std"]

    obs_buf = np.zeros((horizon, env_count, obs_size), dtype=np.float32)
    act_buf = np.zeros((horizon, env_count, act_size), dtype=np.float32)
    logp_buf = np.zeros((horizon, env_count), dtype=np.float32)
    rew_buf = np.zeros((horizon, env_count), dtype=np.float32)
    done_buf = np.zeros((horizon, env_count), dtype=np.float32)
    val_buf = np.zeros((horizon, env_count), dtype=np.float32)

    episode_returns = np.zeros(env_count, dtype=np.float64)
    episode_lengths = np.zeros(env_count, dtype=np.float64)
    finished_returns: list[float] = []
    finished_lengths: list[float] = []

    observations = env.observations
    with torch.no_grad():
        for step in range(horizon):
            obs_t = torch.as_tensor(_normalize(observations, mean, std))
            dist = net.distribution(obs_t)
            u = dist.sample()
            action_t = torch.tanh(u)
            log_prob = squashed_log_prob(dist, u, action_t)
            value = net.value(obs_t)

            action = action_t.numpy().astype(np.float32)
            obs_buf[step] = observations
            act_buf[step] = action
            logp_buf[step] = log_prob.numpy()
            val_buf[step] = value.numpy()

            observations, rewards, dones = env.step(action)
            rew_buf[step] = rewards
            done_buf[step] = dones

            episode_returns += rewards
            episode_lengths += 1.0
            for index in np.nonzero(dones)[0]:
                finished_returns.append(float(episode_returns[index]))
                finished_lengths.append(float(episode_lengths[index]))
                episode_returns[index] = 0.0
                episode_lengths[index] = 0.0

        last_value = net.value(
            torch.as_tensor(_normalize(observations, mean, std))
        ).numpy()

    return {
        "obs": obs_buf,
        "act": act_buf,
        "logp": logp_buf,
        "rew": rew_buf,
        "done": done_buf,
        "val": val_buf,
        "last_value": last_value,
        "returns": finished_returns,
        "lengths": finished_lengths,
        "steps": horizon * env_count,
    }


# --------------------------------------------------------------------------- 训练
def train(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit(
            "CUDA 不可用。当前 torch 可能是 CPU 版，请安装 CUDA 版：\n"
            "  uv pip install --reinstall torch --index-url https://download.pytorch.org/whl/cu126"
        )

    library = Path(args.lib) if args.lib else find_library()
    probe = RagdollEnv(library, seed=args.seed)
    obs_size = probe.obs_size
    act_size = probe.act_size
    max_episode_steps = probe.max_episode_steps
    probe.close()

    if args.eval_only:
        eval_path = Path(args.eval_only)
        if not eval_path.is_absolute():
            eval_path = ROOT / eval_path
        eval_net = ActorCritic(obs_size, act_size, args.hidden).to(device)
        try:
            eval_mean, eval_std, _ = load_policy_into(eval_net, eval_path)
        except (OSError, ValueError) as error:
            raise SystemExit(f"评估失败: {error}") from error
        eval_env = RagdollEnv(library, seed=args.seed)
        mean_return, mean_length = evaluate_policy(
            eval_env, eval_net, eval_mean, eval_std, args.eval_episodes
        )
        eval_env.close()
        print(
            f"评估 {eval_path.name}: 平均回报 {mean_return:.2f}，"
            f"平均存活 {mean_length:.1f} 步（{args.eval_episodes} 个回合，确定性策略）"
        )
        return

    workers = max(1, args.workers)
    parallel = workers * args.envs
    print(f"物理库: {library}")
    print(f"网络设备: {device}")
    print(
        f"观测维度: {obs_size}  动作维度: {act_size}  单回合最多 {max_episode_steps} 步\n"
        f"并行采样: {workers} 个进程 × {args.envs} 个角色 = {parallel} 个角色同时训练"
    )

    norm = RunningNorm(obs_size)
    net = ActorCritic(obs_size, act_size, args.hidden).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, eps=1e-5)

    resume_baseline: float | None = None
    baseline_snapshot: tuple[dict, np.ndarray, np.ndarray] | None = None
    warmup = args.warmup if args.warmup is not None else (3 if args.resume else 0)

    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.is_absolute():
            resume_path = ROOT / resume_path
        try:
            mean, std, log_std = load_policy_into(net, resume_path)
        except (OSError, ValueError) as error:
            raise SystemExit(f"续训失败: {error}") from error
        norm.mean = mean
        norm.var = np.maximum(std, 1e-6) ** 2
        # 给一个较大的伪计数，避免新采样的前几批把统计量冲掉
        norm.count = 1e4
        if log_std is not None:
            with torch.no_grad():
                net.log_std.copy_(torch.as_tensor(log_std, device=device))
            print(f"续训: 已载入 {resume_path.name} 的 actor、归一化参数与探索噪声")
        else:
            # 旧格式文件没有探索噪声字段：训练后期噪声通常已经很小，
            # 用 -1.5（std≈0.22）比默认 -0.8 更接近真实水平，避免一续训就被噪声打崩
            with torch.no_grad():
                net.log_std.fill_(-1.5)
            print(f"续训: {resume_path.name} 是旧格式（无探索噪声字段），用 log_std=-1.5 兜底")

        # 先评估一次基线：续训后如果表现还不如原策略，就不应该覆盖原文件
        baseline_env = RagdollEnv(library, seed=args.seed + 991)
        resume_baseline, baseline_length = evaluate_policy(
            baseline_env, net, norm.mean, norm.std, args.eval_episodes
        )
        baseline_env.close()
        baseline_snapshot = (
            {key: value.detach().clone() for key, value in net.state_dict().items()},
            norm.mean.copy(),
            norm.std.copy(),
        )
        print(
            f"续训基线: 平均回报 {resume_baseline:.2f} / 平均存活 {baseline_length:.1f} 步"
            f"（达不到这个水平不会覆盖 {Path(args.out).name}）"
        )

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = ROOT / out_path
    print(
        f"实时观察训练进展：另开一个终端执行\n"
        f"  build\\msvc\\bin\\Release\\window.exe --watch {out_path.relative_to(ROOT)}\n"
        f"（窗口会自动热重载 policy.bin，摔倒后自动复位）"
    )

    log_stream = None
    if args.log:
        log_path = Path(args.log)
        if not log_path.is_absolute():
            log_path = ROOT / log_path
        log_path.parent.mkdir(parents=True, exist_ok=True)
        starts_empty = not log_path.exists() or log_path.stat().st_size == 0
        log_stream = open(log_path, "a", encoding="utf-8")
        if starts_empty:
            log_stream.write("iteration,total_steps,mean_return,mean_length,policy_loss,value_loss\n")
        print(f"指标日志: {log_path}")

    executor: ProcessPoolExecutor | None = None
    if workers > 1:
        executor = ProcessPoolExecutor(
            max_workers=workers,
            mp_context=mp.get_context("spawn"),
        )

    total_steps = 0
    iteration = 0
    recent_returns: list[float] = []
    recent_lengths: list[float] = []
    best_return = resume_baseline if resume_baseline is not None else -float("inf")
    # 存一份历史最佳快照，避免最后一轮退化把好策略覆盖掉
    best_snapshot: tuple[dict, np.ndarray, np.ndarray] | None = None
    horizon = args.rollout

    try:
        while total_steps < args.steps:
            iteration += 1

            buffer = io.BytesIO()
            torch.save(
                {key: value.detach().cpu() for key, value in net.state_dict().items()},
                buffer,
            )
            state = buffer.getvalue()
            mean = norm.mean.astype(np.float32)
            std = norm.std.astype(np.float32)

            payloads = [
                {
                    "worker_id": index,
                    "library": str(library),
                    "envs": args.envs,
                    "seed": args.seed * 1000 + index * 97,
                    "horizon": horizon,
                    "obs_size": obs_size,
                    "act_size": act_size,
                    "hidden": args.hidden,
                    "state": state,
                    "mean": mean,
                    "std": std,
                }
                for index in range(workers)
            ]

            if executor is None:
                results = [worker_rollout(payloads[0])]
            else:
                results = list(executor.map(worker_rollout, payloads))

            total_steps += sum(result["steps"] for result in results)
            for result in results:
                recent_returns.extend(result["returns"])
                recent_lengths.extend(result["lengths"])
                del result["returns"], result["lengths"], result["steps"]

            obs_all = np.stack([result["obs"] for result in results])
            act_all = np.stack([result["act"] for result in results])
            logp_all = np.stack([result["logp"] for result in results])
            rew_all = np.stack([result["rew"] for result in results])
            done_all = np.stack([result["done"] for result in results])
            val_all = np.stack([result["val"] for result in results])

            norm.update(obs_all.reshape(-1, obs_size))

            # 每个 worker 的每个角色单独做 GAE，done 处切断（auto-reset 后不回传价值）
            advantages = np.zeros_like(rew_all)
            for index in range(workers):
                running = np.zeros(args.envs, dtype=np.float32)
                last_value = results[index]["last_value"]
                for step in reversed(range(horizon)):
                    next_value = last_value if step == horizon - 1 else val_all[index][step + 1]
                    nonterminal = 1.0 - done_all[index][step]
                    delta = rew_all[index][step] + args.gamma * next_value * nonterminal - val_all[index][step]
                    running = delta + args.gamma * args.lam * nonterminal * running
                    advantages[index][step] = running
            returns_all = advantages + val_all

            obs_tensor = torch.as_tensor(
                _normalize(obs_all.reshape(-1, obs_size), norm.mean, norm.std), device=device
            )
            action_tensor = torch.as_tensor(act_all.reshape(-1, act_size), device=device)
            logp_tensor = torch.as_tensor(logp_all.reshape(-1), device=device)
            advantage_tensor = torch.as_tensor(advantages.reshape(-1), device=device)
            advantage_tensor = (advantage_tensor - advantage_tensor.mean()) / (
                advantage_tensor.std() + 1e-8
            )
            return_tensor = torch.as_tensor(returns_all.reshape(-1), device=device)

            batch_size = obs_tensor.shape[0]
            policy_loss = torch.zeros(())
            value_loss = torch.zeros(())
            for _ in range(args.epochs):
                order = torch.randperm(batch_size, device=device)
                for start in range(0, batch_size, args.minibatch):
                    index = order[start:start + args.minibatch]
                    dist = net.distribution(obs_tensor[index])
                    action_mini = action_tensor[index].clamp(-0.999, 0.999)
                    u_mini = torch.atanh(action_mini)
                    log_prob = squashed_log_prob(dist, u_mini, action_mini)
                    ratio = (log_prob - logp_tensor[index]).exp()

                    surrogate_1 = ratio * advantage_tensor[index]
                    surrogate_2 = ratio.clamp(
                        1.0 - args.clip, 1.0 + args.clip
                    ) * advantage_tensor[index]
                    policy_loss = -torch.min(surrogate_1, surrogate_2).mean()
                    value_loss = (net.value(obs_tensor[index]) - return_tensor[index]).pow(2).mean()
                    entropy = dist.entropy().sum(dim=-1).mean()

                    if iteration <= warmup:
                        # 续训前几轮只让 critic 追上真实回报，
                        # 否则新初始化的 critic 会给出错误的 advantage，把好策略更新坏
                        loss = 0.5 * value_loss
                    else:
                        loss = policy_loss + 0.5 * value_loss - args.entropy * entropy
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(net.parameters(), 0.5)
                    optimizer.step()

            mean_return = float(np.mean(recent_returns[-20:])) if recent_returns else 0.0
            mean_length = float(np.mean(recent_lengths[-20:])) if recent_lengths else 0.0
            progress = min(total_steps / max(args.steps, 1), 1.0)
            print(
                f"[{iteration:4d}] 步数 {total_steps:8d} ({progress:5.1%}) | 平均回报 {mean_return:8.2f} "
                f"| 平均存活 {mean_length:6.1f} 步 | 策略损失 {float(policy_loss.detach()):7.3f} "
                f"| 价值损失 {float(value_loss.detach()):8.3f} | batch {batch_size}"
            )
            if log_stream is not None:
                log_stream.write(
                    f"{iteration},{total_steps},{mean_return:.4f},{mean_length:.2f},"
                    f"{float(policy_loss.detach()):.6f},{float(value_loss.detach()):.6f}\n"
                )
                log_stream.flush()

            if recent_returns and mean_return > best_return:
                best_return = mean_return
                best_snapshot = (
                    {key: value.detach().clone() for key, value in net.state_dict().items()},
                    norm.mean.copy(),
                    norm.std.copy(),
                )
                export_policy(net, norm.mean, norm.std, out_path)

    except KeyboardInterrupt:
        print("\n训练被中断，导出当前策略……")
    finally:
        if executor is not None:
            executor.shutdown(cancel_futures=True)

    if log_stream is not None:
        log_stream.close()

    if best_snapshot is not None:
        # 导出的永远是历史最佳（按平均回报），而不是最后一轮
        net.load_state_dict(best_snapshot[0])
        export_policy(net, best_snapshot[1], best_snapshot[2], out_path)
        print(f"已导出历史最佳策略（平均回报 {best_return:.2f}）")
    elif baseline_snapshot is not None:
        # 续训没超过原策略：保留原权重，别用更差的结果覆盖它
        net.load_state_dict(baseline_snapshot[0])
        export_policy(net, baseline_snapshot[1], baseline_snapshot[2], out_path)
        print(f"本次未超过续训基线（{resume_baseline:.2f}），保留原策略权重")
    else:
        export_policy(net, norm.mean, norm.std, out_path)
    print(f"策略已导出: {out_path}")
    print(f"回放: build\\msvc\\bin\\Release\\window.exe --watch {out_path.relative_to(ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="PPO 训练布娃娃站立")
    parser.add_argument("--steps", type=int, default=3_000_000,
                        help="总交互步数（所有角色加起来）；12 个角色并行时 300 万步约 20~40 分钟")
    parser.add_argument("--workers", type=int, default=1, help="并行进程数，建议设为 CPU 物理核数")
    parser.add_argument("--envs", type=int, default=8, help="每个进程内的角色数量")
    parser.add_argument("--rollout", type=int, default=512, help="每个角色每轮采样步数")
    parser.add_argument("--epochs", type=int, default=10, help="每轮 PPO 更新遍数")
    parser.add_argument("--minibatch", type=int, default=256)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lam", type=float, default=0.95)
    parser.add_argument("--clip", type=float, default=0.2)
    parser.add_argument("--entropy", type=float, default=0.005, help="熵系数，越大探索越强，防局部最优")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--device", type=str, default="auto", choices=["auto", "cpu", "cuda"],
        help="网络计算设备；auto 表示有 CUDA 就用 CUDA（需要装 CUDA 版 torch）",
    )
    parser.add_argument("--resume", type=str, default=None,
                        help="从已有的 policy.bin 继续训练（恢复 actor、归一化参数与探索噪声）")
    parser.add_argument("--warmup", type=int, default=None,
                        help="续训时前 N 轮只训练 critic（默认续训为 3，普通训练为 0）")
    parser.add_argument("--eval-only", type=str, default=None,
                        help="只评估指定的 policy.bin（确定性策略），不训练")
    parser.add_argument("--eval-episodes", type=int, default=5,
                        help="评估与续训基线使用的回合数")
    parser.add_argument("--lib", type=str, default=None, help="手动指定 rl_env 动态库路径")
    parser.add_argument("--out", type=str, default="py/policy.bin", help="策略导出路径")
    parser.add_argument("--log", type=str, default="py/runs/metrics.csv",
                        help="每轮指标 CSV 路径，传空字符串可关闭")
    train(parser.parse_args())


if __name__ == "__main__":
    main()
