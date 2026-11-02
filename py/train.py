"""用 PPO 训练 Box2D 布娃娃「站起来」。

C++ 侧只保留物理（无渲染）并编译成 rl_env 动态库，Python 通过 ctypes 调用，
所以 Python 这边只负责神经网络、采样与优化器。

用法：
    python py/train.py                          # 默认 60 万步
    python py/train.py --steps 200000           # 少跑一点，先看效果
    python py/train.py --out py/policy.bin      # 指定导出路径

训练结束后（以及每次刷新最佳成绩时）会导出 py/policy.bin，
用 OpenGL 窗口回放：build/msvc/bin/Release/window.exe py/policy.bin
"""

from __future__ import annotations

import argparse
import ctypes
import struct
from glob import glob
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
MAGIC = 0x504C4D31  # 'P','L','M','1'，必须与 src/policy.cpp 一致


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

    def __init__(self, library: Path, seed: int = 0) -> None:
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

    def __enter__(self) -> "RagdollEnv":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


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
        self.log_std = nn.Parameter(torch.full((act_dim,), -1.2))

    def distribution(self, obs: torch.Tensor) -> torch.distributions.Normal:
        mean = self.actor(obs)
        std = self.log_std.clamp(-3.0, 0.5).exp().expand_as(mean)
        return torch.distributions.Normal(mean, std)

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(obs).squeeze(-1)


def squashed_log_prob(dist: torch.distributions.Normal, u: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    """tanh 压缩后的 log 概率密度（含雅可比修正）。"""
    log_prob = dist.log_prob(u)
    log_prob -= torch.log(1.0 - action.pow(2) + 1e-6)
    return log_prob.sum(dim=-1)


# --------------------------------------------------------------------------- 导出
def export_policy(net: ActorCritic, norm: RunningNorm, obs_dim: int, act_dim: int, path: Path) -> None:
    """把 actor 权重量化成 C++ 可直接读取的二进制（每层 Linear + tanh）。"""
    linears = [module for module in net.actor if isinstance(module, nn.Linear)]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as stream:
        stream.write(struct.pack("<iiii", MAGIC, obs_dim, act_dim, len(linears)))
        stream.write(norm.mean.astype(np.float32).tobytes())
        stream.write(norm.std.astype(np.float32).tobytes())
        for linear in linears:
            weight = linear.weight.detach().cpu().numpy().astype(np.float32)
            bias = linear.bias.detach().cpu().numpy().astype(np.float32)
            stream.write(struct.pack("<ii", linear.in_features, linear.out_features))
            stream.write(weight.tobytes())  # 行优先 [out, in]
            stream.write(bias.tobytes())


# --------------------------------------------------------------------------- 训练
def train(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(1)

    library = Path(args.lib) if args.lib else find_library()
    print(f"物理库: {library}")

    env = RagdollEnv(library, seed=args.seed)
    print(f"观测维度: {env.obs_size}  动作维度: {env.act_size}  单回合最多 {env.max_episode_steps} 步")

    norm = RunningNorm(env.obs_size)
    net = ActorCritic(env.obs_size, env.act_size, args.hidden)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, eps=1e-5)

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = ROOT / out_path

    observation = env.reset()
    total_steps = 0
    iteration = 0
    episode_return = 0.0
    episode_length = 0
    recent_returns: list[float] = []
    recent_lengths: list[float] = []
    best_return = -float("inf")

    try:
        while total_steps < args.steps:
            iteration += 1
            obs_buf, act_buf, logp_buf = [], [], []
            rew_buf, done_buf, val_buf = [], [], []

            for _ in range(args.rollout):
                obs_t = torch.as_tensor(norm(observation)).unsqueeze(0)
                with torch.no_grad():
                    dist = net.distribution(obs_t)
                    u = dist.sample()
                    action_t = torch.tanh(u)
                    log_prob = squashed_log_prob(dist, u, action_t)
                    value = net.value(obs_t)

                action = action_t.squeeze(0).numpy().astype(np.float32)
                next_observation, reward, done = env.step(action)

                obs_buf.append(observation)
                act_buf.append(action)
                logp_buf.append(float(log_prob.item()))
                rew_buf.append(reward)
                done_buf.append(1.0 if done else 0.0)
                val_buf.append(float(value.item()))

                episode_return += reward
                episode_length += 1
                total_steps += 1

                if done:
                    recent_returns.append(episode_return)
                    recent_lengths.append(float(episode_length))
                    episode_return = 0.0
                    episode_length = 0
                    observation = env.reset()
                else:
                    observation = next_observation

            obs_raw = np.asarray(obs_buf, dtype=np.float32)
            norm.update(obs_raw)

            with torch.no_grad():
                last_value = float(
                    net.value(torch.as_tensor(norm(observation)).unsqueeze(0)).item()
                )

            advantages = np.zeros(len(rew_buf), dtype=np.float32)
            running = 0.0
            for index in reversed(range(len(rew_buf))):
                next_value = last_value if index == len(rew_buf) - 1 else val_buf[index + 1]
                nonterminal = 1.0 - done_buf[index]
                delta = rew_buf[index] + args.gamma * next_value * nonterminal - val_buf[index]
                running = delta + args.gamma * args.lam * nonterminal * running
                advantages[index] = running
            returns = advantages + np.asarray(val_buf, dtype=np.float32)

            obs_tensor = torch.as_tensor(norm(obs_raw))
            action_tensor = torch.as_tensor(np.asarray(act_buf, dtype=np.float32))
            logp_tensor = torch.as_tensor(np.asarray(logp_buf, dtype=np.float32))
            advantage_tensor = torch.as_tensor(advantages)
            advantage_tensor = (advantage_tensor - advantage_tensor.mean()) / (advantage_tensor.std() + 1e-8)
            return_tensor = torch.as_tensor(returns)

            batch_size = obs_tensor.shape[0]
            for _ in range(args.epochs):
                order = torch.randperm(batch_size)
                for start in range(0, batch_size, args.minibatch):
                    index = order[start:start + args.minibatch]
                    dist = net.distribution(obs_tensor[index])
                    action_mini = action_tensor[index].clamp(-0.999, 0.999)
                    u_mini = torch.atanh(action_mini)
                    log_prob = squashed_log_prob(dist, u_mini, action_mini)
                    ratio = (log_prob - logp_tensor[index]).exp()

                    surrogate_1 = ratio * advantage_tensor[index]
                    surrogate_2 = ratio.clamp(1.0 - args.clip, 1.0 + args.clip) * advantage_tensor[index]
                    policy_loss = -torch.min(surrogate_1, surrogate_2).mean()
                    value_loss = (net.value(obs_tensor[index]) - return_tensor[index]).pow(2).mean()
                    entropy = dist.entropy().sum(dim=-1).mean()

                    loss = policy_loss + 0.5 * value_loss - args.entropy * entropy
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(net.parameters(), 0.5)
                    optimizer.step()

            mean_return = float(np.mean(recent_returns[-10:])) if recent_returns else 0.0
            mean_length = float(np.mean(recent_lengths[-10:])) if recent_lengths else 0.0
            print(
                f"[{iteration:4d}] 步数 {total_steps:7d} | 平均回报 {mean_return:8.2f} "
                f"| 平均存活 {mean_length:6.1f} 步 | 策略损失 {float(policy_loss):7.3f} "
                f"| 价值损失 {float(value_loss):8.3f}"
            )

            if recent_returns and mean_return > best_return:
                best_return = mean_return
                export_policy(net, norm, env.obs_size, env.act_size, out_path)

    except KeyboardInterrupt:
        print("\n训练被中断，导出当前策略……")

    export_policy(net, norm, env.obs_size, env.act_size, out_path)
    print(f"策略已导出: {out_path}")
    print(f"回放: build\\msvc\\bin\\Release\\window.exe {out_path.relative_to(ROOT)}")
    env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="PPO 训练布娃娃站立")
    parser.add_argument("--steps", type=int, default=600_000, help="总交互步数")
    parser.add_argument("--rollout", type=int, default=2048, help="每轮采样步数")
    parser.add_argument("--epochs", type=int, default=10, help="每轮 PPO 更新遍数")
    parser.add_argument("--minibatch", type=int, default=256)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lam", type=float, default=0.95)
    parser.add_argument("--clip", type=float, default=0.2)
    parser.add_argument("--entropy", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lib", type=str, default=None, help="手动指定 rl_env 动态库路径")
    parser.add_argument("--out", type=str, default="py/policy.bin", help="策略导出路径")
    train(parser.parse_args())


if __name__ == "__main__":
    main()
