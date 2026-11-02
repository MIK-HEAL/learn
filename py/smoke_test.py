"""只用标准库验证 C++ 环境与 Python 的桥接是否正常。

    python py/smoke_test.py

不需要 numpy / torch，适合在装依赖前先确认 rl_env.dll 能加载。
"""

from __future__ import annotations

import ctypes
import random
from glob import glob
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def find_library() -> Path:
    found = sorted(glob(str(ROOT / "build/**/rl_env.dll"), recursive=True))
    found += sorted(glob(str(ROOT / "build/**/librl_env.so"), recursive=True))
    if not found:
        raise SystemExit("找不到 rl_env 动态库，请先执行: cmake --workflow --preset release")
    found.sort(key=lambda path: "elease" not in path)
    return Path(found[0])


def main() -> None:
    path = find_library()
    print(f"加载: {path}")
    lib = ctypes.CDLL(str(path))

    float_p = ctypes.POINTER(ctypes.c_float)
    int_p = ctypes.POINTER(ctypes.c_int)
    lib.env_obs_size.restype = ctypes.c_int
    lib.env_act_size.restype = ctypes.c_int
    lib.env_max_episode_steps.restype = ctypes.c_int
    lib.env_create.restype = ctypes.c_void_p
    lib.env_destroy.argtypes = [ctypes.c_void_p]
    lib.env_reset.argtypes = [ctypes.c_void_p, float_p]
    lib.env_step.argtypes = [ctypes.c_void_p, float_p, float_p, float_p, int_p]

    obs_size = lib.env_obs_size()
    act_size = lib.env_act_size()
    limit = lib.env_max_episode_steps()
    print(f"obs={obs_size} act={act_size} max_steps={limit}")

    handle = lib.env_create()
    if not handle:
        raise SystemExit("env_create 失败")

    observation = (ctypes.c_float * obs_size)()
    actions = (ctypes.c_float * act_size)()
    reward = ctypes.c_float(0.0)
    done = ctypes.c_int(1)
    rng = random.Random(0)

    lib.env_reset(handle, observation)

    total = 0.0
    steps = 0
    episodes = 0
    while episodes < 5 and steps < 2000:
        for index in range(act_size):
            actions[index] = rng.uniform(-1.0, 1.0)
        lib.env_step(handle, actions, observation, ctypes.byref(reward), ctypes.byref(done))
        total += reward.value
        steps += 1
        if done.value:
            episodes += 1
            lib.env_reset(handle, observation)

    lib.env_destroy(handle)
    print(f"随机策略: {steps} 步, {episodes} 个回合, 平均每步奖励 {total / max(steps, 1):.3f}")
    print("接口正常: C++ 物理环境可以被 Python 正常驱动")


if __name__ == "__main__":
    main()
