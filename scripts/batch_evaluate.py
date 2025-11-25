# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Batch evaluation of RSL-RL checkpoints with the same settings as play_race."""

import sys
import os
import re
import glob
import argparse
import numpy as np
import pandas as pd
import torch

# -----------------------------------------------------------------------------
# 1) Local RSL-RL path (same as play_race)
# -----------------------------------------------------------------------------
local_rsl_path = os.path.abspath("src/third_parties/rsl_rl_local")
if os.path.exists(local_rsl_path):
    sys.path.insert(0, local_rsl_path)
    print(f"[INFO] Using local rsl_rl from: {local_rsl_path}")
else:
    print(f"[WARNING] Local rsl_rl not found at: {local_rsl_path}")

# -----------------------------------------------------------------------------
# 2) Isaac App / CLI
# -----------------------------------------------------------------------------
from isaaclab.app import AppLauncher
import cli_args

parser = argparse.ArgumentParser(description="Batch evaluate RL checkpoints with RSL-RL.")

# 自己的参数
parser.add_argument(
    "--run_dir",
    type=str,
    required=True,
    help="Path to the run directory containing checkpoints (e.g. logs/rsl_rl/quadcopter_race/2025-11-19_23-39-21).",
)
parser.add_argument(
    "--num_envs",
    type=int,
    default=None,
    help="Number of environments to simulate during evaluation. If None, use default from task cfg.",
)
parser.add_argument(
    "--task",
    type=str,
    default=None,
    help="Name of the task, e.g. Isaac-Quadcopter-Race-v0. MUST match training task.",
)
parser.add_argument(
    "--seed",
    type=int,
    default=42,
    help="Random seed for reproducibility.",
)
parser.add_argument(
    "--disable_fabric",
    action="store_true",
    default=False,
    help="Disable fabric and use USD I/O operations.",
)

# RSL-RL + AppLauncher 的 CLI（保持和 play_race 一致）
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# 批量评测不需要视频
if hasattr(args_cli, "video"):
    args_cli.video = False

# 启动 Isaac Sim
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# -----------------------------------------------------------------------------
# 3) 其余 import（必须在 SimulationApp 初始化后）
# -----------------------------------------------------------------------------
import gymnasium as gym
from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
import isaaclab_tasks  # noqa: F401  # 注册内置任务
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlVecEnvWrapper,
)

# 注册你自己的自定义任务（和 play_race 一样）
import src.isaac_quad_sim2real.tasks  # noqa: F401


# -----------------------------------------------------------------------------
# 4) 工具函数：扫描 checkpoint
# -----------------------------------------------------------------------------
def get_checkpoints(run_dir: str):
    """Recursively collect all .pt checkpoints in run_dir (excluding policy/jit).

    假设命名格式主要是：
        model_{step}_{reward}.pt
    例如：
        model_0_11.pt      -> step=0,  reward=11
        model_200_-13.pt   -> step=200, reward=-13

    其它名字（例如 best_model.pt）就按老办法兜底。
    """
    run_dir = os.path.abspath(run_dir)
    print(f"\n[INFO] Scanning checkpoints in: {run_dir}")
    search_path = os.path.join(run_dir, "**", "*.pt")
    files = glob.glob(search_path, recursive=True)

    checkpoints = []
    for f in files:
        basename = os.path.basename(f)

        # 排除导出的 policy/jit
        if "policy.pt" in basename or "jit" in basename:
            continue

        step = None
        reward = None

        # 1) 优先按 model_{step}_{reward}.pt 解析
        name_no_ext = os.path.splitext(basename)[0]  # e.g. "model_200_-13"
        parts = name_no_ext.split("_")
        if len(parts) >= 3 and parts[0] == "model":
            try:
                step = int(parts[1])
                reward = int(parts[2])
            except ValueError:
                step = None
                reward = None

        # 2) 特殊处理 best_model
        if step is None and "best_model" in basename:
            step = 999999999
            reward = None

        # 3) 兜底：从文件名里找第一个数字当 step（不一定用得到）
        if step is None:
            nums = re.findall(r"(-?\d+)", basename)
            if nums:
                try:
                    step = int(nums[0])
                except ValueError:
                    continue
            else:
                # 实在解析不出来就跳过
                continue

        checkpoints.append(
            {
                "step": step,
                "reward": reward,  # 可能是 None
                "path": f,
                "name": basename,
            }
        )

    # 按 step 排序
    checkpoints.sort(key=lambda x: x["step"])
    return checkpoints



# -----------------------------------------------------------------------------
# 5) 主逻辑
# -----------------------------------------------------------------------------
def main():
    if args_cli.task is None:
        print("[ERROR] --task must be provided (e.g., --task Isaac-Quadcopter-Race-v0).")
        simulation_app.close()
        return

    run_dir = os.path.abspath(args_cli.run_dir)
    checkpoints = get_checkpoints(run_dir)
    if not checkpoints:
        print(f"[ERROR] No checkpoints found in: {run_dir}")
        simulation_app.close()
        return

    # -------------------------------------------------------------------------
    # 5.1 配置和环境：和 play_race 完全同源
    # -------------------------------------------------------------------------
    # RSL-RL 运行配置
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    # 环境配置
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )

    # 和 play_race 相同的 eval 设置
    env_cfg.is_train = False
    if hasattr(env_cfg, "max_motor_noise_std"):
        env_cfg.max_motor_noise_std = 0.0
    env_cfg.seed = args_cli.seed

    # 创建环境（不录视频）
    env = gym.make(args_cli.task, cfg=env_cfg)

    # 多智能体 -> 单智能体（保持和训练 / play_race 一致）
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # RSL-RL 封装
    env = RslRlVecEnvWrapper(env)

    # 方便访问底层 env
    quad_env = env.unwrapped
    device = quad_env.device
    num_envs = quad_env.num_envs
    dt = quad_env.cfg.sim.dt * quad_env.cfg.decimation
    max_episode_length_s = quad_env.max_episode_length * dt

    print(f"[INFO] Task: {args_cli.task}")
    print(f"[INFO] Num envs: {num_envs}")
    print(f"[INFO] dt: {dt:.4f} s, max episode length: {max_episode_length_s:.2f} s")

    # OnPolicyRunner，和 play_race 完全一致
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)

    # -------------------------------------------------------------------------
    # 5.2 对每个 checkpoint 做一次“跑一局”的评测
    # -------------------------------------------------------------------------
    all_results = []

    print("\n" + "=" * 70)
    print(f"STARTING BATCH EVALUATION OVER {len(checkpoints)} CHECKPOINTS")
    print("=" * 70)

    for i, ckpt in enumerate(checkpoints):
        ckpt_path = ckpt["path"]
        step = ckpt["step"]
        name = ckpt["name"]

        print(f"[{i+1}/{len(checkpoints)}] {name} (step={step}) ... ", end="", flush=True)

        # 载入模型（和 play_race 一样）
        try:
            runner.load(ckpt_path)
        except RuntimeError as e:
            print(f"FAILED (shape mismatch): {e}")
            continue
        except Exception as e:
            print(f"FAILED (error): {e}")
            continue

        policy = runner.get_inference_policy(device=quad_env.device)

        # ---------------------------------------------------------------------
        # 评测逻辑：复刻 play_race 的统计方式
        # ---------------------------------------------------------------------
        # 1) reset 环境（不用返回值，避免 (obs, info) 的 tuple）
        env.reset()

        # 2) 用 get_observations 拿 obs（和 play_race 一模一样）
        obs = env.get_observations()
        if isinstance(obs, tuple):
            obs = obs[0]
        if hasattr(obs, "get"):
            obs = obs["policy"]

        # 统计量（单个 checkpoint）
        current_steps = torch.zeros(num_envs, dtype=torch.long, device=quad_env.device)
        env_has_finished = torch.zeros(num_envs, dtype=torch.bool, device=quad_env.device)

        success_times = []  # 每个成功环境的用时（秒）
        crash_count = 0
        timeout_count = 0
        total_episodes = 0

        # 保守一点，加一点冗余步数
        max_steps = quad_env.max_episode_length + 5

        for _ in range(max_steps):
            with torch.no_grad():   # <<< 关键：用 no_grad 而不是 inference_mode
                # 再保险一次：如果哪一步 obs 又变成 (obs, info)，拆开
                if isinstance(obs, tuple):
                    obs, _ = obs

                actions = policy(obs)
                obs, rewards, dones, infos = env.step(actions)

                # 累积 step 计数
                current_steps += 1

                if torch.any(dones):
                    done_indices = torch.where(dones)[0]
                    for idx in done_indices:
                        if env_has_finished[idx]:
                            current_steps[idx] = 0
                            continue

                        time_taken = current_steps[idx].item() * dt
                        total_episodes += 1
                        env_has_finished[idx] = True

                        # 和 play_race 一样：只看 reset_terminated 和 reset_time_outs
                        is_crash = quad_env.reset_terminated[idx].item()
                        is_timeout = quad_env.reset_time_outs[idx].item()

                        if is_crash:
                            crash_count += 1
                        elif is_timeout:
                            # 时间远小于最大时长 => 认为是“完赛触发的伪 timeout”，算成功
                            if time_taken < (max_episode_length_s - 0.5):
                                success_times.append(time_taken)
                            else:
                                timeout_count += 1
                        else:
                            # 理论上不该发生，保底算超时
                            timeout_count += 1

                        # 重置该 env 的 step 计数器
                        current_steps[idx] = 0

                # 和 play_race 一样，从 TensorDict 里取 policy obs
                if hasattr(obs, "get"):
                    obs = obs["policy"]

                # 所有 env 至少跑完一局就可以停
                if torch.all(env_has_finished):
                    break

        # ---------------------------------------------------------------------
        # 汇总当前 checkpoint 的结果
        # ---------------------------------------------------------------------
        if total_episodes == 0:
            total_episodes = num_envs  # 防止除零，理论上不会发生

        success_count = len(success_times)
        success_rate = success_count / total_episodes * 100.0
        crash_rate = crash_count / total_episodes * 100.0
        timeout_rate = timeout_count / total_episodes * 100.0

        avg_time = float(np.mean(success_times)) if success_times else 0.0

        print(
            f"Success: {success_rate:5.1f}% | Crash: {crash_rate:5.1f}% "
            f"| Timeout: {timeout_rate:5.1f}% | AvgLap: {avg_time:5.2f}s"
        )

        all_results.append(
            {
                "Step": step,
                "Name": name,
                "Success Rate (%)": success_rate,
                "Crash Rate (%)": crash_rate,
                "Timeout Rate (%)": timeout_rate,
                "Avg Lap Time (s)": avg_time,
            }
        )

    # -------------------------------------------------------------------------
    # 5.3 输出最终表格 & CSV
    # -------------------------------------------------------------------------
    if all_results:
        df = pd.DataFrame(all_results).sort_values("Step")

        print("\n" + "=" * 70)
        pd.set_option("display.max_rows", None)
        pd.set_option("display.float_format", "{:.2f}".format)
        print(df.to_string(index=False))
        print("=" * 70)

        csv_path = os.path.join(run_dir, "batch_evaluation_report.csv")
        df.to_csv(csv_path, index=False)
        print(f"\n[INFO] Saved CSV report to: {csv_path}")

    # 关闭环境和 Sim
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
