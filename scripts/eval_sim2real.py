# eval_sim2real_policy.py
from __future__ import annotations

import argparse
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

from isaaclab.app import AppLauncher

from quadcopter_env import QuadcopterEnv, QuadcopterEnvCfg
from ese651_controller_simple_policy import SimpleRacingPolicy


def parse_args():
    parser = argparse.ArgumentParser("Sim2Real eval for ESE651 quadcopter policy")
    parser.add_argument("--track",
                        type=str,
                        default="circle",
                        choices=["circle", "complex", "lemniscate"],
                        help="Track name used inside QuadcopterEnvCfg.")
    parser.add_argument("--ckpt",
                        type=str,
                        required=True,
                        help="Path to trained PPO checkpoint (.pt).")
    parser.add_argument("--device",
                        type=str,
                        default="cuda:0",
                        help="Torch device for policy network.")
    parser.add_argument("--max_steps",
                        type=int,
                        default=6000,
                        help="Maximum env steps for evaluation.")
    # Isaac Lab / AppLauncher 相关参数（和 train_race 一样）
    AppLauncher.add_app_launcher_args(parser)
    args, _ = parser.parse_known_args()
    return args


def make_env(args, simulation_app) -> QuadcopterEnv:
    cfg = QuadcopterEnvCfg()
    cfg.is_train = False                 # eval 模式
    cfg.track_name = args.track          # "circle" / "complex"
    cfg.scene.num_envs = 1               # sim2real 就 1 架机
    cfg.sim.device = args.device

    env = QuadcopterEnv(cfg)
    return env


def make_policy(env: QuadcopterEnv, ckpt_path: str, device: str) -> SimpleRacingPolicy:
    # 从 env 里拿赛道信息，和 ROS 那边用的一样
    waypoints = env._waypoints[:, :3].detach().cpu().numpy()        # (n_wp, 3)
    waypoints_quat = env._waypoints_quat.detach().cpu().numpy()     # (n_wp, 4) [w,x,y,z]
    gate_side = float(env._gate_model_cfg_data.gate_side)

    max_roll_br = float(env.cfg.body_rate_scale_xy)
    max_pitch_br = float(env.cfg.body_rate_scale_xy)
    max_yaw_br = float(env.cfg.body_rate_scale_z)

    params = dict(
        waypoints=waypoints,
        waypoints_quat=waypoints_quat,
        gate_side=gate_side,
        initial_waypoint=0,
        max_roll_br=max_roll_br,
        max_pitch_br=max_pitch_br,
        max_yaw_br=max_yaw_br,
        pass_gate_thr=0.10,
    )

    policy = SimpleRacingPolicy(
        vehicle=None,
        model_path=ckpt_path,
        params=params,
        device=device,
        use_cond=False,
    )
    return policy


def rollout(env: QuadcopterEnv,
            policy: SimpleRacingPolicy,
            max_steps: int,
            simulation_app):
    obs, _ = env.reset()
    step = 0

    while simulation_app.is_running() and step < max_steps:
        # 取出 env 中当前的状态（只用 env 0）
        root_state = env._robot.data.root_link_state_w  # (N, 13)
        pos_w = root_state[0, :3].detach().cpu().numpy()
        quat_w = env._robot.data.root_quat_w[0].detach().cpu().numpy()       # [w,x,y,z]
        lin_vel_b = env._robot.data.root_com_lin_vel_b[0].detach().cpu().numpy()

        # world_from_body 旋转矩阵
        R_wb = R.from_quat(quat_w, scalar_first=True).as_matrix()
        # SimpleRacingPolicy 期望的是 body_from_world
        R_bw = R_wb.T

        state = {
            "x": pos_w,
            "v_b": lin_vel_b,
            "R": R_bw,
        }

        # 实机同款 policy 输出 thrust & body-rates
        ctrl, _ = policy.update(state)
        cmd_thrust = float(ctrl["cmd_thrust"])               # in [0, 1]
        cmd_w = np.asarray(ctrl["cmd_w"], dtype=np.float32)  # [roll,pitch,yaw] rad/s

        # 映射回 Isaac env 的 action ∈ [-1,1]
        max_br = np.array(
            [policy.max_roll_br, policy.max_pitch_br, policy.max_yaw_br],
            dtype=np.float32,
        )
        actions = np.zeros((env.num_envs, 4), dtype=np.float32)
        # thrust [0,1] → [-1,1]
        actions[:, 0] = 2.0 * cmd_thrust - 1.0
        # body-rates [-max_br, max_br] → [-1,1]
        actions[:, 1:] = cmd_w / max_br

        actions_t = torch.from_numpy(actions).to(env.device)

        obs, rewards, terminated, truncated, info = env.step(actions_t)
        env.render()

        step += 1

        done = bool(terminated[0] or truncated[0])
        if done:
            print(f"[sim2real-eval] episode finished at step {step}, "
                  f"reward={rewards[0].item():.3f}")
            obs, _ = env.reset()
            step = 0


def main():
    args = parse_args()
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    env = make_env(args, simulation_app)
    policy = make_policy(env, args.ckpt, args.device)

    rollout(env, policy, args.max_steps, simulation_app)

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
