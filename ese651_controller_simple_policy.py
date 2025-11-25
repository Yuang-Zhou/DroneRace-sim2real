import numpy as np
import torch
import torch.nn as nn
from scipy.spatial.transform import Rotation as R

# import pygame

class Actor(nn.Module):
    def __init__(self, mlp_input_dim, actor_hidden_dims, num_actions, activation):
        super(Actor, self).__init__()

        actor_layers = []
        actor_layers.append(nn.Linear(mlp_input_dim, actor_hidden_dims[0]))
        actor_layers.append(activation())
        for layer_index in range(len(actor_hidden_dims)):
            if layer_index == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[layer_index], num_actions))
                actor_layers.append(nn.Tanh())
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[layer_index], actor_hidden_dims[layer_index + 1]))
                actor_layers.append(activation())
        self.actor = nn.Sequential(*actor_layers)

    def forward(self, x):
        return self.actor(x)

class SimpleRacingPolicy:
    def __init__(self, vehicle, model_path, params, device="cpu", use_cond=False):
        self.quadrotor = vehicle
        self.device = torch.device(device)
        self.use_cond = use_cond
        self.obs_dim = 3 + 9 + 12 + 12 + (2 if use_cond else 0)

        self.action_dim = 4

        self.waypoints = params["waypoints"]
        self.waypoints_quat = params["waypoints_quat"]

        self.gate_side = params["gate_side"]
        d = self.gate_side / 2
        self.local_square = np.array([
            [0,  d,  d],
            [0, -d,  d],
            [0, -d, -d],
            [0,  d, -d]
        ], dtype=np.float32)

        # Create network
        self.model = Actor(self.obs_dim, [512, 512, 256, 128], self.action_dim, nn.ELU).to(self.device)
        if use_cond:
            self.cond_twr = torch.tensor([3.15])
            self.cond_perc = torch.tensor([0.0])
        checkpoint = torch.load(model_path, map_location=self.device)
        # Load checkpoint
        actor_state_dict = {k: v for k, v in checkpoint["model_state_dict"].items() if "actor" in k}
        self.model.load_state_dict(actor_state_dict, strict=True)
        # Compile network
        self.model = torch.compile(self.model)
        self.model.eval()
        # Warm-up network
        with torch.no_grad():
            dummy_obs = torch.zeros(self.obs_dim, dtype=torch.float32, device=self.device)
            _ = self.model(dummy_obs)

        self.idx_wp = params["initial_waypoint"]


        # Set the maximum body rate on each axis (this is hand selected), rad/s
        self.max_roll_br = params["max_roll_br"]
        self.max_pitch_br = params["max_pitch_br"]
        self.max_yaw_br = params["max_yaw_br"]
        self.pass_gate_thr = params.get("pass_gate_thr", 0.10)  # Default value if not in params

        # pygame.init()
        # pygame.joystick.init()
        # self.joystick = pygame.joystick.Joystick(0)
        # self.joystick.init()
        # print(f"Joystick connected: {self.joystick.get_name()}")
        # self.axes_names = {
        #     0: "Left Stick X",
        #     1: "Left Stick Y",
        #     3: "Right Stick X",
        #     4: "Right Stick Y",
        # }

    def update(self, state):
        """
        Compute the control command using the neural network.

        Inputs:
            state, current dictionary with state
         Output:
            control_input and observation vector:
        """
        pos_drone = state['x']
        lin_vel_drone = state['v_b']
        rot_drone = state['R']

        curr_idx = self.idx_wp
        next_idx = (self.idx_wp + 1) % self.waypoints.shape[0]

        wp_curr_pos = self.waypoints[curr_idx, :3]
        wp_next_pos = self.waypoints[next_idx, :3]
        quat_curr = self.waypoints_quat[curr_idx, :]
        quat_next = self.waypoints_quat[next_idx, :]
        rot_curr = R.from_quat(quat_curr, scalar_first=True).as_matrix()
        rot_next = R.from_quat(quat_next, scalar_first=True).as_matrix()

        pose_drone_wrt_gate = self._subtract_frame_transforms(wp_curr_pos, rot_curr, pos_drone)
        if np.linalg.norm(pose_drone_wrt_gate) < self.gate_side and pose_drone_wrt_gate[0] < self.pass_gate_thr:
            self.idx_wp = (self.idx_wp + 1) % self.waypoints.shape[0]

        verts_curr = self.local_square @ rot_curr.T + wp_curr_pos
        verts_next = self.local_square @ rot_next.T + wp_next_pos

        waypoint_pos_b_curr = self._subtract_frame_transforms(pos_drone, rot_drone, verts_curr).reshape(4, 3)
        waypoint_pos_b_next = self._subtract_frame_transforms(pos_drone, rot_drone, verts_next).reshape(4, 3)

        obs = [
            torch.from_numpy(lin_vel_drone).float().flatten(),
            torch.from_numpy(rot_drone).float().flatten(),
            torch.from_numpy(waypoint_pos_b_curr).float().flatten(),
            torch.from_numpy(waypoint_pos_b_next).float().flatten(),
        ]

        if self.use_cond:
            obs.append(self.cond_twr.flatten())
            obs.append(self.cond_perc.flatten())

        obs = torch.cat(obs).float().to(self.device)

        with torch.no_grad():
            actions = self.model(obs).squeeze(0).cpu().numpy()
        actions = np.clip(actions, -1, 1)

        # print('Linear velocity:')
        # print(obs[0:3])
        # print('Rotation:')
        # print(obs[3:12])
        # print('Corners curr:')
        # print(obs[12:15])
        # print(obs[15:18])
        # print(obs[18:21])
        # print(obs[21:24])
        # print('Corners next:')
        # print(obs[24:27])
        # print(obs[27:30])
        # print(obs[30:33])
        # print(obs[33:36])
        # print('Conditioning:')
        # print(obs[36:38])
        # print('Waypoint index:')
        # print(self.idx_wp)
        # print('Actions:')
        # print(actions)
        # print()

        # pygame.event.pump()
        # def deadzone_general(x, threshold=0.03):
        #     return 0.0 if abs(x) < threshold else x

        # actions[0] = -self.joystick.get_axis(1)
        # if actions[0] < -0.95:
        #     actions[0] = -1.0

        # actions[1] = deadzone_general(self.joystick.get_axis(3))
        # actions[2] = deadzone_general(-self.joystick.get_axis(4))
        # actions[3] = deadzone_general(-self.joystick.get_axis(0), threshold=0.1)

        cmd_thrust = 0.5 * (actions[0] + 1.0)

        roll_br  = actions[1] * self.max_roll_br
        pitch_br = actions[2] * self.max_pitch_br
        yaw_br   = actions[3] * self.max_yaw_br

        control_input = {'cmd_thrust': cmd_thrust,
                         'cmd_w': np.array([roll_br, pitch_br, yaw_br])}

        return control_input, obs

    def _subtract_frame_transforms(self, pos, rot, pos_des):
        if pos_des.ndim == 1:
            return rot.T @ (pos_des - pos)
        elif pos_des.ndim == 2:
            return (pos_des - pos) @ rot