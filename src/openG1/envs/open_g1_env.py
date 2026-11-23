"""Open G1 standing environment (Gymnasium + MuJoCo).

Actions are normalised to [-1, 1] for every actuator and mapped onto the actuator's ctrlrange:
  * legs + waist  -> joint torque (motor actuators, G1 limits until your leg actuators are measured)
  * arms (servos) -> joint position targets (position actuators, servo travel / torque limits)
"""
from pathlib import Path

import numpy as np
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box

from ..meshes import check_meshes

DEFAULT_XML = str(Path(__file__).resolve().parent.parent / "assets" / "open_g1.xml")
DEFAULT_CAMERA = {"trackbodyid": 1, "distance": 3.0, "lookat": np.array((0.0, 0.0, 0.8)), "elevation": -15.0}


class OpenG1StandEnv(MujocoEnv, utils.EzPickle):
    metadata = {"render_modes": ["human", "rgb_array", "depth_array"]}

    def __init__(
        self,
        xml_file: str = DEFAULT_XML,
        frame_skip: int = 10,                 # 0.002 s x 10 = 50 Hz control
        healthy_z_range=(0.45, 1.2),          # pelvis height (m)
        healthy_reward: float = 1.0,
        upright_weight: float = 1.0,
        ctrl_cost_weight: float = 1e-3,
        reset_noise_scale: float = 0.02,
        **kwargs,
    ):
        utils.EzPickle.__init__(self, xml_file, frame_skip, healthy_z_range, healthy_reward,
                                upright_weight, ctrl_cost_weight, reset_noise_scale, **kwargs)
        self._healthy_z_range = healthy_z_range
        self._healthy_reward = healthy_reward
        self._upright_weight = upright_weight
        self._ctrl_cost_weight = ctrl_cost_weight
        self._reset_noise_scale = reset_noise_scale

        if xml_file == DEFAULT_XML:
            check_meshes()              # clear message if a mesh is missing
        MujocoEnv.__init__(self, xml_file, frame_skip, observation_space=None,
                           default_camera_config=DEFAULT_CAMERA, **kwargs)
        self.metadata["render_fps"] = int(np.round(1.0 / self.dt))

        obs_size = self._get_obs().shape[0]
        self.observation_space = Box(-np.inf, np.inf, shape=(obs_size,), dtype=np.float64)
        lo, hi = self.model.actuator_ctrlrange.T
        self._ctrl_lo, self._ctrl_hi = lo.copy(), hi.copy()
        self.action_space = Box(-1.0, 1.0, shape=(self.model.nu,), dtype=np.float32)

    # ------------------------------------------------------------------ helpers
    def _scale_action(self, action):
        a = np.clip(action, -1.0, 1.0)
        return self._ctrl_lo + (a + 1.0) * 0.5 * (self._ctrl_hi - self._ctrl_lo)

    @property
    def is_healthy(self):
        z = self.data.qpos[2]
        return self._healthy_z_range[0] < z < self._healthy_z_range[1]

    def _upright(self):
        # z-axis of the torso expressed in world frame, dotted with world up
        return float(self.data.body("torso_link").xmat.reshape(3, 3)[2, 2])

    def _get_obs(self):
        qpos = self.data.qpos.flat[2:]          # drop world x, y
        qvel = self.data.qvel.flat[:]
        return np.array(qpos)

    # ------------------------------------------------------------------ gym API
    def step(self, action):
        ctrl = self._scale_action(action)
        self.do_simulation(ctrl, self.frame_skip)

        upright = self._upright()
        ctrl_cost = self._ctrl_cost_weight * float(np.sum(np.square(action)))
        healthy = self.is_healthy
        reward = (self._healthy_reward if healthy else 0.0) + self._upright_weight * upright - ctrl_cost
        terminated = not healthy
        info = {"upright": upright, "ctrl_cost": ctrl_cost, "pelvis_z": float(self.data.qpos[2])}
        if self.render_mode == "human":
            self.render()
        return self._get_obs(), reward, terminated, False, info

    def reset_model(self):
        noise = self._reset_noise_scale
        qpos = self.init_qpos.copy()
        qvel = self.init_qvel.copy()
        qpos[7:] += self.np_random.uniform(-noise, noise, size=self.model.nq - 7)
        qvel[6:] += self.np_random.uniform(-noise, noise, size=self.model.nv - 6)
        # keep every hinge inside its (servo-clipped) range
        lim = self.model.jnt_range[1:]
        qpos[7:] = np.clip(qpos[7:], lim[:, 0], lim[:, 1])
        self.set_state(qpos, qvel)
        return self._get_obs()
