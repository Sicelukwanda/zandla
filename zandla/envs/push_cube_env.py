import os
import time
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from dm_control import mujoco as dmc_mujoco
from dm_control.rl import control
from dm_env import specs
import mujoco.viewer


class PushCubeTask(control.Task):
    """A custom dm_control Task for pushing a red cube to a green or blue target."""

    def __init__(self, random_state=None):
        self.random = random_state or np.random.RandomState()
        self.current_goal = "green"  # Default goal

    def action_spec(self, physics):
        """Returns the action spec (the control range of the robot's actuators)."""
        ctrlrange = physics.model.actuator_ctrlrange
        return specs.BoundedArray(
            shape=(6,),
            dtype=np.float64,
            minimum=ctrlrange[:, 0],
            maximum=ctrlrange[:, 1],
            name="action"
        )

    def before_step(self, action, physics):
        """Apply actions to the physics simulation."""
        # Set control inputs to the actuators
        physics.set_control(action)

    def initialize_episode(self, physics):
        """Reset the robot joints and place the cube at a random reachable position."""
        # Shift the robot base to height z = 0.15 (sits on the pedestal)
        base_id = physics.model.name2id("base", "body")
        physics.model.body_pos[base_id] = [0.0, 0.0, 0.15]

        # Reset joint positions to neutral posture (from simulate_camera.py)
        physics.data.qpos[0] = 0.0
        physics.data.qpos[1] = 0.5
        physics.data.qpos[2] = 1.0
        physics.data.qpos[3] = 0.5
        physics.data.qpos[4] = 0.0
        physics.data.qpos[5] = 0.5

        # Randomize cube position within a reachable workspace
        cube_x = self.random.uniform(0.16, 0.24)
        cube_y = self.random.uniform(-0.08, 0.08)
        physics.data.qpos[6:9] = [cube_x, cube_y, 0.015]
        physics.data.qpos[9:13] = [1.0, 0.0, 0.0, 0.0]  # identity quaternion

        # Zero out joint velocities
        physics.data.qvel[:] = 0.0

    def get_observation(self, physics):
        """Returns low-level state observations for the environment."""
        obs = {}
        obs["joint_positions"] = physics.data.qpos[0:6].copy()
        obs["joint_velocities"] = physics.data.qvel[0:6].copy()
        obs["cube_position"] = physics.data.qpos[6:9].copy()
        obs["cube_orientation"] = physics.data.qpos[9:13].copy()

        target_green_id = physics.model.name2id("target_green", "geom")
        target_blue_id = physics.model.name2id("target_blue", "geom")
        obs["target_green_position"] = physics.model.geom_pos[target_green_id].copy()
        obs["target_blue_position"] = physics.model.geom_pos[target_blue_id].copy()
        return obs

    def get_reward(self, physics):
        """Computes a shaped reward based on gripper-cube and cube-target distances."""
        # 1. Get gripper site position
        site_id = physics.model.name2id("gripperframe", "site")
        gripper_pos = physics.data.site_xpos[site_id]

        # 2. Get cube position
        cube_pos = physics.data.qpos[6:9]

        # 3. Get selected target position
        target_name = "target_green" if self.current_goal == "green" else "target_blue"
        target_id = physics.model.name2id(target_name, "geom")
        target_pos = physics.model.geom_pos[target_id]

        # Compute distances
        dist_hand_cube = np.linalg.norm(gripper_pos - cube_pos)
        dist_cube_target = np.linalg.norm(cube_pos - target_pos)

        # Shaped rewards
        reward_reach = np.exp(-10.0 * dist_hand_cube)
        reward_push = np.exp(-15.0 * dist_cube_target)

        # Combine: 20% reach reward, 80% push reward
        reward = 0.2 * reward_reach + 0.8 * reward_push
        return float(reward)


class PushCubeGymEnv(gym.Env):
    """Gymnasium Wrapper around the custom dm_control PushCube environment."""
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 25}

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode

        # Define path to the MuJoCo scene XML
        xml_path = os.path.join(
            os.path.dirname(__file__),
            "../", "robot_models", "SO101", "scene_push_cube.xml"
        )
        if not os.path.exists(xml_path):
            raise FileNotFoundError(f"Scene XML not found at {xml_path}")

        # Instantiate task and physics
        self.task = PushCubeTask()
        self.physics = dmc_mujoco.Physics.from_xml_path(xml_path)

        # Initialize the underlying dm_control environment
        # 0.04s control timestep matches the 25 Hz render rate
        self.dmc_env = control.Environment(
            physics=self.physics,
            task=self.task,
            time_limit=10.0,  # 10 second time limit per episode (250 steps)
            control_timestep=0.04
        )

        # Get control range from physics model for action space
        ctrl_range = self.physics.model.actuator_ctrlrange
        self.action_space = spaces.Box(
            low=ctrl_range[:, 0].astype(np.float32),
            high=ctrl_range[:, 1].astype(np.float32),
            dtype=np.float32
        )

        # Define Gym observation space with dict containing state and text instructions
        self.observation_space = spaces.Dict({
            "joint_positions": spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32),
            "joint_velocities": spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32),
            "cube_position": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
            "cube_orientation": spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32),
            "target_green_position": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
            "target_blue_position": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
            "instruction": spaces.Text(max_length=100)
        })

        self._viewer = None

    def _get_obs(self, dmc_obs):
        """Converts raw dm_control task observation arrays to float32 numpy arrays."""
        return {
            "joint_positions": np.array(dmc_obs["joint_positions"], dtype=np.float32),
            "joint_velocities": np.array(dmc_obs["joint_velocities"], dtype=np.float32),
            "cube_position": np.array(dmc_obs["cube_position"], dtype=np.float32),
            "cube_orientation": np.array(dmc_obs["cube_orientation"], dtype=np.float32),
            "target_green_position": np.array(dmc_obs["target_green_position"], dtype=np.float32),
            "target_blue_position": np.array(dmc_obs["target_blue_position"], dtype=np.float32),
        }

    def reset(self, seed=None, options=None):
        """Resets the environment, optionally choosing a target or using seed/options."""
        super().reset(seed=seed)

        # Seed random generators
        if seed is not None:
            self.task.random.seed(seed)

        # Set task goal based on options or choose randomly
        if options is not None and "instruction" in options:
            instruction = options["instruction"]
            if "green" in instruction.lower():
                self.task.current_goal = "green"
            elif "blue" in instruction.lower():
                self.task.current_goal = "blue"
            else:
                self.task.current_goal = self.np_random.choice(["green", "blue"])
        else:
            self.task.current_goal = self.np_random.choice(["green", "blue"])

        # Reset dm_control env
        timestep = self.dmc_env.reset()

        # Build observations
        obs = self._get_obs(timestep.observation)
        instruction_str = f"push the red cube to the {self.task.current_goal} target"
        obs["instruction"] = instruction_str

        info = {
            "instruction": instruction_str,
            "goal": self.task.current_goal
        }

        if self.render_mode == "human":
            self.render()

        return obs, info

    def step(self, action):
        """Steps the environment by applying the action."""
        # Convert action to np.float64 for dm_control
        action = np.asarray(action, dtype=np.float64)
        timestep = self.dmc_env.step(action)

        # Convert observation
        obs = self._get_obs(timestep.observation)
        instruction_str = f"push the red cube to the {self.task.current_goal} target"
        obs["instruction"] = instruction_str

        reward = timestep.reward
        terminated = False
        truncated = timestep.last()

        # Check success condition (cube center within 5 cm of target center)
        cube_pos = obs["cube_position"]
        target_name = "target_green" if self.task.current_goal == "green" else "target_blue"
        target_pos = obs[f"{target_name}_position"]
        dist_cube_target = np.linalg.norm(cube_pos - target_pos)
        success = dist_cube_target < 0.05

        if success:
            terminated = True

        info = {
            "instruction": instruction_str,
            "goal": self.task.current_goal,
            "success": success,
            "dist_cube_target": dist_cube_target
        }

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    def render(self):
        """Renders the environment."""
        if self.render_mode == "human":
            if self._viewer is None:
                # Open the MuJoCo viewer with the underlying raw model and data pointers
                self._viewer = mujoco.viewer.launch_passive(
                    self.physics.model.ptr,
                    self.physics.data.ptr
                )
            self._viewer.sync()
        elif self.render_mode == "rgb_array":
            # Return camera view (id -1 is the default scene camera)
            return self.physics.render(height=480, width=640, camera_id=-1)

    def close(self):
        """Closes the viewer if open."""
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
