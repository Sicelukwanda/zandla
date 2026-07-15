import os
import time
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from dm_control import mujoco as dmc_mujoco
from dm_control.rl import control
from dm_env import specs
import mujoco.viewer

from zandla.envs import PushCubeGymEnv


if __name__ == "__main__":
    print("Initializing PushCube Gymnasium Environment...")
    env = PushCubeGymEnv(render_mode="human")

    # Run for a few episodes
    for episode in range(3):
        print(f"\n--- Episode {episode + 1} ---")
        
        # Test custom instruction options
        target_color = "green" if episode % 2 == 0 else "blue"
        options = {"instruction": f"push the red cube to the {target_color} target"}
        
        obs, info = env.reset(options=options)
        print(f"Goal set: {info['goal']}")
        print(f"Text Instruction: '{obs['instruction']}'")
        print(f"Initial Cube Position: {obs['cube_position']}")
        print(f"Initial Green Target: {obs['target_green_position']}")
        print(f"Initial Blue Target:  {obs['target_blue_position']}")

        step_count = 0
        total_reward = 0.0
        
        while True:
            # Sample random control actions within joint limits
            action = env.action_space.sample()
            
            # Step the simulation
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step_count += 1

            if step_count % 25 == 0:
                print(f"  Step {step_count:3d} | Reward: {reward:.4f} | Cube Dist to Target: {info['dist_cube_target']:.4f} m")

            # Maintain simulation speed (25 Hz control loop)
            time.sleep(0.04)

            if terminated or truncated:
                print(f"Episode Finished! Total Steps: {step_count} | Total Reward: {total_reward:.4f} | Success: {info['success']}")
                break
                
    env.close()
    print("\nEnvironment closed. Demonstration complete.")
