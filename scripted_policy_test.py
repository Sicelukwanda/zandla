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
from utils.scripted_policy import generate_push_instruction, get_next_action


if __name__ == "__main__":
    print("Initializing PushCube Gym Environment...")
    env = PushCubeGymEnv(render_mode="human")
    # Custom initial cube position
    init_cube_pos = [0.2, 0.0, 0.015]
    for episode in range(3):
        direction = "left" if episode % 2 == 0 else "right"
        swap_target = False
        instruction = generate_push_instruction(
            direction=direction, swap_targets=swap_target
        )
        options = {
            "instruction": instruction,
            "init_cube_pos": init_cube_pos,
            "swap_target_colors": swap_target,
        }

        obs, info = env.reset(options=options)

        print(f"\n--- Episode {episode + 1} ({direction.upper()}) ---")
        print(f"Instruction : '{obs['instruction']}'")
        print(f"Cube Pos    : {obs['cube_position']}")

        step_count = 0
        total_reward = 0.0
        reset_eef = False
        #counters for phase 2 delay (move box into the target), and tracking steps in Phase 3
        term_delay_counter = 0
        return_step_count = 0        
        #number of steps each phase should last for, to adjust the robots speed
        approach_steps = 45 
        delay_steps = 5
        push_steps = 45
        return_steps = 45

        while True:
            # compute next action based on specified phases given the current steps
            # action function generates next action based on current phase      
            action, is_at_init = get_next_action(
                obs,
                env.unwrapped.physics,
                direction=direction,
                ee_ini_pos= env.unwrapped.ee_rest, # ee init pos from env
                swap_targets=swap_target,
                step_count=step_count,
                reset_eef_pos=reset_eef,
                return_step_count=return_step_count,
                approach_steps= approach_steps, 
                delay_steps=delay_steps, 
                push_steps = push_steps,
                return_steps = return_steps,
            )

            # Step the simulation
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step_count += 1

            if reset_eef:
                return_step_count += 1

            # Verify task success and trigger return phase
            if terminated or truncated:
                term_delay_counter += 1
                if term_delay_counter >= 2: # delay counter for phase 2 (ensures cube is fully in the target)
                    reset_eef = True

            time.sleep(0.04)
            if step_count >= 100 or (reset_eef and is_at_init):
                print(
                    f"Scripted policy Finished! Total Steps: {step_count} | Total Reward: {total_reward:.4f} | Success: {info['success']}"
                )
                break
    env.close()
    print("\nTest complete.")
