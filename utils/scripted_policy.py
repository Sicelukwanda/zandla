import random
import numpy as np
from dm_control.utils import inverse_kinematics


def mc_ik(
    physics,
    target_pos,
    target_quat=None,
    site_name="gripperframe",
    tol=1e-3,
    max_steps=50,
):
    """
    Inverse Kinematics solver wrapper function
    """
    ik_result = inverse_kinematics.qpos_from_site_pose(
        physics=physics,
        site_name=site_name,
        target_pos=target_pos,
        target_quat=target_quat,
        tol=tol,
        max_steps=max_steps,
    )
    return ik_result.qpos[:6].copy()


def generate_push_instruction(direction="left", swap_targets=False):
    """
    Generates randomly chosen language instructions based on the direction and if the targets are swapped.
    - Standard (swap_targets=False): Left -> green target, Right -> blue target
    - Swapped (swap_targets=True): Left -> blue target, Right -> green target

    Args:
        direction: string specifying the direction of motion between left and right to choose the target colour.
        swap_targets: bool
    """
    if not swap_targets:
        target_color = "green" if direction.lower() == "left" else "blue"
    else:
        target_color = "blue" if direction.lower() == "left" else "green"

    direction_words = {
        "left": ["left", "leftward"],
        "right": ["right", "rightward"],
    }
    direction_word = random.choice(direction_words[direction.lower()])

    templates = [
        f"push the red cube {direction_word} to the {target_color} target",
        f"nudge the red block {direction_word} towards the {target_color} marker",
        f"slide the red cube {direction_word} into the {target_color} zone",
        f"move the block {direction_word} into the {target_color} area",
        f"guide the red cube {direction_word} to the {target_color} target",
        f"shove the block {direction_word}, into the {target_color} spot",
    ]

    return random.choice(templates)


EE_INIT_POS = np.array([0.0, 0.5, 1.0, 0.5, 0.0, 0.5], dtype=np.float32)


def get_next_action(
    obs,
    physics,
    ee_ini_pos=EE_INIT_POS,
    direction="left",
    step_count=0,
    approach_steps=35,
    push_steps=50,
    return_steps=35,
    delay_steps=5,
    swap_targets=False,
    reset_eef_pos=False,
    return_step_count=0,
):
    """
    Generates the next action given the current observations, actions are generated based on the current step counts and phase
    moves the end-effector beside the cube if step_count < 25, and moves the cube to the target otherwise.
    Phase 1: Moves the robot from its initial position to beside the cube opposite the target area.
    Phase 2: Moves the robot to push the cube to the target position.
    Phase 3: Moves the robot after phase 2 to the starting position.
    (n.b. steps is the most reliable option so far)
    Args:
        obs: dictionary containing the current observations
        physics: dm_control physics object
        ee_ini_pos: array of initial joint positions of the robot at reset
        direction: string specifying motion direction ("left" or "right")
        step_count: integer representing total episode step count
        approach_steps: integer representing steps for Phase 1 
        push_steps: integer representing steps for Phase 2 
        return_steps: integer representing total steps for Phase 3 
        delay_steps: integer representing steps to pause for before returning to start position
        swap_targets: bool specifying if target colors are swapped
        reset_eef_pos: bool or int flag indicating Phase 3 is active
        return_step_count: integer representing current step count in Phase 3

    Returns:
        next_action: numpy array of shape (6,) containing action for the next step
        is_at_init: bool indicating if end-effector is at initial position to end capture
    """
    cube_pos = obs["cube_position"]
    arm_joints = obs["joint_positions"]

    # Target joint positions for Phase 3
    target_start_joints = ee_ini_pos

    # Swaps target location if specified
    if not swap_targets:
        target_pos = (
            obs["target_green_position"]
            if direction.lower() == "left"
            else obs["target_blue_position"]
        )
    else:
        target_pos = (
            obs["target_blue_position"]
            if direction.lower() == "left"
            else obs["target_green_position"]
        )

    # Calculation of end-effector position beside the cube
    y_offset = -0.05 if direction.lower() == "left" else 0.05
    wp_beside = np.array([cube_pos[0], cube_pos[1] + y_offset, 0.025])

    # Final target position
    wp_destination = np.array([cube_pos[0], target_pos[1], 0.025])

    # Generate next action based on current phase
    if reset_eef_pos:
        # Phase 3 logic, waits for delay then the robot moves to the inital positoin
        if return_step_count < delay_steps:
            # Keep current position during delay phase
            action = arm_joints.copy()
        else:
            move_step = return_step_count - delay_steps + 1
            move_total = float(max(1, return_steps - delay_steps))
            alpha = min(1.0, move_step / move_total)
            action = (1 - alpha) * arm_joints + alpha * target_start_joints
    elif step_count < approach_steps:
        # Phase 1: Move the robot to the cube's side
        target_joints = mc_ik(physics, target_pos=wp_beside, site_name="gripperframe")
        alpha = (step_count + 1) / float(approach_steps)
        action = (1 - alpha) * arm_joints + alpha * target_joints
    else:
        # Phase 2: Push cube to the target
        push_step = step_count - approach_steps + 1
        alpha = min(1.0, push_step / float(push_steps))
        current_wp = (1 - alpha) * wp_beside + alpha * wp_destination
        action = mc_ik(physics, target_pos=current_wp, site_name="gripperframe")

    # Check if current joint position matches initial position to end capturing
    is_at_init = False
    if step_count >= 10:
        is_at_init = bool(np.allclose(arm_joints, target_start_joints, atol=1e-2))

    return action, is_at_init
