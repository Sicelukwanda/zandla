import os
import time
import argparse
import numpy as np
import glfw
import mujoco

MODEL_PATH = os.path.join("zandla", "robot_models", "SO101", "scene_camera.xml")


def main():
    parser = argparse.ArgumentParser(description="SO101 MuJoCo Simulation CLI with Camera")
    parser.add_argument(
        "--model",
        type=str,
        default=MODEL_PATH,
        help="Path to the MuJoCo scene XML file",
    )
    parser.add_argument(
        "--span",
        type=float,
        default=0.15,
        help="Percentage of the joint range to span (0.0 to 1.0)",
    )
    parser.add_argument(
        "--freq",
        type=float,
        default=2.0,
        help="Frequency of the sinusoidal oscillation in Hz",
    )

    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"Error: Could not find model file at {args.model}")
        return

    # Load the model and data
    model = mujoco.MjModel.from_xml_path(args.model)
    data = mujoco.MjData(model)

    # Set joints to a neutral posture
    data.qpos[0] = 0.0
    data.qpos[1] = 0.5
    data.qpos[2] = 1.0
    data.qpos[3] = 0.5
    data.qpos[4] = 0.0
    data.qpos[5] = 0.5
    mujoco.mj_forward(model, data)

    # Get joint control ranges (ctrlrange) from the model
    ctrl_ranges = model.actuator_ctrlrange
    ctrl_mid = np.mean(ctrl_ranges, axis=1)
    ctrl_width = (ctrl_ranges[:, 1] - ctrl_ranges[:, 0]) * args.span

    # Logging the ranges
    print(f"\nActuator Control Ranges (Span={args.span*100}%):")
    for i in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        orig_min, orig_max = ctrl_ranges[i]
        new_min = ctrl_mid[i] - (ctrl_width[i] / 2.0)
        new_max = ctrl_mid[i] + (ctrl_width[i] / 2.0)
        print(
            f"  {name:15}: [{orig_min:7.3f}, {orig_max:7.3f}] -> [{new_min:7.3f}, {new_max:7.3f}]"
        )
    print("")

    # Verify if wrist camera is in the model
    has_camera = False
    try:
        cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "wrist_camera")
        if cam_id != -1:
            has_camera = True
            print("Detected wrist camera. Initializing viewport...")
        else:
            print("Warning: Camera 'wrist_camera' not found in model.")
    except Exception as e:
        print("Failed to look up camera ID:", e)

    # Initialize GLFW
    if not glfw.init():
        print("Error: Could not initialize GLFW")
        return

    # Create double-width window for side-by-side views
    window_width = 1280 if has_camera else 640
    window_height = 480
    window = glfw.create_window(
        window_width,
        window_height,
        "SO101 Simulation - Main View (Left) | Wrist Camera View (Right)" if has_camera else "SO101 Simulation",
        None,
        None
    )
    if not window:
        glfw.terminate()
        print("Error: Could not create GLFW window")
        return

    glfw.make_context_current(window)

    # Create MuJoCo scene and context
    ctx = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)
    scene = mujoco.MjvScene(model, maxgeom=10000)

    # Main scene camera
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.distance = 1.0
    cam.elevation = -20
    cam.azimuth = 135

    # Wrist camera
    cam_wrist = None
    if has_camera:
        cam_wrist = mujoco.MjvCamera()
        cam_wrist.type = mujoco.mjtCamera.mjCAMERA_FIXED
        cam_wrist.fixedcamid = cam_id

    option = mujoco.MjvOption()
    perturb = mujoco.MjvPerturb()

    # Mouse interaction globals
    button_left = False
    button_middle = False
    button_right = False
    lastx = 0
    lasty = 0

    def mouse_button(window, button, act, mods):
        nonlocal button_left, button_middle, button_right, lastx, lasty
        button_left = (glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS)
        button_middle = (glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS)
        button_right = (glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS)
        lastx, lasty = glfw.get_cursor_pos(window)

    def mouse_move(window, xpos, ypos):
        nonlocal lastx, lasty, button_left, button_middle, button_right
        dx = xpos - lastx
        dy = ypos - lasty
        lastx = xpos
        lasty = ypos
        
        width, height = glfw.get_framebuffer_size(window)
        # Only allow mouse rotation/zoom in the left viewport (main scene)
        if has_camera and xpos > width / 2:
            return
            
        if button_left:
            action = mujoco.mjtMouse.mjMOUSE_ROTATE_V
        elif button_right:
            action = mujoco.mjtMouse.mjMOUSE_MOVE_V
        elif button_middle:
            action = mujoco.mjtMouse.mjMOUSE_ZOOM
        else:
            return
            
        mujoco.mjv_moveCamera(model, action, dx/height, dy/height, scene, cam)

    def scroll(window, xoffset, yoffset):
        mujoco.mjv_moveCamera(model, mujoco.mjtMouse.mjMOUSE_ZOOM, 0.0, -0.05 * yoffset, scene, cam)

    glfw.set_mouse_button_callback(window, mouse_button)
    glfw.set_cursor_pos_callback(window, mouse_move)
    glfw.set_scroll_callback(window, scroll)

    paused = False

    def key_callback(window, key, scancode, action, mods):
        nonlocal paused
        if action == glfw.PRESS:
            if key == glfw.KEY_SPACE:
                paused = not paused
                print("Simulation Paused:", paused)
            elif key == glfw.KEY_ESCAPE:
                glfw.set_window_should_close(window, True)
            elif key == glfw.KEY_BACKSPACE:
                mujoco.mj_resetData(model, data)
                # Restore joints to neutral
                data.qpos[0] = 0.0
                data.qpos[1] = 0.5
                data.qpos[2] = 1.0
                data.qpos[3] = 0.5
                data.qpos[4] = 0.0
                data.qpos[5] = 0.5
                mujoco.mj_forward(model, data)
                print("Reset simulation data")

    glfw.set_key_callback(window, key_callback)

    print(f"Launching simulation loop (Freq={args.freq}Hz)...")
    print("Mouse Controls:")
    print("  Left Drag  : Rotate Main View")
    print("  Right Drag : Pan Main View")
    print("  Scroll     : Zoom Main View")
    print("Keyboard Controls:")
    print("  SPACE      : Pause/Unpause Simulation")
    print("  BACKSPACE  : Reset Simulation")
    print("  ESC        : Exit")
    print("")

    step_start = time.time()
    while not glfw.window_should_close(window):
        # Update simulation joints
        elapsed = data.time
        if not paused:
            data.ctrl[:] = ctrl_mid + (ctrl_width / 2.0) * np.sin(2 * np.pi * args.freq * elapsed)
            mujoco.mj_step(model, data)
        
        # Get frame buffer size
        width, height = glfw.get_framebuffer_size(window)
        
        if has_camera:
            # 1. Render main scene in left half
            mujoco.mjv_updateScene(model, data, option, perturb, cam, mujoco.mjtCatBit.mjCAT_ALL, scene)
            rect_left = mujoco.MjrRect(0, 0, width // 2, height)
            mujoco.mjr_render(rect_left, scene, ctx)
            
            # 2. Render camera view in right half
            mujoco.mjv_updateScene(model, data, option, perturb, cam_wrist, mujoco.mjtCatBit.mjCAT_ALL, scene)
            rect_right = mujoco.MjrRect(width // 2, 0, width // 2, height)
            mujoco.mjr_render(rect_right, scene, ctx)
        else:
            # Render main scene full window
            mujoco.mjv_updateScene(model, data, option, perturb, cam, mujoco.mjtCatBit.mjCAT_ALL, scene)
            rect_full = mujoco.MjrRect(0, 0, width, height)
            mujoco.mjr_render(rect_full, scene, ctx)
            
        # Swap buffers and poll events
        glfw.swap_buffers(window)
        glfw.poll_events()
        
        # Sync with simulation time step
        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
        step_start = time.time()

    glfw.terminate()


if __name__ == "__main__":
    main()
