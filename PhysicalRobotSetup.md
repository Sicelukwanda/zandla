# Physical Robot Setup & Calibration

This guide details setting up, calibrating, and teleoperating physical SO-101 leader-follower robot arms using the LeRobot framework and Seeed RoboController.

## Controller SDK
First, clone the controller SDK repository:
```bash
git clone https://github.com/Seeed-Projects/Seeed_RoboController.git
```

And install it in the virtual environment:
```bash
uv pip install -e .
```

---

## Calibration & Hardware Setup

### 1. Discovering Serial Ports
Use the port scanner target to discover connected robot devices:

```bash
make scan_ports
```

Inspect the output and note the serial device path for your robot (e.g., `/dev/ttyACM0`).

### 2. Performing Motor Calibration
Run the setup target with `SETUP_ARGS` to specify the robot type and serial port:

```bash
make setup SETUP_ARGS="--robot.type=so101_follower --robot.port=/dev/ttyACM0"
```

Replace the values with the ports detected on your machine.

---

## Middle Position Calibration (Seeed RoboController)
If you need to calibrate your servos' middle position, use the Seeed RoboController middle calibration tool:

```bash
cd Seeed_RoboController
uv run python -m src.tools.servo_middle_calibration /dev/ttyACM0
```

If you want to choose the port interactively, omit the device path:

```bash
uv run python -m src.tools.servo_middle_calibration
```

This tool will:
- Scan connected servos
- Optionally disable them
- Allow you to move each servo to the desired center position
- Write the current position as the new middle value (2048)

---

## Teleoperation Example
Use `TELEOP_ARGS` to launch the follower and leader hardware teleoperation:

```bash
make teleop TELEOP_ARGS="--robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=zandla_follower_arm \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM1 \
    --teleop.id=zandla_leader_arm"
```

### Notes & Troubleshooting
- Do not pass `--robot.type`, `--robot.port`, or other runtime args directly to `make`; use `SETUP_ARGS` or `TELEOP_ARGS` instead.
- If your device ports change upon reconnecting, re-run `make scan_ports` before setup or teleoperation.
