
## Calibration

### 1. Finding Ports
Use the port scanner target to discover connected robot devices:

```bash
make scan-ports
```

Inspect the output and note the serial device path for your robot, for example `/dev/ttyACM0`.

### 2. Performing Calibration
Run the setup target with `SETUP_ARGS` to specify the robot type and port:

```bash
make setup SETUP_ARGS="--robot.type=so101_follower --robot.port=/dev/ttyACM0"
```

Replace the values with the ones detected on your machine.

## Middle Position Calibration (Seeed RoboController)
If you need to calibrate your servos' middle position, use the Seeed RoboController middle calibration tool.

```bash
cd Seeed_RoboController
python -m src.tools.servo_middle_calibration /dev/ttyACM0
```

If you want to choose the port interactively, omit the device path:

```bash
python -m src.tools.servo_middle_calibration
```

This tool will:
- scan connected servos
- optionally disable them
- allow you to move each servo to the desired center position
- write the current position as the new middle value (2048)

## Teleoperation Example
Use `TELEOP_ARGS` to launch the follower and leader configuration:

```bash
make teleop TELEOP_ARGS="--robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=zandla_follower_arm \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM1 \
    --teleop.id=zandla_leader_arm"
```

### Notes
- Do not pass `--robot.type`, `--robot.port`, or other runtime args directly to `make`; use `SETUP_ARGS` or `TELEOP_ARGS` instead.
- If your device ports change, re-run `make scan-ports` before setup or teleoperation.
