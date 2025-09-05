"""
DiffusionDrive agent for Bench2Drive CARLA evaluation.
Implements closed-loop evaluation with the DiffusionDrive model trained on Bench2Drive dataset.
"""

import os
import json
import datetime
import pathlib
import time
import math
from collections import deque
import numpy as np
from PIL import Image

import torch
from torchvision import transforms as T

# CARLA imports - assuming they exist at runtime
import carla
from leaderboard.autoagents import autonomous_agent

from .pid_controller import PIDController
from .planner import RoutePlanner
from .diffusion_model_wrapper import DiffusionDriveModelWrapper
from .sensor_processor import SensorProcessor
from .trajectory_planner import TrajectoryPlanner

# Environment variables
SAVE_PATH = os.environ.get("SAVE_PATH", None)
IS_BENCH2DRIVE = os.environ.get("IS_BENCH2DRIVE", None)


def get_entry_point():
    """Entry point for CARLA leaderboard."""
    return "DiffusionDriveAgent"


class DiffusionDriveAgent(autonomous_agent.AutonomousAgent):
    """
    DiffusionDrive agent for closed-loop evaluation in CARLA.
    Processes sensor data and outputs control commands using the DiffusionDrive model.
    """

    def setup(self, path_to_conf_file):
        """
        Setup the agent with configuration and model checkpoint.

        Args:
            path_to_conf_file: Either checkpoint path or "config_path+checkpoint_path+save_name"
        """
        self.track = autonomous_agent.Track.SENSORS

        # Debug print to understand what's being passed
        print(f"[DEBUG] Agent setup received: {path_to_conf_file}")

        # Parse configuration path - handle different formats
        parts = path_to_conf_file.split("+")

        # Find the actual checkpoint file (should end with .ckpt or .pth)
        checkpoint_path = None
        save_name = None
        config_path = None

        for i, part in enumerate(parts):
            if part.endswith((".ckpt", ".pth", ".pt")):
                checkpoint_path = part
            elif part.startswith("RouteScenario_"):
                save_name = part
            elif (
                i == 0
                and not part.startswith("RouteScenario_")
                and not part.endswith((".ckpt", ".pth", ".pt"))
            ):
                # Might be a config file
                config_path = part

        # If no checkpoint found, assume the first non-RouteScenario part is the checkpoint
        if checkpoint_path is None:
            for part in parts:
                if not part.startswith("RouteScenario_"):
                    checkpoint_path = part
                    break

        # Set the values
        self.config_path = config_path
        self.ckpt_path = (
            checkpoint_path if checkpoint_path else "test_files/test_b2d.ckpt"
        )  # Fallback to default

        if save_name:
            self.save_name = save_name
        else:
            now = datetime.datetime.now()
            self.save_name = "_".join(
                map(lambda x: "%02d" % x, (now.month, now.day, now.hour, now.minute, now.second))
            )

        # Initialize components
        self.step = -1
        self.wall_start = time.time()
        self.initialized = False

        # Control state
        self.steer_step = 0
        self.last_moving_status = 0
        self.last_moving_step = -1
        self.last_steer = 0

        # Navigation command tracking
        self.current_command = 3  # Default: STRAIGHT (3)
        self.current_command_text = "STRAIGHT"

        # Initialize PID controller
        self.pidcontroller = PIDController()

        # Initialize model wrapper
        print(f"Loading DiffusionDrive model from {self.ckpt_path}")
        self.model_wrapper = DiffusionDriveModelWrapper(
            checkpoint_path=self.ckpt_path, config_path=self.config_path
        )

        # Initialize sensor processor
        self.sensor_processor = SensorProcessor()

        # Initialize trajectory planner
        self.trajectory_planner = TrajectoryPlanner()

        # Image transformation for normalization
        self._im_transform = T.Compose(
            [T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
        )

        # GPS reference point (CARLA default)
        self.lat_ref, self.lon_ref = 42.0, 2.0

        # Previous control for smoothing
        control = carla.VehicleControl()
        control.steer = 0.0
        control.throttle = 0.0
        control.brake = 0.0
        self.prev_control = control
        self.prev_control_cache = []

        # Setup save path for video generation
        self.save_path = None
        if SAVE_PATH is not None:
            string = pathlib.Path(os.environ["ROUTES"]).stem + "_"
            string += self.save_name
            self.save_path = pathlib.Path(SAVE_PATH) / string
            self.save_path.mkdir(parents=True, exist_ok=False)

            # Create directories for different data types
            (self.save_path / "rgb_front").mkdir()
            (self.save_path / "rgb_front_right").mkdir()
            (self.save_path / "rgb_front_left").mkdir()
            (self.save_path / "rgb_back").mkdir()
            (self.save_path / "rgb_back_right").mkdir()
            (self.save_path / "rgb_back_left").mkdir()
            (self.save_path / "meta").mkdir()
            (self.save_path / "bev").mkdir()

        # Camera transformation matrices (from VAD agent)
        self._setup_camera_transforms()

        # History buffers for temporal features
        self.camera_history = deque(maxlen=4)  # Keep 4 frames of history
        self.ego_history = deque(maxlen=4)

        print("DiffusionDrive agent setup complete")

    def set_global_plan(self, global_plan_gps, global_plan_world_coord):
        """
        Set the route plan and store for navigation command extraction.
        """
        super().set_global_plan(global_plan_gps, global_plan_world_coord)

        # Store full route for command extraction
        # Each element is (waypoint, RoadOption)
        self._global_plan_world_coord_full = global_plan_world_coord
        print(f"Global plan set with {len(global_plan_world_coord)} waypoints")

    def _get_current_nav_command(self, current_location):
        """
        Extract navigation command from route based on current location.

        Returns:
            command_id: Integer command ID for model
            command_text: String description for visualization
        """
        if (
            not hasattr(self, "_global_plan_world_coord_full")
            or not self._global_plan_world_coord_full
        ):
            return 3, "STRAIGHT"  # Default

        # Find nearest waypoint in route
        min_dist = float("inf")
        nearest_cmd = None

        for waypoint, road_option in self._global_plan_world_coord_full[
            :50
        ]:  # Check next 50 waypoints
            dist = waypoint.location.distance(current_location)
            if dist < min_dist and dist < 5.0:  # Within 5 meters
                min_dist = dist
                nearest_cmd = road_option

        # Map RoadOption to command ID and text
        # Import at runtime to avoid circular dependency
        from agents.navigation.local_planner import RoadOption

        if nearest_cmd == RoadOption.LEFT:
            return 1, "LEFT"
        elif nearest_cmd == RoadOption.RIGHT:
            return 2, "RIGHT"
        elif nearest_cmd == RoadOption.STRAIGHT:
            return 3, "STRAIGHT"
        elif nearest_cmd == RoadOption.LANEFOLLOW:
            return 4, "LANEFOLLOW"
        elif nearest_cmd == RoadOption.CHANGELANELEFT:
            return 5, "CHANGELANELEFT"
        elif nearest_cmd == RoadOption.CHANGELANERIGHT:
            return 6, "CHANGELANERIGHT"
        else:
            return 3, "STRAIGHT"  # Default

    def _setup_camera_transforms(self):
        """Setup camera transformation matrices for sensor processing."""
        # These matrices are from the VAD agent - verified for Bench2Drive
        self.lidar2img = {
            "CAM_FRONT": np.array(
                [
                    [1.14251841e03, 8.00000000e02, 0.00000000e00, -9.52000000e02],
                    [0.00000000e00, 4.50000000e02, -1.14251841e03, -8.09704417e02],
                    [0.00000000e00, 1.00000000e00, 0.00000000e00, -1.19000000e00],
                    [0.00000000e00, 0.00000000e00, 0.00000000e00, 1.00000000e00],
                ]
            ),
            "CAM_FRONT_LEFT": np.array(
                [
                    [6.03961325e-14, 1.39475744e03, 0.00000000e00, -9.20539908e02],
                    [-3.68618420e02, 2.58109396e02, -1.14251841e03, -6.47296750e02],
                    [-8.19152044e-01, 5.73576436e-01, 0.00000000e00, -8.29094072e-01],
                    [0.00000000e00, 0.00000000e00, 0.00000000e00, 1.00000000e00],
                ]
            ),
            "CAM_FRONT_RIGHT": np.array(
                [
                    [1.31064327e03, -4.77035138e02, 0.00000000e00, -4.06010608e02],
                    [3.68618420e02, 2.58109396e02, -1.14251841e03, -6.47296750e02],
                    [8.19152044e-01, 5.73576436e-01, 0.00000000e00, -8.29094072e-01],
                    [0.00000000e00, 0.00000000e00, 0.00000000e00, 1.00000000e00],
                ]
            ),
            "CAM_BACK": np.array(
                [
                    [-5.60166031e02, -8.00000000e02, 0.00000000e00, -1.28800000e03],
                    [5.51091060e-14, -4.50000000e02, -5.60166031e02, -8.58939847e02],
                    [1.22464680e-16, -1.00000000e00, 0.00000000e00, -1.61000000e00],
                    [0.00000000e00, 0.00000000e00, 0.00000000e00, 1.00000000e00],
                ]
            ),
            "CAM_BACK_LEFT": np.array(
                [
                    [-1.14251841e03, 8.00000000e02, 0.00000000e00, -6.84385123e02],
                    [-4.22861679e02, -1.53909064e02, -1.14251841e03, -4.96004706e02],
                    [-9.39692621e-01, -3.42020143e-01, 0.00000000e00, -4.92889531e-01],
                    [0.00000000e00, 0.00000000e00, 0.00000000e00, 1.00000000e00],
                ]
            ),
            "CAM_BACK_RIGHT": np.array(
                [
                    [3.60989788e02, -1.34723223e03, 0.00000000e00, -1.04238127e02],
                    [4.22861679e02, -1.53909064e02, -1.14251841e03, -4.96004706e02],
                    [9.39692621e-01, -3.42020143e-01, 0.00000000e00, -4.92889531e-01],
                    [0.00000000e00, 0.00000000e00, 0.00000000e00, 1.00000000e00],
                ]
            ),
        }

    def sensors(self):
        """
        Define the sensor suite required by the agent.

        Returns:
            List of sensor configurations
        """
        sensors = [
            # RGB cameras
            {
                "type": "sensor.camera.rgb",
                "id": "CAM_FRONT",
                "x": 1.5,
                "y": 0.0,
                "z": 2.4,
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": 0.0,
                "width": 1600,
                "height": 900,
                "fov": 100,
            },
            {
                "type": "sensor.camera.rgb",
                "id": "CAM_FRONT_LEFT",
                "x": 1.5,
                "y": -0.5,
                "z": 2.4,
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": -55.0,
                "width": 1600,
                "height": 900,
                "fov": 100,
            },
            {
                "type": "sensor.camera.rgb",
                "id": "CAM_FRONT_RIGHT",
                "x": 1.5,
                "y": 0.5,
                "z": 2.4,
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": 55.0,
                "width": 1600,
                "height": 900,
                "fov": 100,
            },
            {
                "type": "sensor.camera.rgb",
                "id": "CAM_BACK",
                "x": -1.5,
                "y": 0.0,
                "z": 2.4,
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": 180.0,
                "width": 1600,
                "height": 900,
                "fov": 100,
            },
            {
                "type": "sensor.camera.rgb",
                "id": "CAM_BACK_LEFT",
                "x": -1.5,
                "y": -0.5,
                "z": 2.4,
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": -125.0,
                "width": 1600,
                "height": 900,
                "fov": 100,
            },
            {
                "type": "sensor.camera.rgb",
                "id": "CAM_BACK_RIGHT",
                "x": -1.5,
                "y": 0.5,
                "z": 2.4,
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": 125.0,
                "width": 1600,
                "height": 900,
                "fov": 100,
            },
            # Other sensors
            {
                "type": "sensor.other.imu",
                "id": "IMU",
                "x": 0.0,
                "y": 0.0,
                "z": 0.0,
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": 0.0,
            },
            {
                "type": "sensor.other.gnss",
                "id": "GPS",
                "x": 0.0,
                "y": 0.0,
                "z": 0.0,
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": 0.0,
            },
            {
                "type": "sensor.speedometer",
                "id": "SPEED",
                "x": 0.0,
                "y": 0.0,
                "z": 0.0,
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": 0.0,
            },
        ]

        return sensors

    def tick(self, input_data):
        """
        Process sensor data at each tick.

        Args:
            input_data: Dictionary of sensor data
        """
        # Store raw sensor data for processing in run_step
        self.raw_sensor_data = input_data

        # Extract GPS and IMU data
        gps = input_data["GPS"][1]
        imu = input_data["IMU"][1]
        speed = input_data["SPEED"][1]["speed"]

        # Process ego status
        ego_status = self.sensor_processor.process_ego_status(
            gps_data=gps,
            imu_data=imu,
            speed_data=speed,
            lat_ref=self.lat_ref,
            lon_ref=self.lon_ref,
        )

        # Get current navigation command from route
        current_transform = self._get_current_transform(input_data)
        self.current_command, self.current_command_text = self._get_current_nav_command(
            current_transform.location
        )

        # Add driving command to ego_status
        ego_status["driving_command"] = self.current_command

        # Update history buffers
        self.ego_history.append(ego_status)

        return input_data

    def run_step(self, input_data, timestamp):
        """
        Execute one step of the agent.

        Args:
            input_data: Current sensor data
            timestamp: Current timestamp

        Returns:
            carla.VehicleControl object
        """
        self.step += 1

        # Process camera images
        camera_data = {}
        for cam_id in [
            "CAM_FRONT",
            "CAM_FRONT_LEFT",
            "CAM_FRONT_RIGHT",
            "CAM_BACK",
            "CAM_BACK_LEFT",
            "CAM_BACK_RIGHT",
        ]:
            camera_data[cam_id] = input_data[cam_id][1]

        # If ego_history is empty (first frame), process current GPS/IMU data
        if not self.ego_history:
            # Get current ego status from input_data
            gps = input_data["GPS"][1]
            imu = input_data["IMU"][1]
            speed = input_data["SPEED"][1]["speed"] if "SPEED" in input_data else 0.0

            current_ego_status = self.sensor_processor.process_ego_status(
                gps_data=gps,
                imu_data=imu,
                speed_data=speed,
                lat_ref=self.lat_ref,
                lon_ref=self.lon_ref,
            )

            # Get navigation command for initial frame
            current_transform = self._get_current_transform(input_data)
            self.current_command, self.current_command_text = self._get_current_nav_command(
                current_transform.location
            )
            current_ego_status["driving_command"] = self.current_command

            self.ego_history.append(current_ego_status)

        # Process sensors to get model input features
        features = self.sensor_processor.process_carla_sensors(
            camera_data=camera_data,
            ego_history=list(self.ego_history),
            transform_matrices=self.lidar2img,
            im_transform=self._im_transform,
        )

        # Run model inference
        trajectory_output = self.model_wrapper.inference(features)

        # Get current vehicle transform (from GPS/IMU)
        current_transform = self._get_current_transform(input_data)

        # Convert trajectory to waypoints
        waypoints = self.trajectory_planner.trajectory_to_waypoints(
            trajectory=trajectory_output, ego_transform=current_transform
        )

        # Store trajectory for command determination and saving
        self.last_trajectory = trajectory_output
        self.last_waypoints = waypoints

        # Get current speed
        current_speed = input_data["SPEED"][1]["speed"]

        # Run PID controller to get control commands
        # Convert waypoints from carla.Location to numpy arrays for PID controller
        waypoints_np = []
        for wp in waypoints:
            if hasattr(wp, "x"):  # carla.Location or similar
                waypoints_np.append(np.array([wp.x, wp.y, wp.z]))
            else:  # Already numpy array
                waypoints_np.append(wp)
        waypoints_np = np.array(waypoints_np)

        # The PID controller expects waypoints, speed, and target
        # For target, we'll use the last waypoint as our goal
        target = waypoints_np[-1] if len(waypoints_np) > 0 else waypoints_np[0]
        steer, throttle, brake, metadata = self.pidcontroller.control_pid(
            waypoints=waypoints_np, speed=current_speed, target=target
        )

        # Create CARLA control object
        control = carla.VehicleControl()
        control.steer = float(steer)
        control.throttle = float(throttle)
        control.brake = float(brake)
        control.hand_brake = False
        control.manual_gear_shift = False

        # Apply control smoothing
        control = self._smooth_control(control)

        # Save data for video generation if enabled
        if self.save_path is not None and self.step % 10 == 0:
            self._save_evaluation_data(input_data, control)

        # Update previous control
        self.prev_control = control
        self.prev_control_cache.append(control)
        if len(self.prev_control_cache) > 5:
            self.prev_control_cache.pop(0)

        return control

    def _get_current_transform(self, input_data):
        """
        Get current vehicle transform from sensor data.

        Args:
            input_data: Sensor data dictionary

        Returns:
            carla.Transform object
        """
        gps = input_data["GPS"][1]
        imu = input_data["IMU"][1]

        # Handle both array and dictionary format for GPS
        if isinstance(gps, (list, tuple, np.ndarray)):
            # CARLA GPS sensor returns array: [latitude, longitude, altitude]
            lat = gps[0]
            lon = gps[1]
            alt = gps[2] if len(gps) > 2 else 0.0
        else:
            # Dictionary format
            lat = gps["latitude"]
            lon = gps["longitude"]
            alt = gps["altitude"]

        # Simple conversion (CARLA uses a flat earth model internally)
        x = (lon - self.lon_ref) * 111320.0 * math.cos(math.radians(self.lat_ref))
        y = -(lat - self.lat_ref) * 111320.0
        z = alt

        location = carla.Location(x=x, y=y, z=z)

        # Handle both array and dictionary format for IMU
        if isinstance(imu, (list, tuple, np.ndarray)):
            # CARLA IMU sensor returns arrays - need to handle properly
            # IMU typically has accelerometer, gyroscope, and compass data
            # For now, we'll use default orientation since exact format is unclear
            pitch = 0.0
            roll = 0.0
            yaw = 0.0
        else:
            # Dictionary format
            pitch = math.degrees(imu["accelerometer"][0])
            roll = math.degrees(imu["accelerometer"][1])
            yaw = math.degrees(imu["compass"])

        rotation = carla.Rotation(pitch=pitch, yaw=yaw, roll=roll)

        return carla.Transform(location, rotation)

    def _smooth_control(self, control):
        """
        Apply smoothing to control commands.

        Args:
            control: Raw control commands

        Returns:
            Smoothed control commands
        """
        if len(self.prev_control_cache) > 0:
            # Average steering over recent controls
            avg_steer = sum(c.steer for c in self.prev_control_cache) / len(
                self.prev_control_cache
            )
            control.steer = 0.7 * control.steer + 0.3 * avg_steer

            # Clamp steering change rate
            max_steer_change = 0.3
            steer_change = control.steer - self.prev_control.steer
            if abs(steer_change) > max_steer_change:
                control.steer = self.prev_control.steer + np.sign(steer_change) * max_steer_change

        # Clamp control values
        control.steer = np.clip(control.steer, -1.0, 1.0)
        control.throttle = np.clip(control.throttle, 0.0, 0.75)
        control.brake = np.clip(control.brake, 0.0, 1.0)

        return control

    def _save_evaluation_data(self, input_data, control):
        """
        Save data for video generation.

        Args:
            input_data: Sensor data
            control: Control commands
        """
        # Save RGB images
        for cam_id in [
            "CAM_FRONT",
            "CAM_FRONT_LEFT",
            "CAM_FRONT_RIGHT",
            "CAM_BACK",
            "CAM_BACK_LEFT",
            "CAM_BACK_RIGHT",
        ]:
            image = input_data[cam_id][1]

            # Convert to PIL Image if necessary
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)

            # Map camera IDs to directory names
            dir_map = {
                "CAM_FRONT": "rgb_front",
                "CAM_FRONT_LEFT": "rgb_front_left",
                "CAM_FRONT_RIGHT": "rgb_front_right",
                "CAM_BACK": "rgb_back",
                "CAM_BACK_LEFT": "rgb_back_left",
                "CAM_BACK_RIGHT": "rgb_back_right",
            }

            save_dir = self.save_path / dir_map[cam_id]

            # Convert RGBA to RGB if necessary (JPEG doesn't support transparency)
            if image.mode == "RGBA":
                # Create a white background
                rgb_image = Image.new("RGB", image.size, (255, 255, 255))
                rgb_image.paste(image, mask=image.split()[3])  # Use alpha channel as mask
                image = rgb_image
            elif image.mode != "RGB":
                image = image.convert("RGB")

            image.save(save_dir / f"{self.step:05d}.jpg", quality=85)

        # Save control metadata with ACTUAL predicted trajectory
        meta = {
            "step": self.step,
            "timestamp": float(input_data["CAM_FRONT"][0]),
            "control": {
                "steer": float(control.steer),
                "throttle": float(control.throttle),
                "brake": float(control.brake),
            },
            "speed": float(input_data["SPEED"][1]["speed"]),
            "command": self.current_command_text,
            "command_id": self.current_command,
            "predicted_trajectory": (
                self.last_trajectory[0].tolist() if hasattr(self, "last_trajectory") else None
            ),  # Save first mode of trajectory
            "predicted_waypoints": (
                [[w.x, w.y, w.z] for w in self.last_waypoints]
                if hasattr(self, "last_waypoints")
                else None
            ),
        }

        meta_dir = self.save_path / "meta"
        with open(meta_dir / f"{self.step:05d}.json", "w") as f:
            json.dump(meta, f, indent=2)

    def destroy(self):
        """Clean up resources when agent is destroyed."""
        if hasattr(self, "model_wrapper"):
            del self.model_wrapper
        torch.cuda.empty_cache()
