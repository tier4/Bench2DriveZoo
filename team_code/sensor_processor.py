"""
Sensor processor for converting CARLA sensor data to DiffusionDrive format.
"""

import math
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from typing import Dict, List


class SensorProcessor:
    """
    Processes CARLA sensor data into features for DiffusionDrive model.
    """

    def __init__(self):
        """Initialize the sensor processor."""
        # Target dimensions for model input
        self.camera_width = 1024
        self.camera_height = 256

        # LiDAR dimensions (DiffusionDrive expects 256x256)
        self.lidar_size = 256
        self.lidar_range = 64.0  # meters

        # Status feature dimensions - model expects 8 (4 + 2 + 2)
        # 4: trajectory history, 2: velocity, 2: yaw info
        self.status_dim = 8

        # History length
        self.history_length = 4

    def process_carla_sensors(
        self, camera_data: Dict, ego_history: List, transform_matrices: Dict, im_transform
    ) -> Dict[str, torch.Tensor]:
        """
        Process CARLA sensor data into model features.

        Args:
            camera_data: Dictionary of camera images
            ego_history: List of ego status dictionaries
            transform_matrices: Camera transformation matrices
            im_transform: Image normalization transform

        Returns:
            Dictionary with processed features
        """
        features = {}

        # Process camera features
        features["camera_feature"] = self._process_cameras(camera_data, im_transform)

        # Process LiDAR features (placeholder - DiffusionDrive can work without LiDAR)
        features["lidar_feature"] = self._create_placeholder_lidar()

        # Process status features
        features["status_feature"] = self._process_ego_status(ego_history)

        return features

    def _process_cameras(self, camera_data: Dict, im_transform) -> torch.Tensor:
        """
        Process and stitch front cameras.

        Args:
            camera_data: Dictionary of camera images
            im_transform: Normalization transform

        Returns:
            Stitched and normalized camera tensor [3, H, W]
        """
        # Get front cameras in order: left, front, right
        front_cameras = ["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT"]

        processed_images = []
        for cam_id in front_cameras:
            image = camera_data[cam_id]

            # Convert to PIL Image if it's a numpy array
            if isinstance(image, np.ndarray):
                # CARLA returns BGR, convert to RGB
                if len(image.shape) == 3 and image.shape[2] == 4:
                    # BGRA format
                    image = image[:, :, :3]
                    image = image[:, :, ::-1]  # BGR to RGB
                elif len(image.shape) == 3 and image.shape[2] == 3:
                    # BGR format
                    image = image[:, :, ::-1]  # BGR to RGB

                image = Image.fromarray(image.astype("uint8"))

            # Resize to target size (each camera gets 1/3 of width)
            target_width = self.camera_width // 3
            image = image.resize((target_width, self.camera_height), Image.BILINEAR)

            # Apply normalization transform
            image_tensor = im_transform(image)
            processed_images.append(image_tensor)

        # Stitch cameras horizontally
        stitched = torch.cat(processed_images, dim=2)  # Concatenate along width

        # Ensure correct dimensions [3, H, W] - allow for small rounding differences
        if stitched.shape[2] != self.camera_width:
            # Resize to exact dimensions if there's a mismatch
            stitched = F.interpolate(
                stitched.unsqueeze(0),
                size=(self.camera_height, self.camera_width),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)

        return stitched

    def _create_placeholder_lidar(self) -> torch.Tensor:
        """
        Create placeholder LiDAR feature.
        DiffusionDrive can work without LiDAR, so we create a zero tensor.

        Returns:
            Zero tensor of shape [C, H, W] for LiDAR
        """
        # Create zero tensor with expected dimensions
        # Model expects 1 channel for lidar input
        lidar_feature = torch.zeros(1, self.lidar_size, self.lidar_size)

        return lidar_feature

    def _process_ego_status(self, ego_history: List[Dict]) -> torch.Tensor:
        """
        Process ego vehicle status history.

        Args:
            ego_history: List of ego status dictionaries

        Returns:
            Status feature tensor
        """
        # Use the most recent status (model expects single timestep, not history)
        if not ego_history:
            raise ValueError("No ego history available - cannot process status features")
        
        current_status = ego_history[-1]
        
        # Get driving command (default to 0 = STRAIGHT if not present)
        command_value = current_status.get("driving_command", 0)
        # Create one-hot encoding for 4 commands: STRAIGHT, FOLLOW, LEFT, RIGHT
        command_one_hot = torch.zeros(4)
        command_one_hot[min(int(command_value), 3)] = 1.0
        
        # Get velocity (x, y)
        velocity = torch.tensor(current_status["velocity"][:2], dtype=torch.float32)
        
        # Get acceleration (x, y)
        acceleration = torch.tensor(current_status["acceleration"][:2], dtype=torch.float32)
        
        # Concatenate all features [4 + 2 + 2 = 8]
        status_tensor = torch.cat([command_one_hot, velocity, acceleration])

        return status_tensor

    def process_ego_status(
        self, gps_data: Dict, imu_data: Dict, speed_data: float, lat_ref: float, lon_ref: float
    ) -> Dict:
        """
        Process GPS, IMU, and speed data into ego status.

        Args:
            gps_data: GPS sensor data
            imu_data: IMU sensor data
            speed_data: Speed in m/s
            lat_ref: Reference latitude
            lon_ref: Reference longitude

        Returns:
            Dictionary with ego status information
        """
        # Process GPS position
        # CARLA GPS sensor returns array: [latitude, longitude, altitude]
        if isinstance(gps_data, (list, tuple, np.ndarray)):
            lat = gps_data[0]
            lon = gps_data[1]
            alt = gps_data[2] if len(gps_data) > 2 else 0.0
        else:
            # Dictionary format (for compatibility)
            lat = gps_data["latitude"]
            lon = gps_data["longitude"]
            alt = gps_data["altitude"]

        # Convert to local coordinates (simple flat earth approximation)
        x = (lon - lon_ref) * 111320.0 * math.cos(math.radians(lat_ref))
        y = -(lat - lat_ref) * 111320.0
        z = alt

        # Process IMU data
        # CARLA IMU sensor returns array: [acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, compass]
        if isinstance(imu_data, (list, tuple, np.ndarray)):
            acceleration = [
                imu_data[0] if len(imu_data) > 0 else 0.0,
                imu_data[1] if len(imu_data) > 1 else 0.0,
                imu_data[2] if len(imu_data) > 2 else 0.0,
            ]
            angular_velocity = [
                imu_data[3] if len(imu_data) > 3 else 0.0,
                imu_data[4] if len(imu_data) > 4 else 0.0,
                imu_data[5] if len(imu_data) > 5 else 0.0,
            ]
            # Get heading from compass (convert to radians)
            heading = math.radians(imu_data[6]) if len(imu_data) > 6 else 0.0
        else:
            # Dictionary format (for compatibility)
            acceleration = [
                imu_data["accelerometer"][0],
                imu_data["accelerometer"][1],
                imu_data["accelerometer"][2],
            ]
            angular_velocity = [
                imu_data["gyroscope"][0],
                imu_data["gyroscope"][1],
                imu_data["gyroscope"][2],
            ]
            # Get heading from compass (convert to radians)
            heading = math.radians(imu_data["compass"])

        # Calculate velocity from speed and heading
        velocity = [speed_data * math.cos(heading), speed_data * math.sin(heading), 0.0]

        ego_status = {
            "position": [x, y, z],
            "velocity": velocity,
            "acceleration": acceleration,
            "angular_velocity": angular_velocity,
            "heading": heading,
        }

        return ego_status

    def stitch_cameras_for_visualization(self, camera_data: Dict) -> np.ndarray:
        """
        Stitch all 6 cameras for visualization/saving.

        Args:
            camera_data: Dictionary of camera images

        Returns:
            Stitched image array for visualization
        """
        # Arrange cameras in 2x3 grid
        # Top row: front_left, front, front_right
        # Bottom row: back_left, back, back_right

        top_row = []
        bottom_row = []

        for cam_id in ["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT"]:
            image = camera_data[cam_id]
            if isinstance(image, np.ndarray):
                top_row.append(image)
            else:
                top_row.append(np.array(image))

        for cam_id in ["CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT"]:
            image = camera_data[cam_id]
            if isinstance(image, np.ndarray):
                bottom_row.append(image)
            else:
                bottom_row.append(np.array(image))

        # Concatenate horizontally then vertically
        top = np.concatenate(top_row, axis=1)
        bottom = np.concatenate(bottom_row, axis=1)
        stitched = np.concatenate([top, bottom], axis=0)

        return stitched
