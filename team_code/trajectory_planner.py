"""
Trajectory planner for converting DiffusionDrive output to CARLA waypoints.
"""

import numpy as np
import carla
from typing import List


class TrajectoryPlanner:
    """
    Converts DiffusionDrive trajectory predictions to CARLA waypoints.
    """

    def __init__(self):
        """Initialize the trajectory planner."""
        # Trajectory parameters
        self.num_modes = 20
        self.num_timesteps = 8
        self.dt = 0.5  # Time interval between waypoints (seconds)

        # Mode selection strategy
        self.mode_selection = "first"  # Options: 'first', 'center', 'scoring'

        # Trajectory smoothing
        self.smooth_trajectory = True
        self.smoothing_window = 3

    def trajectory_to_waypoints(
        self, trajectory: np.ndarray, ego_transform: carla.Transform
    ) -> List[carla.Location]:
        """
        Convert model trajectory output to CARLA waypoints.

        Args:
            trajectory: Model output of shape [num_modes, num_timesteps, 2]
                       Contains relative positions in ego frame
            ego_transform: Current vehicle transform in world coordinates

        Returns:
            List of carla.Location objects representing waypoints
        """
        # Select best trajectory mode
        selected_trajectory = self._select_best_mode(trajectory)

        # Smooth trajectory if enabled
        if self.smooth_trajectory:
            selected_trajectory = self._smooth_trajectory(selected_trajectory)

        # Convert relative positions to world coordinates
        waypoints = self._convert_to_world_coordinates(selected_trajectory, ego_transform)

        return waypoints

    def _select_best_mode(self, trajectory: np.ndarray) -> np.ndarray:
        """
        Select the best trajectory mode from predictions.

        Args:
            trajectory: Shape [num_modes, num_timesteps, 2]

        Returns:
            Selected trajectory of shape [num_timesteps, 2]
        """
        if self.mode_selection == "first":
            # Simply take the first mode (often the most likely)
            selected = trajectory[0]

        elif self.mode_selection == "center":
            # Take the center mode
            center_idx = self.num_modes // 2
            selected = trajectory[center_idx]

        elif self.mode_selection == "scoring":
            # Score each mode and select best
            # This would require additional scoring logic
            scores = self._score_trajectories(trajectory)
            best_idx = np.argmax(scores)
            selected = trajectory[best_idx]

        else:
            # Default to first mode
            selected = trajectory[0]

        return selected

    def _score_trajectories(self, trajectory: np.ndarray) -> np.ndarray:
        """
        Score trajectory modes for selection.

        Args:
            trajectory: Shape [num_modes, num_timesteps, 2]

        Returns:
            Scores for each mode
        """
        scores = np.zeros(self.num_modes)

        for i in range(self.num_modes):
            mode_traj = trajectory[i]

            # Score based on smoothness (penalize sharp turns)
            velocities = np.diff(mode_traj, axis=0)
            accelerations = np.diff(velocities, axis=0)
            smoothness_score = -np.sum(np.linalg.norm(accelerations, axis=1))

            # Score based on forward progress
            forward_progress = mode_traj[-1, 0]  # X coordinate at final timestep

            # Combined score
            scores[i] = forward_progress + 0.1 * smoothness_score

        return scores

    def _smooth_trajectory(self, trajectory: np.ndarray) -> np.ndarray:
        """
        Apply smoothing to trajectory.

        Args:
            trajectory: Shape [num_timesteps, 2]

        Returns:
            Smoothed trajectory
        """
        if len(trajectory) < self.smoothing_window:
            return trajectory

        smoothed = np.copy(trajectory)

        # Apply moving average
        for i in range(len(trajectory)):
            start_idx = max(0, i - self.smoothing_window // 2)
            end_idx = min(len(trajectory), i + self.smoothing_window // 2 + 1)
            smoothed[i] = np.mean(trajectory[start_idx:end_idx], axis=0)

        return smoothed

    def _convert_to_world_coordinates(
        self, trajectory: np.ndarray, ego_transform: carla.Transform
    ) -> List[carla.Location]:
        """
        Convert trajectory from ego coordinates to world coordinates.

        Args:
            trajectory: Relative positions in ego frame [num_timesteps, 2]
            ego_transform: Current vehicle transform

        Returns:
            List of waypoints in world coordinates
        """
        waypoints = []

        # Get ego rotation matrix
        yaw = np.radians(ego_transform.rotation.yaw)
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)

        for i in range(len(trajectory)):
            # Get relative position
            rel_x = trajectory[i, 0]
            rel_y = trajectory[i, 1]

            # Rotate to world frame
            world_x = cos_yaw * rel_x - sin_yaw * rel_y
            world_y = sin_yaw * rel_x + cos_yaw * rel_y

            # Translate to world position
            world_x += ego_transform.location.x
            world_y += ego_transform.location.y

            # Create waypoint (maintain ego z-coordinate)
            waypoint = carla.Location(x=world_x, y=world_y, z=ego_transform.location.z)

            waypoints.append(waypoint)

        return waypoints

    def compute_target_speed(
        self, waypoints: List[carla.Location], current_speed: float
    ) -> List[float]:
        """
        Compute target speeds for each waypoint.

        Args:
            waypoints: List of waypoints
            current_speed: Current vehicle speed (m/s)

        Returns:
            List of target speeds for each waypoint
        """
        target_speeds = []

        for i in range(len(waypoints)):
            if i == 0:
                # First waypoint - gradual change from current speed
                distance = self._compute_distance(carla.Location(x=0, y=0, z=0), waypoints[0])
            else:
                # Compute distance between consecutive waypoints
                distance = self._compute_distance(waypoints[i - 1], waypoints[i])

            # Target speed based on distance and time interval
            target_speed = distance / self.dt

            # Apply speed limits
            target_speed = np.clip(target_speed, 0, 30)  # Max 30 m/s (~108 km/h)

            # Smooth speed changes
            if i == 0:
                target_speed = 0.7 * current_speed + 0.3 * target_speed
            else:
                target_speed = 0.7 * target_speeds[-1] + 0.3 * target_speed

            target_speeds.append(target_speed)

        return target_speeds

    def _compute_distance(self, loc1: carla.Location, loc2: carla.Location) -> float:
        """
        Compute Euclidean distance between two locations.

        Args:
            loc1: First location
            loc2: Second location

        Returns:
            Distance in meters
        """
        dx = loc2.x - loc1.x
        dy = loc2.y - loc1.y
        dz = loc2.z - loc1.z
        return np.sqrt(dx**2 + dy**2 + dz**2)

    def interpolate_trajectory(
        self, waypoints: List[carla.Location], num_points: int
    ) -> List[carla.Location]:
        """
        Interpolate between waypoints for smoother control.

        Args:
            waypoints: Original waypoints
            num_points: Number of interpolated points

        Returns:
            Interpolated waypoints
        """
        if len(waypoints) < 2:
            return waypoints

        # Convert to numpy array for easier manipulation
        points = np.array([[wp.x, wp.y, wp.z] for wp in waypoints])

        # Create parameter values for original points
        t_original = np.linspace(0, 1, len(points))

        # Create parameter values for interpolated points
        t_interp = np.linspace(0, 1, num_points)

        # Interpolate each dimension
        x_interp = np.interp(t_interp, t_original, points[:, 0])
        y_interp = np.interp(t_interp, t_original, points[:, 1])
        z_interp = np.interp(t_interp, t_original, points[:, 2])

        # Create interpolated waypoints
        interpolated = []
        for i in range(num_points):
            waypoint = carla.Location(x=x_interp[i], y=y_interp[i], z=z_interp[i])
            interpolated.append(waypoint)

        return interpolated
