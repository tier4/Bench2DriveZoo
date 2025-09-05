"""
Metrics computation for DiffusionDrive open-loop evaluation.
Outputs metrics in VAD-compatible format for comparison.
"""

import json
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class TrajectoryMetrics:
    """Container for trajectory evaluation metrics."""
    # L2 distance errors at different time horizons (in seconds)
    L2_05s: float = 0.0  # 0.5 seconds (1 timestep at 2Hz)
    L2_10s: float = 0.0  # 1.0 seconds (2 timesteps at 2Hz)
    L2_15s: float = 0.0  # 1.5 seconds (3 timesteps at 2Hz)
    L2_20s: float = 0.0  # 2.0 seconds (4 timesteps at 2Hz)
    L2_25s: float = 0.0  # 2.5 seconds (5 timesteps at 2Hz)
    L2_30s: float = 0.0  # 3.0 seconds (6 timesteps at 2Hz)
    
    # Average L2 error across all timesteps
    L2_avg: float = 0.0
    
    # Minimum L2 error across modes (for multi-modal predictions)
    minL2_05s: float = 0.0
    minL2_10s: float = 0.0
    minL2_15s: float = 0.0
    minL2_20s: float = 0.0
    minL2_25s: float = 0.0
    minL2_30s: float = 0.0
    minL2_avg: float = 0.0
    
    # Additional metrics
    collision_rate: float = 0.0
    offroad_rate: float = 0.0
    
    # Longitudinal and lateral errors
    long_error_avg: float = 0.0
    lat_error_avg: float = 0.0
    
    # Heading error
    heading_error_avg: float = 0.0  # In radians
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return asdict(self)


class OpenLoopEvaluator:
    """
    Evaluator for open-loop trajectory prediction.
    Computes metrics compatible with VAD evaluation format.
    """
    
    def __init__(self, sampling_rate_hz: float = 2.0):
        """
        Initialize evaluator.
        
        Args:
            sampling_rate_hz: Sampling rate of trajectories (default 2Hz for B2D)
        """
        self.sampling_rate_hz = sampling_rate_hz
        self.timestep_duration = 1.0 / sampling_rate_hz
        
        # Define evaluation horizons in timesteps
        self.eval_horizons_sec = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
        self.eval_horizons_steps = [int(t * sampling_rate_hz) for t in self.eval_horizons_sec]
    
    def compute_trajectory_metrics(
        self,
        pred_trajectory: np.ndarray,
        gt_trajectory: np.ndarray,
        pred_modes: Optional[np.ndarray] = None
    ) -> TrajectoryMetrics:
        """
        Compute trajectory prediction metrics.
        
        Args:
            pred_trajectory: Predicted trajectory [T, 3] (x, y, heading) or [M, T, 3] for multi-modal
            gt_trajectory: Ground truth trajectory [T, 3]
            pred_modes: Optional mode probabilities [M] for multi-modal predictions
        
        Returns:
            TrajectoryMetrics object
        """
        metrics = TrajectoryMetrics()
        
        # Handle single vs multi-modal predictions
        if pred_trajectory.ndim == 2:
            # Single mode prediction
            pred_trajectory = pred_trajectory[np.newaxis, ...]  # Add mode dimension
            pred_modes = np.array([1.0])
        
        num_modes = pred_trajectory.shape[0]
        num_timesteps = min(pred_trajectory.shape[1], gt_trajectory.shape[0])
        
        # Compute L2 errors for each mode
        l2_errors_per_mode = np.zeros((num_modes, num_timesteps))
        
        for mode_idx in range(num_modes):
            for t in range(num_timesteps):
                pred_pos = pred_trajectory[mode_idx, t, :2]  # x, y
                gt_pos = gt_trajectory[t, :2]
                l2_errors_per_mode[mode_idx, t] = np.linalg.norm(pred_pos - gt_pos)
        
        # Compute metrics at different horizons
        for horizon_idx, (horizon_sec, horizon_steps) in enumerate(
            zip(self.eval_horizons_sec, self.eval_horizons_steps)
        ):
            if horizon_steps <= num_timesteps:
                # Average L2 error (weighted by mode probabilities if available)
                if pred_modes is not None:
                    avg_l2 = np.sum(pred_modes[:, np.newaxis] * l2_errors_per_mode[:, horizon_steps-1]) / np.sum(pred_modes)
                else:
                    avg_l2 = np.mean(l2_errors_per_mode[:, horizon_steps-1])
                
                # Minimum L2 error across modes
                min_l2 = np.min(l2_errors_per_mode[:, horizon_steps-1])
                
                # Set metrics
                setattr(metrics, f"L2_{int(horizon_sec*10):02d}s", avg_l2)
                setattr(metrics, f"minL2_{int(horizon_sec*10):02d}s", min_l2)
        
        # Average L2 error across all timesteps
        if pred_modes is not None:
            metrics.L2_avg = np.sum(pred_modes[:, np.newaxis] * np.mean(l2_errors_per_mode[:, :num_timesteps], axis=1)) / np.sum(pred_modes)
        else:
            metrics.L2_avg = np.mean(l2_errors_per_mode[:, :num_timesteps])
        
        metrics.minL2_avg = np.mean(np.min(l2_errors_per_mode[:, :num_timesteps], axis=0))
        
        # Compute longitudinal and lateral errors (in vehicle frame)
        best_mode_idx = np.argmin(np.mean(l2_errors_per_mode[:, :num_timesteps], axis=1))
        best_pred = pred_trajectory[best_mode_idx]
        
        long_errors = []
        lat_errors = []
        heading_errors = []
        
        for t in range(num_timesteps):
            # Transform to vehicle frame at time t
            if t > 0:
                # Use previous position as reference
                ref_pos = gt_trajectory[t-1, :2]
                ref_heading = gt_trajectory[t-1, 2] if gt_trajectory.shape[1] > 2 else 0.0
            else:
                ref_pos = np.array([0.0, 0.0])
                ref_heading = 0.0
            
            # Compute errors in vehicle frame
            pred_delta = best_pred[t, :2] - ref_pos
            gt_delta = gt_trajectory[t, :2] - ref_pos
            
            # Rotate to vehicle frame
            cos_h = np.cos(ref_heading)
            sin_h = np.sin(ref_heading)
            rot_matrix = np.array([[cos_h, sin_h], [-sin_h, cos_h]])
            
            pred_vf = rot_matrix @ pred_delta
            gt_vf = rot_matrix @ gt_delta
            
            long_errors.append(abs(pred_vf[0] - gt_vf[0]))
            lat_errors.append(abs(pred_vf[1] - gt_vf[1]))
            
            # Heading error
            if pred_trajectory.shape[2] > 2 and gt_trajectory.shape[1] > 2:
                heading_diff = best_pred[t, 2] - gt_trajectory[t, 2]
                # Normalize to [-pi, pi]
                heading_diff = np.arctan2(np.sin(heading_diff), np.cos(heading_diff))
                heading_errors.append(abs(heading_diff))
        
        metrics.long_error_avg = np.mean(long_errors) if long_errors else 0.0
        metrics.lat_error_avg = np.mean(lat_errors) if lat_errors else 0.0
        metrics.heading_error_avg = np.mean(heading_errors) if heading_errors else 0.0
        
        return metrics
    
    def compute_collision_rate(
        self,
        pred_trajectory: np.ndarray,
        obstacles: List[Dict[str, Any]],
        vehicle_size: Tuple[float, float] = (4.5, 2.0)
    ) -> float:
        """
        Compute collision rate with obstacles.
        
        Args:
            pred_trajectory: Predicted trajectory [T, 3] or [M, T, 3]
            obstacles: List of obstacle dictionaries with 'position' and 'size'
            vehicle_size: Vehicle dimensions (length, width) in meters
        
        Returns:
            Collision rate (0.0 to 1.0)
        """
        if len(obstacles) == 0:
            return 0.0
        
        # Handle multi-modal predictions - use best mode
        if pred_trajectory.ndim == 3:
            # Use mode with lowest average distance from origin (simple heuristic)
            mode_distances = np.mean(np.linalg.norm(pred_trajectory[:, :, :2], axis=2), axis=1)
            best_mode = np.argmin(mode_distances)
            pred_trajectory = pred_trajectory[best_mode]
        
        collision_detected = False
        
        for t in range(pred_trajectory.shape[0]):
            pred_pos = pred_trajectory[t, :2]
            pred_heading = pred_trajectory[t, 2] if pred_trajectory.shape[1] > 2 else 0.0
            
            # Check collision with each obstacle
            for obstacle in obstacles:
                if 'position' not in obstacle:
                    raise KeyError(f"'position' not found in obstacle: {obstacle}")
                if 'size' not in obstacle:
                    raise KeyError(f"'size' not found in obstacle: {obstacle}")
                    
                obs_pos = obstacle['position']
                obs_size = obstacle['size']
                
                # Simple bounding box collision check
                # TODO: Implement oriented bounding box for more accuracy
                dx = abs(pred_pos[0] - obs_pos[0])
                dy = abs(pred_pos[1] - obs_pos[1])
                
                collision_threshold_x = (vehicle_size[0] + obs_size[0]) / 2
                collision_threshold_y = (vehicle_size[1] + obs_size[1]) / 2
                
                if dx < collision_threshold_x and dy < collision_threshold_y:
                    collision_detected = True
                    break
            
            if collision_detected:
                break
        
        return 1.0 if collision_detected else 0.0
    
    def compute_offroad_rate(
        self,
        pred_trajectory: np.ndarray,
        road_boundaries: Optional[np.ndarray] = None
    ) -> float:
        """
        Compute rate of trajectory going off-road.
        
        Args:
            pred_trajectory: Predicted trajectory [T, 3] or [M, T, 3]
            road_boundaries: Optional road boundary polygon
        
        Returns:
            Off-road rate (0.0 to 1.0)
        """
        # TODO: Implement proper road boundary checking
        # For now, return 0 as placeholder
        return 0.0
    
    def aggregate_metrics(
        self,
        metrics_list: List[TrajectoryMetrics]
    ) -> Dict[str, Any]:
        """
        Aggregate metrics across multiple samples.
        
        Args:
            metrics_list: List of TrajectoryMetrics objects
        
        Returns:
            Aggregated metrics dictionary
        """
        if not metrics_list:
            return {}
        
        # Convert to numpy array for easier computation
        metrics_array = np.array([
            list(m.to_dict().values()) for m in metrics_list
        ])
        
        metric_names = list(metrics_list[0].to_dict().keys())
        
        # Compute statistics
        aggregated = {
            'mean': {},
            'std': {},
            'min': {},
            'max': {},
            'num_samples': len(metrics_list)
        }
        
        for idx, name in enumerate(metric_names):
            values = metrics_array[:, idx]
            aggregated['mean'][name] = float(np.mean(values))
            aggregated['std'][name] = float(np.std(values))
            aggregated['min'][name] = float(np.min(values))
            aggregated['max'][name] = float(np.max(values))
        
        return aggregated
    
    def save_metrics(
        self,
        metrics: Dict[str, Any],
        output_path: str,
        format: str = 'json'
    ):
        """
        Save metrics to file.
        
        Args:
            metrics: Metrics dictionary
            output_path: Output file path
            format: Output format ('json' or 'yaml')
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == 'json':
            with open(output_path, 'w') as f:
                json.dump(metrics, f, indent=2)
        elif format == 'yaml':
            import yaml
            with open(output_path, 'w') as f:
                yaml.dump(metrics, f, default_flow_style=False)
        else:
            raise ValueError(f"Unknown format: {format}")
        
        logger.info(f"Saved metrics to {output_path}")
    
    def format_vad_compatible(
        self,
        metrics_list: List[Tuple[str, TrajectoryMetrics]]
    ) -> Dict[str, Any]:
        """
        Format metrics in VAD-compatible structure.
        
        Args:
            metrics_list: List of (scenario_id, metrics) tuples
        
        Returns:
            VAD-compatible metrics dictionary
        """
        vad_format = {
            'meta': {
                'model': 'DiffusionDrive',
                'dataset': 'Bench2Drive',
                'eval_type': 'open_loop'
            },
            'results': {},
            'aggregated': {}
        }
        
        # Store per-scenario results
        for scenario_id, metrics in metrics_list:
            vad_format['results'][scenario_id] = {
                'L2': {
                    '0.5s': metrics.L2_05s,
                    '1.0s': metrics.L2_10s,
                    '1.5s': metrics.L2_15s,
                    '2.0s': metrics.L2_20s,
                    '2.5s': metrics.L2_25s,
                    '3.0s': metrics.L2_30s,
                    'avg': metrics.L2_avg
                },
                'minL2': {
                    '0.5s': metrics.minL2_05s,
                    '1.0s': metrics.minL2_10s,
                    '1.5s': metrics.minL2_15s,
                    '2.0s': metrics.minL2_20s,
                    '2.5s': metrics.minL2_25s,
                    '3.0s': metrics.minL2_30s,
                    'avg': metrics.minL2_avg
                },
                'collision_rate': metrics.collision_rate,
                'offroad_rate': metrics.offroad_rate
            }
        
        # Compute aggregated metrics
        if metrics_list:
            metrics_objects = [m for _, m in metrics_list]
            vad_format['aggregated'] = self.aggregate_metrics(metrics_objects)
        
        return vad_format


if __name__ == "__main__":
    # Test metrics computation
    import argparse
    
    parser = argparse.ArgumentParser(description="Test metrics computation")
    parser.add_argument("--output", type=str, default="test_metrics.json",
                       help="Output file for test metrics")
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO,
                       format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Create evaluator
    evaluator = OpenLoopEvaluator()
    
    # Create dummy data for testing
    print("\n=== Testing Metrics Computation ===")
    
    # Dummy trajectories
    gt_trajectory = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [2, 0.1, 0.1],
        [3, 0.2, 0.1],
        [4, 0.3, 0.2],
        [5, 0.5, 0.2],
        [6, 0.7, 0.3],
        [7, 1.0, 0.3]
    ])
    
    # Single mode prediction with some error
    pred_trajectory = gt_trajectory + np.random.normal(0, 0.2, gt_trajectory.shape)
    
    # Compute metrics
    metrics = evaluator.compute_trajectory_metrics(pred_trajectory, gt_trajectory)
    
    print("\nSingle Mode Metrics:")
    for key, value in metrics.to_dict().items():
        print(f"  {key}: {value:.4f}")
    
    # Multi-modal prediction
    pred_multi = np.stack([
        pred_trajectory,
        gt_trajectory + np.random.normal(0, 0.5, gt_trajectory.shape),
        gt_trajectory + np.random.normal(0, 1.0, gt_trajectory.shape)
    ])
    
    metrics_multi = evaluator.compute_trajectory_metrics(pred_multi, gt_trajectory)
    
    print("\nMulti-Modal Metrics:")
    for key, value in metrics_multi.to_dict().items():
        print(f"  {key}: {value:.4f}")
    
    # Save test metrics
    test_results = evaluator.format_vad_compatible([
        ("test_scenario_1", metrics),
        ("test_scenario_2", metrics_multi)
    ])
    
    evaluator.save_metrics(test_results, args.output)
    print(f"\nSaved test metrics to {args.output}")