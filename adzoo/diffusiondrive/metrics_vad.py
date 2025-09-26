"""
VAD-compatible metrics implementation.
Supports both VAD (period-average) and UniAD (point-wise) L2 calculation methods.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict


@dataclass
class PlanningMetrics:
    """Planning trajectory metrics compatible with VAD/UniAD format."""
    
    # L2 metrics at specific time horizons (renamed with Planning_ prefix)
    Planning_L2_05s: float = 0.0  # 0.5 seconds
    Planning_L2_10s: float = 0.0  # 1.0 seconds
    Planning_L2_15s: float = 0.0  # 1.5 seconds
    Planning_L2_20s: float = 0.0  # 2.0 seconds
    Planning_L2_25s: float = 0.0  # 2.5 seconds
    Planning_L2_30s: float = 0.0  # 3.0 seconds
    Planning_L2_avg: float = 0.0  # Average across all timesteps
    
    # Collision metrics (object collision)
    Planning_obj_col_05s: float = 0.0
    Planning_obj_col_10s: float = 0.0
    Planning_obj_col_15s: float = 0.0
    Planning_obj_col_20s: float = 0.0
    Planning_obj_col_25s: float = 0.0
    Planning_obj_col_30s: float = 0.0
    Planning_obj_col_avg: float = 0.0
    
    # Collision metrics (bounding box collision)
    Planning_obj_box_col_05s: float = 0.0
    Planning_obj_box_col_10s: float = 0.0
    Planning_obj_box_col_15s: float = 0.0
    Planning_obj_box_col_20s: float = 0.0
    Planning_obj_box_col_25s: float = 0.0
    Planning_obj_box_col_30s: float = 0.0
    Planning_obj_box_col_avg: float = 0.0
    
    # Optional: minimum L2 across modes (for multi-modal predictions)
    Planning_minL2_05s: float = 0.0
    Planning_minL2_10s: float = 0.0
    Planning_minL2_15s: float = 0.0
    Planning_minL2_20s: float = 0.0
    Planning_minL2_25s: float = 0.0
    Planning_minL2_30s: float = 0.0
    Planning_minL2_avg: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary format for JSON output."""
        return asdict(self)


class VADCompatibleMetrics:
    """
    Compute planning metrics compatible with both VAD and UniAD evaluation.
    
    Key difference:
    - UniAD: Point-wise L2 at specific timesteps
    - VAD: Period-average L2 over time intervals [0, t]
    """
    
    def __init__(self, 
                 method: str = 'vad',
                 model_type: str = 'vad',
                 sampling_rate_hz: float = 10.0,
                 compute_collision: bool = True):  # Not used directly anymore
        """
        Initialize metrics calculator.
        
        Args:
            method: 'vad' for period-average or 'uniad' for point-wise
            model_type: 'vad' (6 timesteps), 'uniad' (4 timesteps), or 'diffusiondrive' (8 timesteps)
            sampling_rate_hz: Kept for compatibility but horizons are based on model_type
            compute_collision: Whether to compute collision metrics
        """
        assert method in ['vad', 'uniad'], f"Invalid method: {method}. Must be 'vad' or 'uniad'"
        assert model_type in ['vad', 'uniad', 'diffusiondrive'], f"Invalid model_type: {model_type}"
        
        self.method = method
        self.model_type = model_type
        self.sampling_rate_hz = sampling_rate_hz
        self.timestep_duration = 1.0 / sampling_rate_hz
        self.compute_collision = compute_collision
        
        # Initialize planning metric helper for collision computation
        if self.compute_collision:
            from planning_metric_utils import PlanningMetric
            self.planning_metric = PlanningMetric()
        
        # Configure evaluation based on model type
        if model_type == 'vad':
            # VAD: 6 timesteps covering ~3 seconds
            num_timesteps = 6
            total_time = 3.0
            pass  # VAD: 6 timesteps over 3 seconds
        elif model_type == 'uniad':
            # UniAD: 4 timesteps covering ~2 seconds
            num_timesteps = 4
            total_time = 2.0
            pass  # UniAD: 4 timesteps over 2 seconds
        else:  # diffusiondrive
            # DiffusionDrive: 8 timesteps covering 4.0 seconds (2Hz)
            num_timesteps = 8
            total_time = 4.0
            pass  # DiffusionDrive: 8 timesteps over 4.0 seconds
        
        # Calculate actual timestep duration for this model
        self.model_timestep_duration = total_time / num_timesteps
        
        # Define evaluation horizons based on available timesteps
        self.eval_horizons_steps = list(range(1, num_timesteps + 1))
        self.eval_horizons_sec = [i * self.model_timestep_duration for i in self.eval_horizons_steps]
        
        # Store evaluation configuration silently
    
    def compute_planning_l2(self,
                           pred_trajectory: np.ndarray,
                           gt_trajectory: np.ndarray,
                           pred_modes: Optional[np.ndarray] = None,
                           gt_agent_states: Optional[np.ndarray] = None,
                           gt_agent_labels: Optional[np.ndarray] = None) -> PlanningMetrics:
        """
        Compute L2 and collision metrics with specified method (VAD or UniAD).
        
        Args:
            pred_trajectory: Predicted trajectory [T, 3] or [M, T, 3] for multi-modal
                           Shape: (timesteps, 3) or (modes, timesteps, 3)
                           3 = (x, y, heading)
            gt_trajectory: Ground truth trajectory [T, 3]
            pred_modes: Optional mode probabilities [M] for multi-modal predictions
            gt_agent_states: Optional agent states [N, 5] (x, y, heading, length, width)
            gt_agent_labels: Optional agent validity flags [N]
            
        Returns:
            PlanningMetrics object with L2 and collision errors
        """
        metrics = PlanningMetrics()
        
        # Handle single vs multi-modal predictions
        if pred_trajectory.ndim == 2:
            # Single mode prediction - add mode dimension
            pred_trajectory = pred_trajectory[np.newaxis, ...]
            pred_modes = np.array([1.0])
        
        num_modes = pred_trajectory.shape[0]
        num_timesteps = min(pred_trajectory.shape[1], gt_trajectory.shape[0])
        
        # Debug: Print shape information
        # Process trajectories

        # Handle trajectory format based on evaluation method
        if self.method == 'vad':
            # CRITICAL FIX: Convert offset trajectories to absolute positions using cumsum
            # This matches VAD's approach where trajectories are stored as offsets
            # and converted to absolute positions before L2 calculation

            # Apply cumsum to convert offsets to absolute positions
            # pred_trajectory shape: [modes, timesteps, 3] where 3 = (x, y, heading)
            # gt_trajectory shape: [timesteps, 3]

            # Convert predictions from offsets to absolute positions
            pred_absolute = np.cumsum(pred_trajectory, axis=1)  # Cumsum along time axis

            # Convert GT from offsets to absolute positions
            gt_absolute = np.cumsum(gt_trajectory, axis=0)  # Cumsum along time axis

        elif self.method == 'uniad':
            # UniAD uses absolute positions directly (no cumsum needed)
            # This would require the input trajectories to already be absolute positions
            raise NotImplementedError(
                "UniAD evaluation method requires absolute position trajectories.\n"
                "The current pipeline converts to offsets for VAD compatibility.\n"
                "To use UniAD method: remove offset conversion in test_openloop_vad.py"
            )
        else:
            raise ValueError(f"Unknown method: {self.method}")

        # Compute L2 errors for each mode at each timestep
        l2_errors_per_mode = np.zeros((num_modes, num_timesteps))

        for mode_idx in range(num_modes):
            for t in range(num_timesteps):
                pred_pos = pred_absolute[mode_idx, t, :2]  # x, y only (absolute position)
                gt_pos = gt_absolute[t, :2]  # absolute position
                l2_errors_per_mode[mode_idx, t] = np.linalg.norm(pred_pos - gt_pos)
        
        # Compute metrics at different horizons
        for horizon_idx, (horizon_sec, horizon_steps) in enumerate(
            zip(self.eval_horizons_sec, self.eval_horizons_steps)
        ):
            if horizon_steps <= num_timesteps:
                if self.method == 'vad':
                    # VAD method: Average L2 over period [0, t]
                    period_errors = l2_errors_per_mode[:, :horizon_steps]
                    
                    # Average over time period for each mode
                    period_avg_per_mode = np.mean(period_errors, axis=1)
                    
                    # Weighted average across modes
                    if pred_modes is not None:
                        avg_l2 = np.sum(pred_modes * period_avg_per_mode) / np.sum(pred_modes)
                    else:
                        avg_l2 = np.mean(period_avg_per_mode)
                    
                    # Minimum across modes
                    min_l2 = np.min(period_avg_per_mode)
                    
                else:  # uniad
                    # UniAD method: Point-wise L2 at specific timestep
                    point_errors = l2_errors_per_mode[:, horizon_steps - 1]
                    
                    # Weighted average across modes
                    if pred_modes is not None:
                        avg_l2 = np.sum(pred_modes * point_errors) / np.sum(pred_modes)
                    else:
                        avg_l2 = np.mean(point_errors)
                    
                    # Minimum across modes
                    min_l2 = np.min(point_errors)
                
                # Set metrics with Planning_ prefix
                # Format horizon string based on actual seconds (e.g., "0.4s" -> "04s", "1.0s" -> "10s")
                horizon_str = f"{int(horizon_sec * 10):02d}s"
                setattr(metrics, f"Planning_L2_{horizon_str}", avg_l2)
                setattr(metrics, f"Planning_minL2_{horizon_str}", min_l2)
        
        # Compute overall average
        if self.method == 'vad':
            # For VAD: average of all period averages
            all_period_avgs = []
            for horizon_steps in self.eval_horizons_steps:
                if horizon_steps <= num_timesteps:
                    period_errors = l2_errors_per_mode[:, :horizon_steps]
                    period_avg_per_mode = np.mean(period_errors, axis=1)
                    if pred_modes is not None:
                        avg_l2 = np.sum(pred_modes * period_avg_per_mode) / np.sum(pred_modes)
                    else:
                        avg_l2 = np.mean(period_avg_per_mode)
                    all_period_avgs.append(avg_l2)
            
            metrics.Planning_L2_avg = np.mean(all_period_avgs) if all_period_avgs else 0.0
            
        else:  # uniad
            # For UniAD: average of point-wise errors at evaluation timesteps
            point_errors_at_horizons = []
            for horizon_steps in self.eval_horizons_steps:
                if horizon_steps <= num_timesteps:
                    point_errors = l2_errors_per_mode[:, horizon_steps - 1]
                    if pred_modes is not None:
                        avg_l2 = np.sum(pred_modes * point_errors) / np.sum(pred_modes)
                    else:
                        avg_l2 = np.mean(point_errors)
                    point_errors_at_horizons.append(avg_l2)
            
            metrics.Planning_L2_avg = np.mean(point_errors_at_horizons) if point_errors_at_horizons else 0.0
        
        # Compute minimum average
        min_mode_avg = np.mean(np.min(l2_errors_per_mode[:, :num_timesteps], axis=0))
        metrics.Planning_minL2_avg = min_mode_avg
        
        # Compute collision metrics if agent states are provided
        if self.compute_collision and gt_agent_states is not None and gt_agent_labels is not None:
            
            # Generate occupancy grids
            segmentation, pedestrian = self.planning_metric.get_label(
                gt_agent_states, gt_agent_labels, num_timesteps=num_timesteps
            )
            occupancy = np.logical_or(segmentation, pedestrian)
            
            # Compute collision for best mode (lowest L2)
            best_mode_idx = np.argmin(np.mean(l2_errors_per_mode[:, :num_timesteps], axis=1))
            best_traj = pred_trajectory[best_mode_idx:best_mode_idx+1, :num_timesteps, :]
            
            # Evaluate collision
            import torch
            obj_coll, obj_box_coll = self.planning_metric.evaluate_coll(
                torch.tensor(best_traj), 
                torch.tensor(gt_trajectory[:num_timesteps][np.newaxis, ...]),
                occupancy
            )
            
            # Set collision metrics for each horizon
            for horizon_idx, (horizon_sec, horizon_steps) in enumerate(
                zip(self.eval_horizons_sec, self.eval_horizons_steps)
            ):
                if horizon_steps <= num_timesteps:
                    if self.method == 'vad':
                        # Average collision rate over period [0, t]
                        obj_col_rate = torch.mean(obj_coll[0, :horizon_steps].float()).item()
                        obj_box_col_rate = torch.mean(obj_box_coll[0, :horizon_steps].float()).item()
                    else:
                        # Point-wise collision at specific timestep
                        obj_col_rate = obj_coll[0, horizon_steps - 1].item()
                        obj_box_col_rate = obj_box_coll[0, horizon_steps - 1].item()
                    
                    horizon_str = f"{int(horizon_sec * 10):02d}s"
                    setattr(metrics, f"Planning_obj_col_{horizon_str}", obj_col_rate)
                    setattr(metrics, f"Planning_obj_box_col_{horizon_str}", obj_box_col_rate)
            
            # Compute collision averages
            metrics.Planning_obj_col_avg = torch.mean(obj_coll.float()).item()
            metrics.Planning_obj_box_col_avg = torch.mean(obj_box_coll.float()).item()
        else:
            if not self.compute_collision:
                print("Collision computation disabled")
            else:
                print("No agent states provided, skipping collision metrics")
        
        return metrics
    
    def aggregate_metrics(self, all_sample_metrics: List[PlanningMetrics]) -> Dict[str, float]:
        """
        Aggregate metrics across all samples.
        
        Args:
            all_sample_metrics: List of PlanningMetrics for each sample
            
        Returns:
            Dictionary with aggregated metrics
        """
        if not all_sample_metrics:
            return {}
        
        # Convert all metrics to dictionaries
        all_dicts = [m.to_dict() for m in all_sample_metrics]
        
        # Aggregate by taking mean of each metric
        aggregated = {}
        for key in all_dicts[0].keys():
            values = [d[key] for d in all_dicts]
            aggregated[key] = np.mean(values)
        
        return aggregated


def compare_methods(pred_trajectory: np.ndarray, 
                    gt_trajectory: np.ndarray) -> Dict[str, Dict[str, float]]:
    """
    Compare VAD and UniAD metric calculation methods.
    
    Args:
        pred_trajectory: Predicted trajectory
        gt_trajectory: Ground truth trajectory
        
    Returns:
        Dictionary with metrics from both methods
    """
    # Calculate with VAD method
    vad_calculator = VADCompatibleMetrics(method='vad', sampling_rate_hz=10.0)
    vad_metrics = vad_calculator.compute_planning_l2(pred_trajectory, gt_trajectory)
    
    # Calculate with UniAD method  
    uniad_calculator = VADCompatibleMetrics(method='uniad', sampling_rate_hz=10.0)
    uniad_metrics = uniad_calculator.compute_planning_l2(pred_trajectory, gt_trajectory)
    
    return {
        'vad': vad_metrics.to_dict(),
        'uniad': uniad_metrics.to_dict()
    }


if __name__ == "__main__":
    # Test the metrics implementation
    print("Testing VAD-compatible metrics...")
    
    # Create dummy data (10Hz, 3 seconds = 30 timesteps)
    np.random.seed(42)
    timesteps = 30
    
    # Ground truth: straight line
    gt_trajectory = np.zeros((timesteps, 3))
    gt_trajectory[:, 0] = np.arange(timesteps) * 0.5  # x increases
    
    # Prediction: slightly off with some noise
    pred_trajectory = gt_trajectory.copy()
    pred_trajectory[:, 1] = np.random.normal(0, 0.2, timesteps)  # y noise
    
    print("\n1. Testing VAD method (period-average):")
    vad_calc = VADCompatibleMetrics(method='vad')
    vad_metrics = vad_calc.compute_planning_l2(pred_trajectory, gt_trajectory)
    print("VAD Metrics:")
    for key, value in vad_metrics.to_dict().items():
        if value > 0:
            print(f"  {key}: {value:.4f}")
    
    print("\n2. Testing UniAD method (point-wise):")
    uniad_calc = VADCompatibleMetrics(method='uniad')
    uniad_metrics = uniad_calc.compute_planning_l2(pred_trajectory, gt_trajectory)
    print("UniAD Metrics:")
    for key, value in uniad_metrics.to_dict().items():
        if value > 0:
            print(f"  {key}: {value:.4f}")
    
    print("\n3. Comparing methods:")
    comparison = compare_methods(pred_trajectory, gt_trajectory)
    print("Difference (VAD - UniAD):")
    for key in comparison['vad'].keys():
        if comparison['vad'][key] > 0:
            diff = comparison['vad'][key] - comparison['uniad'][key]
            print(f"  {key}: {diff:+.4f}")