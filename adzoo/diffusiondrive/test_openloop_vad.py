"""
VAD-compatible open-loop evaluation script for DiffusionDrive on Bench2Drive.

CRITICAL: Uses 10Hz sampling (sampling_rate=1) for evaluation to match VAD/UniAD,
even though training uses 2Hz (sampling_rate=5).
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from tqdm import tqdm
import time

# Add DiffusionDrive to path
sys.path.append('/workspace/Bench2Drive/DiffusionDrive')

# Import DiffusionDrive modules
from navsim.common.bench2drive_dataloader import (
    Bench2DriveConfig, 
    Bench2DriveSceneLoader
)
from navsim.agents.diffusiondrive.transfuser_features_b2d import (
    Bench2DriveFeatureBuilder,
    Bench2DriveTargetBuilder
)
from navsim.agents.diffusiondrive.transfuser_model_v2 import V2TransfuserModel

# Import our custom modules
from data_split_loader import ValidationSplitLoader
from metrics_vad import VADCompatibleMetrics, PlanningMetrics

# Import config
from navsim.agents.diffusiondrive.transfuser_config import TransfuserConfig as B2DModelConfig


def load_model(checkpoint_path: str, device: str = 'cuda'):
    """
    Load DiffusionDrive model (same as in original test_openloop.py).
    
    Args:
        checkpoint_path: Path to checkpoint
        device: Device to load on
    
    Returns:
        Tuple of (model, config)
    """
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Create config
    config = B2DModelConfig()
    
    # Find trajectory anchors
    checkpoint_dir = Path(checkpoint_path).parent
    anchor_files = list(checkpoint_dir.glob("kmeans_b2d_*.npy"))
    if not anchor_files:
        anchor_files = list(Path("test_files").glob("kmeans_b2d_*.npy"))
    
    if anchor_files:
        config.plan_anchor_path = str(anchor_files[0])
        print(f"Using trajectory anchors: {config.plan_anchor_path}")
    
    # Create model
    model = V2TransfuserModel(config)
    
    # Load state dict
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    # Clean state dict keys (remove 'model.' prefix if present)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('model.'):
            k = k[6:]  # Remove 'model.' prefix
        new_state_dict[k] = v
    
    # Try loading, but don't fail if incompatible (for testing)
    try:
        model.load_state_dict(new_state_dict, strict=False)
        print(f"Loaded checkpoint weights")
    except Exception as e:
        print(f"Warning: Could not load checkpoint weights: {e}")
        print("Using randomly initialized model for testing")
    
    model = model.to(device)
    model.eval()
    
    return model, config


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='VAD-compatible evaluation for DiffusionDrive')
    
    # Data arguments
    parser.add_argument('--data_root', type=str, 
                       default='/mnt/nvme1/dataset/Bench2Drive-Base',
                       help='Path to Bench2Drive dataset')
    parser.add_argument('--split_file', type=str,
                       default='/workspace/Bench2Drive/Bench2DriveZoo/data/splits/bench2drive_base_train_val_split.json',
                       help='Path to train/val split JSON')
    
    # Model arguments
    parser.add_argument('--checkpoint', type=str,
                       default='/workspace/Bench2Drive/DiffusionDrive/checkpoints/diffusiondrive_b2d.ckpt',
                       help='Path to model checkpoint')
    parser.add_argument('--anchor_file', type=str,
                       default='/workspace/Bench2Drive/DiffusionDrive/checkpoints/transfuser_b2d_anchors.npy',
                       help='Path to trajectory anchors file')
    
    # Evaluation arguments
    parser.add_argument('--dev_mode', action='store_true',
                       help='Use only 2 scenarios for development testing')
    parser.add_argument('--full_eval', action='store_true',
                       help='Evaluate on full validation set')
    parser.add_argument('--metric_method', type=str, default='vad',
                       choices=['vad', 'uniad'],
                       help='Metric calculation method (default: vad)')
    parser.add_argument('--model_type', type=str, default='vad',
                       choices=['vad', 'uniad', 'diffusiondrive'],
                       help='Model type for timestep configuration (VAD:6, UniAD:4, DiffusionDrive:8)')
    parser.add_argument('--num_samples', type=int, default=-1,
                       help='Number of samples per scenario (-1 for all)')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str,
                       default='/workspace/Bench2Drive/Bench2DriveZoo/adzoo/diffusiondrive/eval_results',
                       help='Directory to save evaluation results')
    parser.add_argument('--save_per_sample', action='store_true',
                       help='Save per-sample results (can be large)')
    
    # Device arguments
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to use for inference')
    
    args = parser.parse_args()
    
    # Set evaluation mode
    if args.full_eval:
        args.dev_mode = False
    # If neither --dev_mode nor --full_eval specified, default to full evaluation
    # (dev_mode remains False as initialized)
    
    return args


def evaluate_scenario(
    model: V2TransfuserModel,
    scenario_path: str,
    feature_builder: Bench2DriveFeatureBuilder,
    target_builder: Bench2DriveTargetBuilder,
    metrics_calculator: VADCompatibleMetrics,
    num_samples: int = -1,
    device: str = 'cuda'
) -> Tuple[List[PlanningMetrics], Dict]:
    """
    Evaluate a single scenario.
    
    Args:
        model: The DiffusionDrive model
        scenario_path: Path to scenario directory
        feature_builder: Feature builder for input processing
        target_builder: Target builder for ground truth
        metrics_calculator: Metrics calculator (VAD or UniAD method)
        num_samples: Number of samples to evaluate (-1 for all)
        device: Device for inference
        
    Returns:
        Tuple of (list of per-sample metrics, aggregated metrics dict)
    """
    print(f"Evaluating scenario: {os.path.basename(scenario_path)}")
    
    # Create data loader for this scenario
    # CRITICAL: Use sampling_rate=1 for 10Hz (no frame skipping)
    config = Bench2DriveConfig(
        data_root=Path(scenario_path).parent,  # Parent directory
        scenarios=[Path(scenario_path).name],  # Just this scenario
        sampling_rate=1,  # CRITICAL: 10Hz for evaluation (not 5 like training!)
        num_frames=30,    # 3 seconds at 10Hz
    )
    
    scene_loader = Bench2DriveSceneLoader(config)
    
    # Get all scene tokens
    scene_tokens = scene_loader.get_scene_tokens()
    total_samples = len(scene_tokens)
    
    if num_samples > 0:
        eval_samples = min(num_samples, total_samples)
    else:
        eval_samples = total_samples
    
    print(f"  Total samples: {total_samples}, Evaluating: {eval_samples}")
    
    sample_metrics = []
    
    # Evaluate each sample
    for idx in tqdm(range(eval_samples), desc="Samples"):
        token = scene_tokens[idx]
        scene = scene_loader.get_scene(token)
        
        # Get agent input - using EXISTING scene method
        agent_input = scene.get_agent_input(0)  # Get first frame
        
        # Compute features using EXISTING feature builder method
        features = feature_builder.compute_features(agent_input)
        
        # Add batch dimension and move to device
        for key in features:
            features[key] = features[key].unsqueeze(0).to(device)
        
        # Run model inference
        with torch.no_grad():
            outputs = model(features)
        
        # Process sample silently
        
        # Extract predicted trajectory
        if isinstance(outputs, dict):
            if 'trajectory' in outputs:
                pred_trajectory = outputs['trajectory'].cpu().numpy()[0]
            elif 'plan_trajectory' in outputs:
                pred_trajectory = outputs['plan_trajectory'].cpu().numpy()[0]
            else:
                continue  # Unknown output format
        else:
            pred_trajectory = outputs.cpu().numpy()[0]
        
        # Get ground truth trajectory from scene (pass frame index 0)
        gt_trajectory = np.array(scene.get_future_trajectory(0))  # Get GT future trajectory
        
        # Check trajectory shapes
        
        # Interpolate predicted trajectory from 8 timesteps to 30 timesteps
        # Model outputs 8 timesteps at ~2.5Hz, we need 30 at 10Hz
        if pred_trajectory.shape[0] == 8 and gt_trajectory.shape[0] > 8:
            # Create interpolation indices
            old_indices = np.linspace(0, 3.2, 8)  # 8 points over 3.2 seconds
            new_indices = np.linspace(0, 3.0, 30)  # 30 points over 3.0 seconds
            
            # Interpolate each dimension
            from scipy import interpolate
            interp_traj = np.zeros((30, 3))
            for dim in range(3):
                f = interpolate.interp1d(old_indices, pred_trajectory[:, dim], 
                                        kind='linear', fill_value='extrapolate')
                interp_traj[:, dim] = f(new_indices)
            
            pred_trajectory = interp_traj
            pass  # Trajectory interpolated
        
        # Extract agent states and labels if available
        gt_agent_states = None
        gt_agent_labels = None
        if isinstance(outputs, dict):
            if 'agent_states' in outputs:
                gt_agent_states = outputs['agent_states'].cpu().numpy()[0]  # Remove batch dim
                gt_agent_labels = outputs.get('agent_labels', np.ones(gt_agent_states.shape[0], dtype=bool))
                if torch.is_tensor(gt_agent_labels):
                    gt_agent_labels = gt_agent_labels.cpu().numpy()[0]
                pass  # Agent states loaded
        
        # Compute metrics
        metrics = metrics_calculator.compute_planning_l2(
            pred_trajectory,
            gt_trajectory,
            pred_modes=outputs.get('mode_probs', None) if isinstance(outputs, dict) else None,
            gt_agent_states=gt_agent_states,
            gt_agent_labels=gt_agent_labels
        )
        
        sample_metrics.append(metrics)
    
    # Aggregate metrics
    aggregated = metrics_calculator.aggregate_metrics(sample_metrics)
    
    return sample_metrics, aggregated


def main():
    """Main evaluation function."""
    args = parse_args()
    start_time = time.time()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load validation split
    print("\n=== Loading Validation Split ===")
    split_loader = ValidationSplitLoader(args.split_file)
    scenarios = split_loader.get_validation_scenarios(
        dev_mode=args.dev_mode,
        num_dev_scenarios=2
    )
    
    print(f"Evaluation mode: {'Dev' if args.dev_mode else 'Full'}")
    print(f"Number of scenarios: {len(scenarios)}")
    print(f"Metric method: {args.metric_method.upper()}")
    print(f"Model type: {args.model_type.upper()} (timesteps configuration)")
    
    # Load model
    print("\n=== Loading Model ===")
    model, model_config = load_model(args.checkpoint, args.device)
    print(f"Model loaded from: {args.checkpoint}")
    
    # Create feature and target builders using the model config
    feature_builder = Bench2DriveFeatureBuilder(model_config)
    target_builder = Bench2DriveTargetBuilder(model_config)
    
    # Create metrics calculator
    metrics_calculator = VADCompatibleMetrics(
        method=args.metric_method,
        model_type=args.model_type,
        sampling_rate_hz=10.0  # Kept for compatibility
    )
    
    # Evaluate all scenarios
    print("\n=== Starting Evaluation ===")
    all_scenario_metrics = []
    per_scenario_results = {}
    
    for scenario in scenarios:
        # Get full path to scenario
        scenario_name = scenario[3:] if scenario.startswith('v1/') else scenario
        scenario_path = os.path.join(args.data_root, scenario_name)
        
        if not os.path.exists(scenario_path):
            print(f"Warning: Scenario not found: {scenario_path}")
            continue
        
        # Evaluate scenario
        sample_metrics, aggregated = evaluate_scenario(
            model=model,
            scenario_path=scenario_path,
            feature_builder=feature_builder,
            target_builder=target_builder,
            metrics_calculator=metrics_calculator,
            num_samples=args.num_samples,
            device=args.device
        )
        
        all_scenario_metrics.extend(sample_metrics)
        per_scenario_results[scenario_name] = aggregated
        
        # Print scenario results - show actual metric keys
        print(f"\nScenario: {scenario_name}")
        # Print L2 metrics
        for key in sorted(aggregated.keys()):
            if key.startswith('Planning_L2_') and not key.startswith('Planning_L2_avg'):
                value = aggregated[key]
                if value > 0:  # Only show non-zero metrics
                    print(f"  {key}: {value:.4f}")
        print(f"  Planning_L2_avg: {aggregated.get('Planning_L2_avg', 0):.4f}")
        # Print collision metrics if available
        if 'Planning_obj_col_avg' in aggregated:
            print(f"  Planning_obj_col_avg: {aggregated.get('Planning_obj_col_avg', 0):.4f} ({aggregated.get('Planning_obj_col_avg', 0)*100:.2f}%)")
            print(f"  Planning_obj_box_col_avg: {aggregated.get('Planning_obj_box_col_avg', 0):.4f} ({aggregated.get('Planning_obj_box_col_avg', 0)*100:.2f}%)")
    
    # Compute overall metrics
    print("\n=== Overall Results ===")
    overall_metrics = metrics_calculator.aggregate_metrics(all_scenario_metrics)
    
    # Display actual metric keys
    # L2 metrics
    for key in sorted(overall_metrics.keys()):
        if key.startswith('Planning_L2_') and not key.startswith('Planning_L2_avg'):
            value = overall_metrics[key]
            if value > 0:  # Only show non-zero metrics
                print(f"{key}: {value:.4f}")
    print(f"Planning_L2_avg: {overall_metrics.get('Planning_L2_avg', 0):.4f}")
    # Collision metrics
    if 'Planning_obj_col_avg' in overall_metrics:
        print(f"Planning_obj_col_avg: {overall_metrics.get('Planning_obj_col_avg', 0):.4f} ({overall_metrics.get('Planning_obj_col_avg', 0)*100:.2f}%)")
        print(f"Planning_obj_box_col_avg: {overall_metrics.get('Planning_obj_box_col_avg', 0):.4f} ({overall_metrics.get('Planning_obj_box_col_avg', 0)*100:.2f}%)")
    
    # Prepare output JSON (VAD-compatible format)
    output_data = {
        "meta": {
            "checkpoint": args.checkpoint,
            "eval_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "num_samples": len(all_scenario_metrics),
            "num_scenarios": len(scenarios),
            "metric_method": args.metric_method,
            "model_type": args.model_type,
            "validation_split": "b2d_val",
            "sampling_rate_hz": 10.0,
            "dev_mode": args.dev_mode
        },
        "results": overall_metrics,
        "per_scenario_results": per_scenario_results
    }
    
    # Add average
    output_data["results"]["Planning_L2_avg"] = overall_metrics.get('Planning_L2_avg', 0)
    
    # Save per-sample results if requested
    if args.save_per_sample:
        output_data["per_sample_results"] = [
            m.to_dict() for m in all_scenario_metrics
        ]
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(
        args.output_dir,
        f"eval_results_{args.metric_method}_{timestamp}.json"
    )
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    # Calculate total time
    total_time = time.time() - start_time
    minutes = int(total_time // 60)
    seconds = total_time % 60
    
    print(f"\n=== Evaluation Complete ===")
    print(f"Total evaluation time: {minutes}m {seconds:.1f}s")
    print(f"Results saved to: {output_file}")
    
    return output_data


if __name__ == "__main__":
    main()