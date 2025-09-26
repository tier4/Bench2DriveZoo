"""
VAD-compatible open-loop evaluation script for DiffusionDrive on Bench2Drive.

Evaluation runs at 10Hz input frequency (sampling_rate=1) but trajectories are
compared at 2Hz (model's native output frequency). Both model predictions and
ground truth are 8 waypoints at 0.5s intervals, perfectly aligned for comparison.

No interpolation is performed - following VAD's approach of comparing at the
model's native frequency rather than interpolating to input frequency.
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
sys.path.append("/workspace/Bench2Drive/DiffusionDrive")

# Import DiffusionDrive modules
from navsim.common.bench2drive_dataloader import Bench2DriveConfig, Bench2DriveSceneLoader
from navsim.agents.diffusiondrive.transfuser_features_b2d import (
    Bench2DriveFeatureBuilder,
    Bench2DriveTargetBuilder,
)
from navsim.agents.diffusiondrive.transfuser_model_v2 import V2TransfuserModel

# Import our custom modules
from data_split_loader import ValidationSplitLoader
from metrics_vad import VADCompatibleMetrics, PlanningMetrics

# Import config
from navsim.agents.diffusiondrive.transfuser_config import TransfuserConfig as B2DModelConfig


def load_model(checkpoint_path: str, device: str = "cuda"):
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
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    # Clean state dict keys (remove 'model.' prefix if present)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("model."):
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
    parser = argparse.ArgumentParser(description="VAD-compatible evaluation for DiffusionDrive")

    # Data arguments
    parser.add_argument(
        "--data_root",
        type=str,
        default="/mnt/nvme1/dataset/Bench2Drive-Base",
        help="Path to Bench2Drive dataset",
    )
    parser.add_argument(
        "--split_file",
        type=str,
        default="/workspace/Bench2Drive/Bench2DriveZoo/data/splits/bench2drive_base_train_val_split.json",
        help="Path to train/val split JSON",
    )

    # Model arguments
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="/workspace/Bench2Drive/DiffusionDrive/checkpoints/diffusiondrive_b2d.ckpt",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--anchor_file",
        type=str,
        default="/workspace/Bench2Drive/DiffusionDrive/checkpoints/transfuser_b2d_anchors.npy",
        help="Path to trajectory anchors file",
    )

    # Evaluation arguments
    parser.add_argument(
        "--use_gt_model",
        action="store_true",
        help="Use ground truth as predictions to verify evaluation pipeline",
    )
    parser.add_argument(
        "--dev_mode", action="store_true", help="Use only 2 scenarios for development testing"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["train", "val"],
        help="Which split to evaluate on (default: val)",
    )
    parser.add_argument(
        "--metric_method",
        type=str,
        default="vad",
        choices=["vad", "uniad"],
        help="Metric calculation method (default: vad)",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="vad",
        choices=["vad", "uniad", "diffusiondrive"],
        help="Model type for timestep configuration (VAD:6, UniAD:4, DiffusionDrive:8)",
    )
    parser.add_argument(
        "--num_samples", type=int, default=-1, help="Number of samples per scenario (-1 for all)"
    )

    # Output arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/workspace/Bench2Drive/Bench2DriveZoo/adzoo/diffusiondrive/eval_results",
        help="Directory to save evaluation results",
    )
    parser.add_argument(
        "--output_name",
        type=str,
        default=None,
        help="Custom output filename (without extension). If not specified, auto-generates from timestamp",
    )
    parser.add_argument(
        "--save_per_sample", action="store_true", help="Save per-sample results (can be large)"
    )

    # Device arguments
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use for inference",
    )

    args = parser.parse_args()

    # Default behavior: full evaluation unless --dev_mode is specified

    return args


def evaluate_scenario(
    model: V2TransfuserModel,
    scenario_path: str,
    feature_builder: Bench2DriveFeatureBuilder,
    target_builder: Bench2DriveTargetBuilder,
    metrics_calculator: VADCompatibleMetrics,
    num_samples: int = -1,
    device: str = "cuda",
    use_gt_model: bool = False,
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
    # Use sampling_rate=1 for 10Hz input (evaluation mode)
    # GT extraction will automatically sample every 5th frame to get 2Hz waypoints
    config = Bench2DriveConfig(
        data_root=Path(scenario_path).parent,  # Parent directory
        scenarios=[Path(scenario_path).name],  # Just this scenario
        sampling_rate=1,  # CRITICAL: 10Hz for evaluation (not 5 like training!)
        num_frames=50,  # Need 41+ frames for ground truth extraction
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

        # Get ground truth trajectory first (needed for all cases)
        gt_trajectory = scene.get_future_trajectory(0)
        if gt_trajectory is None:
            # Skip frames without complete future data
            continue
        gt_trajectory = np.array(gt_trajectory)

        # Debug: Check original GT shape
        if idx == 0:
            print(f"  DEBUG: Original GT shape from scene: {gt_trajectory.shape}")

        # Handle GT mode vs normal model inference
        if use_gt_model:
            # For GT verification: use exact GT as prediction (no downsampling/interpolation)
            # This should result in exactly 0.0 error if metrics are calculated correctly
            pred_trajectory = gt_trajectory.copy()
            outputs = {"mode_probs": None}  # Dummy for compatibility
        else:
            # Normal model inference path
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

            # Extract predicted trajectory
            if isinstance(outputs, dict):
                if "trajectory" in outputs:
                    pred_trajectory = outputs["trajectory"].cpu().numpy()[0]
                elif "plan_trajectory" in outputs:
                    pred_trajectory = outputs["plan_trajectory"].cpu().numpy()[0]
                else:
                    continue  # Unknown output format
            else:
                pred_trajectory = outputs.cpu().numpy()[0]

        # CRITICAL FIX: Convert absolute positions to offsets
        # VAD expects offset trajectories (displacements between consecutive timesteps)
        # Both DiffusionDrive and GT typically provide absolute positions
        # We need to convert them to offsets for correct evaluation

        # Convert GT from absolute to offset format
        # Prepend origin (0,0) to compute offsets from t=0
        gt_absolute = gt_trajectory.copy()
        gt_origin = np.zeros((1, gt_trajectory.shape[1]))  # Origin at (0,0,...)
        gt_with_origin = np.vstack([gt_origin, gt_absolute])
        gt_offsets = np.diff(gt_with_origin, axis=0)  # Compute offsets
        gt_trajectory = gt_offsets  # Use offsets as GT

        # Convert predictions from absolute to offset format
        pred_absolute = pred_trajectory.copy()
        pred_origin = np.zeros((1, pred_trajectory.shape[1]))  # Origin at (0,0,...)
        pred_with_origin = np.vstack([pred_origin, pred_absolute])
        pred_offsets = np.diff(pred_with_origin, axis=0)  # Compute offsets
        pred_trajectory = pred_offsets  # Use offsets as predictions

        # Debug: Print conversion results
        if idx == 0:
            print(f"  DEBUG: Converted trajectories to offset format")
            print(f"  DEBUG: GT first 3 absolute: {gt_absolute[:3, :2]}")
            print(f"  DEBUG: GT first 3 offsets: {gt_offsets[:3, :2]}")
            print(f"  DEBUG: Pred first 3 absolute: {pred_absolute[:3, :2]}")
            print(f"  DEBUG: Pred first 3 offsets: {pred_offsets[:3, :2]}")

        # Validate trajectory shapes
        if pred_trajectory.shape[0] != 8:
            print(f"  WARNING: Expected 8 pred waypoints, got {pred_trajectory.shape[0]}")
        if gt_trajectory.shape[0] != 8:
            print(f"  WARNING: Expected 8 GT waypoints, got {gt_trajectory.shape[0]}")

        # Extract agent states and labels if available
        gt_agent_states = None
        gt_agent_labels = None
        if isinstance(outputs, dict):
            if "agent_states" in outputs:
                gt_agent_states = outputs["agent_states"].cpu().numpy()[0]  # Remove batch dim
                gt_agent_labels = outputs.get(
                    "agent_labels", np.ones(gt_agent_states.shape[0], dtype=bool)
                )
                if torch.is_tensor(gt_agent_labels):
                    gt_agent_labels = gt_agent_labels.cpu().numpy()[0]
                pass  # Agent states loaded

        # Debug: Check trajectory shapes before metric calculation
        if idx == 0:  # Only print for first sample to avoid spam
            print(f"  DEBUG: Pred shape: {pred_trajectory.shape}, GT shape: {gt_trajectory.shape}")
            if not use_gt_model:
                print(f"  DEBUG: Pred values (first 3): {pred_trajectory[:3]}")
                print(f"  DEBUG: GT values (first 3): {gt_trajectory[:3]}")
                print(f"  DEBUG: Diff (first 3): {pred_trajectory[:3] - gt_trajectory[:3]}")
                print(f"  DEBUG: L2 distance (first 3): {np.linalg.norm(pred_trajectory[:3] - gt_trajectory[:3], axis=1)}")

        # Compute metrics
        metrics = metrics_calculator.compute_planning_l2(
            pred_trajectory,
            gt_trajectory,
            pred_modes=outputs.get("mode_probs", None) if isinstance(outputs, dict) else None,
            gt_agent_states=gt_agent_states,
            gt_agent_labels=gt_agent_labels,
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

    # Load split data
    print(f"\n=== Loading {args.split.upper()} Split ===")
    split_loader = ValidationSplitLoader(args.split_file, data_root=args.data_root)
    scenarios = split_loader.get_scenarios(
        split=args.split, dev_mode=args.dev_mode, num_dev_scenarios=2
    )

    print(f"Evaluation mode: {'Dev' if args.dev_mode else 'Full'}")
    print(f"Split: {args.split.upper()}")
    print(f"Number of scenarios: {len(scenarios)}")
    print(f"Metric method: {args.metric_method.upper()}")
    print(f"Model type: {args.model_type.upper()} (timesteps configuration)")

    # Load model (skip if using GT model)
    print("\n=== Loading Model ===")
    if args.use_gt_model:
        print("USING GROUND TRUTH AS PREDICTIONS FOR VERIFICATION")
        model = None  # We won't use the model
        from navsim.agents.diffusiondrive.transfuser_config import (
            TransfuserConfig as B2DModelConfig,
        )

        model_config = B2DModelConfig()
    else:
        model, model_config = load_model(args.checkpoint, args.device)
        print(f"Model loaded from: {args.checkpoint}")

    # Create feature and target builders using the model config
    feature_builder = Bench2DriveFeatureBuilder(model_config)
    target_builder = Bench2DriveTargetBuilder(model_config)

    # Create metrics calculator
    metrics_calculator = VADCompatibleMetrics(
        method=args.metric_method,
        model_type=args.model_type,
        sampling_rate_hz=10.0,  # Kept for compatibility
    )

    # Evaluate all scenarios
    print("\n=== Starting Evaluation ===")
    all_scenario_metrics = []
    per_scenario_results = {}

    for scenario in scenarios:
        # Get full path to scenario
        scenario_name = scenario[3:] if scenario.startswith("v1/") else scenario
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
            device=args.device,
            use_gt_model=args.use_gt_model,
        )

        all_scenario_metrics.extend(sample_metrics)
        per_scenario_results[scenario_name] = aggregated

        # Print scenario results - show actual metric keys
        print(f"\nScenario: {scenario_name}")
        # Print L2 metrics
        for key in sorted(aggregated.keys()):
            if key.startswith("Planning_L2_") and not key.startswith("Planning_L2_avg"):
                value = aggregated[key]
                if value > 0:  # Only show non-zero metrics
                    print(f"  {key}: {value:.4f}")
        print(f"  Planning_L2_avg: {aggregated.get('Planning_L2_avg', 0):.4f}")
        # Print collision metrics if available
        if "Planning_obj_col_avg" in aggregated:
            print(
                f"  Planning_obj_col_avg: {aggregated.get('Planning_obj_col_avg', 0):.4f} ({aggregated.get('Planning_obj_col_avg', 0)*100:.2f}%)"
            )
            print(
                f"  Planning_obj_box_col_avg: {aggregated.get('Planning_obj_box_col_avg', 0):.4f} ({aggregated.get('Planning_obj_box_col_avg', 0)*100:.2f}%)"
            )

    # Compute overall metrics
    print("\n=== Overall Results ===")
    overall_metrics = metrics_calculator.aggregate_metrics(all_scenario_metrics)

    # Display actual metric keys
    # L2 metrics
    for key in sorted(overall_metrics.keys()):
        if key.startswith("Planning_L2_") and not key.startswith("Planning_L2_avg"):
            value = overall_metrics[key]
            if value > 0:  # Only show non-zero metrics
                print(f"{key}: {value:.4f}")
    print(f"Planning_L2_avg: {overall_metrics.get('Planning_L2_avg', 0):.4f}")
    # Collision metrics
    if "Planning_obj_col_avg" in overall_metrics:
        print(
            f"Planning_obj_col_avg: {overall_metrics.get('Planning_obj_col_avg', 0):.4f} ({overall_metrics.get('Planning_obj_col_avg', 0)*100:.2f}%)"
        )
        print(
            f"Planning_obj_box_col_avg: {overall_metrics.get('Planning_obj_box_col_avg', 0):.4f} ({overall_metrics.get('Planning_obj_box_col_avg', 0)*100:.2f}%)"
        )

    # Prepare output JSON (VAD-compatible format)
    output_data = {
        "meta": {
            "checkpoint": args.checkpoint,
            "eval_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "num_samples": len(all_scenario_metrics),
            "num_scenarios": len(scenarios),
            "metric_method": args.metric_method,
            "model_type": args.model_type,
            "split": args.split,
            "validation_split": f"b2d_{args.split}",
            "sampling_rate_hz": 10.0,
            "dev_mode": args.dev_mode,
        },
        "results": overall_metrics,
        "per_scenario_results": per_scenario_results,
    }

    # Add average
    output_data["results"]["Planning_L2_avg"] = overall_metrics.get("Planning_L2_avg", 0)

    # Save per-sample results if requested
    if args.save_per_sample:
        output_data["per_sample_results"] = [m.to_dict() for m in all_scenario_metrics]

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Extract weight name from checkpoint path
    weight_name = "no_weight"
    if args.checkpoint and os.path.exists(args.checkpoint):
        # First try to get the checkpoint filename without extension
        checkpoint_filename = os.path.splitext(os.path.basename(args.checkpoint))[0]

        # Check if this is a meaningful name (not just 'model' or 'checkpoint')
        if checkpoint_filename not in ["model", "checkpoint", "latest", "best"]:
            weight_name = checkpoint_filename
        else:
            # Try to get the parent directory name (usually the experiment name)
            parent_dir = os.path.basename(os.path.dirname(args.checkpoint))
            if parent_dir and parent_dir not in ["checkpoints", "weights", "models"]:
                weight_name = parent_dir
            else:
                # Try grandparent directory
                grandparent_dir = os.path.basename(
                    os.path.dirname(os.path.dirname(args.checkpoint))
                )
                if grandparent_dir:
                    weight_name = grandparent_dir

    # Determine mode and split
    mode = "dev" if args.dev_mode else "full"
    split = args.split  # 'train' or 'val'

    # Create subdirectory structure including split type
    output_subdir = os.path.join(
        args.output_dir, f"{weight_name}_{split}_{mode}_{args.metric_method}"
    )
    os.makedirs(output_subdir, exist_ok=True)

    # Determine output filename - include weight name and split in default filename
    if args.output_name:
        output_filename = f"{args.output_name}.json"
    else:
        output_filename = f"{weight_name}_{split}_{timestamp}.json"

    output_file = os.path.join(output_subdir, output_filename)

    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)

    # Calculate total time
    total_time = time.time() - start_time
    minutes = int(total_time // 60)
    seconds = total_time % 60

    print("\n=== Evaluation Complete ===")
    print(f"Total evaluation time: {minutes}m {seconds:.1f}s")
    print(f"Results saved to: {output_file}")

    return output_data


if __name__ == "__main__":
    main()
