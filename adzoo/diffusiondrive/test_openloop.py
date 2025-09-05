#!/usr/bin/env python3
"""
Open-loop evaluation for DiffusionDrive on Bench2Drive dataset.
Uses existing DiffusionDrive modules - no reinventing the wheel.
Produces VAD-compatible metrics for benchmarking comparison.
"""

import sys
import os
from pathlib import Path

# Add DiffusionDrive to path
sys.path.insert(0, '/workspace/Bench2Drive/DiffusionDrive')

import torch
import numpy as np
from typing import Dict, List, Any, Optional
import json
import yaml
from tqdm import tqdm

# Import DiffusionDrive modules - USE WHAT EXISTS!
from navsim.common.bench2drive_dataloader import (
    Bench2DriveConfig, 
    Bench2DriveSceneLoader
)
from navsim.agents.diffusiondrive.transfuser_features_b2d import (
    Bench2DriveFeatureBuilder,
    Bench2DriveTargetBuilder
)
from navsim.agents.diffusiondrive.transfuser_model_v2 import V2TransfuserModel
from navsim.agents.diffusiondrive.bench2drive_config import Bench2DriveConfig as B2DModelConfig

# Only our metrics implementation for VAD compatibility
from metrics import OpenLoopEvaluator


def load_model(checkpoint_path: str, device: str = 'cuda'):
    """
    Load DiffusionDrive model.
    
    Args:
        checkpoint_path: Path to checkpoint
        device: Device to load on
    
    Returns:
        Loaded model
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
    
    # Clean state dict keys
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('model.'):
            k = k[6:]  # Remove 'model.' prefix
        new_state_dict[k] = v
    
    model.load_state_dict(new_state_dict, strict=False)
    model = model.to(device)
    model.eval()
    
    return model, config


def main():
    """Main evaluation using ONLY DiffusionDrive's existing modules."""
    
    # Configuration
    checkpoint_path = "/workspace/Bench2Drive/test_files/exp0_datasetv2_archv0_bs256x8_ep3000_lr1e-5_step2539.ckpt"
    data_root = "/mnt/nvme1/dataset/Bench2Drive-Base"
    scenario = "AccidentTwoWays_Town12_Route1102_Weather10"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_samples = 10  # Limit for testing
    
    output_dir = Path("./eval_results/diffusiondrive_openloop")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=== DiffusionDrive Open-Loop Evaluation ===")
    print(f"Using existing DiffusionDrive modules - no reinventing!\n")
    
    # 1. Create B2D config for data loading
    print("1. Creating Bench2Drive configuration...")
    b2d_config = Bench2DriveConfig(
        data_root=Path(data_root),
        scenarios=[scenario],
        num_history_frames=4,
        num_future_frames=8,
        sampling_rate=5,  # Downsample from 10Hz to 2Hz
        extract_tar=False
    )
    print(f"   Config created for: {scenario}\n")
    
    # 2. Create scene loader using EXISTING infrastructure
    print("2. Creating scene loader (using existing Bench2DriveSceneLoader)...")
    scene_loader = Bench2DriveSceneLoader(b2d_config)
    scene_tokens = scene_loader.get_scene_tokens()
    print(f"   Found {len(scene_tokens)} scenes\n")
    
    if len(scene_tokens) == 0:
        raise ValueError("No scenes found! Check data path and scenario name.")
    
    # 3. Load model
    print("3. Loading DiffusionDrive model...")
    model, model_config = load_model(checkpoint_path, device)
    print(f"   Model loaded on {device}\n")
    
    # 4. Create feature builder using EXISTING implementation
    print("4. Creating feature builder (using existing Bench2DriveFeatureBuilder)...")
    feature_builder = Bench2DriveFeatureBuilder(model_config)
    print("   Feature builder ready\n")
    
    # 5. Create evaluator (our custom metrics for VAD compatibility)
    evaluator = OpenLoopEvaluator(sampling_rate_hz=2.0)
    results = []
    
    # 6. Run evaluation
    print("5. Running evaluation...")
    num_eval = min(num_samples, len(scene_tokens))
    
    for idx in tqdm(range(num_eval), desc="Evaluating"):
        # Get scene using EXISTING infrastructure
        scene_token = scene_tokens[idx]
        scene = scene_loader.get_scene(scene_token)
        
        # Get agent input - this properly loads ALL sensors
        # No manual loading needed!
        agent_input = scene.get_agent_input(0)  # Get first frame
        
        # Compute features using EXISTING feature builder
        # This handles cameras, LiDAR, status - everything!
        features = feature_builder.compute_features(agent_input)
        
        # Add batch dimension and move to device
        for key in features:
            features[key] = features[key].unsqueeze(0).to(device)
        
        # Run model inference
        with torch.no_grad():
            output = model(features)
        
        # Extract trajectory
        if isinstance(output, dict):
            if 'trajectory' in output:
                pred_traj = output['trajectory']
            elif 'plan_trajectory' in output:
                pred_traj = output['plan_trajectory']
            else:
                print(f"Warning: Unknown output keys: {output.keys()}")
                continue
        else:
            pred_traj = output
        
        # Convert to numpy and remove batch dimension
        pred_traj = pred_traj.squeeze(0).cpu().numpy()
        
        # Get ground truth trajectory
        gt_traj = scene.get_future_trajectory(0)
        
        # Convert to numpy if it's a tensor
        if torch.is_tensor(gt_traj):
            gt_traj = gt_traj.cpu().numpy()
        
        # Handle shape differences
        if pred_traj.ndim == 3:  # [M, T, 3]
            # Multi-modal output
            num_modes = pred_traj.shape[0]
            print(f"   Sample {idx}: {num_modes} modes predicted")
        elif pred_traj.ndim == 2:  # [T, 3]
            # Single mode
            pred_traj = pred_traj[np.newaxis, :]  # Add mode dimension
        
        # Compute metrics
        metrics = evaluator.compute_trajectory_metrics(
            pred_trajectory=pred_traj,
            gt_trajectory=gt_traj
        )
        
        result = {
            'scene_token': scene_token,
            'metrics': {
                'L2_0.5s': metrics.L2_05s,
                'L2_1.0s': metrics.L2_10s,
                'L2_1.5s': metrics.L2_15s,
                'L2_2.0s': metrics.L2_20s,
                'L2_2.5s': metrics.L2_25s,
                'L2_3.0s': metrics.L2_30s,
                'L2_avg': metrics.L2_avg
            }
        }
        results.append(result)
        
        # Print first sample metrics
        if idx == 0:
            print(f"\n   First sample metrics:")
            for key, value in result['metrics'].items():
                print(f"     {key}: {value:.3f} m")
    
    # 7. Save results
    print(f"\n6. Saving results...")
    
    # Compute aggregate metrics
    all_metrics = {}
    for result in results:
        for key, value in result['metrics'].items():
            if key not in all_metrics:
                all_metrics[key] = []
            all_metrics[key].append(value)
    
    avg_metrics = {key: np.mean(values) for key, values in all_metrics.items()}
    
    # Save to JSON (VAD-compatible format)
    output_file = output_dir / f"{scenario}_results.json"
    with open(output_file, 'w') as f:
        json.dump({
            'scenario': scenario,
            'checkpoint': checkpoint_path,
            'num_samples': num_eval,
            'aggregate_metrics': avg_metrics,
            'per_scene_results': results
        }, f, indent=2)
    
    print(f"   Results saved to: {output_file}")
    
    # Print summary
    print(f"\n=== Evaluation Summary ===")
    print(f"Evaluated {num_eval} samples")
    print(f"Average metrics:")
    for key, value in avg_metrics.items():
        print(f"  {key}: {value:.3f} m")
    
    print(f"\n✓ Successfully used DiffusionDrive's existing modules!")
    print(f"✓ No wheels were reinvented in this evaluation!")


if __name__ == "__main__":
    main()