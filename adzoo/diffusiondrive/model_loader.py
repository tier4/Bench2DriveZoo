"""
Model loader for DiffusionDrive open-loop evaluation.
Loads DiffusionDrive model without Hydra dependencies.
"""

import os
import sys
import torch
import logging
from pathlib import Path
from typing import Dict, Optional, Any

# Add DiffusionDrive to path for navsim imports
DIFFUSION_DRIVE_PATH = Path(__file__).parent.parent.parent.parent / "DiffusionDrive"
if DIFFUSION_DRIVE_PATH.exists():
    sys.path.insert(0, str(DIFFUSION_DRIVE_PATH))

from navsim.agents.diffusiondrive.transfuser_model_wrapper import V2TransfuserModelWrapper
from navsim.agents.diffusiondrive.bench2drive_config import Bench2DriveConfig

logger = logging.getLogger(__name__)


def load_diffusiondrive_model(
    checkpoint_path: str,
    config_override: Optional[Dict[str, Any]] = None,
    device: str = "cuda"
) -> tuple:
    """
    Load DiffusionDrive model from checkpoint.
    
    Args:
        checkpoint_path: Path to model checkpoint (.ckpt or .pth file)
        config_override: Optional dictionary to override config parameters
        device: Device to load model on ('cuda' or 'cpu')
    
    Returns:
        Tuple of (model, config)
    """
    
    # Check if checkpoint exists
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Create Bench2Drive configuration
    config = Bench2DriveConfig()
    
    # Set B2D-specific parameters
    config.dataset_type = "bench2drive"
    
    # Set default paths if they exist
    # First check in test_files (where the checkpoint is)
    checkpoint_dir = Path(checkpoint_path).parent
    b2d_anchor_path = checkpoint_dir / "kmeans_b2d_v2_traj_20.npy"
    
    if not b2d_anchor_path.exists():
        # Fallback to test_files directory
        b2d_anchor_path = Path("test_files") / "kmeans_b2d_v2_traj_20.npy"
    
    if b2d_anchor_path.exists():
        config.plan_anchor_path = str(b2d_anchor_path)
        logger.info(f"Using B2D anchor file: {b2d_anchor_path}")
    else:
        logger.warning(f"B2D anchor file not found at {b2d_anchor_path}")
    
    # Apply config overrides if provided
    if config_override:
        for key, value in config_override.items():
            if hasattr(config, key):
                setattr(config, key, value)
                logger.info(f"Config override: {key} = {value}")
            else:
                logger.warning(f"Unknown config parameter: {key}")
    
    # Initialize model
    logger.info("Initializing V2TransfuserModelWrapper...")
    model = V2TransfuserModelWrapper(config)
    
    # Load checkpoint
    logger.info(f"Loading checkpoint from {checkpoint_path}...")
    
    if device == "cuda" and torch.cuda.is_available():
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        device = "cpu"
    
    # Handle different checkpoint formats
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        # Assume the checkpoint is the state dict itself
        state_dict = checkpoint
    
    # Remove 'agent.' prefix if present (from training)
    state_dict = {k.replace('agent.', ''): v for k, v in state_dict.items()}
    
    # Remove '_transfuser_model.' prefix if present
    state_dict = {k.replace('_transfuser_model.', ''): v for k, v in state_dict.items()}
    
    # Load state dict with strict=False to handle potential mismatches
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    
    if missing_keys:
        logger.warning(f"Missing keys in checkpoint: {missing_keys[:5]}..." if len(missing_keys) > 5 else missing_keys)
    if unexpected_keys:
        logger.warning(f"Unexpected keys in checkpoint: {unexpected_keys[:5]}..." if len(unexpected_keys) > 5 else unexpected_keys)
    
    # Move model to device
    model = model.to(device)
    model.eval()  # Set to evaluation mode
    
    logger.info(f"Model loaded successfully on {device}")
    
    # Log model configuration
    logger.info(f"Model configuration:")
    logger.info(f"  - Dataset type: {config.dataset_type}")
    if hasattr(config, 'num_modes'):
        logger.info(f"  - Trajectory modes: {config.num_modes}")
    if hasattr(config, 'predict_n'):
        logger.info(f"  - Future timesteps: {config.predict_n}")
    if hasattr(config, 'num_history_images'):
        logger.info(f"  - History frames: {config.num_history_images}")
    
    return model, config


def get_model_info(checkpoint_path: str) -> Dict[str, Any]:
    """
    Get information about a model checkpoint without fully loading it.
    
    Args:
        checkpoint_path: Path to model checkpoint
    
    Returns:
        Dictionary with checkpoint information
    """
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Load checkpoint metadata only
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    
    info = {
        "checkpoint_path": checkpoint_path,
        "has_state_dict": 'state_dict' in checkpoint,
        "has_model_state_dict": 'model_state_dict' in checkpoint,
        "keys": list(checkpoint.keys()) if isinstance(checkpoint, dict) else [],
    }
    
    # Try to get training info if available
    if isinstance(checkpoint, dict):
        if 'epoch' in checkpoint:
            info['epoch'] = checkpoint['epoch']
        if 'global_step' in checkpoint:
            info['global_step'] = checkpoint['global_step']
        if 'pytorch-lightning_version' in checkpoint:
            info['lightning_version'] = checkpoint['pytorch-lightning_version']
    
    return info


if __name__ == "__main__":
    # Test the model loader
    import argparse
    
    parser = argparse.ArgumentParser(description="Test DiffusionDrive model loading")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Get model info
    print("\n=== Checkpoint Info ===")
    info = get_model_info(args.checkpoint)
    for key, value in info.items():
        print(f"{key}: {value}")
    
    # Load model
    print("\n=== Loading Model ===")
    model, config = load_diffusiondrive_model(args.checkpoint, device=args.device)
    
    # Skip forward pass test - will use real data in actual evaluation
    print("\n=== Model Loading Complete ===")
    print("Skipping forward pass test. Use real data for evaluation.")