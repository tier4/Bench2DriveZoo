"""
Model wrapper for DiffusionDrive that loads and runs inference without MMCV dependencies.
"""

import sys
import os
import yaml
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

# Add DiffusionDrive to path for imports
sys.path.insert(0, '/workspace/DiffusionDrive')

from navsim.agents.diffusiondrive.transfuser_model_wrapper import V2TransfuserModelWrapper
from navsim.agents.diffusiondrive.transfuser_config import TransfuserConfig
from navsim.agents.diffusiondrive.bench2drive_config import Bench2DriveConfig


class DiffusionDriveModelWrapper:
    """
    Wrapper for DiffusionDrive model that handles loading and inference.
    Removes MMCV dependencies and provides direct PyTorch interface.
    """
    
    def __init__(self, checkpoint_path: str, config_path: str = None):
        """
        Initialize the model wrapper.
        
        Args:
            checkpoint_path: Path to the trained model checkpoint
            config_path: Optional path to configuration file
        """
        self.checkpoint_path = checkpoint_path
        self.config_path = config_path
        
        # Initialize configuration
        self.config = self._load_config()
        
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize model
        self.model = self._initialize_model()
        
        # Load checkpoint
        self._load_checkpoint()
        
        # Set to evaluation mode
        self.model.eval()
        
        print(f"Model loaded successfully from {checkpoint_path}")

    def _load_config(self):
        """
        Load model configuration.
        
        Returns:
            Bench2DriveConfig object
        """
        # Try to load from YAML if config path provided
        if self.config_path and os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                yaml_config = yaml.safe_load(f)
            
            # Always use Bench2DriveConfig for B2D evaluation
            config = Bench2DriveConfig()
            config.dataset_type = 'bench2drive'
            
            # Override with YAML values if provided
            for key, value in yaml_config.get('config', {}).items():
                if hasattr(config, key):
                    setattr(config, key, value)
        else:
            # Default configuration for Bench2Drive
            config = Bench2DriveConfig()
            config.dataset_type = 'bench2drive'
        
        # Set paths for pretrained weights and anchors from test_files
        # These are the actual test files provided
        config.bkb_path = "/workspace/Bench2Drive/test_files/pytorch_model.bin"
        config.plan_anchor_path = "/workspace/Bench2Drive/test_files/kmeans_b2d_v2_traj_20.npy"
        
        # Verify anchor file exists
        if not os.path.exists(config.plan_anchor_path):
            print(f"Warning: B2D anchors not found at {config.plan_anchor_path}")
            # No fallback since we have the correct file
        
        # Set diffusion parameters
        config.n_timesteps = 50  # Truncated diffusion
        config.num_modes = 20
        config.num_timesteps = 8  # 4 seconds at 0.5s intervals
        
        return config

    def _initialize_model(self):
        """
        Initialize the DiffusionDrive model.
        
        Returns:
            V2TransfuserModelWrapper instance with normalization support
        """
        # Create model instance with wrapper for B2D normalization
        model = V2TransfuserModelWrapper(self.config)
        
        # Move to GPU if available
        if torch.cuda.is_available():
            model = model.cuda()
            print("Model moved to GPU")
        else:
            print("Warning: CUDA not available, using CPU")
        
        return model

    def _load_checkpoint(self):
        """Load model weights from checkpoint."""
        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")
        
        # Load checkpoint
        if torch.cuda.is_available():
            checkpoint = torch.load(self.checkpoint_path)
        else:
            checkpoint = torch.load(self.checkpoint_path, map_location='cpu')
        
        # Handle different checkpoint formats
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            # Assume the checkpoint is the state dict itself
            state_dict = checkpoint
        
        # Remove 'agent.' prefix if present (from Lightning training)
        state_dict = {k.replace('agent.', ''): v for k, v in state_dict.items()}
        
        # Remove '_transfuser_model.' prefix if present
        state_dict = {k.replace('_transfuser_model.', ''): v for k, v in state_dict.items()}
        
        # Load state dict
        missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)
        
        if missing_keys:
            print(f"Warning: Missing keys in checkpoint: {missing_keys[:5]}...")
        if unexpected_keys:
            print(f"Warning: Unexpected keys in checkpoint: {unexpected_keys[:5]}...")
        
        print(f"Checkpoint loaded from {self.checkpoint_path}")

    def inference(self, features: dict):
        """
        Run model inference on processed features.
        
        Args:
            features: Dictionary containing:
                - camera_feature: Tensor of shape [B, 3, H, W] or [B, T, 3, H, W]
                - lidar_feature: Tensor of shape [B, C, H, W] (optional)
                - status_feature: Tensor of shape [B, status_dim]
                
        Returns:
            trajectory: Numpy array of shape [num_modes, num_timesteps, 2]
                       representing predicted trajectories
        """
        with torch.no_grad():
            # Ensure features are on the correct device
            device = next(self.model.parameters()).device
            
            # Move features to device
            for key in features:
                if isinstance(features[key], torch.Tensor):
                    features[key] = features[key].to(device)
                elif isinstance(features[key], np.ndarray):
                    features[key] = torch.from_numpy(features[key]).to(device)
            
            # Add batch dimension if not present
            for key in features:
                if features[key].dim() == 1:  # [D] -> [1, D] (for status_feature)
                    features[key] = features[key].unsqueeze(0)
                elif features[key].dim() == 3:  # [C, H, W] -> [1, C, H, W]
                    features[key] = features[key].unsqueeze(0)
                elif key == 'camera_feature' and features[key].dim() == 4:  # [T, C, H, W] -> [1, T, C, H, W]
                    features[key] = features[key].unsqueeze(0)
            
            # Run model forward pass - the wrapper expects a features dict
            # Just pass the features directly to the model
            output = self.model(features)
            
            # Process output based on model configuration
            if isinstance(output, dict):
                # Model returns dictionary with 'trajectory' key
                trajectory = output['trajectory']
            else:
                # Model directly returns trajectory
                trajectory = output
            
            # Convert to numpy and remove batch dimension
            if isinstance(trajectory, torch.Tensor):
                trajectory = trajectory.cpu().numpy()
            
            # Expected shape: [1, num_modes, num_timesteps, 2]
            # Remove batch dimension: [num_modes, num_timesteps, 2]
            if trajectory.ndim == 4:
                trajectory = trajectory[0]
            
            # Ensure correct shape - can be (1, num_timesteps, 2/3) for single mode or (num_modes, num_timesteps, 2/3) for multi-mode
            # In test mode, the model returns only the best trajectory (1 mode)
            if trajectory.shape[0] == 1:
                # Single mode output - expand to match expected multi-mode format
                # Use numpy repeat with correct syntax: repeat along axis 0
                trajectory = np.repeat(trajectory, self.config.num_modes, axis=0)
            
            assert trajectory.shape[:2] == (self.config.num_modes, self.config.num_timesteps) and trajectory.shape[2] in [2, 3], \
                f"Unexpected trajectory shape: {trajectory.shape}, expected ({self.config.num_modes}, {self.config.num_timesteps}, 2 or 3)"
            
            return trajectory

    def compute_trajectory_with_features(self, agent_input, scene=None):
        """
        Compute trajectory from raw agent input (for compatibility).
        
        Args:
            agent_input: AgentInput object with cameras, lidars, ego_statuses
            scene: Optional scene object (not used in DiffusionDrive)
            
        Returns:
            Trajectory predictions
        """
        # This would require the feature builders
        # For CARLA agent, we use the direct inference method instead
        raise NotImplementedError(
            "This method requires feature builders. Use inference() with preprocessed features instead."
        )

    def get_config(self):
        """Get model configuration."""
        return self.config

    def to(self, device):
        """Move model to specified device."""
        self.model = self.model.to(device)
        return self

    def eval(self):
        """Set model to evaluation mode."""
        self.model.eval()
        return self

    def train(self):
        """Set model to training mode."""
        self.model.train()
        return self