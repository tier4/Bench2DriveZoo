"""
Data split loader for VAD-compatible evaluation.
Loads validation scenarios from the Bench2Drive split JSON file.
"""

import json
import os
from typing import List, Optional
from pathlib import Path


class ValidationSplitLoader:
    """Load and manage train/validation split for Bench2Drive evaluation."""

    def __init__(self, split_file_path: Optional[str] = None, data_root: Optional[str] = None):
        """
        Initialize the validation split loader.
        
        Args:
            split_file_path: Path to the JSON split file. If None, uses default location.
        """
        if split_file_path is None:
            # Default path to the split file
            split_file_path = "/workspace/Bench2Drive/Bench2DriveZoo/data/splits/bench2drive_base_train_val_split.json"

        if data_root is None:
            data_root = "/mnt/nvme1/dataset/Bench2Drive-Base"

        self.split_file_path = split_file_path
        self.data_root = data_root
        self.validation_scenarios = self._load_validation_split()
        self.train_scenarios = self._load_train_split()
        
    def _load_validation_split(self) -> List[str]:
        """
        Load validation scenarios from JSON split file.

        Returns:
            List of validation scenario paths
        """
        if not os.path.exists(self.split_file_path):
            raise FileNotFoundError(f"Split file not found: {self.split_file_path}")

        with open(self.split_file_path, 'r') as f:
            split_data = json.load(f)

        if 'val' not in split_data:
            raise ValueError(f"Split file missing 'val' key: {self.split_file_path}")

        validation_scenarios = split_data['val']
        print(f"Loaded {len(validation_scenarios)} validation scenarios from split file")

        return validation_scenarios

    def _load_train_split(self) -> List[str]:
        """
        Load training scenarios (all scenarios not in validation).

        Returns:
            List of training scenario paths
        """
        # Get all scenarios from data directory
        if not os.path.exists(self.data_root):
            print(f"Warning: Data root not found: {self.data_root}")
            return []

        all_scenarios = [d for d in os.listdir(self.data_root)
                        if os.path.isdir(os.path.join(self.data_root, d))]

        # Remove validation scenarios to get training scenarios
        val_scenarios_clean = set(s.replace('v1/', '') for s in self.validation_scenarios)
        train_scenarios = [f'v1/{s}' for s in all_scenarios if s not in val_scenarios_clean]

        print(f"Found {len(train_scenarios)} training scenarios (total: {len(all_scenarios)}, val: {len(self.validation_scenarios)})")

        return sorted(train_scenarios)
    
    def get_scenarios(self, split: str = 'val', dev_mode: bool = False, num_dev_scenarios: int = 2) -> List[str]:
        """
        Get scenarios for evaluation.

        Args:
            split: 'train' or 'val' to select which split to use
            dev_mode: If True, return only a subset for development/testing
            num_dev_scenarios: Number of scenarios to use in dev mode

        Returns:
            List of scenario paths to evaluate
        """
        if split == 'train':
            scenarios = self.train_scenarios
        elif split == 'val':
            scenarios = self.validation_scenarios
        else:
            raise ValueError(f"Invalid split: {split}. Must be 'train' or 'val'")

        if dev_mode:
            # Select diverse scenarios for development testing
            selected = []
            if len(scenarios) > 0:
                selected.append(scenarios[0])  # First scenario
            if len(scenarios) > 1 and num_dev_scenarios > 1:
                mid_idx = len(scenarios) // 2
                selected.append(scenarios[mid_idx])  # Middle scenario

            print(f"Dev mode ({split}): Using {len(selected)} scenarios for testing")
            for scenario in selected:
                print(f"  - {scenario}")
            return selected[:num_dev_scenarios]
        else:
            print(f"Full evaluation mode ({split}): Using all {len(scenarios)} {split} scenarios")
            return scenarios

    def get_validation_scenarios(self, dev_mode: bool = False, num_dev_scenarios: int = 2) -> List[str]:
        """
        Get validation scenarios for evaluation.
        
        Args:
            dev_mode: If True, return only a subset for development/testing
            num_dev_scenarios: Number of scenarios to use in dev mode
            
        Returns:
            List of scenario paths to evaluate
        """
        if dev_mode:
            # Select diverse scenarios for development testing
            # Pick first and middle scenarios for variety
            selected = []
            if len(self.validation_scenarios) > 0:
                selected.append(self.validation_scenarios[0])  # First scenario
            if len(self.validation_scenarios) > 1 and num_dev_scenarios > 1:
                mid_idx = len(self.validation_scenarios) // 2
                selected.append(self.validation_scenarios[mid_idx])  # Middle scenario
            
            print(f"Dev mode: Using {len(selected)} scenarios for testing")
            for scenario in selected:
                print(f"  - {scenario}")
            return selected[:num_dev_scenarios]
        else:
            print(f"Full evaluation mode: Using all {len(self.validation_scenarios)} validation scenarios")
            return self.validation_scenarios
    
    def get_scenario_names(self, scenarios: Optional[List[str]] = None) -> List[str]:
        """
        Extract clean scenario names from paths.
        
        Args:
            scenarios: List of scenario paths. If None, uses all validation scenarios.
            
        Returns:
            List of clean scenario names (without v1/ prefix)
        """
        if scenarios is None:
            scenarios = self.validation_scenarios
        
        # Remove 'v1/' prefix if present
        clean_names = []
        for scenario in scenarios:
            if scenario.startswith('v1/'):
                clean_names.append(scenario[3:])
            else:
                clean_names.append(scenario)
        
        return clean_names
    
    def get_full_paths(self, base_dir: str, scenarios: Optional[List[str]] = None) -> List[str]:
        """
        Convert scenario names to full file system paths.
        
        Args:
            base_dir: Base directory containing the scenarios
            scenarios: List of scenario names. If None, uses all validation scenarios.
            
        Returns:
            List of full paths to scenario directories
        """
        if scenarios is None:
            scenarios = self.validation_scenarios
        
        full_paths = []
        for scenario in scenarios:
            # Remove 'v1/' prefix if present since it's usually part of the base_dir structure
            scenario_name = scenario[3:] if scenario.startswith('v1/') else scenario
            full_path = os.path.join(base_dir, scenario_name)
            full_paths.append(full_path)
        
        return full_paths


# Convenience functions for direct use
def load_validation_split(split_file_path: Optional[str] = None) -> List[str]:
    """
    Load validation scenarios from JSON split file.
    
    Args:
        split_file_path: Path to JSON file. If None, uses default location.
        
    Returns:
        List of validation scenario names
    """
    loader = ValidationSplitLoader(split_file_path)
    return loader.validation_scenarios


def get_validation_scenarios(dev_mode: bool = False, num_dev_scenarios: int = 2) -> List[str]:
    """
    Get validation scenarios for evaluation.
    
    Args:
        dev_mode: If True, return only a subset for development
        num_dev_scenarios: Number of scenarios in dev mode
        
    Returns:
        List of scenario names to evaluate
    """
    loader = ValidationSplitLoader()
    return loader.get_validation_scenarios(dev_mode, num_dev_scenarios)


if __name__ == "__main__":
    # Test the loader
    print("Testing ValidationSplitLoader...")
    
    # Test loading all validation scenarios
    loader = ValidationSplitLoader()
    all_scenarios = loader.get_validation_scenarios(dev_mode=False)
    print(f"\nTotal validation scenarios: {len(all_scenarios)}")
    print(f"First 3 scenarios: {all_scenarios[:3]}")
    
    # Test dev mode
    dev_scenarios = loader.get_validation_scenarios(dev_mode=True)
    print(f"\nDev mode scenarios: {dev_scenarios}")
    
    # Test name cleaning
    clean_names = loader.get_scenario_names(dev_scenarios)
    print(f"\nClean scenario names: {clean_names}")