# DiffusionDrive Open-Loop Evaluation

Open-loop evaluation implementation for DiffusionDrive on Bench2Drive dataset.

## Quick Start

```bash
# Run evaluation
python3 test_openloop.py

# Or use the shell script
bash dist_test.sh
```

## Files

- `test_openloop.py` - Main evaluation script using DiffusionDrive's existing modules
- `metrics.py` - VAD-compatible metric computation
- `model_loader.py` - Model loading utilities
- `configs/test_config.yaml` - Configuration file
- `dist_test.sh` - Shell script wrapper

## Key Features

- Uses existing DiffusionDrive infrastructure (no reinventing the wheel)
- Processes ALL sensor inputs (cameras, LiDAR, ego status, driving commands)
- Produces VAD-compatible metrics for benchmarking
- No dummy data or placeholders

## Results

Results are saved to `eval_results/diffusiondrive_openloop/` in JSON format with:
- L2 errors at multiple time horizons (0.5s to 3.0s)
- Per-scene and aggregate metrics
- VAD-compatible format for comparison

## Configuration

Edit `test_openloop.py` to modify:
- `checkpoint_path`: Path to model checkpoint
- `data_root`: Path to Bench2Drive dataset  
- `scenario`: Specific scenario to evaluate
- `num_samples`: Number of samples to process

## Dependencies

- DiffusionDrive package (`/workspace/Bench2Drive/DiffusionDrive`)
- Bench2Drive dataset
- PyTorch, numpy, tqdm, laspy

## Important Notes

This implementation properly uses DiffusionDrive's existing modules:
- `Bench2DriveSceneLoader` for data loading
- `Bench2DriveFeatureBuilder` for feature processing
- No manual reimplementation of existing functionality