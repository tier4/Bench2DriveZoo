#!/usr/bin/env bash
# DiffusionDrive Open-Loop Evaluation Script for Bench2Drive
# Single GPU execution (distributed support can be added later)

# Script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Default values
CONFIG="${SCRIPT_DIR}/configs/test_config.yaml"
CHECKPOINT=""
DATA_ROOT="/mnt/nvme1/dataset/Bench2Drive-Base"
OUTPUT_DIR="${SCRIPT_DIR}/eval_results"
MAX_SAMPLES=""
DEVICE="cuda"
GPU_ID=0

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --checkpoint)
            CHECKPOINT="$2"
            shift 2
            ;;
        --data-root)
            DATA_ROOT="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --max-samples)
            MAX_SAMPLES="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --gpu)
            GPU_ID="$2"
            shift 2
            ;;
        --debug)
            DEBUG_FLAG="--debug"
            shift
            ;;
        --help)
            echo "DiffusionDrive Open-Loop Evaluation Script"
            echo ""
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --config PATH         Path to configuration file (default: configs/test_config.yaml)"
            echo "  --checkpoint PATH     Path to model checkpoint (overrides config)"
            echo "  --data-root PATH      Path to B2D dataset (default: /mnt/nvme1/dataset/Bench2Drive-Base)"
            echo "  --output-dir PATH     Output directory for results"
            echo "  --max-samples N       Maximum number of samples to evaluate"
            echo "  --device DEVICE       Device to use: cuda or cpu (default: cuda)"
            echo "  --gpu ID              GPU ID to use (default: 0)"
            echo "  --debug               Enable debug logging"
            echo "  --help                Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Set up environment
export CUDA_VISIBLE_DEVICES=$GPU_ID

# Add Python paths
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"
export PYTHONPATH="${SCRIPT_DIR}/../../..:${PYTHONPATH}"  # Bench2Drive root
export PYTHONPATH="${SCRIPT_DIR}/../../../DiffusionDrive:${PYTHONPATH}"  # DiffusionDrive

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Build command
CMD="python3 ${SCRIPT_DIR}/test.py --config ${CONFIG}"

# Add optional arguments
if [ ! -z "$CHECKPOINT" ]; then
    CMD="$CMD --checkpoint $CHECKPOINT"
fi

if [ ! -z "$DATA_ROOT" ]; then
    CMD="$CMD --data-root $DATA_ROOT"
fi

if [ ! -z "$OUTPUT_DIR" ]; then
    CMD="$CMD --output-dir $OUTPUT_DIR"
fi

if [ ! -z "$MAX_SAMPLES" ]; then
    CMD="$CMD --max-samples $MAX_SAMPLES"
fi

if [ ! -z "$DEVICE" ]; then
    CMD="$CMD --device $DEVICE"
fi

if [ ! -z "$DEBUG_FLAG" ]; then
    CMD="$CMD $DEBUG_FLAG"
fi

# Print configuration
echo "========================================="
echo "DiffusionDrive Open-Loop Evaluation"
echo "========================================="
echo "Config: $CONFIG"
echo "Checkpoint: ${CHECKPOINT:-from config}"
echo "Data Root: $DATA_ROOT"
echo "Output Dir: $OUTPUT_DIR"
echo "Device: $DEVICE (GPU $GPU_ID)"
echo "Max Samples: ${MAX_SAMPLES:-all}"
echo "========================================="
echo ""

# Run evaluation
echo "Starting evaluation..."
echo "Command: $CMD"
echo ""

# Execute
$CMD

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo "========================================="
    echo "Evaluation completed successfully!"
    echo "Results saved to: $OUTPUT_DIR"
    echo "========================================="
else
    echo ""
    echo "========================================="
    echo "Evaluation failed!"
    echo "Check the error messages above."
    echo "========================================="
    exit 1
fi