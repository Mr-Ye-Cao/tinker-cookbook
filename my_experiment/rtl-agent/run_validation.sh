#!/bin/bash
# Validation script for CVDP + Tinker integration

set -e  # Exit on error

echo "====================================================================="
echo "CVDP + Tinker RTL Agent Validation"
echo "====================================================================="
echo ""

# Check if venv exists
if [ ! -d "venv" ]; then
    echo "Error: Virtual environment not found. Run setup first."
    exit 1
fi

# Activate virtual environment
source venv/bin/activate

# Set environment variables
export TINKER_API_KEY="${TINKER_API_KEY:-$(cat ../../.env | grep TINKER_API_KEY | cut -d'=' -f2)}"
export CVDP_DATASET_PATH="/home/ubuntu/peter/benchmark/cvdp_benchmark/example_dataset/cvdp_v1.0.1_example_agentic_code_generation_no_commercial_with_solutions.jsonl"

echo "Step 1: Running integration tests..."
python test_setup.py
echo ""

echo "Step 2: Checking Docker availability..."
if ! command -v docker &> /dev/null; then
    echo "Warning: Docker not found. CVDP evaluation will fail."
    echo "Please install Docker: https://docs.docker.com/get-docker/"
else
    echo "Docker version: $(docker --version)"
fi
echo ""

echo "Step 3: Checking dataset..."
if [ -f "$CVDP_DATASET_PATH" ]; then
    NUM_PROBLEMS=$(wc -l < "$CVDP_DATASET_PATH")
    echo "Found CVDP dataset with $NUM_PROBLEMS problems"
else
    echo "Error: CVDP dataset not found at $CVDP_DATASET_PATH"
    exit 1
fi
echo ""

echo "====================================================================="
echo "Validation Complete!"
echo "====================================================================="
