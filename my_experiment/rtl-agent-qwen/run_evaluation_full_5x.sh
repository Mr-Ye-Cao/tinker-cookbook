#!/bin/bash
# Continuous Evaluation Runner
# Runs full evaluation (all 92 problems) five times per iteration in an infinite loop

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

ITERATION=1

echo "====================================================================="
echo "Continuous Evaluation Runner"
echo "Will run: full (92 problems) x5 in a loop"
echo "Press Ctrl+C to stop"
echo "====================================================================="

while true; do
    echo ""
    echo "############################################################"
    echo "# ITERATION $ITERATION"
    echo "# Started at: $(date)"
    echo "############################################################"
    echo ""

    echo ">>> Running full evaluation (run 1 of 5)..."
    if ./run_evaluation_full.sh; then
        echo ">>> Full evaluation (run 1) completed successfully"
    else
        echo ">>> Full evaluation (run 1) failed with exit code $?"
        echo ">>> Continuing to next run..."
    fi

    echo ""
    echo ">>> Running full evaluation (run 2 of 5)..."
    if ./run_evaluation_full.sh; then
        echo ">>> Full evaluation (run 2) completed successfully"
    else
        echo ">>> Full evaluation (run 2) failed with exit code $?"
        echo ">>> Continuing to next run..."
    fi

    echo ""
    echo ">>> Running full evaluation (run 3 of 5)..."
    if ./run_evaluation_full.sh; then
        echo ">>> Full evaluation (run 3) completed successfully"
    else
        echo ">>> Full evaluation (run 3) failed with exit code $?"
        echo ">>> Continuing to next run..."
    fi

    echo ""
    echo ">>> Running full evaluation (run 4 of 5)..."
    if ./run_evaluation_full.sh; then
        echo ">>> Full evaluation (run 4) completed successfully"
    else
        echo ">>> Full evaluation (run 4) failed with exit code $?"
        echo ">>> Continuing to next run..."
    fi

    echo ""
    echo ">>> Running full evaluation (run 5 of 5)..."
    if ./run_evaluation_full.sh; then
        echo ">>> Full evaluation (run 5) completed successfully"
    else
        echo ">>> Full evaluation (run 5) failed with exit code $?"
        echo ">>> Continuing to next iteration..."
    fi

    echo ""
    echo "############################################################"
    echo "# ITERATION $ITERATION COMPLETE at $(date)"
    echo "############################################################"

    ITERATION=$((ITERATION + 1))

    # Optional: add a small delay between iterations
    # sleep 5
done
