#!/bin/bash
# Continuous Evaluation Runner
# Runs medium_hard followed by 16easy evaluations in an infinite loop

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

ITERATION=1

echo "====================================================================="
echo "Continuous Evaluation Runner"
echo "Will run: medium_hard -> 16easy in a loop"
echo "Press Ctrl+C to stop"
echo "====================================================================="

while true; do
    echo ""
    echo "############################################################"
    echo "# ITERATION $ITERATION"
    echo "# Started at: $(date)"
    echo "############################################################"
    echo ""

    echo ">>> Running medium + hard evaluation..."
    if ./run_evaluation_medium_hard.sh; then
        echo ">>> Medium + hard evaluation completed successfully"
    else
        echo ">>> Medium + hard evaluation failed with exit code $?"
        echo ">>> Continuing to next evaluation..."
    fi

    echo ""
    echo ">>> Running 16 easy evaluation..."
    if ./run_evaluation_16easy.sh; then
        echo ">>> 16 easy evaluation completed successfully"
    else
        echo ">>> 16 easy evaluation failed with exit code $?"
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
