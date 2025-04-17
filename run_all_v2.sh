#!/bin/bash

# Exit on error
set -e

echo "=== Starting RevenaHybrid V2 Pipeline ==="

# Step 1: Train LTM V2 model
echo "Training LTM V2 model..."
python train_v2.py

# Step 2: Evaluate models
echo "Evaluating models..."
python eval_v2.py

# Step 3: Plot results
echo "Generating plots..."
python plot_eval.py --v2

echo "=== Pipeline completed successfully ==="
echo "Results saved to:"
echo "- results_v2.json (metrics)"
echo "- results_v2.png (visualization)" 