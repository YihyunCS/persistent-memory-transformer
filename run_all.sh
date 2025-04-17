#!/bin/bash

# Exit on error
set -e

echo "=== Starting RevenaHybrid Pipeline ==="

# Step 1: Prepare data
echo "Preparing data..."
python prepare_data.py

# Step 2: Train baseline model
echo "Training baseline model..."
python train.py --baseline

# Step 3: Train LTM model
echo "Training LTM model..."
python train.py --with-ltm

# Step 4: Evaluate models
echo "Evaluating models..."
python eval.py

# Step 5: Plot results
echo "Generating plots..."
python plot_eval.py

echo "=== Pipeline completed successfully ==="
echo "Results saved to:"
echo "- results.json (metrics)"
echo "- results.png (visualization)"