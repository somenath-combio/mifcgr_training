#!/bin/bash

# Training script for miRNA-mRNA interaction prediction
# This script runs training with the specified parameters

echo "======================================================"
echo "miRNA-mRNA Interaction Prediction Training"
echo "======================================================"
echo ""

# Default parameters (matching requirements)
K=6
BATCH_SIZE=64
LR=0.003
DROPOUT=0.1
EPOCHS=100
PATIENCE=15

# Run training
python main.py \
    --data_path data/miraw.csv \
    --k $K \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --dropout $DROPOUT \
    --epochs $EPOCHS \
    --patience $PATIENCE \
    --checkpoint_path checkpoints/best_model.pth \
    --results_dir results \
    --log_file training.log

echo ""
echo "======================================================"
echo "Training Complete!"
echo "======================================================"
echo "Results saved to: results/"
echo "Best model saved to: checkpoints/best_model.pth"
echo "Training log saved to: logs/training.log"
