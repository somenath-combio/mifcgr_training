# miRNA-mRNA Interaction Prediction Training

This repository contains a complete training pipeline for predicting miRNA-mRNA interactions using Frequency Chaos Game Representation (FCGR) and deep learning.

## Project Structure

```
mifcgr_training/
├── data/
│   └── miraw.csv              # Training data
├── core/
│   ├── model.py               # Model architecture (InteractionModel)
│   ├── dataset.py             # PyTorch dataset class
│   └── trainer.py             # Training loop implementation
├── utils/
│   ├── fcgr.py                # FCGR generation utility
│   ├── early_stop.py          # Early stopping and model checkpointing
│   ├── metrics.py             # Metrics calculation
│   └── visualization.py       # Training visualization
├── main.py                    # Main training script
└── requirements.txt           # Python dependencies
```

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Training

Run training with default parameters:
```bash
python main.py
```

### Custom Parameters

Train with custom parameters:
```bash
python main.py \
    --data_path data/miraw.csv \
    --k 6 \
    --batch_size 64 \
    --lr 0.003 \
    --dropout 0.1 \
    --epochs 100 \
    --patience 15
```

### Command Line Arguments

- `--data_path`: Path to CSV data file (default: `data/miraw.csv`)
- `--k`: K-mer size for FCGR generation (default: `6`)
- `--batch_size`: Batch size for training (default: `64`)
- `--lr`: Learning rate (default: `0.003`)
- `--dropout`: Dropout rate (default: `0.1`)
- `--epochs`: Maximum number of epochs (default: `100`)
- `--patience`: Patience for early stopping (default: `15`)
- `--train_split`: Train/validation split ratio (default: `0.8`)
- `--checkpoint_path`: Path to save best model (default: `checkpoints/best_model.pth`)
- `--results_dir`: Directory for results (default: `results`)

## Training Pipeline

1. **Data Loading**: Reads CSV file and removes duplicates based on miRNA, mRNA sequences, and validation column
2. **Preprocessing**: Converts RNA sequences (U) to DNA sequences (T)
3. **FCGR Generation**: Generates 64x64 FCGR images for both miRNA and mRNA sequences (k=6)
4. **Model Training**: Trains InteractionModel with:
   - Loss function: CrossEntropyLoss
   - Optimizer: Adam (lr=0.003)
   - Dropout: 0.1
5. **Monitoring**: Tracks metrics (accuracy, precision, recall, F1, AUC, MCC)
6. **Early Stopping**: Stops training if validation accuracy doesn't improve
7. **Visualization**: Saves training curves and metrics

## Output

After training, you will find:
- `checkpoints/best_model.pth`: Best model checkpoint (saved based on highest validation accuracy)
- `results/training_progress.png`: Training and validation curves
- `results/training_history.txt`: Detailed training history
- `logs/training.log`: Complete training log

## Model Architecture

- **Input**: Two FCGR images (64x64) for miRNA and mRNA
- **Backbone**: Separate CNN branches for miRNA and mRNA (ModelK6)
- **Features**: Combined features from both branches
- **Output**: Binary classification (interaction/no interaction)

## Notes

- The model automatically selects GPU if available
- Training uses gradient clipping to prevent exploding gradients
- Learning rate scheduler reduces LR on validation loss plateau
- Best model is saved based on highest validation accuracy
- Early stopping monitors validation accuracy (not loss)
- All results are saved automatically during training
