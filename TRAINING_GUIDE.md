# Training Script Implementation Guide

## Overview
The complete training pipeline for miRNA-mRNA interaction prediction is already implemented and ready to use.

## Requirements Verification

### ✓ Data Processing (Implemented in `core/dataset.py`)
- **Read CSV**: Lines 28-29 use `pandas.read_csv()` to read `data/miraw.csv`
- **Drop Duplicates**: Line 33 drops duplicates based on `['mature_miRNA_Transcript', 'mRNA_Site_Transcript', 'validation']`
- **RNA to DNA Conversion**: Lines 44-54 convert RNA sequences to DNA by replacing 'U' with 'T'
- **FCGR Generation**: Lines 56-73 generate FCGR for both miRNA and mRNA sequences

### ✓ Dataset & DataLoader (Implemented in `core/dataset.py`)
- **PyTorch Dataset**: `MiRNAInteractionDataset` class (lines 9-107)
- **DataLoader Creation**: `prepare_data_loaders()` function (lines 110-160)
- **Batch Size**: Default is 64 (line 222 in `main.py`)

### ✓ Model Configuration (Implemented in `main.py` and `core/model.py`)
- **K-mer Size**: k=6 (default, line 207 in `main.py`)
- **Model Architecture**: `ModelK6` for 64x64 FCGR input (lines 223-294 in `core/model.py`)
- **Dropout Rate**: 0.1 (default, line 214 in `main.py`)

### ✓ Training Parameters (Implemented in `main.py`)
- **Loss Function**: CrossEntropyLoss (line 125)
- **Optimizer**: Adam (line 129)
- **Learning Rate**: 0.003 (default, line 229)
- **Validation Monitoring**: Tracks validation accuracy, F1, AUC, and more (lines 216-218 in `core/trainer.py`)

### ✓ Training Loop (Implemented in `core/trainer.py`)
- **Training Epoch**: Lines 71-135
- **Validation Epoch**: Lines 137-190
- **Metrics Tracking**: Uses `MetricsCalculator` to track accuracy, precision, recall, F1, AUC, MCC
- **Early Stopping**: Monitors validation loss and stops if no improvement (lines 238-241)
- **Model Checkpointing**: Saves best model based on validation loss (line 236)

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Training with Default Parameters
```bash
python main.py
```

This will use:
- Data: `data/miraw.csv`
- K-mer size: 6
- Batch size: 64
- Learning rate: 0.003
- Dropout: 0.1
- Loss: CrossEntropyLoss

### 3. Monitor Training
The script will:
- Print training progress with tqdm progress bars
- Log metrics for each epoch (train/val loss, accuracy, F1, AUC)
- Save visualizations to `results/`
- Save best model to `checkpoints/best_model.pth`
- Save detailed logs to `logs/training.log`

### 4. Custom Parameters Example
```bash
python main.py \
    --data_path data/miraw.csv \
    --k 6 \
    --batch_size 64 \
    --lr 0.003 \
    --dropout 0.1 \
    --epochs 100 \
    --patience 15 \
    --checkpoint_path checkpoints/best_model.pth \
    --results_dir results
```

## Training Flow Detailed

### Step 1: Data Loading (`core/dataset.py`)
```python
# Read CSV
df = pd.read_csv(csv_path)

# Drop duplicates
df = df.drop_duplicates(subset=['mature_miRNA_Transcript', 'mRNA_Site_Transcript', 'validation'])
```

### Step 2: Sequence Processing
```python
# RNA to DNA conversion
def _rna_to_dna(self, sequence: str) -> str:
    return sequence.replace('U', 'T')

# FCGR generation
fcgr = FCGR(sequence=dna_sequence, k=6)
fcgr_matrix = fcgr.generate_fcgr()  # Returns 64x64 numpy array
```

### Step 3: Dataset Creation
```python
# Create DataLoaders with batch_size=64
train_loader, val_loader = prepare_data_loaders(
    csv_path='data/miraw.csv',
    k=6,
    batch_size=64,
    train_split=0.8
)
```

### Step 4: Model Initialization
```python
# Initialize model with k=6 (uses ModelK6 for 64x64 FCGR)
model = InteractionModel(dropout_rate=0.1, k=6)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.003)
```

### Step 5: Training Loop
```python
# Trainer handles everything
trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=criterion,
    optimizer=optimizer,
    device=device,
    num_epochs=100,
    patience=15
)

# Train and monitor
history = trainer.train()
```

## Validation Accuracy Monitoring

The training script monitors validation accuracy in multiple ways:

1. **Real-time Logging** (`core/trainer.py` line 216):
   ```python
   logging.info(f"Train Acc: {train_metrics.get('accuracy', 0):.4f} | Val Acc: {val_metrics.get('accuracy', 0):.4f}")
   ```

2. **Visualization** (`core/trainer.py` lines 227-233):
   - Plots validation accuracy vs epoch
   - Saves to `results/accuracy_plot.png`
   - Combined plot in `results/training_progress.png`

3. **History Tracking** (`core/trainer.py` line 224):
   - Stores all validation metrics in `history['val_metrics']`
   - Includes: accuracy, precision, recall, F1, AUC, MCC, specificity, sensitivity

4. **Best Model Selection** (`core/trainer.py` lines 250-255):
   - Reports accuracy of best model
   - Based on lowest validation loss

## Output Files

After training completes:

```
mifcgr_training/
├── checkpoints/
│   └── best_model.pth          # Best model weights
├── results/
│   ├── training_progress.png   # Loss and accuracy curves
│   ├── loss_plot.png          # Detailed loss plot
│   ├── accuracy_plot.png      # Detailed accuracy plot
│   └── training_history.txt   # CSV format history
└── logs/
    └── training.log           # Complete training log
```

## Key Features

1. **Automatic GPU Detection**: Uses CUDA if available
2. **Early Stopping**: Prevents overfitting
3. **Gradient Clipping**: Prevents exploding gradients (max_norm=1.0)
4. **Learning Rate Scheduling**: ReduceLROnPlateau (factor=0.5, patience=5)
5. **Comprehensive Metrics**: Accuracy, Precision, Recall, F1, AUC, MCC, Specificity, Sensitivity
6. **Progress Bars**: tqdm for real-time training progress
7. **Reproducibility**: Fixed random seed (default: 42)

## Example Training Output

```
==============================================================
Starting Training
==============================================================
Device: cuda:0
Number of epochs: 100
Training samples: 800
Validation samples: 200
Batch size: 64
==============================================================

Epoch 1/100 [Train]: 100%|████████| 13/13 [00:05<00:00]
Epoch 1/100 [Val]: 100%|████████| 4/4 [00:01<00:00]

Epoch 1/100
Train Loss: 0.6543 | Val Loss: 0.6234
Train Acc: 0.6250 | Val Acc: 0.6500
Train F1: 0.6123 | Val F1: 0.6421
Val AUC: 0.6789

Model checkpoint saved at epoch 1 with score: 0.6234

...

Early stopping triggered at epoch 45

Best model at epoch 32:
Validation Loss: 0.4123
Validation Accuracy: 0.8250
Validation F1: 0.8156
Validation AUC: 0.8934
```

## All Command Line Options

```bash
python main.py --help
```

Available arguments:
- `--data_path`: Path to CSV file (default: `data/miraw.csv`)
- `--train_split`: Train/validation split (default: `0.8`)
- `--k`: K-mer size for FCGR (default: `6`)
- `--dropout`: Dropout rate (default: `0.1`)
- `--batch_size`: Training batch size (default: `64`)
- `--lr`: Learning rate (default: `0.003`)
- `--weight_decay`: L2 regularization (default: `1e-5`)
- `--epochs`: Maximum epochs (default: `100`)
- `--patience`: Early stopping patience (default: `15`)
- `--seed`: Random seed (default: `42`)
- `--checkpoint_path`: Model save path (default: `checkpoints/best_model.pth`)
- `--results_dir`: Results directory (default: `results`)
- `--log_file`: Log filename (default: `training.log`)

## Summary

**Everything you requested is already implemented!** The training script is production-ready with:
- ✓ Data preprocessing (pandas, duplicate removal)
- ✓ RNA to DNA conversion
- ✓ FCGR generation (k=6)
- ✓ PyTorch Dataset and DataLoader (batch_size=64)
- ✓ CrossEntropyLoss
- ✓ Adam optimizer (lr=0.003)
- ✓ Dropout (0.1)
- ✓ Validation accuracy monitoring
- ✓ Early stopping, checkpointing, visualization
- ✓ Comprehensive metrics and logging

Simply run `python main.py` to start training!
