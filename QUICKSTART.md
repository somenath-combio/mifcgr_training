# Quick Start Guide

## Prerequisites

1. Python 3.8 or higher
2. CUDA-capable GPU (optional, but recommended)

## Installation

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- PyTorch (>= 2.0.0)
- NumPy (>= 1.24.0)
- Pandas (>= 2.0.0)
- scikit-learn (>= 1.3.0)
- matplotlib (>= 3.7.0)
- tqdm (>= 4.65.0)

### Step 2: Verify Installation

```bash
python -c "import torch; print(f'PyTorch {torch.__version__} installed successfully')"
```

## Running Training

### Default Configuration

The default configuration matches your requirements:
- K-mer size: 6
- Batch size: 64
- Learning rate: 0.003
- Dropout rate: 0.1
- Loss function: CrossEntropyLoss
- Optimizer: Adam

Simply run:

```bash
python main.py
```

### Custom Configuration

Override default parameters:

```bash
python main.py --k 6 --batch_size 64 --lr 0.003 --dropout 0.1 --epochs 100
```

### Minimal Test Run

For a quick test (fewer epochs):

```bash
python main.py --epochs 5 --patience 3
```

## Expected Output

During training, you will see:
1. Data loading information
2. Model architecture details
3. Training progress bars
4. Epoch-by-epoch metrics (loss, accuracy, F1, AUC)
5. Early stopping notifications

After training:
- `checkpoints/best_model.pth` - Best model weights
- `results/training_progress.png` - Visualization of training/validation curves
- `results/training_history.txt` - Complete training history
- `logs/training.log` - Detailed training log

## Troubleshooting

### CUDA Out of Memory

Reduce batch size:
```bash
python main.py --batch_size 32
```

### Slow Training

Enable GPU if available:
```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Data Issues

Verify your CSV file has these columns:
- mature_miRNA_Transcript
- mRNA_Site_Transcript
- validation

## Training Flow Summary

1. **Data Loading**: Reads `data/miraw.csv`
2. **Preprocessing**:
   - Removes duplicates based on miRNA, mRNA, and validation columns
   - Converts RNA sequences (U → T)
3. **FCGR Generation**: Creates 64×64 images for each sequence (k=6)
4. **Model Training**:
   - Dual CNN branches for miRNA and mRNA
   - Combined features for classification
5. **Evaluation**: Tracks multiple metrics and saves best model
6. **Visualization**: Generates training curves

## Next Steps

After training completes:

1. Check the best model performance in the logs
2. View training curves in `results/training_progress.png`
3. Load the best model for inference:

```python
import torch
from core.model import InteractionModel

# Load model
model = InteractionModel(dropout_rate=0.1, k=6)
checkpoint = torch.load('checkpoints/best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Use for prediction
# ... your inference code ...
```
