# Deep Learning Interpretability for FCGR-based miRNA-mRNA Models

This directory contains a comprehensive interpretability pipeline for analyzing CNN models trained on Frequency Chaos Game Representation (FCGR) images for miRNA-mRNA interaction prediction.

## Overview

The `interpretability_analysis.py` script implements multiple state-of-the-art deep learning interpretability methods adapted specifically for FCGR image inputs:

### Implemented Methods

#### 1. **Gradient-based Attribution Methods**
- **Integrated Gradients (IG)**: Computes attribution by integrating gradients along a path from a baseline (zero FCGR) to the input
- **Saliency Maps**: Vanilla gradient-based attribution showing pixel importance
- **DeepLift**: Attribution method that handles neuron saturation better than gradients
- **GradientSHAP**: Combines Integrated Gradients with SHAP framework using multiple baselines
- **SmoothGrad**: Reduces noise by averaging gradients over multiple noisy samples

#### 2. **Gradient Correction for FCGR**
Since FCGR images are continuous-valued frequency representations (not one-hot sequences), we implement three correction strategies:
- **Normalize**: Divides attribution by FCGR magnitude to reduce bias toward high-frequency regions
- **Filter**: Applies median filtering to reduce gradient noise
- **Tangent**: Projects gradients onto FCGR manifold tangent space

#### 3. **SHAP DeepExplainer**
Uses game-theoretic SHAP values to explain predictions:
- Builds background distribution from real FCGR samples
- Computes SHAP values for both miRNA and mRNA branches
- Provides correlation analysis with Integrated Gradients

#### 4. **CNN Filter Visualization**
- Extracts and visualizes first convolutional layer filters
- Computes spatial activation maps for input samples
- Shows which patterns each filter detects

#### 5. **Grad-CAM (Gradient-weighted Class Activation Mapping)**
- Produces coarse localization maps showing important regions
- Works on final convolutional layer of each branch
- Provides overlays on original FCGR images

#### 6. **Interaction-level Heatmap**
- Combines miRNA and mRNA attributions into 2D interaction map
- Shows co-importance between sequence regions
- Similar to attention/alignment visualization

#### 7. **Analysis Utilities**
- Per-pixel attribution statistics
- K-means clustering of important regions
- Comparison between positive and negative samples
- Export to .npy and .png formats

## Installation

### Dependencies

Install the required packages:

```bash
pip install -r requirements_interpretability.txt
```

This includes:
- `captum` - PyTorch interpretability library
- `shap` - SHAP values implementation
- `scikit-image` - Image processing
- `scipy` - Scientific computing
- `scikit-learn` - Clustering and analysis

## Usage

### Basic Usage

Run interpretability analysis on your trained model:

```bash
python interpretability_analysis.py \
    --checkpoint checkpoints/best_model.pth \
    --data data/miraw.csv \
    --k 6 \
    --dropout 0.1 \
    --n_samples 5 \
    --n_background 20 \
    --output results/interpretability
```

### Arguments

- `--checkpoint`: Path to trained model checkpoint (default: `checkpoints/best_model.pth`)
- `--data`: Path to dataset CSV file (default: `data/miraw.csv`)
- `--k`: K-mer size for FCGR generation (default: 6)
- `--dropout`: Dropout rate used during training (default: 0.1)
- `--n_samples`: Number of samples to analyze in detail (default: 5)
- `--n_background`: Number of background samples for SHAP (default: 20)
- `--output`: Output directory for results (default: `results/interpretability`)

### Example: Analyze 10 samples with 50 background samples

```bash
python interpretability_analysis.py \
    --checkpoint checkpoints/best_model.pth \
    --n_samples 10 \
    --n_background 50 \
    --output results/detailed_analysis
```

## Output Structure

The script generates a comprehensive set of outputs:

```
results/interpretability/
├── logs/
│   └── interpretability.log                      # Detailed logging
│
├── numpy_exports/                                 # Raw attribution data
│   ├── sample_0_IntegratedGradients_mirna.npy
│   ├── sample_0_IntegratedGradients_mrna.npy
│   └── ...
│
├── analysis_summary.json                          # Analysis metadata
├── attribution_statistics.csv                     # Statistical summary
│
├── mirna_conv1_filters.png                       # CNN filter visualizations
├── mrna_conv1_filters.png
├── mirna_conv1_activations.png
├── mrna_conv1_activations.png
│
├── average_attribution_IntegratedGradients.png   # Average attributions
├── average_attribution_SHAP.png
└── ...
│
└── Per-sample outputs (for each analyzed sample):
    ├── sample_0_mirna_comparison.png              # Compare all methods
    ├── sample_0_mrna_comparison.png
    ├── sample_0_gradcam.png                       # Grad-CAM overlay
    ├── sample_0_interaction_heatmap.png           # Interaction map
    ├── sample_0_mirna_clusters.png                # Clustered regions
    ├── sample_0_mrna_clusters.png
    ├── sample_0_mirna_IntegratedGradients.png     # Individual methods
    ├── sample_0_mrna_IntegratedGradients.png
    └── ... (for each attribution method)
```

## Interpretation Guide

### Understanding Attribution Maps

**Color Schemes:**
- **Red/Positive values**: Regions that increase prediction confidence for the target class
- **Blue/Negative values**: Regions that decrease prediction confidence
- **White/Zero values**: Regions with minimal impact

**Magnitude:**
- Larger absolute values indicate stronger influence on the prediction
- Attribution maps are normalized and centered around zero

### Comparing Methods

Different attribution methods have different properties:

1. **Integrated Gradients**: Most theoretically grounded, satisfies important axioms
2. **Saliency**: Fast but can be noisy, shows local gradient information
3. **DeepLift**: Good for handling saturation in ReLU networks
4. **GradientSHAP**: Robust to baseline choice, based on game theory
5. **SmoothGrad**: Smoothed version of saliency, less noisy
6. **SHAP**: Game-theoretic, computationally expensive but theoretically sound

**Best Practice**: Compare multiple methods and look for consensus on important regions.

### FCGR-Specific Considerations

FCGR images differ from standard images:

1. **Frequency values**: Pixel values represent k-mer frequencies, not colors
2. **Spatial structure**: Nearby pixels in FCGR represent similar k-mer patterns
3. **Interpretation**: Important regions indicate k-mer patterns critical for prediction

**Gradient Correction**: The pipeline applies FCGR-specific corrections to account for continuous-valued nature:
- Normalization reduces bias toward high-frequency regions
- Tangent projection respects the FCGR manifold structure

### Grad-CAM vs Attribution Methods

- **Grad-CAM**: Coarse spatial localization (lower resolution)
- **Attribution methods**: Fine-grained pixel-level importance
- **Use together**: Grad-CAM for overall regions, attributions for details

### Interaction Heatmaps

The interaction heatmap shows co-importance between miRNA and mRNA regions:
- **High values**: Both regions are important together (potential binding sites)
- **Interpretation**: Similar to attention weights or alignment scores
- **Note**: This is a synthetic visualization (not learned attention)

## Advanced Usage

### Programmatic Access

You can also use the interpretability functions programmatically:

```python
import torch
from pathlib import Path
from interpretability_analysis import (
    load_trained_model,
    compute_integrated_gradients,
    compute_gradcam,
    visualize_attribution_heatmap
)

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = load_trained_model('checkpoints/best_model.pth', k=6, dropout_rate=0.1, device=device)

# Load your data
mirna_input = ...  # Shape: [1, 1, 64, 64]
mrna_input = ...   # Shape: [1, 1, 64, 64]

# Compute attributions
ig_mirna, ig_mrna = compute_integrated_gradients(
    model, mirna_input, mrna_input, target_class=1, device=device
)

# Visualize
visualize_attribution_heatmap(
    mirna_input.squeeze().cpu().numpy(),
    ig_mirna,
    title='miRNA Integrated Gradients',
    save_path=Path('my_attribution.png')
)
```

### Analyzing Specific Samples

To analyze specific sample indices instead of random samples:

```python
from interpretability_analysis import analyze_sample
from core.dataset import MiRNAInteractionDataset

# Load dataset
dataset = MiRNAInteractionDataset('data/miraw.csv', k=6)

# Select specific sample
sample_idx = 42
mirna, mrna, label = dataset[sample_idx]

# Analyze
results = analyze_sample(
    model,
    mirna.unsqueeze(0).to(device),
    mrna.unsqueeze(0).to(device),
    label.item(),
    sample_idx,
    background_data,
    device,
    save_dir=Path('results/sample_42')
)
```

## Performance Considerations

**Computational Cost:**
- SHAP DeepExplainer is most expensive (scales with background size)
- Integrated Gradients cost depends on n_steps (default: 50)
- Saliency and Grad-CAM are fastest

**Memory:**
- Large background datasets for SHAP may require significant GPU memory
- Reduce `n_background` if you encounter OOM errors

**Recommendations:**
- Start with `n_samples=3` and `n_background=10` for initial exploration
- Use CPU if GPU memory is limited
- For production: `n_samples=10`, `n_background=50`

## Citation

If you use this interpretability pipeline in your research, please cite the relevant papers:

```bibtex
@article{sundararajan2017axiomatic,
  title={Axiomatic attribution for deep networks},
  author={Sundararajan, Mukund and Taly, Ankur and Yan, Qiqi},
  journal={ICML},
  year={2017}
}

@inproceedings{lundberg2017unified,
  title={A unified approach to interpreting model predictions},
  author={Lundberg, Scott M and Lee, Su-In},
  booktitle={NeurIPS},
  year={2017}
}

@article{selvaraju2017grad,
  title={Grad-cam: Visual explanations from deep networks via gradient-based localization},
  author={Selvaraju, Ramprasaath R and Cogswell, Michael and Das, Abhishek and Vedantam, Ramakrishna and Parikh, Devi and Batra, Dhruv},
  journal={ICCV},
  year={2017}
}
```

## Troubleshooting

### Common Issues

**1. Import errors for captum/shap**
```bash
pip install captum shap
```

**2. CUDA out of memory**
- Reduce `n_background` parameter
- Use CPU: Set `device = torch.device('cpu')`
- Process fewer samples at once

**3. Checkpoint loading errors**
- Ensure k and dropout values match training configuration
- Check checkpoint format (should contain 'model_state_dict')

**4. Visualization issues**
- Ensure matplotlib backend is configured
- For headless servers: `export MPLBACKEND=Agg`

## Architecture-Specific Notes

### Two-Branch CNN Model

The pipeline handles the dual-branch architecture by:
1. Creating wrapper classes that fix one input while varying the other
2. Computing attributions separately for each branch
3. Combining results for interaction analysis

### InceptionBlock Support

The model uses InceptionBlocks with multiple kernel sizes:
- Attributions are computed end-to-end through all layers
- Filter visualizations show the first conv layer
- Grad-CAM targets the final conv layer (conv3)

## Future Enhancements

Potential extensions to this pipeline:

1. **Sequence-level interpretation**: Map FCGR attributions back to sequence positions
2. **Motif discovery**: Cluster important patterns and extract k-mer motifs
3. **Comparative analysis**: Compare positive vs negative samples systematically
4. **Interactive visualization**: Web-based dashboard for exploring results
5. **Statistical significance**: Bootstrap confidence intervals for attributions

## Contact & Support

For questions or issues:
1. Check the logs in `logs/interpretability.log`
2. Review the output `analysis_summary.json`
3. Ensure all dependencies are correctly installed

## License

This interpretability pipeline is provided as-is for research and educational purposes.
