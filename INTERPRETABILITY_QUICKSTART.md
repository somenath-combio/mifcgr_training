# Interpretability Analysis - Quick Start Guide

## What You Get

A complete deep learning interpretability pipeline for your FCGR-based miRNA-mRNA interaction model with **9 different analysis methods** and **publication-quality visualizations**.

## Files Created

```
mifcgr_training/
├── interpretability_analysis.py              # Main interpretability script (60KB)
├── example_interpretability.py                # Quick start examples (12KB)
├── requirements_interpretability.txt          # Dependencies
├── INTERPRETABILITY_README.md                 # Full documentation (12KB)
└── INTERPRETABILITY_QUICKSTART.md             # This file
```

## Installation (One-Time Setup)

```bash
# Install interpretability dependencies
pip install -r requirements_interpretability.txt
```

This installs:
- `captum` - PyTorch interpretability (Integrated Gradients, DeepLift, etc.)
- `shap` - SHAP values for deep learning
- `scikit-image`, `scipy`, `scikit-learn` - Analysis utilities

## Usage

### Option 1: Command Line (Recommended for First Use)

Analyze your trained model with one command:

```bash
python interpretability_analysis.py \
    --checkpoint checkpoints/best_model.pth \
    --data data/miraw.csv \
    --k 6 \
    --n_samples 5 \
    --output results/interpretability
```

**Note**: You need a trained model checkpoint first. If you don't have one:
```bash
# Train your model first
python main.py --epochs 50 --k 6
```

### Option 2: Quick Example Script

See results faster with the example script:

```bash
python example_interpretability.py
```

This analyzes 1 sample with 3 methods and saves to `results/quick_example/`.

### Option 3: Python API (For Custom Analysis)

```python
from interpretability_analysis import (
    load_trained_model,
    compute_integrated_gradients,
    visualize_attribution_heatmap
)

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = load_trained_model('checkpoints/best_model.pth', k=6, dropout_rate=0.1, device=device)

# Compute attributions
ig_mirna, ig_mrna = compute_integrated_gradients(
    model, mirna_input, mrna_input, target_class=1, device=device
)

# Visualize
visualize_attribution_heatmap(
    fcgr_mirna, ig_mirna,
    title='miRNA Attribution',
    save_path='my_result.png'
)
```

## What Gets Analyzed

### 1. **Gradient-based Attributions** (5 methods)
- **Integrated Gradients** ⭐ Most reliable, theoretically grounded
- **Saliency Maps** - Fast vanilla gradients
- **DeepLift** - Handles ReLU saturation
- **GradientSHAP** - Game-theoretic, robust
- **SmoothGrad** - Noise-reduced gradients

### 2. **FCGR-Specific Corrections**
- Normalization (removes frequency bias)
- Filtering (reduces noise)
- Tangent projection (respects FCGR manifold)

### 3. **SHAP DeepExplainer**
- Game-theoretic feature importance
- Comparison with Integrated Gradients
- Correlation analysis

### 4. **CNN Filter Visualization**
- First layer filters (learned patterns)
- Activation maps (what fires for your input)

### 5. **Grad-CAM**
- Spatial localization heatmaps
- Overlays on FCGR images

### 6. **Interaction Heatmaps**
- Combined miRNA-mRNA co-importance
- 2D visualization (like attention maps)

### 7. **Cluster Analysis**
- K-means clustering of important regions
- Identifies spatial patterns

## Output Structure

After running analysis on 5 samples:

```
results/interpretability/
├── logs/
│   └── interpretability.log                      # Detailed logs
│
├── numpy_exports/                                 # Raw data (.npy files)
│   ├── sample_0_IntegratedGradients_mirna.npy
│   └── ...
│
├── analysis_summary.json                          # Metadata
├── attribution_statistics.csv                     # Statistics table
│
├── CNN Filters:
│   ├── mirna_conv1_filters.png
│   ├── mrna_conv1_filters.png
│   ├── mirna_conv1_activations.png
│   └── mrna_conv1_activations.png
│
├── Average Attributions:
│   ├── average_attribution_IntegratedGradients.png
│   ├── average_attribution_SHAP.png
│   └── ...
│
└── Per-Sample Results (×5 samples):
    ├── sample_0_mirna_comparison.png              # All methods side-by-side
    ├── sample_0_mrna_comparison.png
    ├── sample_0_gradcam.png                       # Grad-CAM overlays
    ├── sample_0_interaction_heatmap.png           # miRNA-mRNA interaction
    ├── sample_0_mirna_clusters.png                # Clustered regions
    ├── sample_0_mrna_clusters.png
    └── sample_0_{mirna|mrna}_{method}.png        # Individual methods

Total: ~100+ visualization files + statistics
```

## Interpreting Results

### Attribution Maps

**Colors:**
- 🔴 Red = Increases prediction confidence
- 🔵 Blue = Decreases prediction confidence
- ⚪ White = Neutral/no effect

**Magnitude:**
- Brighter = Stronger influence
- Look for consensus across methods

### Which Method to Trust?

**For publication**: Use **Integrated Gradients** + **SHAP**
- Most theoretically sound
- Well-established in literature
- Best combination of speed and reliability

**For exploration**: Compare all methods
- Consensus regions = highly reliable
- Method-specific patterns = investigate further

### FCGR-Specific Tips

1. **High-frequency regions**: May show higher attribution due to magnitude bias
   - Use gradient correction to adjust
   - Look at normalized attributions

2. **Spatial patterns**: FCGR has hierarchical structure
   - Similar k-mers appear nearby
   - Cluster analysis helps identify patterns

3. **Interaction heatmaps**: Show co-importance
   - High values = both regions important together
   - May indicate binding sites or functional regions

## Common Workflows

### Workflow 1: Understand Model Predictions

```bash
# Analyze 10 samples to see what model focuses on
python interpretability_analysis.py --n_samples 10 --n_background 30

# Check results/interpretability/sample_*_mirna_comparison.png
# Look for patterns across samples
```

### Workflow 2: Debug Model Performance

```bash
# Analyze misclassified samples
# (You'll need to modify script to select specific samples)

# Compare positive vs negative class attributions
# Look for bias or spurious correlations
```

### Workflow 3: Publication Figures

```bash
# High-quality analysis with large background set
python interpretability_analysis.py \
    --n_samples 20 \
    --n_background 100 \
    --output results/publication_figs

# Use generated PNG files at 300 DPI
# See average_attribution_IntegratedGradients.png
```

### Workflow 4: Extract Biological Insights

```python
# Use cluster analysis to find important k-mer regions
from interpretability_analysis import cluster_attribution_regions

cluster_map, centers = cluster_attribution_regions(
    ig_mirna, n_clusters=5, threshold_percentile=75
)

# Map cluster centers back to k-mer sequences
# (You'll need to implement reverse FCGR mapping)
```

## Performance Tips

**For Speed:**
- Start with `--n_samples 3 --n_background 10`
- Skip SHAP if too slow (most expensive method)
- Use CPU for small analyses

**For Quality:**
- Use `--n_background 50-100` for stable SHAP values
- Analyze 10+ samples for robust averages
- Use GPU for large batches

**Memory Issues:**
- Reduce `--n_background` (SHAP memory intensive)
- Use smaller batches
- Switch to CPU if GPU OOM

## Troubleshooting

### "No module named 'captum'"
```bash
pip install captum shap
```

### "Checkpoint loading error"
Make sure `--k` and `--dropout` match your training configuration.

### "CUDA out of memory"
```bash
# Use CPU
export CUDA_VISIBLE_DEVICES=""
python interpretability_analysis.py ...
```

### "No model checkpoint found"
Train your model first:
```bash
python main.py --epochs 50 --k 6
```

## Next Steps

1. **Read full documentation**: See `INTERPRETABILITY_README.md`
2. **Try example script**: Run `python example_interpretability.py`
3. **Run analysis**: `python interpretability_analysis.py`
4. **Explore results**: Check `results/interpretability/`
5. **Customize**: Modify scripts for your specific needs

## Key Features

✅ **9 interpretability methods** (IG, Saliency, DeepLift, GradientSHAP, SmoothGrad, SHAP, Grad-CAM, Clusters, Interactions)
✅ **FCGR-specific corrections** (normalized, filtered, tangent-projected)
✅ **Two-branch CNN support** (separate miRNA/mRNA attribution)
✅ **Publication-quality visualizations** (300 DPI, multiple formats)
✅ **Batch processing** (analyze multiple samples)
✅ **Statistical analysis** (correlations, clusters, aggregates)
✅ **Modular Python API** (use functions independently)
✅ **Comprehensive logging** (track all operations)
✅ **Export formats** (PNG images + NumPy arrays)

## Citation

If you use this interpretability pipeline, cite the key papers:

- **Integrated Gradients**: Sundararajan et al. (ICML 2017)
- **SHAP**: Lundberg & Lee (NeurIPS 2017)
- **Grad-CAM**: Selvaraju et al. (ICCV 2017)

See `INTERPRETABILITY_README.md` for full citations.

---

**Questions?** Check the logs in `logs/interpretability.log` or review the comprehensive documentation in `INTERPRETABILITY_README.md`.
