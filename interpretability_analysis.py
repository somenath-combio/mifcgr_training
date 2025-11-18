"""
Deep Learning Interpretability Pipeline for FCGR-based miRNA-mRNA Interaction Model

This script implements comprehensive interpretability methods for CNN models trained on
Frequency Chaos Game Representation (FCGR) images, including:
- Gradient-based attribution methods (Integrated Gradients, Saliency, DeepLift, GradientSHAP)
- SHAP DeepExplainer for CNN models
- CNN filter visualization
- Gradient-weighted Class Activation Mapping (Grad-CAM)
- Interaction-level heatmaps
- Statistical analysis utilities

Author: AI-generated interpretability pipeline
Date: 2025
"""

import logging
import sys
from pathlib import Path
import argparse
from typing import Tuple, Dict, List, Optional
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from sklearn.cluster import KMeans

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

# Captum imports for attribution methods
from captum.attr import (
    IntegratedGradients,
    Saliency,
    DeepLift,
    GradientShap,
    NoiseTunnel,
)

# SHAP imports
import shap

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from core.model import InteractionModel
from core.dataset import MiRNAInteractionDataset

warnings.filterwarnings('ignore')

# ============================================================================
# SETUP AND CONFIGURATION
# ============================================================================

def setup_logging(log_file: str = "interpretability.log"):
    """
    Set up logging configuration for interpretability analysis.

    Args:
        log_file: Name of log file
    """
    Path("logs").mkdir(exist_ok=True)
    log_path = Path("logs") / log_file

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )

    logging.info("Interpretability analysis logging initialized")
    logging.info(f"Log file: {log_path}")


def get_device():
    """Get the best available device (GPU if available, else CPU)."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        logging.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        logging.info("Using CPU")

    return device


def load_trained_model(checkpoint_path: str, k: int, dropout_rate: float, device: torch.device):
    """
    Load trained model from checkpoint.

    Args:
        checkpoint_path: Path to model checkpoint
        k: K-mer size used for FCGR generation
        dropout_rate: Dropout rate used during training
        device: Device to load model on

    Returns:
        Loaded model in evaluation mode
    """
    logging.info(f"Loading model from {checkpoint_path}")

    # Initialize model
    model = InteractionModel(dropout_rate=dropout_rate, k=k)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        logging.info(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
        logging.info(f"Best validation loss: {checkpoint.get('best_val_loss', 'unknown')}")
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    logging.info("Model loaded successfully")
    return model


# ============================================================================
# WRAPPER MODEL FOR CAPTUM (handles two-input architecture)
# ============================================================================

class TwoInputWrapper(nn.Module):
    """
    Wrapper for two-input model to work with Captum attribution methods.

    This wrapper handles the InteractionModel's dual-input architecture
    (miRNA FCGR and mRNA FCGR) by allowing attribution for one input while
    keeping the other fixed.
    """

    def __init__(self, model: nn.Module, target_branch: str = 'mirna',
                 fixed_input: Optional[torch.Tensor] = None):
        """
        Initialize wrapper.

        Args:
            model: The InteractionModel to wrap
            target_branch: Which branch to compute attribution for ('mirna' or 'mrna')
            fixed_input: The input to keep fixed (for the non-target branch)
        """
        super().__init__()
        self.model = model
        self.target_branch = target_branch
        self.fixed_input = fixed_input

    def forward(self, x):
        """
        Forward pass through model.

        Args:
            x: Variable input (for the target branch)

        Returns:
            Model output (logits for both classes)
        """
        if self.target_branch == 'mirna':
            # x is miRNA, fixed_input is mRNA
            return self.model(self.fixed_input, x)
        else:
            # x is mRNA, fixed_input is miRNA
            return self.model(x, self.fixed_input)


# ============================================================================
# 1. GRADIENT-BASED ATTRIBUTION METHODS
# ============================================================================

def compute_integrated_gradients(
    model: nn.Module,
    mirna_input: torch.Tensor,
    mrna_input: torch.Tensor,
    target_class: int,
    device: torch.device,
    n_steps: int = 50
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Integrated Gradients attribution for both miRNA and mRNA FCGR images.

    Integrated Gradients computes the integral of gradients along a straight-line
    path from a baseline (typically zeros) to the input. This satisfies important
    axioms like sensitivity and implementation invariance.

    For FCGR images (continuous-valued frequency representations), we use a
    zero baseline which represents absence of all k-mer frequencies.

    Args:
        model: Trained InteractionModel
        mirna_input: miRNA FCGR image tensor [1, 1, 64, 64]
        mrna_input: mRNA FCGR image tensor [1, 1, 64, 64]
        target_class: Class index to compute attribution for
        device: Device to run computation on
        n_steps: Number of integration steps

    Returns:
        Tuple of (mirna_attribution, mrna_attribution) as numpy arrays
    """
    logging.info("Computing Integrated Gradients...")

    # Baseline is zero FCGR (no k-mer frequencies)
    baseline_mirna = torch.zeros_like(mirna_input).to(device)
    baseline_mrna = torch.zeros_like(mrna_input).to(device)

    # Compute IG for miRNA branch
    wrapper_mirna = TwoInputWrapper(model, target_branch='mirna', fixed_input=mrna_input)
    ig_mirna = IntegratedGradients(wrapper_mirna)
    mirna_attr = ig_mirna.attribute(
        mirna_input,
        baselines=baseline_mirna,
        target=target_class,
        n_steps=n_steps
    )

    # Compute IG for mRNA branch
    wrapper_mrna = TwoInputWrapper(model, target_branch='mrna', fixed_input=mirna_input)
    ig_mrna = IntegratedGradients(wrapper_mrna)
    mrna_attr = ig_mrna.attribute(
        mrna_input,
        baselines=baseline_mrna,
        target=target_class,
        n_steps=n_steps
    )

    # Convert to numpy
    mirna_attribution = mirna_attr.squeeze().cpu().detach().numpy()
    mrna_attribution = mrna_attr.squeeze().cpu().detach().numpy()

    logging.info(f"IG - miRNA attribution range: [{mirna_attribution.min():.4f}, {mirna_attribution.max():.4f}]")
    logging.info(f"IG - mRNA attribution range: [{mrna_attribution.min():.4f}, {mrna_attribution.max():.4f}]")

    return mirna_attribution, mrna_attribution


def compute_saliency(
    model: nn.Module,
    mirna_input: torch.Tensor,
    mrna_input: torch.Tensor,
    target_class: int,
    device: torch.device
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Saliency (vanilla gradients) attribution.

    Saliency maps compute the magnitude of gradients of the output with respect
    to the input. High gradient magnitude indicates regions where small changes
    have large impact on predictions.

    Args:
        model: Trained InteractionModel
        mirna_input: miRNA FCGR image tensor
        mrna_input: mRNA FCGR image tensor
        target_class: Class index to compute attribution for
        device: Device to run computation on

    Returns:
        Tuple of (mirna_attribution, mrna_attribution) as numpy arrays
    """
    logging.info("Computing Saliency maps...")

    # Compute Saliency for miRNA branch
    wrapper_mirna = TwoInputWrapper(model, target_branch='mirna', fixed_input=mrna_input)
    saliency_mirna = Saliency(wrapper_mirna)
    mirna_attr = saliency_mirna.attribute(mirna_input, target=target_class, abs=False)

    # Compute Saliency for mRNA branch
    wrapper_mrna = TwoInputWrapper(model, target_branch='mrna', fixed_input=mirna_input)
    saliency_mrna = Saliency(wrapper_mrna)
    mrna_attr = saliency_mrna.attribute(mrna_input, target=target_class, abs=False)

    # Convert to numpy
    mirna_attribution = mirna_attr.squeeze().cpu().detach().numpy()
    mrna_attribution = mrna_attr.squeeze().cpu().detach().numpy()

    logging.info(f"Saliency - miRNA attribution range: [{mirna_attribution.min():.4f}, {mirna_attribution.max():.4f}]")
    logging.info(f"Saliency - mRNA attribution range: [{mrna_attribution.min():.4f}, {mrna_attribution.max():.4f}]")

    return mirna_attribution, mrna_attribution


def compute_deeplift(
    model: nn.Module,
    mirna_input: torch.Tensor,
    mrna_input: torch.Tensor,
    target_class: int,
    device: torch.device
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute DeepLift attribution.

    DeepLift compares the activation of each neuron to its 'reference activation'
    and assigns contribution scores based on the difference. It handles saturation
    better than vanilla gradients.

    Args:
        model: Trained InteractionModel
        mirna_input: miRNA FCGR image tensor
        mrna_input: mRNA FCGR image tensor
        target_class: Class index to compute attribution for
        device: Device to run computation on

    Returns:
        Tuple of (mirna_attribution, mrna_attribution) as numpy arrays
    """
    logging.info("Computing DeepLift...")

    # Baseline is zero FCGR
    baseline_mirna = torch.zeros_like(mirna_input).to(device)
    baseline_mrna = torch.zeros_like(mrna_input).to(device)

    # Compute DeepLift for miRNA branch
    wrapper_mirna = TwoInputWrapper(model, target_branch='mirna', fixed_input=mrna_input)
    dl_mirna = DeepLift(wrapper_mirna)
    mirna_attr = dl_mirna.attribute(
        mirna_input,
        baselines=baseline_mirna,
        target=target_class
    )

    # Compute DeepLift for mRNA branch
    wrapper_mrna = TwoInputWrapper(model, target_branch='mrna', fixed_input=mirna_input)
    dl_mrna = DeepLift(wrapper_mrna)
    mrna_attr = dl_mrna.attribute(
        mrna_input,
        baselines=baseline_mrna,
        target=target_class
    )

    # Convert to numpy
    mirna_attribution = mirna_attr.squeeze().cpu().detach().numpy()
    mrna_attribution = mrna_attr.squeeze().cpu().detach().numpy()

    logging.info(f"DeepLift - miRNA attribution range: [{mirna_attribution.min():.4f}, {mirna_attribution.max():.4f}]")
    logging.info(f"DeepLift - mRNA attribution range: [{mrna_attribution.min():.4f}, {mrna_attribution.max():.4f}]")

    return mirna_attribution, mrna_attribution


def compute_gradient_shap(
    model: nn.Module,
    mirna_input: torch.Tensor,
    mrna_input: torch.Tensor,
    target_class: int,
    device: torch.device,
    n_samples: int = 50,
    stdevs: float = 0.05
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute GradientSHAP attribution.

    GradientSHAP is a method that combines Integrated Gradients with SHAP values.
    It uses random baselines sampled from a distribution and averages the
    Integrated Gradients computed for each baseline.

    For FCGR images, we use Gaussian noise around zero baseline to represent
    uncertainty in the absence of k-mer patterns.

    Args:
        model: Trained InteractionModel
        mirna_input: miRNA FCGR image tensor
        mrna_input: mRNA FCGR image tensor
        target_class: Class index to compute attribution for
        device: Device to run computation on
        n_samples: Number of baseline samples
        stdevs: Standard deviation for baseline sampling

    Returns:
        Tuple of (mirna_attribution, mrna_attribution) as numpy arrays
    """
    logging.info("Computing GradientSHAP...")

    # Create baseline distributions (Gaussian noise around zero)
    baseline_mirna = torch.randn(n_samples, *mirna_input.shape[1:]).to(device) * stdevs
    baseline_mrna = torch.randn(n_samples, *mrna_input.shape[1:]).to(device) * stdevs

    # Compute GradientSHAP for miRNA branch
    wrapper_mirna = TwoInputWrapper(model, target_branch='mirna', fixed_input=mrna_input)
    gs_mirna = GradientShap(wrapper_mirna)
    mirna_attr = gs_mirna.attribute(
        mirna_input,
        baselines=baseline_mirna,
        target=target_class,
        n_samples=n_samples
    )

    # Compute GradientSHAP for mRNA branch
    wrapper_mrna = TwoInputWrapper(model, target_branch='mrna', fixed_input=mirna_input)
    gs_mrna = GradientShap(wrapper_mrna)
    mrna_attr = gs_mrna.attribute(
        mrna_input,
        baselines=baseline_mrna,
        target=target_class,
        n_samples=n_samples
    )

    # Convert to numpy
    mirna_attribution = mirna_attr.squeeze().cpu().detach().numpy()
    mrna_attribution = mrna_attr.squeeze().cpu().detach().numpy()

    logging.info(f"GradientSHAP - miRNA attribution range: [{mirna_attribution.min():.4f}, {mirna_attribution.max():.4f}]")
    logging.info(f"GradientSHAP - mRNA attribution range: [{mrna_attribution.min():.4f}, {mrna_attribution.max():.4f}]")

    return mirna_attribution, mrna_attribution


def compute_smoothgrad(
    model: nn.Module,
    mirna_input: torch.Tensor,
    mrna_input: torch.Tensor,
    target_class: int,
    device: torch.device,
    n_samples: int = 50,
    stdevs: float = 0.05
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute SmoothGrad attribution (noise tunnel around saliency).

    SmoothGrad adds Gaussian noise to the input multiple times and averages
    the resulting gradients. This reduces noise and provides smoother, more
    interpretable attribution maps.

    Args:
        model: Trained InteractionModel
        mirna_input: miRNA FCGR image tensor
        mrna_input: mRNA FCGR image tensor
        target_class: Class index to compute attribution for
        device: Device to run computation on
        n_samples: Number of noisy samples
        stdevs: Standard deviation of noise

    Returns:
        Tuple of (mirna_attribution, mrna_attribution) as numpy arrays
    """
    logging.info("Computing SmoothGrad...")

    # Compute SmoothGrad for miRNA branch
    wrapper_mirna = TwoInputWrapper(model, target_branch='mirna', fixed_input=mrna_input)
    saliency_mirna = Saliency(wrapper_mirna)
    nt_mirna = NoiseTunnel(saliency_mirna)
    mirna_attr = nt_mirna.attribute(
        mirna_input,
        target=target_class,
        nt_type='smoothgrad',
        nt_samples=n_samples,
        stdevs=stdevs
    )

    # Compute SmoothGrad for mRNA branch
    wrapper_mrna = TwoInputWrapper(model, target_branch='mrna', fixed_input=mirna_input)
    saliency_mrna = Saliency(wrapper_mrna)
    nt_mrna = NoiseTunnel(saliency_mrna)
    mrna_attr = nt_mrna.attribute(
        mrna_input,
        target=target_class,
        nt_type='smoothgrad',
        nt_samples=n_samples,
        stdevs=stdevs
    )

    # Convert to numpy
    mirna_attribution = mirna_attr.squeeze().cpu().detach().numpy()
    mrna_attribution = mrna_attr.squeeze().cpu().detach().numpy()

    logging.info(f"SmoothGrad - miRNA attribution range: [{mirna_attribution.min():.4f}, {mrna_attribution.max():.4f}]")

    return mirna_attribution, mrna_attribution


# ============================================================================
# 2. GRADIENT CORRECTION FOR FCGR
# ============================================================================

def apply_gradient_correction(
    attribution: np.ndarray,
    fcgr_image: np.ndarray,
    correction_type: str = 'normalize'
) -> np.ndarray:
    """
    Apply gradient correction adapted for FCGR continuous-valued representations.

    Unlike one-hot encoded sequences, FCGR images contain continuous frequency values
    that represent k-mer patterns. This function implements correction strategies
    to handle the continuous nature of FCGR:

    1. 'normalize': Normalize attributions by FCGR values (prevents bias toward
       high-frequency regions)
    2. 'filter': Apply median filtering to reduce gradient noise
    3. 'tangent': Project gradients onto FCGR manifold tangent space

    Args:
        attribution: Raw attribution map (H, W)
        fcgr_image: Original FCGR image (H, W)
        correction_type: Type of correction ('normalize', 'filter', 'tangent')

    Returns:
        Corrected attribution map
    """
    if correction_type == 'normalize':
        # Normalize by FCGR magnitude to reduce bias toward high-frequency regions
        # Add small epsilon to avoid division by zero
        epsilon = 1e-8
        corrected = attribution / (np.abs(fcgr_image) + epsilon)

    elif correction_type == 'filter':
        # Apply median filter to reduce noise
        from scipy.ndimage import median_filter
        corrected = median_filter(attribution, size=3)

    elif correction_type == 'tangent':
        # Tangent space projection for continuous-valued FCGR
        # This removes components orthogonal to the FCGR manifold

        # Compute local gradient of FCGR manifold
        from scipy.ndimage import sobel
        grad_x = sobel(fcgr_image, axis=0)
        grad_y = sobel(fcgr_image, axis=1)

        # Project attribution onto tangent space
        grad_mag = np.sqrt(grad_x**2 + grad_y**2) + 1e-8
        projection = (attribution * grad_x * grad_x + attribution * grad_y * grad_y) / (grad_mag**2 + 1e-8)
        corrected = projection

    else:
        logging.warning(f"Unknown correction type: {correction_type}. Returning original attribution.")
        corrected = attribution

    return corrected


# ============================================================================
# 3. SHAP DEEPEXPLAINER FOR CNN
# ============================================================================

def compute_shap_values(
    model: nn.Module,
    mirna_input: torch.Tensor,
    mrna_input: torch.Tensor,
    background_data: List[Tuple[torch.Tensor, torch.Tensor]],
    device: torch.device
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute SHAP values using DeepExplainer for CNN models.

    SHAP (SHapley Additive exPlanations) provides a unified measure of feature
    importance based on cooperative game theory. DeepExplainer is an efficient
    approximation for deep learning models.

    Args:
        model: Trained InteractionModel
        mirna_input: miRNA FCGR image tensor [1, 1, 64, 64]
        mrna_input: mRNA FCGR image tensor [1, 1, 64, 64]
        background_data: List of (mirna, mrna) tuples for background distribution
        device: Device to run computation on

    Returns:
        Tuple of (mirna_shap_values, mrna_shap_values) as numpy arrays
    """
    logging.info("Computing SHAP values with DeepExplainer...")

    # Prepare background data
    background_mirna = torch.stack([item[0] for item in background_data]).to(device)
    background_mrna = torch.stack([item[1] for item in background_data]).to(device)

    logging.info(f"Background dataset size: {len(background_data)} samples")

    # Create SHAP explainer for miRNA branch
    wrapper_mirna = TwoInputWrapper(model, target_branch='mirna', fixed_input=mrna_input)
    explainer_mirna = shap.DeepExplainer(wrapper_mirna, background_mirna)
    shap_values_mirna = explainer_mirna.shap_values(mirna_input)

    # Create SHAP explainer for mRNA branch
    wrapper_mrna = TwoInputWrapper(model, target_branch='mrna', fixed_input=mirna_input)
    explainer_mrna = shap.DeepExplainer(wrapper_mrna, background_mrna)
    shap_values_mrna = explainer_mrna.shap_values(mrna_input)

    # Extract SHAP values for positive class (index 1)
    # shap_values is a list with one array per output class
    mirna_shap = shap_values_mirna[1].squeeze() if isinstance(shap_values_mirna, list) else shap_values_mirna.squeeze()
    mrna_shap = shap_values_mrna[1].squeeze() if isinstance(shap_values_mrna, list) else shap_values_mrna.squeeze()

    logging.info(f"SHAP - miRNA values range: [{mirna_shap.min():.4f}, {mirna_shap.max():.4f}]")
    logging.info(f"SHAP - mRNA values range: [{mrna_shap.min():.4f}, {mrna_shap.max():.4f}]")

    return mirna_shap, mrna_shap


def compare_shap_ig_correlation(
    shap_mirna: np.ndarray,
    shap_mrna: np.ndarray,
    ig_mirna: np.ndarray,
    ig_mrna: np.ndarray
) -> Dict[str, float]:
    """
    Compare SHAP and Integrated Gradients using correlation metrics.

    Args:
        shap_mirna: SHAP values for miRNA
        shap_mrna: SHAP values for mRNA
        ig_mirna: IG attribution for miRNA
        ig_mrna: IG attribution for mRNA

    Returns:
        Dictionary with correlation statistics
    """
    # Flatten arrays
    shap_mirna_flat = shap_mirna.flatten()
    shap_mrna_flat = shap_mrna.flatten()
    ig_mirna_flat = ig_mirna.flatten()
    ig_mrna_flat = ig_mrna.flatten()

    # Compute correlations
    pearson_mirna, _ = pearsonr(shap_mirna_flat, ig_mirna_flat)
    pearson_mrna, _ = pearsonr(shap_mrna_flat, ig_mrna_flat)
    spearman_mirna, _ = spearmanr(shap_mirna_flat, ig_mirna_flat)
    spearman_mrna, _ = spearmanr(shap_mrna_flat, ig_mrna_flat)

    results = {
        'pearson_mirna': pearson_mirna,
        'pearson_mrna': pearson_mrna,
        'spearman_mirna': spearman_mirna,
        'spearman_mrna': spearman_mrna
    }

    logging.info("SHAP vs IG Correlation:")
    logging.info(f"  miRNA - Pearson: {pearson_mirna:.4f}, Spearman: {spearman_mirna:.4f}")
    logging.info(f"  mRNA - Pearson: {pearson_mrna:.4f}, Spearman: {spearman_mrna:.4f}")

    return results


# ============================================================================
# 4. CNN FILTER VISUALIZATION
# ============================================================================

def visualize_conv_filters(model: nn.Module, save_dir: Path):
    """
    Extract and visualize first convolutional layer filters.

    Args:
        model: Trained InteractionModel
        save_dir: Directory to save visualizations
    """
    logging.info("Visualizing CNN filters...")

    # Get first conv layer from both branches
    mirna_conv1 = model.mi_rna_model.conv1[0]  # First layer of Sequential
    mrna_conv1 = model.m_rna_model.conv1[0]

    # Get weights [out_channels, in_channels, height, width]
    mirna_filters = mirna_conv1.weight.data.cpu().numpy()
    mrna_filters = mrna_conv1.weight.data.cpu().numpy()

    logging.info(f"miRNA filters shape: {mirna_filters.shape}")
    logging.info(f"mRNA filters shape: {mrna_filters.shape}")

    # Visualize first 16 filters for each branch
    n_filters = min(16, mirna_filters.shape[0])

    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    fig.suptitle('miRNA Branch - First Conv Layer Filters', fontsize=16)

    for i, ax in enumerate(axes.flat):
        if i < n_filters:
            filter_img = mirna_filters[i, 0, :, :]  # [in_channel=0, :, :]
            im = ax.imshow(filter_img, cmap='viridis', aspect='auto')
            ax.set_title(f'Filter {i}')
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046)

    plt.tight_layout()
    plt.savefig(save_dir / 'mirna_conv1_filters.png', dpi=300, bbox_inches='tight')
    plt.close()

    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    fig.suptitle('mRNA Branch - First Conv Layer Filters', fontsize=16)

    for i, ax in enumerate(axes.flat):
        if i < n_filters:
            filter_img = mrna_filters[i, 0, :, :]
            im = ax.imshow(filter_img, cmap='viridis', aspect='auto')
            ax.set_title(f'Filter {i}')
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046)

    plt.tight_layout()
    plt.savefig(save_dir / 'mrna_conv1_filters.png', dpi=300, bbox_inches='tight')
    plt.close()

    logging.info(f"Filter visualizations saved to {save_dir}")


def compute_filter_activations(
    model: nn.Module,
    mirna_input: torch.Tensor,
    mrna_input: torch.Tensor,
    device: torch.device
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute activations for each filter in the first conv layer.

    Args:
        model: Trained InteractionModel
        mirna_input: miRNA FCGR image
        mrna_input: mRNA FCGR image
        device: Device to run computation on

    Returns:
        Tuple of (mirna_activations, mrna_activations)
    """
    # Hook to capture activations
    activations = {}

    def get_activation(name):
        def hook(model, input, output):
            activations[name] = output.detach()
        return hook

    # Register hooks
    handle_mirna = model.mi_rna_model.conv1.register_forward_hook(get_activation('mirna_conv1'))
    handle_mrna = model.m_rna_model.conv1.register_forward_hook(get_activation('mrna_conv1'))

    # Forward pass
    with torch.no_grad():
        _ = model(mrna_input, mirna_input)

    # Remove hooks
    handle_mirna.remove()
    handle_mrna.remove()

    # Get activations
    mirna_act = activations['mirna_conv1'].cpu().numpy()
    mrna_act = activations['mrna_conv1'].cpu().numpy()

    return mirna_act, mrna_act


def visualize_filter_activations(
    mirna_activations: np.ndarray,
    mrna_activations: np.ndarray,
    save_dir: Path
):
    """
    Visualize spatial activation patterns for each filter.

    Args:
        mirna_activations: Activations from miRNA branch [1, n_filters, H, W]
        mrna_activations: Activations from mRNA branch [1, n_filters, H, W]
        save_dir: Directory to save visualizations
    """
    n_filters = min(16, mirna_activations.shape[1])

    # miRNA activations
    fig, axes = plt.subplots(4, 4, figsize=(14, 14))
    fig.suptitle('miRNA - First Conv Layer Activations', fontsize=16)

    for i, ax in enumerate(axes.flat):
        if i < n_filters:
            act_map = mirna_activations[0, i, :, :]
            im = ax.imshow(act_map, cmap='hot', aspect='auto')
            ax.set_title(f'Filter {i}\nMax: {act_map.max():.2f}')
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046)

    plt.tight_layout()
    plt.savefig(save_dir / 'mirna_conv1_activations.png', dpi=300, bbox_inches='tight')
    plt.close()

    # mRNA activations
    fig, axes = plt.subplots(4, 4, figsize=(14, 14))
    fig.suptitle('mRNA - First Conv Layer Activations', fontsize=16)

    for i, ax in enumerate(axes.flat):
        if i < n_filters:
            act_map = mrna_activations[0, i, :, :]
            im = ax.imshow(act_map, cmap='hot', aspect='auto')
            ax.set_title(f'Filter {i}\nMax: {act_map.max():.2f}')
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046)

    plt.tight_layout()
    plt.savefig(save_dir / 'mrna_conv1_activations.png', dpi=300, bbox_inches='tight')
    plt.close()

    logging.info(f"Activation maps saved to {save_dir}")


# ============================================================================
# 5. GRAD-CAM (Gradient-weighted Class Activation Mapping)
# ============================================================================

class GradCAM:
    """
    Gradient-weighted Class Activation Mapping for CNN interpretability.

    Grad-CAM produces a coarse localization map highlighting important regions
    for prediction by using gradients flowing into the final convolutional layer.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        """
        Initialize Grad-CAM.

        Args:
            model: The model to explain
            target_layer: The convolutional layer to compute CAM from
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Register hooks
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        """Hook to save forward pass activations."""
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        """Hook to save backward pass gradients."""
        self.gradients = grad_output[0].detach()

    def generate_cam(self, input_tensor: torch.Tensor, target_class: int) -> np.ndarray:
        """
        Generate Grad-CAM heatmap.

        Args:
            input_tensor: Input to the model
            target_class: Target class index

        Returns:
            CAM heatmap as numpy array
        """
        # Forward pass
        output = self.model(input_tensor)

        # Zero gradients
        self.model.zero_grad()

        # Backward pass
        class_score = output[0, target_class]
        class_score.backward()

        # Get gradients and activations
        gradients = self.gradients.cpu().numpy()[0]  # [C, H, W]
        activations = self.activations.cpu().numpy()[0]  # [C, H, W]

        # Compute weights (global average pooling of gradients)
        weights = np.mean(gradients, axis=(1, 2))  # [C]

        # Compute weighted combination of activation maps
        cam = np.zeros(activations.shape[1:], dtype=np.float32)  # [H, W]
        for i, w in enumerate(weights):
            cam += w * activations[i]

        # Apply ReLU (only positive contributions)
        cam = np.maximum(cam, 0)

        # Normalize to [0, 1]
        if cam.max() > 0:
            cam = cam / cam.max()

        return cam


def compute_gradcam(
    model: nn.Module,
    mirna_input: torch.Tensor,
    mrna_input: torch.Tensor,
    target_class: int,
    device: torch.device
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Grad-CAM for both miRNA and mRNA branches.

    Args:
        model: Trained InteractionModel
        mirna_input: miRNA FCGR image
        mrna_input: mRNA FCGR image
        target_class: Target class index
        device: Device to run computation on

    Returns:
        Tuple of (mirna_cam, mrna_cam) as numpy arrays
    """
    logging.info("Computing Grad-CAM...")

    # Get last conv layer for each branch (conv3)
    mirna_target_layer = model.mi_rna_model.conv3[0]  # Conv2d layer
    mrna_target_layer = model.m_rna_model.conv3[0]

    # Create wrapped models for each branch
    wrapper_mirna = TwoInputWrapper(model, target_branch='mirna', fixed_input=mrna_input)
    wrapper_mrna = TwoInputWrapper(model, target_branch='mrna', fixed_input=mirna_input)

    # Compute Grad-CAM for miRNA
    gradcam_mirna = GradCAM(wrapper_mirna, mirna_target_layer)
    mirna_input_grad = mirna_input.clone().requires_grad_(True)
    mirna_cam = gradcam_mirna.generate_cam(mirna_input_grad, target_class)

    # Compute Grad-CAM for mRNA
    gradcam_mrna = GradCAM(wrapper_mrna, mrna_target_layer)
    mrna_input_grad = mrna_input.clone().requires_grad_(True)
    mrna_cam = gradcam_mrna.generate_cam(mrna_input_grad, target_class)

    logging.info(f"Grad-CAM - miRNA CAM range: [{mirna_cam.min():.4f}, {mirna_cam.max():.4f}]")
    logging.info(f"Grad-CAM - mRNA CAM range: [{mrna_cam.min():.4f}, {mrna_cam.max():.4f}]")

    return mirna_cam, mrna_cam


def overlay_gradcam(
    fcgr_image: np.ndarray,
    cam: np.ndarray,
    alpha: float = 0.5
) -> np.ndarray:
    """
    Overlay Grad-CAM heatmap on FCGR image.

    Args:
        fcgr_image: Original FCGR image [H, W]
        cam: Grad-CAM heatmap [H_cam, W_cam]
        alpha: Transparency for overlay

    Returns:
        Overlayed image as RGB array
    """
    from skimage.transform import resize

    # Resize CAM to match FCGR size
    cam_resized = resize(cam, fcgr_image.shape, mode='reflect', anti_aliasing=True)

    # Normalize FCGR to [0, 1]
    fcgr_norm = (fcgr_image - fcgr_image.min()) / (fcgr_image.max() - fcgr_image.min() + 1e-8)

    # Convert to RGB
    fcgr_rgb = plt.cm.gray(fcgr_norm)[:, :, :3]
    cam_rgb = plt.cm.jet(cam_resized)[:, :, :3]

    # Blend
    overlayed = alpha * cam_rgb + (1 - alpha) * fcgr_rgb

    return overlayed


# ============================================================================
# 6. INTERACTION-LEVEL HEATMAP
# ============================================================================

def create_interaction_heatmap(
    mirna_attribution: np.ndarray,
    mrna_attribution: np.ndarray,
    method: str = 'outer_product'
) -> np.ndarray:
    """
    Create interaction-level heatmap combining miRNA and mRNA attributions.

    This produces a 2D heatmap showing co-importance between miRNA and mRNA
    regions, similar to an attention or alignment matrix.

    Args:
        mirna_attribution: Attribution map for miRNA [64, 64]
        mrna_attribution: Attribution map for mRNA [64, 64]
        method: Method for combining attributions ('outer_product', 'correlation')

    Returns:
        Interaction heatmap [64, 64] (or flattened size depending on method)
    """
    if method == 'outer_product':
        # Flatten attributions and compute outer product
        mirna_flat = mirna_attribution.flatten()
        mrna_flat = mrna_attribution.flatten()

        # Compute outer product (represents pairwise interactions)
        interaction_matrix = np.outer(mirna_flat, mrna_flat)

        # Aggregate back to 64x64 by averaging
        # For simplicity, we can reshape or use the raw outer product
        # Here we'll average the outer product in blocks
        n = 64
        block_size = len(mirna_flat) // n
        aggregated = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                block_i = slice(i * block_size, (i + 1) * block_size)
                block_j = slice(j * block_size, (j + 1) * block_size)
                aggregated[i, j] = interaction_matrix[block_i, block_j].mean()

        return aggregated

    elif method == 'correlation':
        # Compute local correlation between attribution patterns
        from scipy.signal import correlate2d

        # Compute 2D correlation
        correlation = correlate2d(mirna_attribution, mrna_attribution, mode='same')
        return correlation

    else:
        # Simple element-wise product
        return mirna_attribution * mrna_attribution


# ============================================================================
# 7. ANALYSIS UTILITIES
# ============================================================================

def compute_attribution_statistics(
    attributions: Dict[str, np.ndarray],
    label: str
) -> pd.DataFrame:
    """
    Compute statistics for attribution maps.

    Args:
        attributions: Dictionary mapping method names to attribution arrays
        label: Label for the attribution set ('miRNA' or 'mRNA')

    Returns:
        DataFrame with statistics
    """
    stats = []

    for method, attr in attributions.items():
        stats.append({
            'Method': method,
            'Branch': label,
            'Mean': np.mean(attr),
            'Std': np.std(attr),
            'Min': np.min(attr),
            'Max': np.max(attr),
            'AbsMean': np.mean(np.abs(attr)),
            'L1_Norm': np.sum(np.abs(attr)),
            'L2_Norm': np.sqrt(np.sum(attr ** 2))
        })

    return pd.DataFrame(stats)


def cluster_attribution_regions(
    attribution: np.ndarray,
    n_clusters: int = 5,
    threshold_percentile: float = 75
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Cluster important regions in attribution map using k-means.

    Args:
        attribution: Attribution map [H, W]
        n_clusters: Number of clusters
        threshold_percentile: Percentile threshold for important pixels

    Returns:
        Tuple of (cluster_labels, cluster_centers)
    """
    # Get important pixels (above threshold)
    threshold = np.percentile(np.abs(attribution), threshold_percentile)
    important_mask = np.abs(attribution) >= threshold

    # Get coordinates of important pixels
    coords = np.argwhere(important_mask)

    if len(coords) < n_clusters:
        logging.warning(f"Not enough important pixels ({len(coords)}) for {n_clusters} clusters")
        return np.zeros_like(attribution), np.array([])

    # Perform k-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(coords)
    centers = kmeans.cluster_centers_

    # Create cluster map
    cluster_map = np.zeros_like(attribution, dtype=int)
    for coord, label in zip(coords, labels):
        cluster_map[coord[0], coord[1]] = label + 1  # +1 to distinguish from background (0)

    logging.info(f"Clustered {len(coords)} important pixels into {n_clusters} regions")

    return cluster_map, centers


def export_attributions(
    attributions: Dict[str, Tuple[np.ndarray, np.ndarray]],
    sample_idx: int,
    save_dir: Path
):
    """
    Export attribution maps as .npy files.

    Args:
        attributions: Dictionary mapping method names to (mirna_attr, mrna_attr) tuples
        sample_idx: Sample index
        save_dir: Directory to save files
    """
    for method, (mirna_attr, mrna_attr) in attributions.items():
        np.save(save_dir / f'sample_{sample_idx}_{method}_mirna.npy', mirna_attr)
        np.save(save_dir / f'sample_{sample_idx}_{method}_mrna.npy', mrna_attr)

    logging.info(f"Exported attributions to {save_dir}")


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def visualize_attribution_heatmap(
    fcgr_image: np.ndarray,
    attribution: np.ndarray,
    title: str,
    save_path: Path,
    cmap: str = 'RdBu_r',
    show_colorbar: bool = True
):
    """
    Visualize attribution as heatmap with FCGR overlay.

    Args:
        fcgr_image: Original FCGR image
        attribution: Attribution map
        title: Plot title
        save_path: Path to save figure
        cmap: Colormap for attribution
        show_colorbar: Whether to show colorbar
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Original FCGR
    im0 = axes[0].imshow(fcgr_image, cmap='gray', aspect='auto')
    axes[0].set_title('Original FCGR')
    axes[0].axis('off')
    plt.colorbar(im0, ax=axes[0], fraction=0.046)

    # Attribution heatmap
    # Center colormap at zero for diverging attribution values
    vmax = np.max(np.abs(attribution))
    im1 = axes[1].imshow(attribution, cmap=cmap, aspect='auto', vmin=-vmax, vmax=vmax)
    axes[1].set_title('Attribution Map')
    axes[1].axis('off')
    if show_colorbar:
        plt.colorbar(im1, ax=axes[1], fraction=0.046)

    # Overlay
    alpha = 0.5
    fcgr_norm = (fcgr_image - fcgr_image.min()) / (fcgr_image.max() - fcgr_image.min() + 1e-8)
    attr_norm = (attribution - attribution.min()) / (attribution.max() - attribution.min() + 1e-8)

    fcgr_rgb = plt.cm.gray(fcgr_norm)[:, :, :3]
    attr_rgb = plt.cm.get_cmap(cmap)(attr_norm)[:, :, :3]
    overlay = alpha * attr_rgb + (1 - alpha) * fcgr_rgb

    axes[2].imshow(overlay)
    axes[2].set_title('Attribution Overlay')
    axes[2].axis('off')

    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def visualize_all_methods_comparison(
    fcgr_mirna: np.ndarray,
    fcgr_mrna: np.ndarray,
    attributions: Dict[str, Tuple[np.ndarray, np.ndarray]],
    sample_idx: int,
    save_dir: Path
):
    """
    Create comprehensive comparison visualization of all attribution methods.

    Args:
        fcgr_mirna: miRNA FCGR image
        fcgr_mrna: mRNA FCGR image
        attributions: Dictionary of attribution results
        sample_idx: Sample index
        save_dir: Directory to save visualizations
    """
    n_methods = len(attributions)

    # miRNA comparison
    fig, axes = plt.subplots(2, n_methods, figsize=(5 * n_methods, 10))

    for idx, (method, (mirna_attr, _)) in enumerate(attributions.items()):
        # Original FCGR
        axes[0, idx].imshow(fcgr_mirna, cmap='gray', aspect='auto')
        axes[0, idx].set_title(f'{method}\nOriginal FCGR')
        axes[0, idx].axis('off')

        # Attribution
        vmax = np.max(np.abs(mirna_attr))
        im = axes[1, idx].imshow(mirna_attr, cmap='RdBu_r', aspect='auto', vmin=-vmax, vmax=vmax)
        axes[1, idx].set_title(f'{method}\nAttribution')
        axes[1, idx].axis('off')
        plt.colorbar(im, ax=axes[1, idx], fraction=0.046)

    fig.suptitle(f'miRNA Attribution Methods Comparison - Sample {sample_idx}',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_dir / f'sample_{sample_idx}_mirna_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    # mRNA comparison
    fig, axes = plt.subplots(2, n_methods, figsize=(5 * n_methods, 10))

    for idx, (method, (_, mrna_attr)) in enumerate(attributions.items()):
        # Original FCGR
        axes[0, idx].imshow(fcgr_mrna, cmap='gray', aspect='auto')
        axes[0, idx].set_title(f'{method}\nOriginal FCGR')
        axes[0, idx].axis('off')

        # Attribution
        vmax = np.max(np.abs(mrna_attr))
        im = axes[1, idx].imshow(mrna_attr, cmap='RdBu_r', aspect='auto', vmin=-vmax, vmax=vmax)
        axes[1, idx].set_title(f'{method}\nAttribution')
        axes[1, idx].axis('off')
        plt.colorbar(im, ax=axes[1, idx], fraction=0.046)

    fig.suptitle(f'mRNA Attribution Methods Comparison - Sample {sample_idx}',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_dir / f'sample_{sample_idx}_mrna_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()


def visualize_gradcam_overlay(
    fcgr_mirna: np.ndarray,
    fcgr_mrna: np.ndarray,
    mirna_cam: np.ndarray,
    mrna_cam: np.ndarray,
    sample_idx: int,
    save_dir: Path
):
    """
    Visualize Grad-CAM overlays.

    Args:
        fcgr_mirna: miRNA FCGR image
        fcgr_mrna: mRNA FCGR image
        mirna_cam: miRNA Grad-CAM heatmap
        mrna_cam: mRNA Grad-CAM heatmap
        sample_idx: Sample index
        save_dir: Directory to save visualization
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # miRNA row
    axes[0, 0].imshow(fcgr_mirna, cmap='gray')
    axes[0, 0].set_title('miRNA FCGR')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(mirna_cam, cmap='jet')
    axes[0, 1].set_title('miRNA Grad-CAM')
    axes[0, 1].axis('off')

    overlay_mirna = overlay_gradcam(fcgr_mirna, mirna_cam)
    axes[0, 2].imshow(overlay_mirna)
    axes[0, 2].set_title('miRNA Overlay')
    axes[0, 2].axis('off')

    # mRNA row
    axes[1, 0].imshow(fcgr_mrna, cmap='gray')
    axes[1, 0].set_title('mRNA FCGR')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(mrna_cam, cmap='jet')
    axes[1, 1].set_title('mRNA Grad-CAM')
    axes[1, 1].axis('off')

    overlay_mrna = overlay_gradcam(fcgr_mrna, mrna_cam)
    axes[1, 2].imshow(overlay_mrna)
    axes[1, 2].set_title('mRNA Overlay')
    axes[1, 2].axis('off')

    fig.suptitle(f'Grad-CAM Visualization - Sample {sample_idx}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_dir / f'sample_{sample_idx}_gradcam.png', dpi=300, bbox_inches='tight')
    plt.close()


def visualize_interaction_heatmap(
    interaction_matrix: np.ndarray,
    sample_idx: int,
    save_dir: Path
):
    """
    Visualize interaction-level heatmap.

    Args:
        interaction_matrix: Interaction heatmap [64, 64]
        sample_idx: Sample index
        save_dir: Directory to save visualization
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    im = ax.imshow(interaction_matrix, cmap='YlOrRd', aspect='auto')
    ax.set_xlabel('mRNA Position (aggregated)', fontsize=12)
    ax.set_ylabel('miRNA Position (aggregated)', fontsize=12)
    ax.set_title(f'miRNA-mRNA Interaction Heatmap - Sample {sample_idx}',
                 fontsize=14, fontweight='bold')

    plt.colorbar(im, ax=ax, label='Co-importance Score')
    plt.tight_layout()
    plt.savefig(save_dir / f'sample_{sample_idx}_interaction_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()


def visualize_cluster_analysis(
    attribution: np.ndarray,
    cluster_map: np.ndarray,
    centers: np.ndarray,
    title: str,
    save_path: Path
):
    """
    Visualize clustered attribution regions.

    Args:
        attribution: Original attribution map
        cluster_map: Cluster assignment map
        centers: Cluster centers
        title: Plot title
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Attribution map
    vmax = np.max(np.abs(attribution))
    axes[0].imshow(attribution, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    axes[0].set_title('Attribution Map')
    axes[0].axis('off')

    # Cluster map
    axes[1].imshow(cluster_map, cmap='tab10')
    axes[1].scatter(centers[:, 1], centers[:, 0], c='red', marker='x', s=200, linewidths=3)
    axes[1].set_title(f'Clustered Regions (k={len(centers)})')
    axes[1].axis('off')

    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


# ============================================================================
# MAIN ANALYSIS PIPELINE
# ============================================================================

def analyze_sample(
    model: nn.Module,
    mirna_input: torch.Tensor,
    mrna_input: torch.Tensor,
    label: int,
    sample_idx: int,
    background_data: List[Tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    save_dir: Path
) -> Dict:
    """
    Run complete interpretability analysis on a single sample.

    Args:
        model: Trained model
        mirna_input: miRNA FCGR tensor
        mrna_input: mRNA FCGR tensor
        label: Ground truth label
        sample_idx: Sample index
        background_data: Background dataset for SHAP
        device: Device to run on
        save_dir: Directory to save results

    Returns:
        Dictionary with all analysis results
    """
    logging.info(f"\n{'='*60}")
    logging.info(f"Analyzing Sample {sample_idx}")
    logging.info(f"{'='*60}")

    # Get model prediction
    with torch.no_grad():
        output = model(mrna_input, mirna_input)
        probabilities = F.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0, predicted_class].item()

    logging.info(f"True label: {label}, Predicted: {predicted_class}, Confidence: {confidence:.4f}")

    # Convert FCGR to numpy for visualization
    fcgr_mirna = mirna_input.squeeze().cpu().numpy()
    fcgr_mrna = mrna_input.squeeze().cpu().numpy()

    # Target class for attribution (use predicted class)
    target_class = predicted_class

    results = {
        'sample_idx': sample_idx,
        'true_label': label,
        'predicted_label': predicted_class,
        'confidence': confidence,
        'attributions': {}
    }

    # 1. Integrated Gradients
    logging.info("\n--- Computing Integrated Gradients ---")
    ig_mirna, ig_mrna = compute_integrated_gradients(
        model, mirna_input, mrna_input, target_class, device
    )
    results['attributions']['IntegratedGradients'] = (ig_mirna, ig_mrna)

    # 2. Saliency
    logging.info("\n--- Computing Saliency ---")
    sal_mirna, sal_mrna = compute_saliency(
        model, mirna_input, mrna_input, target_class, device
    )
    results['attributions']['Saliency'] = (sal_mirna, sal_mrna)

    # 3. DeepLift
    logging.info("\n--- Computing DeepLift ---")
    dl_mirna, dl_mrna = compute_deeplift(
        model, mirna_input, mrna_input, target_class, device
    )
    results['attributions']['DeepLift'] = (dl_mirna, dl_mrna)

    # 4. GradientSHAP
    logging.info("\n--- Computing GradientSHAP ---")
    gs_mirna, gs_mrna = compute_gradient_shap(
        model, mirna_input, mrna_input, target_class, device
    )
    results['attributions']['GradientSHAP'] = (gs_mirna, gs_mrna)

    # 5. SmoothGrad
    logging.info("\n--- Computing SmoothGrad ---")
    sg_mirna, sg_mrna = compute_smoothgrad(
        model, mirna_input, mrna_input, target_class, device
    )
    results['attributions']['SmoothGrad'] = (sg_mirna, sg_mrna)

    # 6. SHAP DeepExplainer
    logging.info("\n--- Computing SHAP values ---")
    shap_mirna, shap_mrna = compute_shap_values(
        model, mirna_input, mrna_input, background_data, device
    )
    results['attributions']['SHAP'] = (shap_mirna, shap_mrna)

    # 7. Compare SHAP vs IG
    logging.info("\n--- Comparing SHAP vs IG ---")
    correlation_stats = compare_shap_ig_correlation(
        shap_mirna, shap_mrna, ig_mirna, ig_mrna
    )
    results['shap_ig_correlation'] = correlation_stats

    # 8. Grad-CAM
    logging.info("\n--- Computing Grad-CAM ---")
    gradcam_mirna, gradcam_mrna = compute_gradcam(
        model, mirna_input, mrna_input, target_class, device
    )
    results['gradcam'] = (gradcam_mirna, gradcam_mrna)

    # 9. Gradient correction (apply to IG)
    logging.info("\n--- Applying Gradient Correction ---")
    ig_mirna_corrected = apply_gradient_correction(ig_mirna, fcgr_mirna, 'normalize')
    ig_mrna_corrected = apply_gradient_correction(ig_mrna, fcgr_mrna, 'normalize')
    results['attributions']['IG_Corrected'] = (ig_mirna_corrected, ig_mrna_corrected)

    # 10. Interaction heatmap
    logging.info("\n--- Creating Interaction Heatmap ---")
    interaction_matrix = create_interaction_heatmap(ig_mirna, ig_mrna, method='outer_product')
    results['interaction_matrix'] = interaction_matrix

    # 11. Cluster analysis
    logging.info("\n--- Performing Cluster Analysis ---")
    cluster_mirna, centers_mirna = cluster_attribution_regions(ig_mirna, n_clusters=5)
    cluster_mrna, centers_mrna = cluster_attribution_regions(ig_mrna, n_clusters=5)
    results['clusters'] = {
        'mirna': (cluster_mirna, centers_mirna),
        'mrna': (cluster_mrna, centers_mrna)
    }

    # ============================================================================
    # VISUALIZATIONS
    # ============================================================================

    logging.info("\n--- Generating Visualizations ---")

    # Comprehensive comparison
    visualize_all_methods_comparison(
        fcgr_mirna, fcgr_mrna, results['attributions'], sample_idx, save_dir
    )

    # Grad-CAM overlay
    visualize_gradcam_overlay(
        fcgr_mirna, fcgr_mrna, gradcam_mirna, gradcam_mrna, sample_idx, save_dir
    )

    # Interaction heatmap
    visualize_interaction_heatmap(interaction_matrix, sample_idx, save_dir)

    # Cluster visualization
    visualize_cluster_analysis(
        ig_mirna, cluster_mirna, centers_mirna,
        f'miRNA Clustered Regions - Sample {sample_idx}',
        save_dir / f'sample_{sample_idx}_mirna_clusters.png'
    )

    visualize_cluster_analysis(
        ig_mrna, cluster_mrna, centers_mrna,
        f'mRNA Clustered Regions - Sample {sample_idx}',
        save_dir / f'sample_{sample_idx}_mrna_clusters.png'
    )

    # Individual method visualizations
    for method, (mirna_attr, mrna_attr) in results['attributions'].items():
        visualize_attribution_heatmap(
            fcgr_mirna, mirna_attr,
            f'miRNA - {method} - Sample {sample_idx}',
            save_dir / f'sample_{sample_idx}_mirna_{method}.png'
        )

        visualize_attribution_heatmap(
            fcgr_mrna, mrna_attr,
            f'mRNA - {method} - Sample {sample_idx}',
            save_dir / f'sample_{sample_idx}_mrna_{method}.png'
        )

    # Export attributions
    export_attributions(results['attributions'], sample_idx, save_dir / 'numpy_exports')

    logging.info(f"Analysis complete for sample {sample_idx}")

    return results


def run_interpretability_analysis(
    checkpoint_path: str,
    data_path: str,
    k: int,
    dropout_rate: float,
    n_samples: int,
    n_background: int,
    output_dir: str
):
    """
    Main function to run complete interpretability analysis.

    Args:
        checkpoint_path: Path to trained model checkpoint
        data_path: Path to dataset CSV
        k: K-mer size for FCGR
        dropout_rate: Dropout rate used in model
        n_samples: Number of samples to analyze
        n_background: Number of background samples for SHAP
        output_dir: Output directory for results
    """
    # Setup
    setup_logging()
    device = get_device()

    # Create output directories
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / 'numpy_exports').mkdir(exist_ok=True)

    logging.info(f"\n{'='*60}")
    logging.info("FCGR-based Model Interpretability Analysis")
    logging.info(f"{'='*60}\n")

    # Load model
    model = load_trained_model(checkpoint_path, k, dropout_rate, device)

    # Load dataset
    logging.info(f"Loading dataset from {data_path}")
    dataset = MiRNAInteractionDataset(csv_path=data_path, k=k, train=False)
    logging.info(f"Dataset size: {len(dataset)}")

    # Sample indices for analysis
    np.random.seed(42)
    sample_indices = np.random.choice(len(dataset), size=min(n_samples, len(dataset)), replace=False)

    # Create background dataset for SHAP
    logging.info(f"Creating background dataset with {n_background} samples")
    background_indices = np.random.choice(len(dataset), size=min(n_background, len(dataset)), replace=False)
    background_data = []

    for idx in background_indices:
        mirna, mrna, _ = dataset[idx]
        background_data.append((mirna.unsqueeze(0).to(device), mrna.unsqueeze(0).to(device)))

    # Visualize CNN filters
    logging.info("\n--- Visualizing CNN Filters ---")
    visualize_conv_filters(model, output_path)

    # Analyze each sample
    all_results = []
    all_mirna_attributions = {method: [] for method in ['IntegratedGradients', 'Saliency', 'DeepLift', 'GradientSHAP', 'SHAP']}
    all_mrna_attributions = {method: [] for method in ['IntegratedGradients', 'Saliency', 'DeepLift', 'GradientSHAP', 'SHAP']}

    for idx in sample_indices:
        mirna, mrna, label = dataset[idx]

        # Add batch dimension and move to device
        mirna_input = mirna.unsqueeze(0).to(device)
        mrna_input = mrna.unsqueeze(0).to(device)

        # Analyze sample
        results = analyze_sample(
            model, mirna_input, mrna_input, label.item(), int(idx),
            background_data, device, output_path
        )

        all_results.append(results)

        # Collect attributions for aggregate analysis
        for method in all_mirna_attributions.keys():
            if method in results['attributions']:
                mirna_attr, mrna_attr = results['attributions'][method]
                all_mirna_attributions[method].append(mirna_attr)
                all_mrna_attributions[method].append(mrna_attr)

        # Visualize filter activations for first sample
        if idx == sample_indices[0]:
            mirna_act, mrna_act = compute_filter_activations(model, mirna_input, mrna_input, device)
            visualize_filter_activations(mirna_act, mrna_act, output_path)

    # ============================================================================
    # AGGREGATE ANALYSIS
    # ============================================================================

    logging.info(f"\n{'='*60}")
    logging.info("Aggregate Analysis Across All Samples")
    logging.info(f"{'='*60}\n")

    # Compute statistics
    mirna_stats_list = []
    mrna_stats_list = []

    for method in all_mirna_attributions.keys():
        if all_mirna_attributions[method]:
            mirna_attrs = {method: np.array(all_mirna_attributions[method])}
            mrna_attrs = {method: np.array(all_mrna_attributions[method])}

            mirna_stats = compute_attribution_statistics(mirna_attrs, 'miRNA')
            mrna_stats = compute_attribution_statistics(mrna_attrs, 'mRNA')

            mirna_stats_list.append(mirna_stats)
            mrna_stats_list.append(mrna_stats)

    # Combine statistics
    all_stats = pd.concat(mirna_stats_list + mrna_stats_list, ignore_index=True)

    # Save statistics
    stats_path = output_path / 'attribution_statistics.csv'
    all_stats.to_csv(stats_path, index=False)
    logging.info(f"Attribution statistics saved to {stats_path}")

    # Print summary
    logging.info("\n--- Attribution Statistics Summary ---")
    logging.info(f"\n{all_stats.to_string()}")

    # Compute average attributions
    logging.info("\n--- Computing Average Attributions ---")

    for method in all_mirna_attributions.keys():
        if all_mirna_attributions[method]:
            avg_mirna = np.mean(all_mirna_attributions[method], axis=0)
            avg_mrna = np.mean(all_mrna_attributions[method], axis=0)

            # Visualize average attribution
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))

            vmax_mirna = np.max(np.abs(avg_mirna))
            im1 = axes[0].imshow(avg_mirna, cmap='RdBu_r', vmin=-vmax_mirna, vmax=vmax_mirna)
            axes[0].set_title(f'Average miRNA Attribution - {method}')
            axes[0].axis('off')
            plt.colorbar(im1, ax=axes[0], fraction=0.046)

            vmax_mrna = np.max(np.abs(avg_mrna))
            im2 = axes[1].imshow(avg_mrna, cmap='RdBu_r', vmin=-vmax_mrna, vmax=vmax_mrna)
            axes[1].set_title(f'Average mRNA Attribution - {method}')
            axes[1].axis('off')
            plt.colorbar(im2, ax=axes[1], fraction=0.046)

            plt.tight_layout()
            plt.savefig(output_path / f'average_attribution_{method}.png', dpi=300, bbox_inches='tight')
            plt.close()

    # Generate summary report
    logging.info("\n--- Generating Summary Report ---")

    summary = {
        'n_samples_analyzed': len(sample_indices),
        'n_background_samples': n_background,
        'model_checkpoint': checkpoint_path,
        'k_mer': k,
        'methods_used': list(all_mirna_attributions.keys()),
        'output_directory': str(output_path)
    }

    import json
    with open(output_path / 'analysis_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    logging.info(f"\n{'='*60}")
    logging.info("Interpretability Analysis Complete!")
    logging.info(f"{'='*60}")
    logging.info(f"Results saved to: {output_path}")
    logging.info(f"Total samples analyzed: {len(sample_indices)}")
    logging.info(f"Methods used: {', '.join(summary['methods_used'])}")


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def main():
    """Main entry point for interpretability analysis."""
    parser = argparse.ArgumentParser(
        description="Deep Learning Interpretability Analysis for FCGR-based miRNA-mRNA Model"
    )

    parser.add_argument(
        '--checkpoint',
        type=str,
        default='checkpoints/best_model.pth',
        help='Path to trained model checkpoint'
    )

    parser.add_argument(
        '--data',
        type=str,
        default='data/miraw.csv',
        help='Path to dataset CSV'
    )

    parser.add_argument(
        '--k',
        type=int,
        default=6,
        help='K-mer size for FCGR generation'
    )

    parser.add_argument(
        '--dropout',
        type=float,
        default=0.1,
        help='Dropout rate used in model'
    )

    parser.add_argument(
        '--n_samples',
        type=int,
        default=5,
        help='Number of samples to analyze in detail'
    )

    parser.add_argument(
        '--n_background',
        type=int,
        default=20,
        help='Number of background samples for SHAP'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='results/interpretability',
        help='Output directory for results'
    )

    args = parser.parse_args()

    # Run analysis
    run_interpretability_analysis(
        checkpoint_path=args.checkpoint,
        data_path=args.data,
        k=args.k,
        dropout_rate=args.dropout,
        n_samples=args.n_samples,
        n_background=args.n_background,
        output_dir=args.output
    )


if __name__ == '__main__':
    main()
