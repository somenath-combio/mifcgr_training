"""
Quick Start Example: FCGR Model Interpretability Analysis

This script demonstrates how to use the interpretability pipeline programmatically
for analyzing specific samples or conducting custom analyses.
"""

import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# Import from interpretability_analysis module
from interpretability_analysis import (
    load_trained_model,
    get_device,
    compute_integrated_gradients,
    compute_saliency,
    compute_gradcam,
    visualize_attribution_heatmap,
    visualize_gradcam_overlay,
    create_interaction_heatmap,
    visualize_interaction_heatmap,
    cluster_attribution_regions,
    visualize_cluster_analysis
)

from core.dataset import MiRNAInteractionDataset


def quick_analysis_example():
    """
    Quick example: Analyze a single sample with multiple methods.
    """
    print("="*60)
    print("FCGR Model Interpretability - Quick Start Example")
    print("="*60)

    # Configuration
    checkpoint_path = 'checkpoints/best_model.pth'
    data_path = 'data/miraw.csv'
    k = 6
    dropout_rate = 0.1
    sample_idx = 0  # Analyze first sample

    # Setup
    device = get_device()
    output_dir = Path('results/quick_example')
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nLoading model from {checkpoint_path}")
    model = load_trained_model(checkpoint_path, k, dropout_rate, device)

    print(f"Loading dataset from {data_path}")
    dataset = MiRNAInteractionDataset(csv_path=data_path, k=k, train=False)

    # Get sample
    print(f"\nAnalyzing sample {sample_idx}")
    mirna, mrna, label = dataset[sample_idx]

    # Add batch dimension and move to device
    mirna_input = mirna.unsqueeze(0).to(device)
    mrna_input = mrna.unsqueeze(0).to(device)

    print(f"True label: {label.item()}")

    # Get prediction
    with torch.no_grad():
        output = model(mrna_input, mirna_input)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0, predicted_class].item()

    print(f"Predicted class: {predicted_class}")
    print(f"Confidence: {confidence:.4f}")

    # Target class for attribution
    target_class = predicted_class

    # Convert to numpy for visualization
    fcgr_mirna = mirna.squeeze().cpu().numpy()
    fcgr_mrna = mrna.squeeze().cpu().numpy()

    # ========================================================================
    # METHOD 1: Integrated Gradients
    # ========================================================================
    print("\n" + "="*60)
    print("Computing Integrated Gradients...")
    print("="*60)

    ig_mirna, ig_mrna = compute_integrated_gradients(
        model, mirna_input, mrna_input, target_class, device, n_steps=50
    )

    print(f"miRNA IG range: [{ig_mirna.min():.4f}, {ig_mirna.max():.4f}]")
    print(f"mRNA IG range: [{ig_mrna.min():.4f}, {ig_mrna.max():.4f}]")

    # Visualize
    visualize_attribution_heatmap(
        fcgr_mirna, ig_mirna,
        title=f'miRNA - Integrated Gradients - Sample {sample_idx}',
        save_path=output_dir / 'mirna_integrated_gradients.png'
    )

    visualize_attribution_heatmap(
        fcgr_mrna, ig_mrna,
        title=f'mRNA - Integrated Gradients - Sample {sample_idx}',
        save_path=output_dir / 'mrna_integrated_gradients.png'
    )

    print(f"✓ Integrated Gradients visualizations saved to {output_dir}")

    # ========================================================================
    # METHOD 2: Saliency Maps
    # ========================================================================
    print("\n" + "="*60)
    print("Computing Saliency Maps...")
    print("="*60)

    sal_mirna, sal_mrna = compute_saliency(
        model, mirna_input, mrna_input, target_class, device
    )

    print(f"miRNA Saliency range: [{sal_mirna.min():.4f}, {sal_mirna.max():.4f}]")
    print(f"mRNA Saliency range: [{sal_mrna.min():.4f}, {sal_mrna.max():.4f}]")

    # Visualize
    visualize_attribution_heatmap(
        fcgr_mirna, sal_mirna,
        title=f'miRNA - Saliency - Sample {sample_idx}',
        save_path=output_dir / 'mirna_saliency.png'
    )

    visualize_attribution_heatmap(
        fcgr_mrna, sal_mrna,
        title=f'mRNA - Saliency - Sample {sample_idx}',
        save_path=output_dir / 'mrna_saliency.png'
    )

    print(f"✓ Saliency maps saved to {output_dir}")

    # ========================================================================
    # METHOD 3: Grad-CAM
    # ========================================================================
    print("\n" + "="*60)
    print("Computing Grad-CAM...")
    print("="*60)

    gradcam_mirna, gradcam_mrna = compute_gradcam(
        model, mirna_input, mrna_input, target_class, device
    )

    print(f"miRNA Grad-CAM range: [{gradcam_mirna.min():.4f}, {gradcam_mirna.max():.4f}]")
    print(f"mRNA Grad-CAM range: [{gradcam_mrna.min():.4f}, {gradcam_mrna.max():.4f}]")

    # Visualize
    visualize_gradcam_overlay(
        fcgr_mirna, fcgr_mrna, gradcam_mirna, gradcam_mrna,
        sample_idx, output_dir
    )

    print(f"✓ Grad-CAM visualizations saved to {output_dir}")

    # ========================================================================
    # METHOD 4: Interaction Heatmap
    # ========================================================================
    print("\n" + "="*60)
    print("Creating Interaction Heatmap...")
    print("="*60)

    interaction_matrix = create_interaction_heatmap(
        ig_mirna, ig_mrna, method='outer_product'
    )

    print(f"Interaction matrix shape: {interaction_matrix.shape}")
    print(f"Interaction matrix range: [{interaction_matrix.min():.4f}, {interaction_matrix.max():.4f}]")

    # Visualize
    visualize_interaction_heatmap(interaction_matrix, sample_idx, output_dir)

    print(f"✓ Interaction heatmap saved to {output_dir}")

    # ========================================================================
    # METHOD 5: Cluster Analysis
    # ========================================================================
    print("\n" + "="*60)
    print("Performing Cluster Analysis...")
    print("="*60)

    cluster_mirna, centers_mirna = cluster_attribution_regions(
        ig_mirna, n_clusters=5, threshold_percentile=75
    )

    cluster_mrna, centers_mrna = cluster_attribution_regions(
        ig_mrna, n_clusters=5, threshold_percentile=75
    )

    print(f"miRNA: Found {len(centers_mirna)} cluster centers")
    print(f"mRNA: Found {len(centers_mrna)} cluster centers")

    # Visualize
    visualize_cluster_analysis(
        ig_mirna, cluster_mirna, centers_mirna,
        title=f'miRNA Clustered Regions - Sample {sample_idx}',
        save_path=output_dir / 'mirna_clusters.png'
    )

    visualize_cluster_analysis(
        ig_mrna, cluster_mrna, centers_mrna,
        title=f'mRNA Clustered Regions - Sample {sample_idx}',
        save_path=output_dir / 'mrna_clusters.png'
    )

    print(f"✓ Cluster analysis saved to {output_dir}")

    # ========================================================================
    # COMPARISON VISUALIZATION
    # ========================================================================
    print("\n" + "="*60)
    print("Creating Method Comparison Visualization...")
    print("="*60)

    fig, axes = plt.subplots(3, 2, figsize=(14, 18))

    methods = [
        ('Integrated Gradients', ig_mirna, ig_mrna),
        ('Saliency', sal_mirna, sal_mrna),
        ('Grad-CAM', gradcam_mirna, gradcam_mrna)
    ]

    for idx, (method_name, mirna_attr, mrna_attr) in enumerate(methods):
        # miRNA
        vmax_mirna = np.max(np.abs(mirna_attr))
        im1 = axes[idx, 0].imshow(mirna_attr, cmap='RdBu_r', vmin=-vmax_mirna, vmax=vmax_mirna)
        axes[idx, 0].set_title(f'miRNA - {method_name}', fontsize=12, fontweight='bold')
        axes[idx, 0].axis('off')
        plt.colorbar(im1, ax=axes[idx, 0], fraction=0.046)

        # mRNA
        vmax_mrna = np.max(np.abs(mrna_attr))
        im2 = axes[idx, 1].imshow(mrna_attr, cmap='RdBu_r', vmin=-vmax_mrna, vmax=vmax_mrna)
        axes[idx, 1].set_title(f'mRNA - {method_name}', fontsize=12, fontweight='bold')
        axes[idx, 1].axis('off')
        plt.colorbar(im2, ax=axes[idx, 1], fraction=0.046)

    fig.suptitle(f'Attribution Methods Comparison - Sample {sample_idx}\n'
                 f'True: {label.item()}, Predicted: {predicted_class} (conf: {confidence:.2f})',
                 fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_dir / 'methods_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Comparison visualization saved to {output_dir}")

    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n" + "="*60)
    print("Analysis Complete!")
    print("="*60)
    print(f"\nAll results saved to: {output_dir}")
    print("\nGenerated files:")
    print("  - mirna_integrated_gradients.png")
    print("  - mrna_integrated_gradients.png")
    print("  - mirna_saliency.png")
    print("  - mrna_saliency.png")
    print("  - sample_0_gradcam.png")
    print("  - sample_0_interaction_heatmap.png")
    print("  - mirna_clusters.png")
    print("  - mrna_clusters.png")
    print("  - methods_comparison.png")

    print("\n" + "="*60)
    print("Next Steps:")
    print("="*60)
    print("1. View the visualizations in the output directory")
    print("2. Run full analysis: python interpretability_analysis.py")
    print("3. Customize parameters in this script for your needs")
    print("4. See INTERPRETABILITY_README.md for detailed documentation")


def batch_analysis_example():
    """
    Example: Analyze multiple samples and compute average attributions.
    """
    print("\n" + "="*60)
    print("Batch Analysis Example")
    print("="*60)

    # Configuration
    checkpoint_path = 'checkpoints/best_model.pth'
    data_path = 'data/miraw.csv'
    k = 6
    dropout_rate = 0.1
    n_samples = 3

    # Setup
    device = get_device()
    model = load_trained_model(checkpoint_path, k, dropout_rate, device)
    dataset = MiRNAInteractionDataset(csv_path=data_path, k=k, train=False)

    # Collect attributions
    all_ig_mirna = []
    all_ig_mrna = []

    print(f"\nAnalyzing {n_samples} samples...")

    for i in range(min(n_samples, len(dataset))):
        mirna, mrna, label = dataset[i]
        mirna_input = mirna.unsqueeze(0).to(device)
        mrna_input = mrna.unsqueeze(0).to(device)

        # Get prediction
        with torch.no_grad():
            output = model(mrna_input, mirna_input)
            predicted_class = torch.argmax(output, dim=1).item()

        # Compute IG
        ig_mirna, ig_mrna = compute_integrated_gradients(
            model, mirna_input, mrna_input, predicted_class, device, n_steps=30
        )

        all_ig_mirna.append(ig_mirna)
        all_ig_mrna.append(ig_mrna)

        print(f"  Sample {i}: label={label.item()}, pred={predicted_class}")

    # Compute averages
    avg_ig_mirna = np.mean(all_ig_mirna, axis=0)
    avg_ig_mrna = np.mean(all_ig_mrna, axis=0)

    print(f"\nAverage attribution ranges:")
    print(f"  miRNA: [{avg_ig_mirna.min():.4f}, {avg_ig_mirna.max():.4f}]")
    print(f"  mRNA: [{avg_ig_mrna.min():.4f}, {avg_ig_mrna.max():.4f}]")

    # Visualize averages
    output_dir = Path('results/batch_example')
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    vmax_mirna = np.max(np.abs(avg_ig_mirna))
    im1 = axes[0].imshow(avg_ig_mirna, cmap='RdBu_r', vmin=-vmax_mirna, vmax=vmax_mirna)
    axes[0].set_title(f'Average miRNA Attribution (n={n_samples})')
    axes[0].axis('off')
    plt.colorbar(im1, ax=axes[0], fraction=0.046)

    vmax_mrna = np.max(np.abs(avg_ig_mrna))
    im2 = axes[1].imshow(avg_ig_mrna, cmap='RdBu_r', vmin=-vmax_mrna, vmax=vmax_mrna)
    axes[1].set_title(f'Average mRNA Attribution (n={n_samples})')
    axes[1].axis('off')
    plt.colorbar(im2, ax=axes[1], fraction=0.046)

    plt.tight_layout()
    plt.savefig(output_dir / 'average_attributions.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n✓ Batch analysis saved to {output_dir}")


if __name__ == '__main__':
    # Run quick analysis example
    quick_analysis_example()

    # Uncomment to run batch analysis example
    # batch_analysis_example()
