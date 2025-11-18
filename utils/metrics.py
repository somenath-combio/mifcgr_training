import logging
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    matthews_corrcoef
)


class MetricsCalculator:
    """
    Calculate and store various metrics for binary classification.
    """

    def __init__(self):
        """Initialize metrics calculator."""
        self.reset()

    def reset(self):
        """Reset all stored predictions and labels."""
        self.all_predictions = []
        self.all_labels = []
        self.all_probabilities = []

    def update(self, predictions: np.ndarray, labels: np.ndarray, probabilities: np.ndarray = None):
        """
        Update metrics with new batch of predictions and labels.

        Args:
            predictions: Predicted class labels
            labels: True class labels
            probabilities: Predicted probabilities (optional)
        """
        self.all_predictions.extend(predictions.tolist())
        self.all_labels.extend(labels.tolist())

        if probabilities is not None:
            self.all_probabilities.extend(probabilities.tolist())

    def compute(self) -> dict:
        """
        Compute all metrics.

        Returns:
            Dictionary containing all computed metrics
        """
        if len(self.all_predictions) == 0:
            logging.warning("No predictions available for metric calculation")
            return {}

        predictions = np.array(self.all_predictions)
        labels = np.array(self.all_labels)

        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(labels, predictions),
            'precision': precision_score(labels, predictions, zero_division=0),
            'recall': recall_score(labels, predictions, zero_division=0),
            'f1': f1_score(labels, predictions, zero_division=0),
            'mcc': matthews_corrcoef(labels, predictions)
        }

        # Calculate specificity from confusion matrix
        tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0

        # Calculate AUC if probabilities are available
        if len(self.all_probabilities) > 0:
            probabilities = np.array(self.all_probabilities)
            try:
                metrics['auc'] = roc_auc_score(labels, probabilities)
            except Exception as e:
                logging.warning(f"Could not calculate AUC: {e}")
                metrics['auc'] = 0.0

        return metrics

    def log_metrics(self, prefix: str = ""):
        """
        Compute and log all metrics.

        Args:
            prefix: Prefix to add to metric names (e.g., 'train_' or 'val_')
        """
        metrics = self.compute()

        logging.info(f"\n{'='*50}")
        logging.info(f"{prefix}Metrics:")
        logging.info(f"{'='*50}")

        for metric_name, value in metrics.items():
            logging.info(f"{prefix}{metric_name}: {value:.4f}")

        logging.info(f"{'='*50}\n")

        return metrics


def calculate_loss_metrics(losses: list) -> dict:
    """
    Calculate statistics for losses.

    Args:
        losses: List of loss values

    Returns:
        Dictionary containing loss statistics
    """
    if len(losses) == 0:
        return {'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0}

    losses_array = np.array(losses)

    return {
        'mean': np.mean(losses_array),
        'std': np.std(losses_array),
        'min': np.min(losses_array),
        'max': np.max(losses_array)
    }
