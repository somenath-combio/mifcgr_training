import logging
import numpy as np
import torch


class EarlyStopping:
    """
    Early stopping to stop training when validation loss doesn't improve.
    """

    def __init__(self, patience: int = 10, min_delta: float = 0.0, mode: str = 'min'):
        """
        Initialize early stopping.

        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as an improvement
            mode: 'min' for loss, 'max' for accuracy
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0

        if mode == 'min':
            self.monitor_op = np.less
            self.min_delta *= -1
        elif mode == 'max':
            self.monitor_op = np.greater
            self.min_delta *= 1
        else:
            raise ValueError(f"Invalid mode: {mode}. Must be 'min' or 'max'")

    def __call__(self, score: float, epoch: int) -> bool:
        """
        Check if training should stop.

        Args:
            score: Current validation score (loss or accuracy)
            epoch: Current epoch number

        Returns:
            True if training should stop, False otherwise
        """
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            return False

        if self.monitor_op(score - self.min_delta, self.best_score):
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
            logging.info(f"Early stopping counter reset. Best score: {self.best_score:.4f}")
        else:
            self.counter += 1
            logging.info(f"Early stopping counter: {self.counter}/{self.patience}")

            if self.counter >= self.patience:
                self.early_stop = True
                logging.info(f"Early stopping triggered! Best score: {self.best_score:.4f} at epoch {self.best_epoch}")
                return True

        return False


class ModelCheckpoint:
    """
    Save the best model during training.
    """

    def __init__(self, filepath: str, mode: str = 'min', save_best_only: bool = True):
        """
        Initialize model checkpoint.

        Args:
            filepath: Path to save the model
            mode: 'min' for loss, 'max' for accuracy
            save_best_only: Whether to save only the best model
        """
        self.filepath = filepath
        self.mode = mode
        self.save_best_only = save_best_only
        self.best_score = None

        if mode == 'min':
            self.monitor_op = np.less
        elif mode == 'max':
            self.monitor_op = np.greater
        else:
            raise ValueError(f"Invalid mode: {mode}. Must be 'min' or 'max'")

    def __call__(self, model: torch.nn.Module, score: float, epoch: int, optimizer=None):
        """
        Save model if score improved.

        Args:
            model: PyTorch model to save
            score: Current validation score
            epoch: Current epoch number
            optimizer: Optimizer state to save (optional)
        """
        if self.best_score is None or self.monitor_op(score, self.best_score):
            self.best_score = score

            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'score': score
            }

            if optimizer is not None:
                checkpoint['optimizer_state_dict'] = optimizer.state_dict()

            torch.save(checkpoint, self.filepath)
            logging.info(f"Model checkpoint saved at epoch {epoch} with score: {score:.4f}")
        elif not self.save_best_only:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'score': score
            }

            if optimizer is not None:
                checkpoint['optimizer_state_dict'] = optimizer.state_dict()

            torch.save(checkpoint, self.filepath)
            logging.info(f"Model checkpoint saved at epoch {epoch}")
