import logging
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


class TrainingVisualizer:
    """
    Visualize training progress and results.
    """

    def __init__(self, save_dir: str = "results"):
        """
        Initialize visualizer.

        Args:
            save_dir: Directory to save visualizations
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.epochs = []

    def update(self, epoch: int, train_loss: float, val_loss: float,
               train_acc: float = None, val_acc: float = None):
        """
        Update visualization data.

        Args:
            epoch: Current epoch number
            train_loss: Training loss
            val_loss: Validation loss
            train_acc: Training accuracy (optional)
            val_acc: Validation accuracy (optional)
        """
        self.epochs.append(epoch)
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)

        if train_acc is not None:
            self.train_accuracies.append(train_acc)
        if val_acc is not None:
            self.val_accuracies.append(val_acc)

    def plot_losses(self, save: bool = True):
        """
        Plot training and validation losses.

        Args:
            save: Whether to save the plot
        """
        if len(self.epochs) == 0:
            logging.warning("No data available for plotting")
            return

        plt.figure(figsize=(10, 6))
        plt.plot(self.epochs, self.train_losses, 'b-', label='Training Loss', linewidth=2)
        plt.plot(self.epochs, self.val_losses, 'r-', label='Validation Loss', linewidth=2)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.title('Training and Validation Loss', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        if save:
            save_path = self.save_dir / 'loss_plot.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logging.info(f"Loss plot saved to {save_path}")

        plt.close()

    def plot_accuracies(self, save: bool = True):
        """
        Plot training and validation accuracies.

        Args:
            save: Whether to save the plot
        """
        if len(self.train_accuracies) == 0 or len(self.val_accuracies) == 0:
            logging.warning("No accuracy data available for plotting")
            return

        plt.figure(figsize=(10, 6))
        plt.plot(self.epochs, self.train_accuracies, 'b-', label='Training Accuracy', linewidth=2)
        plt.plot(self.epochs, self.val_accuracies, 'r-', label='Validation Accuracy', linewidth=2)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        if save:
            save_path = self.save_dir / 'accuracy_plot.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logging.info(f"Accuracy plot saved to {save_path}")

        plt.close()

    def plot_all(self, save: bool = True):
        """
        Plot both losses and accuracies in a single figure.

        Args:
            save: Whether to save the plot
        """
        if len(self.epochs) == 0:
            logging.warning("No data available for plotting")
            return

        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        # Plot losses
        axes[0].plot(self.epochs, self.train_losses, 'b-', label='Training Loss', linewidth=2)
        axes[0].plot(self.epochs, self.val_losses, 'r-', label='Validation Loss', linewidth=2)
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Loss', fontsize=12)
        axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)

        # Plot accuracies
        if len(self.train_accuracies) > 0 and len(self.val_accuracies) > 0:
            axes[1].plot(self.epochs, self.train_accuracies, 'b-', label='Training Accuracy', linewidth=2)
            axes[1].plot(self.epochs, self.val_accuracies, 'r-', label='Validation Accuracy', linewidth=2)
            axes[1].set_xlabel('Epoch', fontsize=12)
            axes[1].set_ylabel('Accuracy', fontsize=12)
            axes[1].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
            axes[1].legend(fontsize=10)
            axes[1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save:
            save_path = self.save_dir / 'training_progress.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logging.info(f"Training progress plot saved to {save_path}")

        plt.close()

    def save_history(self):
        """Save training history to a text file."""
        history_path = self.save_dir / 'training_history.txt'

        with open(history_path, 'w') as f:
            f.write("Epoch,Train_Loss,Val_Loss,Train_Acc,Val_Acc\n")

            for i, epoch in enumerate(self.epochs):
                train_acc = self.train_accuracies[i] if i < len(self.train_accuracies) else 'N/A'
                val_acc = self.val_accuracies[i] if i < len(self.val_accuracies) else 'N/A'

                f.write(f"{epoch},{self.train_losses[i]:.4f},{self.val_losses[i]:.4f},{train_acc},{val_acc}\n")

        logging.info(f"Training history saved to {history_path}")
