import logging
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from tqdm import tqdm

from utils.early_stop import EarlyStopping, ModelCheckpoint
from utils.metrics import MetricsCalculator
from utils.visualization import TrainingVisualizer


class Trainer:
    """
    Trainer class for miRNA-mRNA interaction prediction model.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        num_epochs: int = 100,
        patience: int = 15,
        checkpoint_path: str = "checkpoints/best_model.pth",
        results_dir: str = "results"
    ):
        """
        Initialize the trainer.

        Args:
            model: PyTorch model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            criterion: Loss function
            optimizer: Optimizer
            device: Device to run training on
            num_epochs: Maximum number of epochs
            patience: Patience for early stopping
            checkpoint_path: Path to save best model
            results_dir: Directory to save results
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.num_epochs = num_epochs

        # Create directories
        Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
        Path(results_dir).mkdir(parents=True, exist_ok=True)

        # Initialize utilities (monitor validation accuracy)
        self.early_stopping = EarlyStopping(patience=patience, mode='max')
        self.model_checkpoint = ModelCheckpoint(filepath=checkpoint_path, mode='max')
        self.visualizer = TrainingVisualizer(save_dir=results_dir)

        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_metrics': [],
            'val_metrics': []
        }

    def train_epoch(self, epoch: int) -> tuple:
        """
        Train for one epoch.

        Args:
            epoch: Current epoch number

        Returns:
            Tuple of (average_loss, metrics_dict)
        """
        self.model.train()
        epoch_losses = []
        metrics_calc = MetricsCalculator()

        # Progress bar
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs} [Train]")

        for batch_idx, (mirna, mrna, labels) in enumerate(pbar):
            # Move data to device
            mirna = mirna.to(self.device)
            mrna = mrna.to(self.device)
            labels = labels.to(self.device)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(mrna, mirna)  # Note: model expects (mrna, mirna)

            # Calculate loss
            loss = self.criterion(outputs, labels)

            # Backward pass
            loss.backward()

            # Clip gradients to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            # Update weights
            self.optimizer.step()

            # Record loss
            epoch_losses.append(loss.item())

            # Get predictions
            probabilities = torch.softmax(outputs, dim=1)[:, 1]  # Probability of class 1
            predictions = torch.argmax(outputs, dim=1)

            # Update metrics
            metrics_calc.update(
                predictions=predictions.cpu().detach().numpy(),
                labels=labels.cpu().detach().numpy(),
                probabilities=probabilities.cpu().detach().numpy()
            )

            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        # Calculate average loss
        avg_loss = np.mean(epoch_losses)

        # Calculate metrics
        metrics = metrics_calc.compute()

        return avg_loss, metrics

    def validate_epoch(self, epoch: int) -> tuple:
        """
        Validate for one epoch.

        Args:
            epoch: Current epoch number

        Returns:
            Tuple of (average_loss, metrics_dict)
        """
        self.model.eval()
        epoch_losses = []
        metrics_calc = MetricsCalculator()

        # Progress bar
        pbar = tqdm(self.val_loader, desc=f"Epoch {epoch+1}/{self.num_epochs} [Val]")

        with torch.no_grad():
            for batch_idx, (mirna, mrna, labels) in enumerate(pbar):
                # Move data to device
                mirna = mirna.to(self.device)
                mrna = mrna.to(self.device)
                labels = labels.to(self.device)

                # Forward pass
                outputs = self.model(mrna, mirna)

                # Calculate loss
                loss = self.criterion(outputs, labels)

                # Record loss
                epoch_losses.append(loss.item())

                # Get predictions
                probabilities = torch.softmax(outputs, dim=1)[:, 1]
                predictions = torch.argmax(outputs, dim=1)

                # Update metrics
                metrics_calc.update(
                    predictions=predictions.cpu().numpy(),
                    labels=labels.cpu().numpy(),
                    probabilities=probabilities.cpu().numpy()
                )

                # Update progress bar
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        # Calculate average loss
        avg_loss = np.mean(epoch_losses)

        # Calculate metrics
        metrics = metrics_calc.compute()

        return avg_loss, metrics

    def train(self):
        """
        Main training loop.
        """
        logging.info("="*60)
        logging.info("Starting Training")
        logging.info("="*60)
        logging.info(f"Device: {self.device}")
        logging.info(f"Number of epochs: {self.num_epochs}")
        logging.info(f"Training samples: {len(self.train_loader.dataset)}")
        logging.info(f"Validation samples: {len(self.val_loader.dataset)}")
        logging.info(f"Batch size: {self.train_loader.batch_size}")
        logging.info("="*60)

        for epoch in range(self.num_epochs):
            # Train
            train_loss, train_metrics = self.train_epoch(epoch)

            # Validate
            val_loss, val_metrics = self.validate_epoch(epoch)

            # Log results
            logging.info(f"\nEpoch {epoch+1}/{self.num_epochs}")
            logging.info(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            logging.info(f"Train Acc: {train_metrics.get('accuracy', 0):.4f} | Val Acc: {val_metrics.get('accuracy', 0):.4f}")
            logging.info(f"Train F1: {train_metrics.get('f1', 0):.4f} | Val F1: {val_metrics.get('f1', 0):.4f}")
            logging.info(f"Val AUC: {val_metrics.get('auc', 0):.4f}")

            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_metrics'].append(train_metrics)
            self.history['val_metrics'].append(val_metrics)

            # Update visualizer
            self.visualizer.update(
                epoch=epoch + 1,
                train_loss=train_loss,
                val_loss=val_loss,
                train_acc=train_metrics.get('accuracy', 0),
                val_acc=val_metrics.get('accuracy', 0)
            )

            # Save best model based on validation accuracy
            val_acc = val_metrics.get('accuracy', 0)
            self.model_checkpoint(self.model, val_acc, epoch + 1, self.optimizer)

            # Check early stopping based on validation accuracy
            if self.early_stopping(val_acc, epoch + 1):
                logging.info(f"\nEarly stopping triggered at epoch {epoch + 1}")
                break

        # Save final visualizations
        logging.info("\nTraining completed!")
        logging.info("Saving visualizations...")
        self.visualizer.plot_all(save=True)
        self.visualizer.save_history()

        # Print best metrics (based on validation accuracy)
        val_accuracies = [m.get('accuracy', 0) for m in self.history['val_metrics']]
        best_epoch = np.argmax(val_accuracies)
        logging.info(f"\nBest model at epoch {best_epoch + 1}:")
        logging.info(f"Validation Accuracy: {self.history['val_metrics'][best_epoch].get('accuracy', 0):.4f}")
        logging.info(f"Validation Loss: {self.history['val_loss'][best_epoch]:.4f}")
        logging.info(f"Validation F1: {self.history['val_metrics'][best_epoch].get('f1', 0):.4f}")
        logging.info(f"Validation AUC: {self.history['val_metrics'][best_epoch].get('auc', 0):.4f}")

        return self.history

    def load_best_model(self, checkpoint_path: str = None):
        """
        Load the best model from checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
        """
        if checkpoint_path is None:
            checkpoint_path = self.model_checkpoint.filepath

        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        logging.info(f"Loaded best model from {checkpoint_path}")
        logging.info(f"Best validation accuracy: {checkpoint['score']:.4f}")
