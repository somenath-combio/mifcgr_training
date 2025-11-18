import logging
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from core.model import InteractionModel
from core.dataset import prepare_data_loaders
from core.trainer import Trainer


def setup_logging(log_file: str = "training.log"):
    """
    Set up logging configuration.

    Args:
        log_file: Path to log file
    """
    # Create logs directory
    Path("logs").mkdir(exist_ok=True)
    log_path = Path("logs") / log_file

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )

    logging.info("Logging initialized")
    logging.info(f"Log file: {log_path}")


def get_device():
    """
    Get the best available device (GPU if available, else CPU).

    Returns:
        torch.device
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
        logging.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        logging.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        device = torch.device('cpu')
        logging.info("Using CPU")

    return device


def count_parameters(model):
    """
    Count the number of trainable parameters in the model.

    Args:
        model: PyTorch model

    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main(args):
    """
    Main training function.

    Args:
        args: Command line arguments
    """
    # Set up logging
    setup_logging(args.log_file)

    logging.info("="*60)
    logging.info("miRNA-mRNA Interaction Prediction Training")
    logging.info("="*60)

    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Get device
    device = get_device()

    # Load data
    logging.info("\n" + "="*60)
    logging.info("Loading Data")
    logging.info("="*60)

    train_loader, val_loader = prepare_data_loaders(
        csv_path=args.data_path,
        k=args.k,
        batch_size=args.batch_size,
        train_split=args.train_split,
        shuffle=True
    )

    logging.info(f"Data loaded successfully")
    logging.info(f"K-mer size: {args.k}")
    logging.info(f"Batch size: {args.batch_size}")
    logging.info(f"Train/Val split: {args.train_split}/{1-args.train_split}")

    # Initialize model
    logging.info("\n" + "="*60)
    logging.info("Initializing Model")
    logging.info("="*60)

    model = InteractionModel(dropout_rate=args.dropout, k=args.k)

    # Count parameters
    num_params = count_parameters(model)
    logging.info(f"Model initialized with {num_params:,} trainable parameters")

    # Define loss function
    criterion = nn.CrossEntropyLoss()
    logging.info(f"Loss function: CrossEntropyLoss")

    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    logging.info(f"Optimizer: Adam")
    logging.info(f"Learning rate: {args.lr}")
    logging.info(f"Weight decay: {args.weight_decay}")

    # Optional: Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        verbose=True
    )
    logging.info(f"LR Scheduler: ReduceLROnPlateau (factor=0.5, patience=5)")

    # Initialize trainer
    logging.info("\n" + "="*60)
    logging.info("Initializing Trainer")
    logging.info("="*60)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_epochs=args.epochs,
        patience=args.patience,
        checkpoint_path=args.checkpoint_path,
        results_dir=args.results_dir
    )

    # Train model
    logging.info("\n" + "="*60)
    logging.info("Starting Training Loop")
    logging.info("="*60)

    history = trainer.train()

    # Load best model
    logging.info("\n" + "="*60)
    logging.info("Loading Best Model")
    logging.info("="*60)

    trainer.load_best_model()

    logging.info("\n" + "="*60)
    logging.info("Training Complete!")
    logging.info("="*60)
    logging.info(f"Results saved to: {args.results_dir}")
    logging.info(f"Best model saved to: {args.checkpoint_path}")
    logging.info(f"Training log saved to: logs/{args.log_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train miRNA-mRNA interaction prediction model")

    # Data parameters
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/miraw.csv",
        help="Path to the CSV data file"
    )

    parser.add_argument(
        "--train_split",
        type=float,
        default=0.8,
        help="Fraction of data to use for training (default: 0.8)"
    )

    # Model parameters
    parser.add_argument(
        "--k",
        type=int,
        default=6,
        help="K-mer size for FCGR generation (default: 6)"
    )

    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="Dropout rate (default: 0.1)"
    )

    # Training parameters
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for training (default: 64)"
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=0.003,
        help="Learning rate (default: 0.003)"
    )

    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-5,
        help="Weight decay for regularization (default: 1e-5)"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Maximum number of epochs (default: 100)"
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=15,
        help="Patience for early stopping (default: 15)"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )

    # Output parameters
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="checkpoints/best_model.pth",
        help="Path to save the best model checkpoint"
    )

    parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="Directory to save training results and visualizations"
    )

    parser.add_argument(
        "--log_file",
        type=str,
        default="training.log",
        help="Name of the log file"
    )

    args = parser.parse_args()

    # Run training
    main(args)
