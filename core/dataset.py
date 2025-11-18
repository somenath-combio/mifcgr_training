import logging
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from utils.fcgr import FCGR


class MiRNAInteractionDataset(Dataset):
    """
    Dataset class for miRNA-mRNA interaction prediction.
    Handles data loading, preprocessing, and FCGR generation.
    """

    def __init__(self, csv_path: str, k: int = 6, train: bool = True):
        """
        Initialize the dataset.

        Args:
            csv_path: Path to the CSV file containing the data
            k: K-mer size for FCGR generation
            train: Whether this is training data (for logging purposes)
        """
        self.k = k
        self.train = train

        # Read CSV file
        logging.info(f"Reading data from {csv_path}")
        df = pd.read_csv(csv_path)
        logging.info(f"Initial dataset size: {len(df)}")

        # Drop duplicates based on miRNA, mRNA, and validation columns
        df = df.drop_duplicates(subset=['mature_miRNA_Transcript', 'mRNA_Site_Transcript', 'validation'])
        logging.info(f"Dataset size after removing duplicates: {len(df)}")

        # Extract required columns
        self.mirna_sequences = df['mature_miRNA_Transcript'].values
        self.mrna_sequences = df['mRNA_Site_Transcript'].values
        self.labels = df['validation'].values

        logging.info(f"Unique labels: {np.unique(self.labels)}")
        logging.info(f"Label distribution: {np.bincount(self.labels)}")

    def _rna_to_dna(self, sequence: str) -> str:
        """
        Convert RNA sequence to DNA by replacing U with T.

        Args:
            sequence: RNA sequence string

        Returns:
            DNA sequence string
        """
        return sequence.replace('U', 'T')

    def _generate_fcgr(self, sequence: str) -> np.ndarray:
        """
        Generate FCGR for a given sequence.

        Args:
            sequence: DNA sequence string

        Returns:
            FCGR as numpy array
        """
        try:
            fcgr = FCGR(sequence=sequence, k=self.k)
            return fcgr.generate_fcgr()
        except Exception as e:
            logging.error(f"Error generating FCGR for sequence {sequence[:20]}...: {e}")
            # Return zero matrix in case of error
            size = 2 ** self.k
            return np.zeros((size, size), dtype=float)

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.labels)

    def __getitem__(self, idx):
        """
        Get a single sample from the dataset.

        Args:
            idx: Index of the sample

        Returns:
            Tuple of (mirna_fcgr, mrna_fcgr, label)
        """
        # Get sequences
        mirna_seq = self.mirna_sequences[idx]
        mrna_seq = self.mrna_sequences[idx]
        label = self.labels[idx]

        # Convert RNA to DNA
        mirna_dna = self._rna_to_dna(mirna_seq)
        mrna_dna = self._rna_to_dna(mrna_seq)

        # Generate FCGR
        mirna_fcgr = self._generate_fcgr(mirna_dna)
        mrna_fcgr = self._generate_fcgr(mrna_dna)

        # Convert to torch tensors and add channel dimension
        mirna_tensor = torch.FloatTensor(mirna_fcgr).unsqueeze(0)  # Shape: (1, 64, 64)
        mrna_tensor = torch.FloatTensor(mrna_fcgr).unsqueeze(0)    # Shape: (1, 64, 64)
        label_tensor = torch.LongTensor([label])[0]  # Convert to scalar tensor

        return mirna_tensor, mrna_tensor, label_tensor


def prepare_data_loaders(csv_path: str, k: int = 6, batch_size: int = 64,
                         train_split: float = 0.8, shuffle: bool = True):
    """
    Prepare train and validation data loaders.

    Args:
        csv_path: Path to the CSV file
        k: K-mer size for FCGR generation
        batch_size: Batch size for data loaders
        train_split: Fraction of data to use for training
        shuffle: Whether to shuffle the data

    Returns:
        Tuple of (train_loader, val_loader)
    """
    from torch.utils.data import DataLoader, random_split

    # Create full dataset
    full_dataset = MiRNAInteractionDataset(csv_path=csv_path, k=k, train=True)

    # Split into train and validation
    train_size = int(train_split * len(full_dataset))
    val_size = len(full_dataset) - train_size

    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    logging.info(f"Train dataset size: {len(train_dataset)}")
    logging.info(f"Validation dataset size: {len(val_dataset)}")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,  # Set to 0 for compatibility
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    return train_loader, val_loader
