"""
PyTorch Dataset for Protein Subcellular Localization

Loads pre-computed ESM-2 and ProstT5 embeddings
Handles variable-length sequences with padding
"""
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Label encoding for 6 localization classes
LABEL_ENCODER = {
    'Cytoplasmic': 0,
    'Cytoplasmic_membrane': 1,
    'Extracellular': 2,
    'Periplasmic': 3,
    'Outer_membrane': 4,
    'Cell_wall_and_surface': 5
}

LABEL_DECODER = {v: k for k, v in LABEL_ENCODER.items()}
NUM_CLASSES = len(LABEL_ENCODER)


class ProteinLocalizationDataset(Dataset):
    """
    Dataset for protein subcellular localization

    Loads ESM-2 and ProstT5 embeddings for each protein sequence
    """

    def __init__(
        self,
        csv_file: str,
        esm_dir: str = "data/embeddings/esm2",
        prost_dir: str = "data/embeddings/prostt5",
        fold_ids: list = None,
        max_length: int = None
    ):
        """
        Args:
            csv_file: Path to dataset CSV with columns: sequence_id, location_normalized, fold
            esm_dir: Directory containing ESM-2 embeddings (.pt files)
            prost_dir: Directory containing ProstT5 embeddings (.pt files)
            fold_ids: List of fold IDs to include (None = all folds)
            max_length: Maximum sequence length (truncate longer sequences)
        """
        self.esm_dir = Path(esm_dir)
        self.prost_dir = Path(prost_dir)
        self.max_length = max_length

        # Load dataset
        df = pd.read_csv(csv_file)

        # Filter by folds if specified
        if fold_ids is not None:
            df = df[df['fold'].isin(fold_ids)]

        # Filter sequences that have BOTH embeddings
        valid_ids = []
        for seq_id in df['sequence_id']:
            esm_exists = (self.esm_dir / f"{seq_id}.pt").exists()
            prost_exists = (self.prost_dir / f"{seq_id}.pt").exists()
            if esm_exists and prost_exists:
                valid_ids.append(seq_id)

        df = df[df['sequence_id'].isin(valid_ids)]
        self.data = df.reset_index(drop=True)

        logger.info(f"Dataset loaded: {len(self.data)} sequences")
        if fold_ids:
            logger.info(f"  Folds: {fold_ids}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        sequence_id = row['sequence_id']

        # Load embeddings
        esm_emb = torch.load(self.esm_dir / f"{sequence_id}.pt")
        prost_emb = torch.load(self.prost_dir / f"{sequence_id}.pt")

        # Truncate if needed
        if self.max_length is not None:
            esm_emb = esm_emb[:self.max_length]
            prost_emb = prost_emb[:self.max_length]

        # Get label
        label = LABEL_ENCODER[row['location_normalized']]

        return {
            'esm_embeddings': esm_emb,          # (seq_len, 480)
            'prost_embeddings': prost_emb,      # (seq_len, 1024)
            'label': label,
            'sequence_id': sequence_id,
            'seq_length': esm_emb.shape[0]
        }

    def get_class_weights(self):
        """
        Compute inverse frequency class weights for imbalanced classes

        Returns:
            torch.Tensor of shape (num_classes,)
        """
        labels = [LABEL_ENCODER[loc] for loc in self.data['location_normalized']]
        class_counts = torch.bincount(torch.tensor(labels), minlength=NUM_CLASSES)

        # Inverse frequency weighting
        weights = 1.0 / (class_counts.float() + 1e-6)
        weights = weights / weights.sum() * NUM_CLASSES

        return weights

    def get_label_distribution(self):
        """Get label distribution for logging"""
        dist = self.data['location_normalized'].value_counts()
        return dist.to_dict()


def collate_fn(batch):
    """
    Custom collate function for variable-length sequences

    Pads sequences to the same length within a batch
    Creates attention mask for padded positions

    Args:
        batch: List of samples from dataset

    Returns:
        Dictionary with padded tensors and metadata
    """
    # Extract embeddings
    esm_embs = [item['esm_embeddings'] for item in batch]
    prost_embs = [item['prost_embeddings'] for item in batch]
    labels = torch.tensor([item['label'] for item in batch])
    seq_lengths = torch.tensor([item['seq_length'] for item in batch])
    sequence_ids = [item['sequence_id'] for item in batch]

    # Pad sequences (batch_first=True)
    esm_padded = pad_sequence(esm_embs, batch_first=True, padding_value=0)
    prost_padded = pad_sequence(prost_embs, batch_first=True, padding_value=0)

    # Create attention mask (1 for real tokens, 0 for padding)
    max_len = esm_padded.shape[1]
    attention_mask = torch.arange(max_len).unsqueeze(0) < seq_lengths.unsqueeze(1)
    attention_mask = attention_mask.float()

    return {
        'esm_embeddings': esm_padded,       # (batch, max_len, 480)
        'prost_embeddings': prost_padded,   # (batch, max_len, 1024)
        'attention_mask': attention_mask,   # (batch, max_len)
        'labels': labels,                   # (batch,)
        'seq_lengths': seq_lengths,         # (batch,)
        'sequence_ids': sequence_ids        # List[str]
    }


def create_fold_datasets(
    csv_file: str,
    esm_dir: str = "data/embeddings/esm2",
    prost_dir: str = "data/embeddings/prostt5",
    test_fold: int = 0,
    val_fold: int = 1,
    max_length: int = None
):
    """
    Create train/val/test datasets for a specific fold configuration

    Args:
        csv_file: Path to dataset CSV
        esm_dir: ESM-2 embeddings directory
        prost_dir: ProstT5 embeddings directory
        test_fold: Fold to use as test set
        val_fold: Fold to use as validation set
        max_length: Maximum sequence length

    Returns:
        train_dataset, val_dataset, test_dataset
    """
    all_folds = [0, 1, 2, 3, 4]
    train_folds = [f for f in all_folds if f not in [test_fold, val_fold]]

    train_dataset = ProteinLocalizationDataset(
        csv_file, esm_dir, prost_dir,
        fold_ids=train_folds,
        max_length=max_length
    )

    val_dataset = ProteinLocalizationDataset(
        csv_file, esm_dir, prost_dir,
        fold_ids=[val_fold],
        max_length=max_length
    )

    test_dataset = ProteinLocalizationDataset(
        csv_file, esm_dir, prost_dir,
        fold_ids=[test_fold],
        max_length=max_length
    )

    logger.info(f"Train: {len(train_dataset)} (folds {train_folds})")
    logger.info(f"Val: {len(val_dataset)} (fold {val_fold})")
    logger.info(f"Test: {len(test_dataset)} (fold {test_fold})")

    return train_dataset, val_dataset, test_dataset
