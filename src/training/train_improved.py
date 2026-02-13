"""
Training script for Improved Fusion Network

Architecture: Bidirectional Cross-Attention + BiLSTM + Multi-Head Attention Pooling

Based on state-of-the-art 2024-2025:
- LocPro 2025: ESM2 + BiLSTM (+10% F1)
- BioLangFusion: Cross-modal multi-head attention
- HEAL: Hierarchical attention pooling
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report
)
import logging
import argparse
from pathlib import Path
import json
from datetime import datetime
from multiprocessing import Pool, cpu_count, get_context
import signal
import time
import fcntl
from tqdm import tqdm

def init_worker_folds():
    signal.signal(signal.SIGINT, signal.SIG_IGN)

import sys
# Add project root to path
PROJECT_ROOT = str(Path(__file__).parent.parent.parent)
sys.path.insert(0, PROJECT_ROOT)

from src.models.dataset import (
    ProteinLocalizationDataset,
    collate_fn,
    LABEL_ENCODER,
    LABEL_DECODER,
    NUM_CLASSES
)

from src.models.improved_fusion import ImprovedFusionNetwork, SimpleCrossAttentionNetwork

logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(asctime)s - %(processName)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Hyperparameters for improved model
# Slightly adjusted for the more complex architecture
BEST_HPARAMS = {
    'learning_rate': 0.0001,
    'batch_size': 32,
    'dropout': 0.2
}


def compute_per_class_mcc(y_true, y_pred, num_classes=6):
    mcc_per_class = []
    for c in range(num_classes):
        binary_true = (np.array(y_true) == c).astype(int)
        binary_pred = (np.array(y_pred) == c).astype(int)
        if binary_true.sum() == 0 or binary_pred.sum() == 0:
            mcc_per_class.append(0.0)
        else:
            mcc = matthews_corrcoef(binary_true, binary_pred)
            mcc_per_class.append(mcc)
    return mcc_per_class


def is_fold_completed(output_dir: Path, test_fold: int):
    """Check if fold has completed testing (results file exists)"""
    fold_result_file = output_dir / f'fold_{test_fold}_results.json'
    if fold_result_file.exists():
        try:
            with open(fold_result_file, 'r') as f:
                results = json.load(f)
            if 'test_accuracy' in results and 'test_f1' in results:
                return True
        except:
            pass
    return False


def is_training_done_but_not_tested(output_dir: Path, test_fold: int):
    """Check if training is done (best_model.pt exists) but testing isn't (no results.json)"""
    # Check both possible folder naming conventions
    fold_dir_1 = output_dir / f'fold_{test_fold}'
    fold_dir_2 = output_dir / f'fold{test_fold}_final'

    best_model_path_1 = fold_dir_1 / 'best_model.pt'
    best_model_path_2 = fold_dir_2 / 'best_model.pt'
    fold_result_file = output_dir / f'fold_{test_fold}_results.json'

    training_done = best_model_path_1.exists() or best_model_path_2.exists()
    testing_done = fold_result_file.exists()

    return training_done and not testing_done


def save_fold_results(output_dir: Path, test_fold: int, fold_results: dict):
    fold_result_file = output_dir / f'fold_{test_fold}_results.json'
    with open(fold_result_file, 'w') as f:
        json.dump(fold_results, f, indent=2)


def load_completed_folds(output_dir: Path):
    completed_results = []
    for fold_idx in range(5):
        fold_result_file = output_dir / f'fold_{fold_idx}_results.json'
        if fold_result_file.exists():
            try:
                with open(fold_result_file, 'r') as f:
                    results = json.load(f)
                completed_results.append(results)
            except:
                pass
    return completed_results


def update_progress_file(output_dir: Path, fold_id: int, phase: str, current: int = 0,
                         total: int = 0, details: str = "", metrics: dict = None):
    progress_file = output_dir / 'progress.json'
    try:
        with open(progress_file, 'r') as f:
            progress = json.load(f)
    except:
        progress = {'folds': {}, 'status': 'running'}

    fold_key = f'fold_{fold_id}'
    if fold_key not in progress.get('folds', {}):
        progress['folds'] = progress.get('folds', {})
        progress['folds'][fold_key] = {
            'start_time': datetime.now().isoformat(),
            'status': 'in_progress'
        }

    fold_data = progress['folds'][fold_key]
    fold_data['phase'] = phase
    fold_data['current'] = current
    fold_data['total'] = total
    fold_data['progress_pct'] = round(current / total * 100, 1) if total > 0 else 0
    fold_data['details'] = details
    fold_data['last_update'] = datetime.now().isoformat()
    if metrics:
        fold_data['metrics'] = metrics

    completed_folds = sum(1 for f in progress['folds'].values() if f.get('status') == 'completed')
    in_progress = sum(f.get('progress_pct', 0) / 100 for f in progress['folds'].values()
                      if f.get('status') == 'in_progress')
    progress['overall_progress_pct'] = round((completed_folds + in_progress) / 5 * 100, 1)
    progress['completed_folds'] = completed_folds

    try:
        with open(progress_file, 'w') as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            json.dump(progress, f, indent=2)
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    except:
        pass


class ImprovedTrainer:
    """Trainer for Improved Fusion Network"""

    def __init__(self, model, train_dataset, val_dataset, config):
        self.model = model
        self.config = config

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config.get('batch_size', 32),
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=0,
            pin_memory=False,
            drop_last=True
        )

        self.val_loader = DataLoader(
            val_dataset,
            batch_size=config.get('batch_size', 32),
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0,
            pin_memory=False
        )

        if config.get('use_class_weights', True):
            class_weights = train_dataset.get_class_weights()
            self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        else:
            self.criterion = nn.CrossEntropyLoss()

        # Use AdamW for better regularization with complex model
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.get('learning_rate', 0.0001),
            weight_decay=config.get('weight_decay', 1e-4),
            betas=(0.9, 0.999)
        )

        # Cosine annealing scheduler - often better for transformers
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2, eta_min=1e-6
        )

        self.early_stopping_patience = config.get('early_stopping_patience', 15)
        self.best_val_f1 = 0
        self.best_val_mcc = 0
        self.patience_counter = 0

        self.output_dir = Path(config.get('output_dir', 'outputs'))
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def train_epoch(self, fold_id=0, epoch_num=0, total_epochs=30):
        self.model.train()
        total_loss = 0
        all_preds, all_labels = [], []

        pbar = tqdm(
            self.train_loader,
            desc=f"[Fold {fold_id}] Epoch {epoch_num+1:2d}/{total_epochs}",
            leave=False,
            ncols=100,
            bar_format='{desc}: {percentage:3.0f}%|{bar:20}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
        )
        for batch in pbar:
            esm_emb = batch['esm_embeddings']
            prost_emb = batch['prost_embeddings']
            mask = batch['attention_mask']
            labels = batch['labels']

            self.optimizer.zero_grad()
            logits, _ = self.model(esm_emb, prost_emb, mask)
            loss = self.criterion(logits, labels)
            loss.backward()

            # Gradient clipping - important for transformers
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            total_loss += loss.item()
            preds = logits.argmax(dim=-1).numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

        avg_loss = total_loss / len(self.train_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='macro')
        return avg_loss, accuracy, f1

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        total_loss = 0
        all_preds, all_labels, all_probs = [], [], []

        for batch in self.val_loader:
            esm_emb = batch['esm_embeddings']
            prost_emb = batch['prost_embeddings']
            mask = batch['attention_mask']
            labels = batch['labels']

            logits, _ = self.model(esm_emb, prost_emb, mask)
            loss = self.criterion(logits, labels)

            total_loss += loss.item()
            probs = torch.softmax(logits, dim=-1)
            preds = logits.argmax(dim=-1).numpy()

            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.numpy())

        avg_loss = total_loss / len(self.val_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='macro')
        mcc = matthews_corrcoef(all_labels, all_preds)

        return avg_loss, accuracy, f1, mcc, np.array(all_preds), np.array(all_labels), np.array(all_probs)

    def save_checkpoint(self, epoch, val_f1, val_mcc, is_best=False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_f1': val_f1,
            'val_mcc': val_mcc,
            'config': self.config
        }
        if is_best:
            torch.save(checkpoint, self.output_dir / 'best_model.pt')

    def load_checkpoint(self, path):
        # weights_only=False needed for PyTorch 2.6+ compatibility with numpy objects
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        return checkpoint


def run_single_fold_improved(args):
    """Run training for a single fold with Improved Fusion Network"""
    test_fold, csv_file, esm_dir, prost_dir, output_dir, config = args

    num_threads = config.get('threads_per_fold', 12)
    torch.set_num_threads(num_threads)
    try:
        torch.set_num_interop_threads(num_threads)
    except RuntimeError:
        pass

    output_dir = Path(output_dir)

    if is_fold_completed(output_dir, test_fold):
        logger.info(f"[FOLD {test_fold}] Already completed, skipping...")
        return None

    # Check if training is done but testing failed (resume testing only)
    resume_testing_only = is_training_done_but_not_tested(output_dir, test_fold)
    if resume_testing_only:
        logger.info(f"[FOLD {test_fold}] Training done, resuming testing phase...")

    model_type = config.get('model_type', 'full')  # 'full' or 'simple'

    logger.info(f"")
    logger.info(f"{'='*60}")
    logger.info(f"[FOLD {test_fold}] IMPROVED FUSION - {model_type.upper()} MODEL")
    if resume_testing_only:
        logger.info(f"[FOLD {test_fold}] RESUMING TESTING ONLY (training already done)")
    logger.info(f"[FOLD {test_fold}] Using {num_threads} threads")
    logger.info(f"[FOLD {test_fold}] Hparams: lr={BEST_HPARAMS['learning_rate']}, "
                f"bs={BEST_HPARAMS['batch_size']}, do={BEST_HPARAMS['dropout']}")
    logger.info(f"{'='*60}")

    inner_folds = [f for f in range(5) if f != test_fold]
    train_folds = inner_folds[:-1]
    val_fold = inner_folds[-1]

    logger.info(f"[FOLD {test_fold}] Train folds: {train_folds}, Val fold: {val_fold}, Test fold: {test_fold}")

    train_dataset = ProteinLocalizationDataset(
        csv_file, esm_dir, prost_dir,
        fold_ids=train_folds,
        max_length=config.get('max_length')
    )
    val_dataset = ProteinLocalizationDataset(
        csv_file, esm_dir, prost_dir,
        fold_ids=[val_fold],
        max_length=config.get('max_length')
    )
    test_dataset = ProteinLocalizationDataset(
        csv_file, esm_dir, prost_dir,
        fold_ids=[test_fold],
        max_length=config.get('max_length')
    )

    logger.info(f"[FOLD {test_fold}] Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    num_epochs = config.get('num_epochs', 30)

    train_config = {
        **config,
        **BEST_HPARAMS,
        'output_dir': str(output_dir / f'fold{test_fold}_final'),
        'num_threads': num_threads
    }

    # Create model based on type
    if model_type == 'simple':
        model = SimpleCrossAttentionNetwork(
            esm_dim=config.get('esm_dim', 1280),
            prost_dim=config.get('prost_dim', 1024),
            hidden_dim=config.get('hidden_dim', 512),
            num_heads=config.get('num_heads', 8),
            dropout=BEST_HPARAMS['dropout']
        )
        logger.info(f"[FOLD {test_fold}] Using SimpleCrossAttentionNetwork (no BiLSTM)")
    else:
        model = ImprovedFusionNetwork(
            esm_dim=config.get('esm_dim', 1280),
            prost_dim=config.get('prost_dim', 1024),
            hidden_dim=config.get('hidden_dim', 512),
            lstm_hidden=config.get('lstm_hidden', 256),
            num_heads=config.get('num_heads', 8),
            num_lstm_layers=config.get('num_lstm_layers', 2),
            dropout=BEST_HPARAMS['dropout']
        )
        logger.info(f"[FOLD {test_fold}] Using ImprovedFusionNetwork (Cross-Attn + BiLSTM)")

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"[FOLD {test_fold}] Model parameters: {num_params:,}")

    trainer = ImprovedTrainer(model, train_dataset, val_dataset, train_config)

    # Skip training if resuming testing only
    if not resume_testing_only:
        logger.info(f"[FOLD {test_fold}] Training for up to {num_epochs} epochs...")

        for epoch in range(num_epochs):
            train_loss, train_acc, train_f1 = trainer.train_epoch(
                fold_id=test_fold, epoch_num=epoch, total_epochs=num_epochs
            )
            val_loss, val_acc, val_f1, val_mcc, _, _, _ = trainer.validate()

            # Step scheduler
            trainer.scheduler.step()

            update_progress_file(
                output_dir, test_fold, 'training',
                current=epoch + 1, total=num_epochs,
                details=f"Epoch {epoch+1}/{num_epochs}",
                metrics={'val_f1': round(val_f1, 4), 'val_acc': round(val_acc, 4), 'val_mcc': round(val_mcc, 4)}
            )

            # Print epoch summary
            print(f"[Fold {test_fold}] Epoch {epoch+1:2d}/{num_epochs} | "
                  f"Train: loss={train_loss:.4f} F1={train_f1:.4f} | "
                  f"Val: loss={val_loss:.4f} F1={val_f1:.4f} MCC={val_mcc:.4f}")

            # Save best model based on MCC (more robust for imbalanced data)
            if val_mcc > trainer.best_val_mcc:
                trainer.best_val_f1 = val_f1
                trainer.best_val_mcc = val_mcc
                trainer.patience_counter = 0
                trainer.save_checkpoint(epoch, val_f1, val_mcc, is_best=True)
                print(f"[Fold {test_fold}] *** New best MCC: {val_mcc:.4f} ***")
            else:
                trainer.patience_counter += 1
                if trainer.patience_counter >= trainer.early_stopping_patience:
                    logger.info(f"[FOLD {test_fold}] Early stopping at epoch {epoch+1}")
                    break
    else:
        logger.info(f"[FOLD {test_fold}] Skipping training (checkpoint exists)")

    # Testing phase
    logger.info(f"[FOLD {test_fold}] Testing on held-out fold...")

    trainer.load_checkpoint(Path(train_config['output_dir']) / 'best_model.pt')

    test_loader = DataLoader(
        test_dataset,
        batch_size=BEST_HPARAMS['batch_size'],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )
    trainer.val_loader = test_loader

    test_loss, test_acc, test_f1, test_mcc, preds, labels, probs = trainer.validate()
    mcc_per_class = compute_per_class_mcc(labels, preds)

    fold_results = {
        'test_fold': test_fold,
        'best_hparams': BEST_HPARAMS,
        'test_accuracy': float(test_acc),
        'test_f1': float(test_f1),
        'test_mcc': float(test_mcc),
        'mcc_per_class': [float(m) for m in mcc_per_class],
        'predictions': preds.tolist(),
        'labels': labels.tolist(),
        'config': {
            'esm_dim': config.get('esm_dim', 1280),
            'prost_dim': config.get('prost_dim', 1024),
            'model_type': model_type,
            'architecture': 'ImprovedFusionNetwork' if model_type == 'full' else 'SimpleCrossAttentionNetwork'
        }
    }

    # Mark completed
    try:
        progress_file = output_dir / 'progress.json'
        with open(progress_file, 'r') as f:
            progress = json.load(f)
        fold_key = f'fold_{test_fold}'
        if fold_key in progress.get('folds', {}):
            progress['folds'][fold_key]['status'] = 'completed'
            progress['folds'][fold_key]['end_time'] = datetime.now().isoformat()
            progress['folds'][fold_key]['metrics'] = {
                'test_acc': round(test_acc, 4),
                'test_f1': round(test_f1, 4),
                'test_mcc': round(test_mcc, 4)
            }
        progress['completed_folds'] = sum(1 for f in progress['folds'].values() if f.get('status') == 'completed')
        with open(progress_file, 'w') as f:
            json.dump(progress, f, indent=2)
    except:
        pass

    logger.info(f"")
    logger.info(f"{'='*60}")
    logger.info(f"[FOLD {test_fold}] COMPLETED!")
    logger.info(f"[FOLD {test_fold}] Results: Acc={test_acc:.4f}, F1={test_f1:.4f}, MCC={test_mcc:.4f}")
    logger.info(f"{'='*60}")

    save_fold_results(output_dir, test_fold, fold_results)
    return fold_results


def aggregate_results(outer_results, output_dir, logger, model_type='full'):
    accuracies = [r['test_accuracy'] for r in outer_results]
    f1_scores = [r['test_f1'] for r in outer_results]
    mccs = [r['test_mcc'] for r in outer_results]

    logger.info(f"\n{'='*60}")
    logger.info(f"FINAL RESULTS - IMPROVED FUSION ({model_type.upper()})")
    logger.info(f"{'='*60}")
    logger.info(f"Accuracy: {np.mean(accuracies):.4f} +/- {np.std(accuracies):.4f}")
    logger.info(f"Macro F1: {np.mean(f1_scores):.4f} +/- {np.std(f1_scores):.4f}")
    logger.info(f"MCC: {np.mean(mccs):.4f} +/- {np.std(mccs):.4f}")

    logger.info("\nPer-class MCC:")
    mcc_per_class_all = np.array([r['mcc_per_class'] for r in outer_results])
    for i, class_name in LABEL_DECODER.items():
        mean_mcc = np.mean(mcc_per_class_all[:, i])
        std_mcc = np.std(mcc_per_class_all[:, i])
        logger.info(f"  {class_name}: {mean_mcc:.4f} +/- {std_mcc:.4f}")

    # Comparison with baseline
    logger.info(f"\n{'='*60}")
    logger.info("COMPARISON WITH BASELINES:")
    logger.info(f"{'='*60}")
    logger.info(f"DeepLocPro (paper):     F1=0.80, MCC=0.86")
    logger.info(f"Our Gated Fusion:       F1=0.76, MCC=0.82")
    logger.info(f"IMPROVED ({model_type}):      F1={np.mean(f1_scores):.2f}, MCC={np.mean(mccs):.2f}")

    f1_improvement = (np.mean(f1_scores) - 0.76) / 0.76 * 100
    mcc_improvement = (np.mean(mccs) - 0.82) / 0.82 * 100
    logger.info(f"\nImprovement vs Gated Fusion: F1={f1_improvement:+.1f}%, MCC={mcc_improvement:+.1f}%")

    all_predictions = []
    all_labels = []
    for r in outer_results:
        all_predictions.extend(r['predictions'])
        all_labels.extend(r['labels'])

    cm = confusion_matrix(np.array(all_labels), np.array(all_predictions))
    logger.info("\nConfusion Matrix:")
    logger.info(str(cm))

    report = classification_report(
        np.array(all_labels), np.array(all_predictions),
        target_names=list(LABEL_ENCODER.keys())
    )
    logger.info("\nClassification Report:")
    logger.info(report)

    final_results = {
        'nested_cv_results': {
            'accuracy': {
                'mean': float(np.mean(accuracies)),
                'std': float(np.std(accuracies)),
                'per_fold': [float(a) for a in accuracies]
            },
            'macro_f1': {
                'mean': float(np.mean(f1_scores)),
                'std': float(np.std(f1_scores)),
                'per_fold': [float(f) for f in f1_scores]
            },
            'mcc': {
                'mean': float(np.mean(mccs)),
                'std': float(np.std(mccs)),
                'per_fold': [float(m) for m in mccs]
            }
        },
        'best_hparams_per_fold': [r['best_hparams'] for r in outer_results],
        'model_info': {
            'esm_model': 'ESM-2 650M (1280-D)',
            'prost_model': 'ProstT5 (1024-D)',
            'architecture': model_type,
            'fusion': 'BidirectionalCrossAttention',
            'sequence_encoder': 'BiLSTM' if model_type == 'full' else 'None',
            'pooling': 'MultiHeadAttentionPooling',
            'partitioning': 'GraphPart 30% identity'
        },
        'comparison': {
            'deeplocpro': {'f1': 0.80, 'mcc': 0.86},
            'our_gated_fusion': {'f1': 0.76, 'mcc': 0.82},
            'our_improved': {'f1': float(np.mean(f1_scores)), 'mcc': float(np.mean(mccs))}
        }
    }

    with open(output_dir / 'final_summary.json', 'w') as f:
        json.dump(final_results, f, indent=2)

    # Update progress file
    try:
        progress_file = output_dir / 'progress.json'
        with open(progress_file, 'r') as f:
            progress = json.load(f)
        progress['status'] = 'completed'
        progress['end_time'] = datetime.now().isoformat()
        progress['overall_progress_pct'] = 100.0
        progress['final_summary'] = final_results['nested_cv_results']
        with open(progress_file, 'w') as f:
            json.dump(progress, f, indent=2)
    except:
        pass

    logger.info(f"\nResults saved to {output_dir / 'final_summary.json'}")


def main():
    parser = argparse.ArgumentParser(description='Train Improved Fusion Network (Cross-Attn + BiLSTM)')
    parser.add_argument('--dataset', type=str, default='data/processed/dataset_with_folds.csv')
    parser.add_argument('--esm_dir', type=str, default='data/embeddings/esm2_650m')
    parser.add_argument('--prost_dir', type=str, default='data/embeddings/prostt5')
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--num_epochs', type=int, default=30)
    parser.add_argument('--num_parallel_folds', type=int, default=5)
    parser.add_argument('--threads_per_fold', type=int, default=9)
    parser.add_argument('--model_type', type=str, default='full', choices=['full', 'simple'],
                        help='full=CrossAttn+BiLSTM, simple=CrossAttn only')
    parser.add_argument('--max_length', type=int, default=None)
    args = parser.parse_args()

    total_cpus = cpu_count()
    logger.info(f"System has {total_cpus} CPU threads available")
    logger.info(f"Configuration: {args.num_parallel_folds} parallel folds x {args.threads_per_fold} threads")
    logger.info(f"")
    logger.info(f"{'='*60}")
    logger.info(f"IMPROVED FUSION NETWORK TRAINING")
    logger.info(f"Model type: {args.model_type}")
    if args.model_type == 'full':
        logger.info(f"Architecture: Bidirectional Cross-Attention + BiLSTM + Multi-Head Pool")
    else:
        logger.info(f"Architecture: Bidirectional Cross-Attention + Multi-Head Pool")
    logger.info(f"{'='*60}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize progress file
    progress_file = output_dir / 'progress.json'
    initial_progress = {
        'start_time': datetime.now().isoformat(),
        'total_folds': 5,
        'model_type': args.model_type,
        'status': 'running',
        'folds': {},
        'completed_folds': 0,
        'overall_progress_pct': 0.0,
        'config': {
            'num_parallel_folds': args.num_parallel_folds,
            'threads_per_fold': args.threads_per_fold,
            'num_epochs': args.num_epochs,
            'hparams': BEST_HPARAMS
        }
    }
    with open(progress_file, 'w') as f:
        json.dump(initial_progress, f, indent=2)

    completed_results = load_completed_folds(output_dir)
    completed_fold_ids = [r['test_fold'] for r in completed_results]

    logger.info(f"ESM-2 650M (dim=1280)")
    logger.info(f"ProstT5 (dim=1024)")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Completed folds: {completed_fold_ids}")

    folds_to_compute = [f for f in range(5) if f not in completed_fold_ids]
    logger.info(f"Folds to compute: {folds_to_compute}")

    if not folds_to_compute:
        logger.info("All folds already completed!")
        if len(completed_results) == 5:
            aggregate_results(completed_results, output_dir, logger, args.model_type)
        return

    config = {
        'num_epochs': args.num_epochs,
        'use_class_weights': True,
        'early_stopping_patience': 15,
        'max_length': args.max_length,
        'esm_dim': 1280,
        'prost_dim': 1024,
        'hidden_dim': 512,
        'lstm_hidden': 256,
        'num_heads': 8,
        'num_lstm_layers': 2,
        'threads_per_fold': args.threads_per_fold,
        'model_type': args.model_type
    }

    fold_args = [
        (fold, args.dataset, args.esm_dir, args.prost_dir, str(output_dir), config)
        for fold in folds_to_compute
    ]

    logger.info(f"Running {len(folds_to_compute)} folds in PARALLEL with {args.num_parallel_folds} processes")

    ctx = get_context('spawn')
    with ctx.Pool(processes=args.num_parallel_folds, initializer=init_worker_folds) as pool:
        try:
            results = pool.map(run_single_fold_improved, fold_args)
            for result in results:
                if result:
                    completed_results.append(result)
        except KeyboardInterrupt:
            logger.warning("Interrupted!")
            pool.terminate()
            pool.join()

    if len(completed_results) == 5:
        logger.info("\n" + "="*60)
        logger.info("ALL FOLDS COMPLETE - COMPUTING FINAL RESULTS")
        logger.info("="*60)
        aggregate_results(completed_results, output_dir, logger, args.model_type)

    logger.info("Done!")


if __name__ == "__main__":
    main()
