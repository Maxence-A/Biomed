#!/usr/bin/env python3
"""
Evaluation des modeles entraines - Visualisation de la puissance du modele
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import sys
import torch
import numpy as np
from pathlib import Path
from collections import Counter

# Add project root to path
PROJECT_ROOT = str(Path(__file__).parent.parent.parent)
sys.path.insert(0, PROJECT_ROOT)

from src.models.dataset import (
    ProteinLocalizationDataset, collate_fn, LABEL_ENCODER, LABEL_DECODER, NUM_CLASSES
)
from src.models.improved_fusion import ImprovedFusionNetwork
from sklearn.metrics import (
    accuracy_score, f1_score, matthews_corrcoef,
    classification_report, confusion_matrix
)
from torch.utils.data import DataLoader

# Paths (relative to project root)
DATASET_CSV = "data/processed/dataset_full_1000.csv"
ESM_DIR = "data/embeddings/esm2_650m"
PROST_DIR = "data/embeddings/prostt5"
OUTPUT_DIR = Path("results/checkpoints")

BEST_HPARAMS = {
    'learning_rate': 0.0001,
    'batch_size': 32,
    'dropout': 0.2
}


def load_model(fold_idx):
    """Charge le modele pour un fold donne"""
    model = ImprovedFusionNetwork(
        esm_dim=1280, prost_dim=1024, hidden_dim=512,
        lstm_hidden=256, num_heads=8, num_lstm_layers=2,
        dropout=BEST_HPARAMS['dropout']
    )
    ckpt_path = OUTPUT_DIR / f"fold{fold_idx}_final" / "best_model.pt"
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"  Modele fold {fold_idx} charge (epoch {ckpt['epoch']}, val_mcc={ckpt['val_mcc']:.4f})")
    return model


@torch.no_grad()
def evaluate_fold(model, dataloader):
    """Evalue un modele sur un dataloader"""
    all_preds = []
    all_labels = []
    all_probs = []

    for batch in dataloader:
        esm_emb = batch['esm_embeddings']
        prost_emb = batch['prost_embeddings']
        mask = batch['attention_mask']
        labels = batch['labels']

        logits, _ = model(esm_emb, prost_emb, mask)
        probs = torch.softmax(logits, dim=-1)
        preds = logits.argmax(dim=-1)

        all_preds.extend(preds.numpy())
        all_labels.extend(labels.numpy())
        all_probs.extend(probs.numpy())

    return np.array(all_preds), np.array(all_labels), np.array(all_probs)


def print_confusion_matrix(cm, labels):
    """Affiche la confusion matrix de facon lisible"""
    # Header
    max_label = max(len(l) for l in labels)
    header = " " * (max_label + 2) + "  ".join(f"{l[:6]:>6}" for l in labels)
    print(header)
    print("-" * len(header))

    for i, row in enumerate(cm):
        row_str = f"{labels[i]:<{max_label}}  " + "  ".join(f"{v:>6}" for v in row)
        total = sum(row)
        correct = row[i]
        pct = 100 * correct / total if total > 0 else 0
        print(f"{row_str}  | {pct:.1f}%")


def show_confidence_examples(preds, labels, probs, dataset, n=5):
    """Montre des exemples de predictions avec confiance"""
    # Predictions correctes avec haute confiance
    correct_mask = preds == labels
    confidences = np.max(probs, axis=1)

    print("\n--- Predictions CORRECTES (haute confiance) ---")
    correct_indices = np.where(correct_mask)[0]
    if len(correct_indices) > 0:
        correct_conf = confidences[correct_indices]
        top_correct = correct_indices[np.argsort(-correct_conf)[:n]]
        for idx in top_correct:
            true_label = LABEL_DECODER[labels[idx]]
            conf = confidences[idx]
            print(f"  Proteine #{idx}: {true_label} -> predit {true_label} (confiance: {conf:.1%})")

    # Predictions INCORRECTES
    print(f"\n--- Predictions INCORRECTES (erreurs) ---")
    wrong_indices = np.where(~correct_mask)[0]
    if len(wrong_indices) > 0:
        # Les plus confiantes (le modele se trompe et est sur de lui)
        wrong_conf = confidences[wrong_indices]
        top_wrong = wrong_indices[np.argsort(-wrong_conf)[:n]]
        for idx in top_wrong:
            true_label = LABEL_DECODER[labels[idx]]
            pred_label = LABEL_DECODER[preds[idx]]
            conf = confidences[idx]
            print(f"  Proteine #{idx}: {true_label} -> predit {pred_label} (confiance: {conf:.1%})")
    else:
        print("  Aucune erreur!")


def main():
    os.chdir(PROJECT_ROOT)

    print("=" * 70)
    print("  EVALUATION DES MODELES ENTRAINES")
    print("  Run: best_full_20260126_093236")
    print("  Architecture: ImprovedFusionNetwork (CrossAttn + BiLSTM)")
    print("=" * 70)

    all_preds = []
    all_labels = []
    all_probs = []

    class_names = [LABEL_DECODER[i] for i in range(NUM_CLASSES)]

    for fold_idx in range(5):
        print(f"\n{'='*60}")
        print(f"FOLD {fold_idx}")
        print(f"{'='*60}")

        # Charger modele
        model = load_model(fold_idx)

        # Charger dataset de test (le fold tenu a l'ecart)
        test_dataset = ProteinLocalizationDataset(
            DATASET_CSV, ESM_DIR, PROST_DIR,
            fold_ids=[fold_idx],
            max_length=None
        )
        test_loader = DataLoader(
            test_dataset, batch_size=32, shuffle=False,
            collate_fn=collate_fn, num_workers=0
        )
        print(f"  Test set: {len(test_dataset)} sequences")

        # Evaluer
        preds, labels, probs = evaluate_fold(model, test_loader)

        acc = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds, average='macro', zero_division=0)
        mcc = matthews_corrcoef(labels, preds)

        print(f"\n  Accuracy: {acc:.4f} ({acc:.1%})")
        print(f"  Macro F1: {f1:.4f} ({f1:.1%})")
        print(f"  MCC:      {mcc:.4f}")

        # Exemples
        show_confidence_examples(preds, labels, probs, test_dataset)

        all_preds.extend(preds)
        all_labels.extend(labels)
        all_probs.extend(probs)

        # Liberer memoire
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # =====================================================================
    # RESULTATS GLOBAUX (tous les folds combines)
    # =====================================================================
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    print("\n" + "=" * 70)
    print("  RESULTATS GLOBAUX (5 folds combines = 100% du dataset)")
    print("=" * 70)

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    mcc = matthews_corrcoef(all_labels, all_preds)

    print(f"\n  Total sequences testees: {len(all_labels)}")
    print(f"  Accuracy: {acc:.4f} ({acc:.1%})")
    print(f"  Macro F1: {f1:.4f} ({f1:.1%})")
    print(f"  MCC:      {mcc:.4f}")

    # Distribution des predictions vs realite
    print(f"\n  Distribution des classes:")
    print(f"  {'Classe':<20} {'Reelles':>8} {'Predites':>8} {'Correctes':>10} {'Precision':>10}")
    print(f"  {'-'*60}")
    for i in range(NUM_CLASSES):
        real = (all_labels == i).sum()
        pred = (all_preds == i).sum()
        correct = ((all_labels == i) & (all_preds == i)).sum()
        prec = correct / pred if pred > 0 else 0
        print(f"  {class_names[i]:<20} {real:>8} {pred:>8} {correct:>10} {prec:>9.1%}")

    # Classification report
    print(f"\n  Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names, zero_division=0))

    # Confusion matrix
    print(f"  Confusion Matrix (lignes=reel, colonnes=predit):")
    cm = confusion_matrix(all_labels, all_preds)
    print_confusion_matrix(cm, class_names)

    # Confiance du modele
    confidences = np.max(all_probs, axis=1)
    correct_mask = all_preds == all_labels
    print(f"\n  Analyse de confiance:")
    print(f"  Confiance moyenne (predictions correctes): {confidences[correct_mask].mean():.1%}")
    print(f"  Confiance moyenne (predictions incorrectes): {confidences[~correct_mask].mean():.1%}")

    # Distribution de confiance
    for threshold in [0.5, 0.7, 0.9, 0.95, 0.99]:
        above = confidences >= threshold
        acc_above = accuracy_score(all_labels[above], all_preds[above]) if above.sum() > 0 else 0
        print(f"  Confiance >= {threshold:.0%}: {above.sum():>5} sequences ({above.mean():.1%}), accuracy: {acc_above:.1%}")

    # Comparaison
    print(f"\n{'='*70}")
    print(f"  COMPARAISON AVEC DeepLocPro")
    print(f"{'='*70}")
    print(f"  {'Metrique':<12} {'DeepLocPro':>12} {'Notre modele':>14} {'Difference':>12}")
    print(f"  {'-'*52}")
    print(f"  {'Accuracy':<12} {'0.920':>12} {acc:>14.3f} {acc-0.92:>+12.3f}")
    print(f"  {'Macro F1':<12} {'0.800':>12} {f1:>14.3f} {f1-0.80:>+12.3f}")
    print(f"  {'MCC':<12} {'0.860':>12} {mcc:>14.3f} {mcc-0.86:>+12.3f}")
    print()


if __name__ == "__main__":
    main()
