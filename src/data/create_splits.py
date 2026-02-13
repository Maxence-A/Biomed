"""
Script pour creer les splits homology-based avec GraphPart ou methode alternative
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
import logging

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def compute_sequence_identity(seq1, seq2):
    """
    Calcule l'identite de sequence entre deux sequences
    Methode simple: pourcentage de positions identiques apres alignement

    Args:
        seq1: Premiere sequence
        seq2: Deuxieme sequence

    Returns:
        Pourcentage d'identite (0-100)
    """
    if len(seq1) != len(seq2):
        # Pour simplifier, on utilise la plus courte
        min_len = min(len(seq1), len(seq2))
        seq1 = seq1[:min_len]
        seq2 = seq2[:min_len]

    matches = sum(a == b for a, b in zip(seq1, seq2))
    identity = (matches / len(seq1)) * 100

    return identity


def create_homology_aware_splits(df, n_splits=5, max_identity=30):
    """
    Cree des splits en respectant un seuil max d'identite de sequence

    Methode simplifiee (pour GraphPart, utiliser leur API)

    Args:
        df: DataFrame avec sequences
        n_splits: Nombre de folds
        max_identity: Identite maximale entre train et test

    Returns:
        DataFrame avec colonne 'fold' ajoutee
    """
    logger.info(f"Creation de {n_splits} folds avec max {max_identity}% identite")

    # Pour l'implementation initiale, on utilise StratifiedKFold
    # TODO: Remplacer par GraphPart pour vraie homology-based partition
    logger.info("ATTENTION: Utilisation de StratifiedKFold simple")
    logger.info("TODO: Implementer GraphPart pour partition basee sur homologie")

    # Creer des stratified folds bases sur location + organism_group
    df['strat_key'] = df['location_normalized'] + '_' + df['organism_group']

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    df['fold'] = -1

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(df, df['strat_key'])):
        df.loc[test_idx, 'fold'] = fold_idx

    df = df.drop('strat_key', axis=1)

    return df


def validate_splits(df):
    """
    Valide que les splits sont bien balances

    Args:
        df: DataFrame avec folds
    """
    logger.info("\n=== Validation des Splits ===")

    for fold in sorted(df['fold'].unique()):
        fold_df = df[df['fold'] == fold]
        logger.info(f"\nFold {fold}:")
        logger.info(f"  Total: {len(fold_df)}")

        if 'location_normalized' in fold_df.columns:
            logger.info(f"  Locations: {fold_df['location_normalized'].value_counts().to_dict()}")

        if 'organism_group_normalized' in fold_df.columns:
            logger.info(f"  Organisms: {fold_df['organism_group_normalized'].value_counts().to_dict()}")


def save_splits(df, output_dir):
    """
    Sauvegarde les folds dans des fichiers separes

    Args:
        df: DataFrame avec folds
        output_dir: Repertoire de sortie
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for fold in sorted(df['fold'].unique()):
        fold_df = df[df['fold'] == fold]
        fold_file = output_path / f"fold_{fold}.csv"
        fold_df.to_csv(fold_file, index=False)
        logger.info(f"Fold {fold} sauvegarde: {fold_file}")

    # Sauvegarder aussi le dataset complet avec les folds
    full_file = output_path / "dataset_with_folds.csv"
    df.to_csv(full_file, index=False)
    logger.info(f"Dataset complet sauvegarde: {full_file}")


def main():
    """
    Pipeline principal de creation des splits
    """
    logger.info("Demarrage de la creation des splits")

    # Charger le dataset prepare avec folds deja assignes
    input_file = "data/processed/dataset_with_folds.csv"
    output_dir = "data/processed"

    df = pd.read_csv(input_file)
    logger.info(f"Dataset charge: {len(df)} sequences")

    # Verifier que les folds sont presents
    if 'fold' not in df.columns:
        logger.error("Colonne 'fold' manquante dans le dataset")
        logger.info("Creation de splits avec StratifiedKFold")
        df = create_homology_aware_splits(df, n_splits=5, max_identity=30)
    else:
        logger.info("Utilisation des folds GraphPart pre-assignes")

    # Valider
    validate_splits(df)

    # Sauvegarder
    save_splits(df, output_dir)

    logger.info("Creation des splits terminee")


if __name__ == "__main__":
    main()
