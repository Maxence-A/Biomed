"""
Script pour preparer et nettoyer le dataset de localisation subcellulaire
"""
import pandas as pd
from pathlib import Path
from Bio import SeqIO
import logging

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


LOCATION_MAPPING = {
    'cytoplasm': 'Cytoplasmic',
    'cytoplasmic': 'Cytoplasmic',
    'cytosol': 'Cytoplasmic',
    'membrane': 'Cytoplasmic_membrane',
    'cytoplasmic membrane': 'Cytoplasmic_membrane',
    'cytoplasmicmembrane': 'Cytoplasmic_membrane',
    'inner membrane': 'Cytoplasmic_membrane',
    'plasma membrane': 'Cytoplasmic_membrane',
    'extracellular': 'Extracellular',
    'secreted': 'Extracellular',
    'periplasm': 'Periplasmic',
    'periplasmic': 'Periplasmic',
    'outer membrane': 'Outer_membrane',
    'outermembrane': 'Outer_membrane',
    'cell wall': 'Cell_wall_and_surface',
    'cellwall': 'Cell_wall_and_surface',
    'cell surface': 'Cell_wall_and_surface',
}


ORGANISM_GROUPS = {
    'archaea': 'Archaea',
    'gram_positive': 'Gram_positive',
    'gram_negative': 'Gram_negative',
    'positive': 'Gram_positive',
    'negative': 'Gram_negative',
}


def load_deeplocpro_fasta(fasta_file):
    """
    Charge les donnees DeepLocPro au format FASTA
    Format du header: >ID|Location|organism_group|fold

    Args:
        fasta_file: Chemin vers le fichier FASTA DeepLocPro

    Returns:
        DataFrame avec colonnes: sequence_id, sequence, location, organism_group, fold, source
    """
    logger.info(f"Chargement DeepLocPro depuis {fasta_file}")

    data = []
    for record in SeqIO.parse(fasta_file, "fasta"):
        header_parts = record.id.split("|")

        if len(header_parts) >= 4:
            seq_id = header_parts[0]
            location = header_parts[1]
            organism = header_parts[2]
            fold = int(header_parts[3])

            data.append({
                'sequence_id': seq_id,
                'sequence': str(record.seq),
                'location': location,
                'organism_group': organism,
                'fold': fold,
                'source': 'deeplocpro'
            })

    df = pd.DataFrame(data)
    logger.info(f"DeepLocPro charge: {len(df)} sequences")

    return df


def normalize_location(location):
    """
    Normalise le nom de la location vers une des 6 categories

    Args:
        location: Nom de location brut

    Returns:
        Nom de location normalise ou None si inconnu
    """
    location_lower = location.lower().strip()
    return LOCATION_MAPPING.get(location_lower, None)


def normalize_organism_group(organism):
    """
    Normalise le nom de l'organism group

    Args:
        organism: Nom organism group brut

    Returns:
        Nom organism group normalise ou None si inconnu
    """
    organism_lower = organism.lower().strip()
    return ORGANISM_GROUPS.get(organism_lower, None)


def clean_dataset(df, min_length=40):
    """
    Nettoie le dataset:
    - Supprime sequences trop courtes
    - Supprime multi-labels
    - Normalise les locations et organism groups
    - Supprime les duplicats

    Args:
        df: DataFrame brut
        min_length: Longueur minimale des sequences

    Returns:
        DataFrame nettoye
    """
    logger.info(f"Nettoyage du dataset, taille initiale: {len(df)}")

    # Filtrer sequences trop courtes
    df['seq_length'] = df['sequence'].str.len()
    df = df[df['seq_length'] >= min_length].copy()
    logger.info(f"Apres filtrage longueur >= {min_length}: {len(df)}")

    # Normaliser locations
    df['location_normalized'] = df['location'].apply(normalize_location)
    df = df[df['location_normalized'].notna()]
    logger.info(f"Apres normalisation locations: {len(df)}")

    # Normaliser organism groups
    df['organism_group_normalized'] = df['organism_group'].apply(normalize_organism_group)
    df = df[df['organism_group_normalized'].notna()]
    logger.info(f"Apres normalisation organism groups: {len(df)}")

    # Supprimer duplicats (meme sequence)
    df = df.drop_duplicates(subset=['sequence'])
    logger.info(f"Apres suppression duplicats: {len(df)}")

    return df




def get_dataset_statistics(df):
    """
    Calcule et affiche les statistiques du dataset

    Args:
        df: DataFrame
    """
    logger.info("\n=== Statistiques du Dataset ===")
    logger.info(f"Total sequences: {len(df)}")
    logger.info(f"\nDistribution par location:")
    logger.info(df['location_normalized'].value_counts())
    logger.info(f"\nDistribution par organism group:")
    logger.info(df['organism_group_normalized'].value_counts())
    logger.info(f"\nLongueur sequences:")
    logger.info(f"  Min: {df['seq_length'].min()}")
    logger.info(f"  Max: {df['seq_length'].max()}")
    logger.info(f"  Moyenne: {df['seq_length'].mean():.1f}")
    logger.info(f"  Mediane: {df['seq_length'].median():.1f}")

    if 'fold' in df.columns:
        logger.info(f"\nDistribution par fold:")
        logger.info(df['fold'].value_counts().sort_index())


def save_dataset(df, output_file):
    """
    Sauvegarde le dataset prepare

    Args:
        df: DataFrame a sauvegarder
        output_file: Chemin de sortie
    """
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_file, index=False)
    logger.info(f"Dataset sauvegarde: {output_file}")


def main():
    """
    Pipeline principal de preparation des donnees
    """
    logger.info("Demarrage de la preparation du dataset")

    # Chemins
    deeplocpro_file = "data/raw/deeplocpro_graphpart_set.fasta"
    output_file = "data/processed/dataset_with_folds.csv"

    # Charger les donnees DeepLocPro
    df = load_deeplocpro_fasta(deeplocpro_file)

    # Nettoyer
    df = clean_dataset(df, min_length=40)

    # Statistiques
    get_dataset_statistics(df)

    # Sauvegarder
    save_dataset(df, output_file)

    logger.info("Preparation du dataset terminee")


if __name__ == "__main__":
    main()
