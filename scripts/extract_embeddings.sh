#!/bin/bash
# =============================================================================
# EXTRACTION DES EMBEDDINGS ESM-2 650M ET ProstT5
# A lancer AVANT l'entrainement
# =============================================================================

cd "$(dirname "$0")/.."

DATASET="data/processed/dataset_full_1000.csv"
DEVICE="${1:-cpu}"  # cpu par defaut, passer "cuda" en argument pour GPU

echo "=============================================================="
echo "EXTRACTION DES EMBEDDINGS"
echo "Device: ${DEVICE}"
echo "=============================================================="

# --- Verifier le dataset ---
if [ ! -f "${DATASET}" ]; then
    echo "Dataset non trouve. Creation a partir du FASTA..."
    python3 src/data/prepare_dataset.py
    # Puis filtrer <= 1000 AA
    python3 -c "
import pandas as pd
df = pd.read_csv('data/processed/dataset_with_folds.csv')
df = df[df['seq_length'] <= 1000]
df.to_csv('${DATASET}', index=False)
print(f'Dataset cree: {len(df)} sequences (<= 1000 AA)')
"
fi

# --- Extraction ESM-2 650M ---
echo ""
echo "=== ETAPE 1/2: Extraction ESM-2 650M ==="
echo "Dimension: 1280 par position"
echo "Sortie: data/embeddings/esm2_650m/"
echo ""

python3 src/models/extract_esm2_650m.py \
    --input ${DATASET} \
    --output_dir data/embeddings/esm2_650m \
    --device ${DEVICE} \
    --checkpoint_every 100

echo ""
echo "=== ETAPE 2/2: Extraction ProstT5 ==="
echo "Dimension: 1024 par position"
echo "Sortie: data/embeddings/prostt5/"
echo ""

python3 src/models/extract_prost_embeddings.py \
    --input ${DATASET} \
    --output_dir data/embeddings/prostt5 \
    --device ${DEVICE}

echo ""
echo "=============================================================="
echo "EXTRACTION TERMINEE!"
echo "ESM-2: $(ls data/embeddings/esm2_650m/*.pt 2>/dev/null | wc -l) fichiers"
echo "ProstT5: $(ls data/embeddings/prostt5/*.pt 2>/dev/null | wc -l) fichiers"
echo "=============================================================="
