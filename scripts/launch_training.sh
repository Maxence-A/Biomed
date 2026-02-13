#!/bin/bash
# =============================================================================
# LANCEMENT DE L'ENTRAINEMENT - ImprovedFusionNetwork
# 5-Fold Cross-Validation sur CPU
# =============================================================================

# Se placer a la racine du projet biomed/
cd "$(dirname "$0")/.."

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# =============================================================================
# CONFIGURATION
# =============================================================================

MODEL_TYPE="full"            # full = CrossAttn+BiLSTM, simple = CrossAttn seul
NUM_PARALLEL=5               # 5 folds en parallele
THREADS_PER_FOLD=9           # 9 threads par fold (ajuster selon CPU)
NUM_EPOCHS=50                # Max epochs (early stopping arrete avant)

# Chemins relatifs a la racine du projet
DATASET_CSV="data/processed/dataset_full_1000.csv"
ESM_DIR="data/embeddings/esm2_650m"
PROST_DIR="data/embeddings/prostt5"
OUTPUT_DIR="results/run_${TIMESTAMP}"

# =============================================================================
# VERIFICATION DES DONNEES
# =============================================================================

echo "=============================================================="
echo "ENTRAINEMENT - ImprovedFusionNetwork"
echo "=============================================================="
echo ""
echo "Date: $(date)"
echo ""

# Verifier que les fichiers existent
if [ ! -f "${DATASET_CSV}" ]; then
    echo "ERREUR: Dataset non trouve: ${DATASET_CSV}"
    echo "Lancez d'abord: python src/data/prepare_dataset.py"
    exit 1
fi

ESM_COUNT=$(ls ${ESM_DIR}/*.pt 2>/dev/null | wc -l)
PROST_COUNT=$(ls ${PROST_DIR}/*.pt 2>/dev/null | wc -l)

echo "Verification des donnees:"
echo "  - Dataset: ${DATASET_CSV}"
echo "  - Embeddings ESM-2: ${ESM_COUNT} fichiers dans ${ESM_DIR}"
echo "  - Embeddings ProstT5: ${PROST_COUNT} fichiers dans ${PROST_DIR}"

if [ "${ESM_COUNT}" -eq 0 ]; then
    echo "ERREUR: Aucun embedding ESM-2 trouve dans ${ESM_DIR}/"
    echo "Lancez d'abord: python src/models/extract_esm2_650m.py --input ${DATASET_CSV} --output_dir ${ESM_DIR}"
    exit 1
fi

if [ "${PROST_COUNT}" -eq 0 ]; then
    echo "ERREUR: Aucun embedding ProstT5 trouve dans ${PROST_DIR}/"
    echo "Lancez d'abord: python src/models/extract_prost_embeddings.py --input ${DATASET_CSV} --output_dir ${PROST_DIR}"
    exit 1
fi

echo ""
echo "Configuration:"
echo "  - Model: ImprovedFusionNetwork (${MODEL_TYPE})"
echo "  - Epochs: ${NUM_EPOCHS}"
echo "  - Folds paralleles: ${NUM_PARALLEL}"
echo "  - Threads/fold: ${THREADS_PER_FOLD}"
echo "  - Output: ${OUTPUT_DIR}"
echo ""

# Stats du dataset
python3 -c "
import pandas as pd
df = pd.read_csv('${DATASET_CSV}')
print(f'Dataset: {len(df)} sequences')
print(f'Longueur: min={df[\"seq_length\"].min()}, max={df[\"seq_length\"].max()}, moy={df[\"seq_length\"].mean():.1f}')
print(f'Classes:')
for cls, count in df['location_normalized'].value_counts().items():
    pct = 100 * count / len(df)
    print(f'  {cls}: {count} ({pct:.1f}%)')
"
echo ""
echo "=============================================================="
echo ""

# =============================================================================
# CREATION DOSSIER OUTPUT
# =============================================================================

mkdir -p ${OUTPUT_DIR}

# =============================================================================
# LANCEMENT ENTRAINEMENT
# =============================================================================

python3 src/training/train_improved.py \
    --dataset ${DATASET_CSV} \
    --esm_dir ${ESM_DIR} \
    --prost_dir ${PROST_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --num_epochs ${NUM_EPOCHS} \
    --num_parallel_folds ${NUM_PARALLEL} \
    --threads_per_fold ${THREADS_PER_FOLD} \
    --model_type ${MODEL_TYPE} \
    2>&1

echo ""
echo "=============================================================="
echo "ENTRAINEMENT TERMINE!"
echo "Date fin: $(date)"
echo "Resultats: ${OUTPUT_DIR}"
echo "=============================================================="
