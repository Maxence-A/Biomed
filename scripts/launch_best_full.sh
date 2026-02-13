#!/bin/bash
# =============================================================================
# LANCEMENT DU MEILLEUR MODELE (ImprovedFusion) SUR DATASET COMPLET
# Dataset: 11,531 sequences (<= 1000 AA)
# =============================================================================

cd /data/padawans/e2121u/mom/Biomed---Project

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# =============================================================================
# CONFIGURATION OPTIMISEE
# =============================================================================

MODEL_TYPE="full"
NUM_PARALLEL=5           # 5 folds en parallele
THREADS_PER_FOLD=9       # 9 threads par fold
NUM_EPOCHS=50            # Plus d'epochs (early stopping arretera si besoin)

# Dataset complet
DATASET_CSV="try/dataset_full_1000.csv"

OUTPUT_DIR="try/outputs/best_full_${TIMESTAMP}"

# =============================================================================
# AFFICHAGE INFO
# =============================================================================

echo "=============================================================="
echo "ENTRAINEMENT DU MEILLEUR MODELE - DATASET COMPLET"
echo "=============================================================="
echo ""
echo "Date: $(date)"
echo ""
echo "Configuration:"
echo "  - Dataset: ${DATASET_CSV}"
echo "  - Model: ImprovedFusionNetwork (${MODEL_TYPE})"
echo "  - Epochs: ${NUM_EPOCHS}"
echo "  - Folds paralleles: ${NUM_PARALLEL}"
echo "  - Threads/fold: ${THREADS_PER_FOLD}"
echo "  - Output: ${OUTPUT_DIR}"
echo ""

# Stats du dataset
echo "Dataset stats:"
python3 -c "
import pandas as pd
df = pd.read_csv('${DATASET_CSV}')
print(f'  Total: {len(df)} sequences')
print(f'  Longueur: min={df[\"seq_length\"].min()}, max={df[\"seq_length\"].max()}, moy={df[\"seq_length\"].mean():.1f}')
print(f'  Classes:')
for cls, count in df['location_normalized'].value_counts().items():
    pct = 100 * count / len(df)
    print(f'    {cls}: {count} ({pct:.1f}%)')
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

python try/training/train_improved.py \
    --dataset ${DATASET_CSV} \
    --esm_dir data/embeddings/esm2_650m \
    --prost_dir data/embeddings/prostt5 \
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
