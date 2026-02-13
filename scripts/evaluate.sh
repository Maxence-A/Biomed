#!/bin/bash
# =============================================================================
# EVALUATION DES MODELES ENTRAINES
# Charge les 5 modeles et evalue sur chaque fold de test
# =============================================================================

cd "$(dirname "$0")/.."

echo "=============================================================="
echo "EVALUATION DES MODELES"
echo "=============================================================="

# Verifier que les checkpoints existent
if [ ! -f "results/checkpoints/fold0_final/best_model.pt" ]; then
    echo "ERREUR: Checkpoints non trouves dans results/checkpoints/"
    echo "Lancez d'abord l'entrainement avec: bash scripts/launch_training.sh"
    exit 1
fi

python3 src/training/evaluate_models.py
