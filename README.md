# Prediction de Localisation Subcellulaire des Proteines Procaryotes

Prediction de la localisation subcellulaire (6 classes) a partir d'embeddings ESM-2 650M + ProstT5.
Architecture : Bidirectional Cross-Attention + BiLSTM + Multi-Head Attention Pooling.

## Resultats obtenus (run du 26 janvier 2026)

| Metrique | Score | vs DeepLocPro |
|----------|-------|---------------|
| Accuracy | 0.932 +/- 0.004 | +0.012 |
| Macro F1 | 0.819 +/- 0.018 | +0.019 |
| MCC | 0.884 +/- 0.007 | +0.024 |

Voir `RESULTS.md` pour l'analyse complete et `results/` pour les donnees brutes.

---

## Structure du projet

```
biomed/
|-- README.md                              # Ce fichier
|-- RESULTS.md                             # Analyse complete des resultats
|-- requirements.txt                       # Dependances Python
|
|-- src/
|   |-- models/
|   |   |-- improved_fusion.py             # Architecture du modele (ImprovedFusionNetwork)
|   |   |-- dataset.py                     # Dataset PyTorch + collate_fn + LABEL_ENCODER
|   |   |-- extract_esm2_650m.py           # Extraction embeddings ESM-2 650M
|   |   +-- extract_prost_embeddings.py    # Extraction embeddings ProstT5
|   |
|   |-- data/
|   |   |-- prepare_dataset.py             # Preparation du dataset depuis FASTA
|   |   +-- create_splits.py               # Creation des folds
|   |
|   +-- training/
|       |-- train_improved.py              # Script d'entrainement (5-fold CV)
|       +-- evaluate_models.py             # Script d'evaluation des modeles
|
|-- scripts/
|   |-- extract_embeddings.sh              # Lancer l'extraction des embeddings
|   |-- launch_training.sh                 # Lancer l'entrainement
|   +-- evaluate.sh                        # Lancer l'evaluation
|
|-- data/
|   |-- raw/
|   |   +-- deeplocpro_graphpart_set.fasta # Donnees source (11,906 sequences)
|   |
|   |-- processed/
|   |   +-- dataset_full_1000.csv          # Dataset prepare (11,531 seq, <= 1000 AA)
|   |
|   +-- embeddings/                        # DOSSIER VIDE - a remplir (voir etape 2)
|       |-- esm2_650m/                     # Ici: {sequence_id}.pt (dim 1280)
|       +-- prostt5/                       # Ici: {sequence_id}.pt (dim 1024)
|
|-- results/
|   |-- final_summary.json                 # Resume des metriques finales
|   |-- progress.json                      # Progression detaillee de l'entrainement
|   |-- fold_0_results.json                # Resultats fold 0 (predictions, MCC/classe)
|   |-- fold_1_results.json                # Resultats fold 1
|   |-- fold_2_results.json                # Resultats fold 2
|   |-- fold_3_results.json                # Resultats fold 3
|   |-- fold_4_results.json                # Resultats fold 4
|   |-- best_full_output.log               # Log complet d'entrainement (4 MB)
|   +-- checkpoints/                       # Modeles entraines (~500 MB total)
|       |-- fold0_final/best_model.pt      # Checkpoint fold 0 (~97 MB)
|       |-- fold1_final/best_model.pt      # Checkpoint fold 1
|       |-- fold2_final/best_model.pt      # Checkpoint fold 2
|       |-- fold3_final/best_model.pt      # Checkpoint fold 3
|       +-- fold4_final/best_model.pt      # Checkpoint fold 4
|
+-- docs/
    +-- mail.txt                           # Consignes du projet
```

---

## Execution sur une nouvelle machine

### Etape 0 : Installation

```bash
cd biomed/
pip install -r requirements.txt
```

### Etape 1 : Preparation du dataset (deja fait)

Le dataset `data/processed/dataset_full_1000.csv` est deja inclus.
Il a ete cree a partir du FASTA source :

```bash
# Si besoin de regenerer :
python3 src/data/prepare_dataset.py
# Puis filtrer <= 1000 AA :
python3 -c "
import pandas as pd
df = pd.read_csv('data/processed/dataset_with_folds.csv')
df = df[df['seq_length'] <= 1000]
df.to_csv('data/processed/dataset_full_1000.csv', index=False)
print(f'Dataset: {len(df)} sequences')
"
```

**Source des donnees** : Le FASTA (`data/raw/deeplocpro_graphpart_set.fasta`) provient de DeepLocPro.
Format du header FASTA : `>UniProtID|Location|OrganismGroup|FoldNumber`
Les folds sont pre-calcules par **GraphPart** avec 30% d'identite de sequence max.

**Colonnes du CSV** :
- `sequence_id` : ID UniProt (ex: P02968)
- `sequence` : sequence d'acides amines
- `location_normalized` : une des 6 classes
- `fold` : numero du fold (0-4)
- `seq_length` : longueur de la sequence
- `organism_group_normalized` : Gram_negative / Gram_positive / Archaea

### Etape 2 : Extraction des embeddings (OBLIGATOIRE, ~45 GB)

Les embeddings ne sont PAS inclus car trop lourds (45 GB).
Chaque proteine produit un fichier `{sequence_id}.pt` contenant un tensor `(seq_length, dim)`.

```bash
# Avec GPU (rapide, quelques heures) :
bash scripts/extract_embeddings.sh cuda

# Avec CPU (lent, plusieurs jours) :
bash scripts/extract_embeddings.sh cpu
```

Ou separement :

```bash
# ESM-2 650M (1280 dimensions par position, ~25 GB total)
python3 src/models/extract_esm2_650m.py \
    --input data/processed/dataset_full_1000.csv \
    --output_dir data/embeddings/esm2_650m \
    --device cuda \
    --checkpoint_every 100

# ProstT5 (1024 dimensions par position, ~20 GB total)
python3 src/models/extract_prost_embeddings.py \
    --input data/processed/dataset_full_1000.csv \
    --output_dir data/embeddings/prostt5 \
    --device cpu
```

**Resultat attendu** : 11,800 fichiers `.pt` dans chaque dossier d'embeddings.

### Etape 3 : Entrainement

```bash
# Lancement (en arriere-plan recommande) :
nohup bash scripts/launch_training.sh > training.log 2>&1 &

# Suivre la progression :
tail -f training.log
```

**Configuration par defaut** (dans `scripts/launch_training.sh`) :
- 5 folds en parallele, 9 threads/fold
- 50 epochs max, early stopping patience 15 (sur MCC)
- lr=0.0001, batch_size=32, dropout=0.2
- Optimiseur: AdamW, Scheduler: CosineAnnealingWarmRestarts
- Loss: CrossEntropyLoss avec class weights

**Duree estimee** :
- GPU : ~2-4 heures
- CPU : ~7-10 jours

### Etape 4 : Evaluation

```bash
# Evaluer les modeles entraines (checkpoints dans results/checkpoints/) :
bash scripts/evaluate.sh

# Ou directement :
python3 src/training/evaluate_models.py
```

Ce script charge les 5 modeles, evalue sur chaque fold de test, affiche :
- Metriques par fold et globales
- Confusion matrix
- Classification report par classe
- Analyse de confiance
- Comparaison avec DeepLocPro

---

## Pour utiliser les modeles deja entraines (sans re-entrainer)

Les checkpoints sont dans `results/checkpoints/`. Pour evaluer il faut quand meme les embeddings.

```python
import torch
import sys
sys.path.insert(0, '.')
from src.models.improved_fusion import ImprovedFusionNetwork

# Charger le modele du fold 0
model = ImprovedFusionNetwork(
    esm_dim=1280, prost_dim=1024, hidden_dim=512,
    lstm_hidden=256, num_heads=8, num_lstm_layers=2, dropout=0.2
)
ckpt = torch.load('results/checkpoints/fold0_final/best_model.pt',
                   map_location='cpu', weights_only=False)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

# Charger les embeddings d'une proteine
esm = torch.load('data/embeddings/esm2_650m/P02968.pt').unsqueeze(0)
prost = torch.load('data/embeddings/prostt5/P02968.pt').unsqueeze(0)
mask = torch.ones(1, esm.size(1))

with torch.no_grad():
    logits, _ = model(esm, prost, mask)
    pred = logits.argmax(dim=-1).item()

CLASSES = {0:'Cytoplasmic', 1:'Cytoplasmic_membrane', 2:'Extracellular',
           3:'Periplasmic', 4:'Outer_membrane', 5:'Cell_wall_and_surface'}
print(f"Prediction: {CLASSES[pred]}")
```

---

## Embeddings utilises (ce run)

| Embedding | Modele | Dimension | Package |
|-----------|--------|-----------|---------|
| ESM-2 650M | esm2_t33_650M_UR50D | 1280 | `fair-esm` |
| ProstT5 AA | Rostlab/ProstT5 | 1024 | `transformers` |

**Note** : Le sujet demande ESM-C (1152d) + ProstT5 3Di. Ce run utilise ESM-2 + ProstT5 AA
car le serveur avait Python 3.9 (ESM-C requiert 3.12+) et pas de Foldseek (pour 3Di).

---

## Deadline : 13 fevrier 2026
