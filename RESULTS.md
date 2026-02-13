# Resultats - Prediction de Localisation Subcellulaire des Proteines

**Projet** : Biomed Project - Prediction de la localisation subcellulaire des proteines procaryotes
**Run** : `best_full_20260126_093236`
**Date** : 26 janvier 2026 → 5 fevrier 2026 (10 jours sur CPU)
**Statut** : COMPLETE

---

## 1. Objectif

Predire la localisation subcellulaire d'une proteine procaryote (6 classes) a partir de sa sequence d'acides amines, en combinant deux types d'embeddings pre-entraines :
- **ESM-2 650M** : information sequentielle (1280 dimensions)
- **ProstT5** : information structurale (1024 dimensions)

---

## 2. Dataset

**Fichier** : `try/dataset_full_1000.csv` (5.5 MB)

| Info | Valeur |
|------|--------|
| Total sequences | 11,531 |
| Longueur min | 40 AA |
| Longueur max | 999 AA |
| Longueur moyenne | 417.7 AA |
| Longueur mediane | 397 AA |
| Source | DeepLocPro |
| Partitionnement | GraphPart a 30% d'identite de sequence |

### Distribution des classes

| Classe | Nombre | Pourcentage |
|--------|--------|-------------|
| Cytoplasmic | 6,788 | 58.9% |
| Cytoplasmic_membrane | 2,455 | 21.3% |
| Extracellular | 954 | 8.3% |
| Outer_membrane | 717 | 6.2% |
| Periplasmic | 557 | 4.8% |
| Cell_wall_and_surface | 60 | 0.5% |

Le dataset est desequilibre : Cytoplasmic represente 59% des donnees, Cell_wall_and_surface seulement 0.5%.

### Repartition des folds (5-fold CV)

| Fold | Sequences |
|------|-----------|
| Fold 0 | 2,307 |
| Fold 1 | 2,306 |
| Fold 2 | 2,306 |
| Fold 3 | 2,306 |
| Fold 4 | 2,306 |

Les folds sont crees par **GraphPart** avec un seuil de 30% d'identite de sequence, ce qui garantit qu'aucune proteine similaire ne se retrouve dans le train ET le test (evite le data leakage).

### Colonnes du CSV

- `sequence_id` : Identifiant UniProt (ex: A0A0C5CJR8)
- `sequence` : Sequence d'acides amines
- `location` : Localisation originale
- `location_normalized` : Localisation normalisee (6 classes)
- `organism_group` : Groupe d'organisme
- `organism_group_normalized` : Gram_negative / Gram_positive
- `fold` : Numero de fold (0-4)
- `seq_length` : Longueur de la sequence
- `source` : Source des donnees (deeplocpro)

---

## 3. Embeddings

### ESM-2 650M

| Info | Valeur |
|------|--------|
| Modele | `esm2_t33_650M_UR50D` (Meta/Facebook) |
| Parametres | 650 millions |
| Dimension | 1280 par position |
| Dossier | `data/embeddings/esm2_650m/` |
| Fichiers | 11,800 fichiers .pt |
| Taille totale | 25 GB |
| Format | Un fichier `{sequence_id}.pt` par proteine |
| Contenu | Tensor de shape `(seq_length, 1280)` |

Script d'extraction : `src/models/extract_esm2_650m.py`

### ProstT5

| Info | Valeur |
|------|--------|
| Modele | ProstT5 (Heinzinger et al.) |
| Mode | AA (acides amines, pas 3Di) |
| Dimension | 1024 par position |
| Dossier | `data/embeddings/prostt5/` |
| Fichiers | 11,800 fichiers .pt |
| Taille totale | 20 GB |
| Format | Un fichier `{sequence_id}.pt` par proteine |
| Contenu | Tensor de shape `(seq_length, 1024)` |

Script d'extraction : `src/models/extract_prost_embeddings.py`

**Note** : ProstT5 est utilise en mode AA (acides amines) et non en mode 3Di (sequences structurales) car Foldseek/ESMFold ne sont pas disponibles sur le serveur. Le mode AA capture neanmoins l'information structurale apprise lors du pre-entrainement.

---

## 4. Architecture du modele : ImprovedFusionNetwork

### Vue d'ensemble

```
ESM-2 (batch, seq_len, 1280) ---+
                                 |---> [1] BidirectionalCrossAttention (batch, seq_len, 512)
ProstT5 (batch, seq_len, 1024) -+           |
                                             v
                                  [2] BiLSTM 2 couches (batch, seq_len, 512)
                                             |
                                             v
                                  [3] MultiHeadAttentionPooling (batch, 512)
                                             |
                                             v
                                  [4] ClassificationHead (batch, 6)
```

### Detail des composants

#### 4.1 BidirectionalCrossAttention (4,071,426 parametres)

Fusion bidirectionnelle entre ESM-2 et ProstT5 :

1. **Projection** : ESM (1280 -> 512) et ProstT5 (1024 -> 512) vers un espace commun
2. **Direction 1** : ESM query sur ProstT5 key/value → enrichit la sequence avec l'info structurale
3. **Direction 2** : ProstT5 query sur ESM key/value → enrichit la structure avec l'info sequentielle
4. **Connexions residuelles** : ajoutees aux deux directions
5. **Gate appris** : un reseau apprend a ponderer les deux directions (softmax sur 2 poids)
6. **Projection de sortie** : Linear + LayerNorm + ReLU + Dropout

Configuration : 8 tetes d'attention, dimension 512

#### 4.2 BiLSTM Encoder (3,154,944 parametres)

Capture les dependances sequentielles le long de la proteine :

- 2 couches de LSTM bidirectionnel
- Hidden size : 256 par direction (512 total)
- `pack_padded_sequence` pour gerer les longueurs variables efficacement
- LayerNorm en sortie pour stabiliser l'entrainement

Inspire de **LocPro 2025** qui a obtenu +10% F1 avec cette approche.

#### 4.3 MultiHeadAttentionPooling (789,504 parametres)

Agregation de la sequence variable en un vecteur fixe :

- 4 tetes d'attention avec un **query learnable** (comme un token CLS de BERT)
- Chaque tete apprend a "regarder" un aspect different de la sequence
- Concatenation des 4 tetes → vecteur de 512 dimensions
- Linear + LayerNorm + Dropout en sortie

Inspire de **HEAL** (Hierarchical Attention Pooling).

#### 4.4 ClassificationHead (397,062 parametres)

MLP a 2 couches :

```
512 -> LayerNorm -> ReLU -> Dropout -> 256 -> LayerNorm -> ReLU -> Dropout -> 6
```

### Total : 8,412,936 parametres trainables

---

## 5. Configuration d'entrainement

### Hyperparametres

| Parametre | Valeur |
|-----------|--------|
| Optimiseur | AdamW |
| Learning rate | 0.0001 |
| Weight decay | 1e-4 |
| Betas | (0.9, 0.999) |
| Scheduler | CosineAnnealingWarmRestarts (T0=10, T_mult=2, eta_min=1e-6) |
| Loss | CrossEntropyLoss avec class weights |
| Batch size | 32 |
| Dropout | 0.2 |
| Max epochs | 50 |
| Early stopping | 15 epochs patience (sur MCC) |
| Gradient clipping | max_norm=1.0 |
| Seed | 42 |

### Cross-validation

5-Fold Cross-Validation :
- Pour chaque fold : 3 folds train, 1 fold validation (early stopping), 1 fold test
- Le meilleur modele est sauvegarde selon le **MCC** (Matthews Correlation Coefficient)
- Chaque proteine est testee exactement UNE fois sur l'ensemble des 5 folds

### Execution

| Parametre | Valeur |
|-----------|--------|
| Device | CPU (GPU non disponible) |
| Parallelisme | 5 folds en parallele |
| Threads par fold | 9 |
| Methode | multiprocessing spawn |

---

## 6. Deroulement de l'entrainement

### Timeline

| Fold | Debut | Fin | Duree | Epochs | Early stop |
|------|-------|-----|-------|--------|------------|
| Fold 0 | 26 jan 14:35 | 3 fev 13:22 | ~8 jours | 38/50 | Epoch 22 (best) |
| Fold 1 | 26 jan 14:34 | 2 fev 05:21 | ~7 jours | 32/50 | Epoch 16 (best) |
| Fold 2 | 26 jan 14:34 | 3 fev 23:50 | ~8 jours | 42/50 | Epoch 26 (best) |
| Fold 3 | 26 jan 14:48 | 5 fev 01:16 | ~10 jours | 50/50 | Epoch 47 (best) |
| Fold 4 | 26 jan 14:48 | 3 fev 05:00 | ~8 jours | 36/50 | Epoch 20 (best) |

**Duree totale** : ~10 jours (26 jan → 5 fev)
**Temps par epoch** : ~4-5 heures (216 batches par epoch)
**Raison de la lenteur** : execution sur CPU (pas de GPU disponible)

---

## 7. Resultats

### 7.1 Resultats par fold

| Fold | Accuracy | Macro F1 | MCC |
|------|----------|----------|-----|
| Fold 0 | 0.9376 | 0.8348 | 0.8942 |
| Fold 1 | 0.9271 | 0.7985 | 0.8769 |
| Fold 2 | 0.9271 | 0.8383 | 0.8764 |
| Fold 3 | 0.9358 | 0.8251 | 0.8914 |
| Fold 4 | 0.9315 | 0.7960 | 0.8835 |
| **Moyenne** | **0.9318 +/- 0.0043** | **0.8185 +/- 0.0179** | **0.8845 +/- 0.0073** |

### 7.2 MCC par classe et par fold

| Classe | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Moyenne |
|--------|--------|--------|--------|--------|--------|---------|
| Cytoplasmic | 0.927 | 0.919 | 0.913 | 0.931 | 0.937 | 0.925 |
| Cytoplasmic_membrane | 0.903 | 0.908 | 0.901 | 0.909 | 0.902 | 0.905 |
| Extracellular | 0.883 | 0.849 | 0.837 | 0.830 | 0.856 | 0.851 |
| Periplasmic | 0.820 | 0.805 | 0.797 | 0.861 | 0.746 | 0.806 |
| Outer_membrane | 0.833 | 0.743 | 0.783 | 0.813 | 0.774 | 0.789 |
| Cell_wall_and_surface | 0.588 | 0.460 | 0.693 | 0.520 | 0.515 | 0.555 |

### 7.3 Resultats globaux (11,531 sequences testees)

**Classification Report :**

| Classe | Precision | Recall | F1-Score | Support |
|--------|-----------|--------|----------|---------|
| Cytoplasmic | 0.96 | 0.98 | 0.97 | 6,788 |
| Cytoplasmic_membrane | 0.94 | 0.91 | 0.92 | 2,455 |
| Extracellular | 0.87 | 0.86 | 0.86 | 954 |
| Periplasmic | 0.81 | 0.82 | 0.82 | 557 |
| Outer_membrane | 0.85 | 0.75 | 0.80 | 717 |
| Cell_wall_and_surface | 0.60 | 0.50 | 0.55 | 60 |
| **Weighted avg** | **0.93** | **0.93** | **0.93** | **11,531** |

**Confusion Matrix** (lignes = reel, colonnes = predit) :

|  | Cytopl. | Cyto_memb | Extrac. | Peripl. | Outer_m | Cell_w |
|--|---------|-----------|---------|---------|---------|--------|
| **Cytoplasmic** | **6669** | 46 | 24 | 20 | 29 | 0 |
| **Cyto_membrane** | 125 | **2228** | 29 | 34 | 32 | 7 |
| **Extracellular** | 57 | 26 | **819** | 28 | 17 | 7 |
| **Periplasmic** | 31 | 20 | 31 | **459** | 16 | 0 |
| **Outer_membrane** | 79 | 38 | 28 | 26 | **540** | 6 |
| **Cell_wall** | 6 | 8 | 13 | 2 | 1 | **30** |

### 7.4 Analyse de confiance

| Seuil de confiance | Sequences | % du dataset | Accuracy |
|--------------------|-----------|--------------|----------|
| >= 50% | 11,523 | 99.9% | 93.2% |
| >= 70% | 11,425 | 99.1% | 93.7% |
| >= 90% | 11,274 | 97.8% | 94.3% |
| >= 95% | 11,187 | 97.0% | 94.7% |
| >= 99% | 10,749 | 93.2% | 95.3% |

- Confiance moyenne sur predictions correctes : **99.5%**
- Confiance moyenne sur predictions incorrectes : **93.7%**

Le modele est tres confiant dans ses predictions, meme quand il se trompe. Cela signifie qu'il ne faut pas se fier uniquement a la confiance pour detecter les erreurs.

### 7.5 Comparaison avec DeepLocPro (papier de reference)

| Metrique | DeepLocPro (papier) | Notre modele | Difference |
|----------|---------------------|--------------|------------|
| Accuracy | 0.920 | **0.932** | **+0.012** |
| Macro F1 | 0.800 | **0.819** | **+0.019** |
| MCC | 0.860 | **0.884** | **+0.024** |

Notre modele surpasse DeepLocPro sur toutes les metriques.

### 7.6 Comparaison avec le run precedent (Gated Fusion, 17 janvier)

| Metrique | Gated Fusion (17 jan) | Improved Fusion (26 jan) | Amelioration |
|----------|-----------------------|--------------------------|-------------|
| Accuracy | 0.887 | **0.932** | +4.5% |
| Macro F1 | 0.764 | **0.819** | +5.5% |
| MCC | 0.815 | **0.884** | +6.9% |

L'amelioration vient de :
1. **Cross-attention bidirectionnelle** vs simple gate fusion
2. **BiLSTM** pour les dependances sequentielles
3. **Dataset plus grand** (11,531 vs ~3,500 sequences)
4. **Plus d'epochs** (50 vs 30)
5. **Early stopping sur MCC** au lieu de F1

---

## 8. Erreurs les plus frequentes

Les principales confusions du modele :

1. **Cytoplasmic_membrane → Cytoplasmic** (125 erreurs) : proteines membranaires classees comme cytoplasmiques
2. **Outer_membrane → Cytoplasmic** (79 erreurs) : membrane externe confondue avec cytoplasme
3. **Extracellular → Cytoplasmic** (57 erreurs) : proteines extracellulaires classees comme cytoplasmiques
4. **Outer_membrane → Cytoplasmic_membrane** (38 erreurs) : les deux types de membranes confondus

Le modele a tendance a sur-predire la classe Cytoplasmic (6,967 predictions vs 6,788 reelles) car c'est la classe majoritaire.

---

## 9. Arborescence des fichiers

```
Biomed---Project/
|
|-- try/                                    # Dossier de travail principal
|   |-- RESULTS.md                          # Ce fichier
|   |-- dataset_full_1000.csv               # Dataset (5.5 MB)
|   |-- evaluate_models.py                  # Script d'evaluation
|   |-- launch_best_full.sh                 # Script de lancement
|   |-- explication.txt                     # Explications pedagogiques
|   |-- best_full_output.log                # Log complet d'entrainement (4 MB)
|   |
|   |-- models/
|   |   |-- improved_fusion.py              # Architecture du modele
|   |   |-- advanced_fusion.py              # Architecture alternative (non utilisee ici)
|   |   |-- dataset.py                      # Dataset PyTorch + collate_fn
|   |   +-- fusion_network.py               # Ancien modele (Gated Fusion)
|   |
|   |-- training/
|   |   |-- train_improved.py               # Script d'entrainement principal
|   |   |-- train_advanced.py               # Script alternatif
|   |   +-- fast_nested_cv_cpu.py           # CV rapide
|   |
|   +-- outputs/
|       +-- best_full_20260126_093236/      # Resultats du meilleur run (482 MB)
|           |-- progress.json               # Progression detaillee
|           |-- final_summary.json          # Resume final des metriques
|           |-- fold_0_results.json         # Resultats fold 0 (predictions, labels, MCC)
|           |-- fold_1_results.json         # Resultats fold 1
|           |-- fold_2_results.json         # Resultats fold 2
|           |-- fold_3_results.json         # Resultats fold 3
|           |-- fold_4_results.json         # Resultats fold 4
|           |-- fold0_final/best_model.pt   # Checkpoint modele fold 0 (~100 MB)
|           |-- fold1_final/best_model.pt   # Checkpoint modele fold 1
|           |-- fold2_final/best_model.pt   # Checkpoint modele fold 2
|           |-- fold3_final/best_model.pt   # Checkpoint modele fold 3
|           +-- fold4_final/best_model.pt   # Checkpoint modele fold 4
|
|-- data/
|   |-- embeddings/
|   |   |-- esm2_650m/                     # Embeddings ESM-2 (25 GB, 11,800 fichiers)
|   |   +-- prostt5/                       # Embeddings ProstT5 (20 GB, 11,800 fichiers)
|   +-- processed/
|       |-- dataset_with_folds.csv         # Dataset original avec folds
|       +-- dataset_full_1000.csv          # Dataset filtre <= 1000 AA
|
|-- src/
|   |-- models/
|   |   |-- dataset.py                     # Dataset + LABEL_ENCODER/DECODER
|   |   |-- extract_esm2_650m.py           # Extraction embeddings ESM-2
|   |   |-- extract_prost_embeddings.py    # Extraction embeddings ProstT5
|   |   |-- fusion_network.py              # Ancien modele
|   |   +-- validate_embeddings.py         # Validation des embeddings
|   +-- training/
|       |-- train.py                       # Ancien script d'entrainement
|       +-- nested_cv_optimal.py           # Ancien nested CV
|
+-- configs/
    +-- optimal_config.yaml                # Configuration YAML
```

---

## 10. Comment utiliser

### 10.1 Lancer l'entrainement

```bash
cd /data/padawans/e2121u/mom/Biomed---Project

# Lancement en arriere-plan
nohup bash try/launch_best_full.sh > try/best_full_output.log 2>&1 &

# Suivre la progression
tail -f try/best_full_output.log

# Verifier la progression (JSON)
python3 -c "import json; print(json.dumps(json.load(open('try/outputs/best_full_20260126_093236/progress.json')), indent=2))"

# Verifier si le process tourne
ps aux | grep train_improved
```

### 10.2 Evaluer les modeles entraines

```bash
cd /data/padawans/e2121u/mom/Biomed---Project
python3 try/evaluate_models.py
```

Ce script :
1. Charge les 5 modeles sauvegardes
2. Evalue chacun sur son fold de test
3. Affiche les metriques par fold
4. Affiche les resultats globaux (confusion matrix, classification report, confiance)
5. Compare avec DeepLocPro

### 10.3 Charger un modele pour faire des predictions

```python
import torch
import sys
sys.path.insert(0, 'try')
from models.improved_fusion import ImprovedFusionNetwork

# Charger le modele
model = ImprovedFusionNetwork(
    esm_dim=1280, prost_dim=1024, hidden_dim=512,
    lstm_hidden=256, num_heads=8, num_lstm_layers=2, dropout=0.2
)
ckpt = torch.load('try/outputs/best_full_20260126_093236/fold0_final/best_model.pt',
                   map_location='cpu', weights_only=False)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

# Charger les embeddings d'une proteine
esm_emb = torch.load('data/embeddings/esm2_650m/P02968.pt', map_location='cpu')
prost_emb = torch.load('data/embeddings/prostt5/P02968.pt', map_location='cpu')

# Prediction
esm_emb = esm_emb.unsqueeze(0)       # (1, seq_len, 1280)
prost_emb = prost_emb.unsqueeze(0)    # (1, seq_len, 1024)
mask = torch.ones(1, esm_emb.size(1)) # (1, seq_len)

with torch.no_grad():
    logits, _ = model(esm_emb, prost_emb, mask)
    probs = torch.softmax(logits, dim=-1)
    pred = logits.argmax(dim=-1).item()

CLASSES = {
    0: 'Cytoplasmic',
    1: 'Cytoplasmic_membrane',
    2: 'Extracellular',
    3: 'Periplasmic',
    4: 'Outer_membrane',
    5: 'Cell_wall_and_surface'
}
print(f"Prediction: {CLASSES[pred]} (confiance: {probs[0][pred]:.1%})")
```

### 10.4 Lire les resultats sauvegardes

```python
import json

# Resume global
with open('try/outputs/best_full_20260126_093236/final_summary.json') as f:
    summary = json.load(f)
print(f"Accuracy: {summary['nested_cv_results']['accuracy']['mean']:.4f}")
print(f"F1: {summary['nested_cv_results']['macro_f1']['mean']:.4f}")
print(f"MCC: {summary['nested_cv_results']['mcc']['mean']:.4f}")

# Resultats d'un fold specifique
with open('try/outputs/best_full_20260126_093236/fold_0_results.json') as f:
    fold0 = json.load(f)
print(f"Fold 0 predictions: {len(fold0['predictions'])}")
print(f"Fold 0 MCC par classe: {fold0['mcc_per_class']}")
```

---

## 11. Dependances

```
torch >= 1.12
numpy
pandas
scikit-learn
tqdm
```

Pas besoin de GPU. L'entrainement et l'evaluation fonctionnent sur CPU.

---

## 12. Limitations et pistes d'amelioration

### Limitations actuelles

1. **ProstT5 en mode AA** au lieu de 3Di (Foldseek non disponible sur le serveur)
2. **ESM-2 650M** au lieu de ESM-C (Python 3.9 sur le serveur, ESM-C requiert 3.12+)
3. **Cell_wall_and_surface** : seulement 60 exemples → 50% recall
4. **Confiance trop elevee sur les erreurs** (93.7%) → calibration a ameliorer
5. **CPU uniquement** : 10 jours d'entrainement vs quelques heures sur GPU

### Pistes d'amelioration

1. Utiliser **ESM-C** + **ProstT5 3Di** sur un ordi avec Python 3.12 + Foldseek
2. **Data augmentation** : masquage d'acides amines, bruit sur embeddings
3. **Ensemble** : combiner les predictions des 5 modeles (voting)
4. **Focal Loss** : remplacer CrossEntropyLoss pour mieux gerer les classes rares
5. **Label smoothing** : reduire la sur-confiance du modele

---

## 13. References

- **DeepLocPro** : Thumuluri et al. (2024) - Baseline de reference
- **LocPro 2025** : ESM2 + BiLSTM (+10% F1)
- **BioLangFusion** : Cross-modal multi-head attention
- **HEAL** : Hierarchical attention pooling
- **ESM-2** : Lin et al. (2023) - Meta AI protein language model
- **ProstT5** : Heinzinger et al. (2023) - Protein structure-aware language model
- **GraphPart** : Partitionnement par identite de sequence pour cross-validation
