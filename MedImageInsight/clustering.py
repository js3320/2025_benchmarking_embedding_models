# -*- coding: utf-8 -*-
# source .venv/bin/activate
# python plot_labels_only.py

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize

# Try UMAP; fallback to t-SNE
try:
    import umap
    HAVE_UMAP = True
except Exception:
    from sklearn.manifold import TSNE
    HAVE_UMAP = False

# === CONFIG ===
csv_path = '/rds/general/user/js3320/home/(2025_summer)embeddings/codes/MedImageInsights/ATELECTASIS/mimic/atelectasis_embeddings_with_aug.csv'
output_dir = '/rds/general/user/js3320/home/(2025_summer)embeddings/codes/MedImageInsights/ATELECTASIS/mimic/'
os.makedirs(output_dir, exist_ok=True)

ID_COL_PRIORITY = ['patient_id', 'subject_id', 'dicom_id']
USE_ONLY_ORIGINAL_FOR_EMBED = True
RANDOM_STATE = 42

print(f"Loading: {csv_path}")
df = pd.read_csv(csv_path)

# Choose an ID column
id_col = None
for c in ID_COL_PRIORITY:
    if c in df.columns:
        id_col = c
        break
if id_col is None:
    raise RuntimeError("No ID column found among: 'patient_id','subject_id','dicom_id'")
print(f"Using ID: {id_col}")

# Require labels
if 'label' not in df.columns:
    raise RuntimeError("No 'label' column found. This plot needs binary labels 0/1.")

# Keep only original images if requested
non_feat = set(['label', 'augmentation', 'dicom_id', 'study_id', 'patient_id', 'subject_id'])
if USE_ONLY_ORIGINAL_FOR_EMBED and 'augmentation' in df.columns:
    df = df[df['augmentation'] == 'original'].copy()
    print(f"Filtered to original images. Remaining rows: {len(df)}")

# Feature columns
feature_cols = [c for c in df.columns if c not in non_feat]
if not feature_cols:
    raise RuntimeError("No embedding feature columns found.")

# Build patient-level mean embedding and majority label
embeddings = df.groupby(id_col)[feature_cols].mean()
labels = df.groupby(id_col)['label'].agg(lambda s: int(round(s.mean())))
labels = labels.loc[embeddings.index]

X = embeddings.values
ids = embeddings.index.to_numpy()
y = labels.to_numpy()

# L2-normalize (cosine geometry)
X = normalize(X, norm='l2')

# 2D projection
if HAVE_UMAP:
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine', random_state=RANDOM_STATE)
    X2 = reducer.fit_transform(X)
    dr_name = 'UMAP'
else:
    tsne = TSNE(n_components=2, perplexity=30, learning_rate='auto', init='pca', random_state=RANDOM_STATE)
    X2 = tsne.fit_transform(X)
    dr_name = 't-SNE'

# Plot: only two colors by label
plt.figure(figsize=(9, 7))
mask0 = (y == 0)
mask1 = (y == 1)

plt.scatter(X2[mask0, 0], X2[mask0, 1], s=18, alpha=0.9, label='label=0')
plt.scatter(X2[mask1, 0], X2[mask1, 1], s=18, alpha=0.9, label='label=1')

plt.legend(title='Label')
plt.title(f'Patient Embeddings colored by Label ({dr_name})')
plt.xlabel('Dim 1'); plt.ylabel('Dim 2')
out_path = os.path.join(output_dir, 'patient_labels_2colors.png')
plt.tight_layout()
plt.savefig(out_path, dpi=300)
print(f"✓ Saved: {out_path}")
