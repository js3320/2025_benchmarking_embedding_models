# MIMIC-CXR Atelectasis Benchmark  
### MedImageInsight vs CXR Foundation (2025 Foundation Model Comparison)

This repository presents benchmarking experiments for **Atelectasis detection** using the **MIMIC-CXR** dataset.  
Two foundation models were evaluated:

- **MedImageInsight** — a lightweight, clinically aligned foundation model (embeddings available for use)  
- **CXR Foundation** — a large-scale vision–language model (embeddings too large to host here)

The purpose of this project is to explore how well foundation-model-derived image embeddings can classify Atelectasis and visualize disease-related latent spaces in de-identified ICU chest X-rays.

---

## Overview

| Model | Dataset | Task | Embedding Dim | Data Availability |
|--------|----------|------|----------------|-------------------|
| **MedImageInsight** | MIMIC-CXR | Atelectasis (present vs absent) | 512-D | Available upon request |
| **CXR Foundation** | MIMIC-CXR | Atelectasis (present vs absent) | 2048-D | Not hosted due to large file size |

- All embeddings were generated from de-identified **MIMIC-CXR** images (1,000 positive + 1,000 negative samples).  
- Each embedding vector corresponds to a single chest X-ray study.  
- Only derived data (feature embeddings, no pixel data or identifiers) are used here.

---

## Data Availability

**Raw image data (MIMIC-CXR)** are restricted under PhysioNet credentialed access.  
They are not included in this repository and cannot be redistributed.

- The **MedImageInsight embedding CSV** (balanced 1,000/1,000) may be shared upon request for research purposes.  
  Contact the maintainer if you require access.  
- The **CXR Foundation embeddings** are large (several GBs) and are not stored here.  
  They can be regenerated locally using the provided scripts if you have access to the same dataset.

---

## Repository Structure

```
.
├── MedImageInsight/
│   ├── 01_generate_embeddings_mimic.py      # reference script for embedding generation
│   ├── 02_train_eval_atelectasis.py         # training and evaluation on available embeddings
│   ├── 03_cluster_umap.py                   # clustering and visualization of embeddings
│   ├── data/                                # small sample embedding CSV (if included)
│   └── results/
│       ├── kfold_augmented_results.png
│       ├── figures/
│       │   ├── patient_clusters_2D.png
│       │   └── patient_labels_2colors.png
│       └── patient_clusters.csv
│
├── CXR_foundation/
│   ├── 01_generate_embeddings_mimic.py      # reference script for embedding generation
│   ├── 02_train_eval_atelectasis.py         # evaluation logic (embeddings not included)
│   └── results/
│       ├── kfold_augmented_results.png
│       └── kfold_augmented_results.txt
│
└── tools/
    └── inspect_embeddings.py                # utility to check embedding dimensionality
```

---

## What You Can Do with This Repository

You can:

1. Load and train models using the **MedImageInsight embeddings**  
   (`MedImageInsight/02_train_eval_atelectasis.py`)
2. Visualize and cluster embedding spaces using UMAP or t-SNE  
   (`MedImageInsight/03_cluster_umap.py`)
3. Inspect embeddings to verify their dimensionality  
   ```bash
   python tools/inspect_embeddings.py MedImageInsight/data/embeddings_mimic_mii_atelectasis_balanced.csv
   ```

You cannot:

- Access or regenerate the original MIMIC-CXR images through this repository  
- Obtain the **CXR Foundation** embeddings directly (they are too large and subject to data-use agreements)

---

## Example Workflow

### 1. Training and Evaluation
```bash
cd MedImageInsight
python 02_train_eval_atelectasis.py     --data data/embeddings_mimic_mii_atelectasis_balanced.csv     --output results/
```

**Outputs**
- Cross-validation metrics (AUROC, AUPRC, F1)
- ROC and performance summary plots in `results/`

### 2. Clustering and Visualization
```bash
python 03_cluster_umap.py     --data data/embeddings_mimic_mii_atelectasis_balanced.csv     --save results/figures/
```

**Outputs**
- `patient_clusters_2D.png`  
- `patient_labels_2colors.png`

---

## Environment Setup

```bash
conda create -n cxr-benchmark python=3.10 -y
conda activate cxr-benchmark
pip install numpy pandas scikit-learn matplotlib umap-learn tqdm
```

---

## Notes

- This repository provides scripts and example embeddings for research and reproducibility only.  
- No patient identifiers, PHI, or pixel-level data are stored or shared.  
- For other disease labels (edema, effusion, opacity, etc.), the same pipeline can be reused with corresponding embeddings.

---

## Contact

**Maintainer:** Jiho Shin  
Biomedical Engineering, Imperial College London  
Email: jiho.shin20@imperial.ac.uk  
GitHub: [js3320](https://github.com/js3320)

---

*This repository is part of ongoing research on foundation model benchmarking in clinical imaging. Data are shared under institutional agreements. Only MedImageInsight embeddings are available for exploratory analysis and visualization.*
