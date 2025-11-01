# -*- coding: utf-8 -*-
# source .venv/bin/activate
# python mimic_dicom_augmentation_train.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    roc_curve
)
from lightgbm import LGBMClassifier
import os
from io import StringIO

# === CONFIG ===
csv_path = '/rds/general/user/js3320/home/(2025_summer)embeddings/codes/cxr_foundation/ATELECTASIS/mimic/mimic_dicom_embeddings/mimic_dicom_embeddings_with_augmentation.csv'

# Define output directory and ensure it exists
output_dir = '/rds/general/user/js3320/home/(2025_summer)embeddings/codes/cxr_foundation/ATELECTASIS/mimic'
os.makedirs(output_dir, exist_ok=True)

n_splits = 5                             # K in KFold

# === STEP 1: Load embeddings CSV ===
print(f"Loading embeddings from {csv_path}")

emb_df = pd.read_csv(csv_path)
print(f"Shape of loaded data: {emb_df.shape}")

# Separate features and label
if 'label' not in emb_df.columns:
    raise RuntimeError("CSV file must contain a 'label' column.")

# Drop non-feature columns (label, dicom_id, augmentation)
columns_to_drop = ['label']
if 'dicom_id' in emb_df.columns:
    columns_to_drop.append('dicom_id')
    print("Dropped 'dicom_id' column (non-feature)")
if 'augmentation' in emb_df.columns:
    columns_to_drop.append('augmentation')
    print("Dropped 'augmentation' column (non-feature)")

print(f"Non-feature columns to drop: {columns_to_drop}")
print(f"Augmentation distribution: {emb_df['augmentation'].value_counts().to_dict()}")

X = emb_df.drop(columns=columns_to_drop).values
y = emb_df['label'].values

print(f"Feature matrix shape: {X.shape}")
print(f"Labels shape: {y.shape}")
print(f"Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")

if len(y) < 20:
    print("Dataset has fewer than 20 samples. Exiting.")
    exit()

# Check if we have both classes
unique = np.unique(y)
if len(unique) < 2:
    print("Warning: Only one class found. Cannot train binary classifier.")
    exit()

# === STEP 2: K-Fold Split by DICOM IDs (no data leakage) ===
from sklearn.model_selection import StratifiedKFold

# Get unique DICOM IDs and their labels
unique_ids = emb_df['dicom_id'].unique()
id_labels = emb_df.groupby('dicom_id')['label'].first()  # Get label for each ID

print(f"\n=== DICOM ID K-FOLD SETUP ===")
print(f"Total unique DICOM IDs: {len(unique_ids)}")
print(f"Class distribution by ID: {id_labels.value_counts().to_dict()}")
print(f"K-Fold splits: {n_splits}")

# Setup K-fold on DICOM IDs (not individual samples)
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

results = {
    'accuracy': [],
    'roc_auc': [],
    'precision': [],
    'recall': [],
    'specificity': [],
    'f1': []
}

all_cms = []
all_y_test = []
all_y_prob = []
all_test_ids = []

def specificity_score(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp)

print(f"\n=== K-FOLD CROSS-VALIDATION ===")

# Perform K-fold on DICOM IDs
for fold_idx, (train_id_indices, test_id_indices) in enumerate(skf.split(unique_ids, id_labels)):
    print(f"\nFold {fold_idx+1}/{n_splits}")
    
    # Get train and test DICOM IDs for this fold
    train_ids = unique_ids[train_id_indices]
    test_ids = unique_ids[test_id_indices]
    
    print(f"  Train IDs: {len(train_ids)} ({len(train_ids)/len(unique_ids)*100:.1f}%)")
    print(f"  Test IDs: {len(test_ids)} ({len(test_ids)/len(unique_ids)*100:.1f}%)")
    
    # Verify no overlap
    overlap = set(train_ids).intersection(set(test_ids))
    if len(overlap) > 0:
        print(f"  ❌ Data leakage detected! Overlapping IDs: {overlap}")
        continue
    
    # Create train set: ALL augmented data for train IDs
    train_mask = emb_df['dicom_id'].isin(train_ids)
    train_df = emb_df[train_mask].copy()
    
    # Create test set: ONLY original images for test IDs
    test_mask = (emb_df['dicom_id'].isin(test_ids)) & (emb_df['augmentation'] == 'original')
    test_df = emb_df[test_mask].copy()
    
    print(f"  Training samples: {len(train_df)} (augmented)")
    print(f"  Test samples: {len(test_df)} (original only)")
    print(f"  Train class dist: {train_df['label'].value_counts().to_dict()}")
    print(f"  Test class dist: {test_df['label'].value_counts().to_dict()}")
    
    # Prepare features and labels for this fold
    X_train = train_df.drop(columns=columns_to_drop).values
    y_train = train_df['label'].values
    X_test = test_df.drop(columns=columns_to_drop).values
    y_test = test_df['label'].values
    
    # Train model
    clf = LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)
    clf.fit(X_train, y_train)
    
    # Make predictions
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]
    
    # Store for aggregation
    all_y_test.extend(y_test)
    all_y_prob.extend(y_prob)
    all_test_ids.extend(test_ids)
    
    # Calculate metrics
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    spec = specificity_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    results['accuracy'].append(acc)
    results['roc_auc'].append(auc)
    results['precision'].append(prec)
    results['recall'].append(rec)
    results['specificity'].append(spec)
    results['f1'].append(f1)
    
    cm = confusion_matrix(y_test, y_pred)
    all_cms.append(cm)
    
    print(f"  Accuracy: {acc:.4f}, ROC AUC: {auc:.4f}, F1: {f1:.4f}")

# Verify all IDs were tested exactly once
all_test_ids_unique = set(all_test_ids)
print(f"\n=== K-FOLD VALIDATION ===")
print(f"Total unique test IDs across all folds: {len(all_test_ids_unique)}")
print(f"Expected: {len(unique_ids)}")
if len(all_test_ids_unique) == len(unique_ids):
    print("✅ All DICOM IDs tested exactly once across all folds!")
else:
    print("❌ Some DICOM IDs missing or tested multiple times!")

# Start capturing output for text file
output_buffer = StringIO()

# Calculate final metrics
results_header = f"\n=== AGGREGATE RESULTS ===\n"
print(results_header.strip())
output_buffer.write(results_header)

for metric_name, values in results.items():
    mean_val = np.mean(values)
    std_val = np.std(values)
    ci_95 = 1.96 * std_val / np.sqrt(len(values))
    result_line = f"{metric_name.upper():>12}: {mean_val:.4f} ± {std_val:.4f} (95% CI: ±{ci_95:.4f})"
    print(result_line)
    output_buffer.write(result_line + "\n")

avg_cm = np.mean(all_cms, axis=0).astype(int)
cm_output = f"\nAVERAGE CONFUSION MATRIX:\nPredicted:    0    1\n"
cm_output += f"Actual 0: {avg_cm[0,0]:4d} {avg_cm[0,1]:4d}\n"
cm_output += f"Actual 1: {avg_cm[1,0]:4d} {avg_cm[1,1]:4d}\n"
print(cm_output.strip())
output_buffer.write(cm_output)

# === STEP 3: Generate plots using final fold data ===
# Use the last fold's data for detailed plots
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
final_auc = roc_auc_score(y_test, y_prob)

fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Plot 1: Confusion Matrix Heatmap (Final Fold)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Airspace Opacity', 'Airspace Opacity'],
            yticklabels=['No Airspace Opacity', 'Airspace Opacity'],
            ax=axes[0,0])
axes[0,0].set_title('Confusion Matrix (Final Fold)')
axes[0,0].set_xlabel('Predicted')
axes[0,0].set_ylabel('Actual')

# Plot 2: ROC Curve (Final Fold)
axes[0,1].plot(fpr, tpr, color='darkorange', lw=2,
               label=f'ROC curve (AUC = {final_auc:.4f})')
axes[0,1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
               label='Random classifier')
axes[0,1].set_xlim([0.0, 1.0])
axes[0,1].set_ylim([0.0, 1.05])
axes[0,1].set_xlabel('False Positive Rate')
axes[0,1].set_ylabel('True Positive Rate')
axes[0,1].set_title('ROC Curve (Final Fold)')
axes[0,1].legend(loc="lower right")
axes[0,1].grid(True, alpha=0.3)

# Plot 3: Metrics with Error Bars (K-Fold Results)
metrics_names = ['Accuracy', 'ROC AUC', 'Precision', 'Recall', 'Specificity', 'F1-Score']
metrics_values = [results['accuracy'], results['roc_auc'], results['precision'],
                  results['recall'], results['specificity'], results['f1']]

means = [np.mean(vals) for vals in metrics_values]
stds = [np.std(vals) for vals in metrics_values]

x_pos = np.arange(len(metrics_names))
bars = axes[1,0].bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7,
                     color=['blue', 'orange', 'green', 'red', 'purple', 'brown'])
axes[1,0].set_xlabel('Metrics')
axes[1,0].set_ylabel('Score')
axes[1,0].set_title('Performance Metrics (K-Fold)\nError bars show ±1 standard deviation')
axes[1,0].set_xticks(x_pos)
axes[1,0].set_xticklabels(metrics_names, rotation=45, ha='right')
axes[1,0].grid(True, alpha=0.3)
axes[1,0].set_ylim([0, 1])

# Plot 4: Metrics Distribution Boxplot
metrics_data = pd.DataFrame(results)
metrics_data.boxplot(ax=axes[1,1])
axes[1,1].set_title('Metric Distribution Across Folds')
axes[1,1].set_ylabel('Score')
axes[1,1].tick_params(axis='x', rotation=45)
axes[1,1].grid(True, alpha=0.3)

# === CONFIG: Set your desired output directory ===
output_path = f"{output_dir}/kfold_augmented_results.png"

plt.tight_layout()
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\n✓ Saved K-fold analysis plots to '{output_path}'")
plt.close()  # Close the figure to free memory

# === STEP 5: Feature importance ===
importances = clf.feature_importances_
top_indices = np.argsort(importances)[-10:][::-1]
feature_importance_text = f"\nTop 10 most important features:\n"
print(feature_importance_text.strip())
output_buffer.write(feature_importance_text)
for i, idx in enumerate(top_indices):
    importance_line = f"  {i+1}. Feature {idx}: {importances[idx]:.4f}"
    print(importance_line)
    output_buffer.write(importance_line + "\n")

# === STEP 6: Final Summary ===
summary_header = "\n" + "="*60 + "\n"
summary_header += "           AUGMENTED TRAINING RESULTS\n"
summary_header += "="*60 + "\n"
print(summary_header.strip())
output_buffer.write(summary_header)

training_info = f"Training Strategy: Augmented data for training, original only for testing\n"
training_info += f"Training set: {len(X_train)} samples ({train_df['dicom_id'].nunique()} unique IDs)\n"
training_info += f"Test set: {len(X_test)} samples ({test_df['dicom_id'].nunique()} unique IDs)\n"
training_info += f"No data leakage: ✅ (train and test IDs are completely separate)\n"
print(training_info.strip())
output_buffer.write(training_info)

performance_header = f"\nPERFORMANCE METRICS:\n"
print(performance_header.strip())
output_buffer.write(performance_header)

performance_metrics = [
    f"  {'ACCURACY':>12}: {acc:.4f}",
    f"  {'ROC AUC':>12}: {auc:.4f}",
    f"  {'PRECISION':>12}: {prec:.4f}",
    f"  {'RECALL':>12}: {rec:.4f}",
    f"  {'SPECIFICITY':>12}: {spec:.4f}",
    f"  {'F1-SCORE':>12}: {f1:.4f}"
]

for metric in performance_metrics:
    print(metric)
    output_buffer.write(metric + "\n")

training_dist_header = f"\nTRAINING DATA DISTRIBUTION:\n"
print(training_dist_header.strip())
output_buffer.write(training_dist_header)

train_aug_counts = train_df['augmentation'].value_counts()
for aug_type, count in train_aug_counts.items():
    dist_line = f"  {aug_type}: {count} samples"
    print(dist_line)
    output_buffer.write(dist_line + "\n")

benefits_info = f"\nMODEL BENEFITS:\n"
benefits_info += f"  ✅ Realistic evaluation (no test data leakage)\n"
benefits_info += f"  ✅ Robust training ({len(X_train)} augmented samples)\n"
benefits_info += f"  ✅ Unbiased testing (original images only)\n"
final_separator = "="*60 + "\n"
print(benefits_info.strip())
print(final_separator.strip())
output_buffer.write(benefits_info)
output_buffer.write(final_separator)

# Save all results to text file
results_text_path = os.path.join(output_dir, 'kfold_augmented_results.txt')
with open(results_text_path, 'w') as f:
    f.write(output_buffer.getvalue())

print(f"\n✓ Saved statistical results to '{results_text_path}'")
print(f"✓ All files saved to directory: '{output_dir}'")
print(f"   - kfold_augmented_results.png")
print(f"   - kfold_augmented_results.txt")
