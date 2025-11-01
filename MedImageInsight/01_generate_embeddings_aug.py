# -*- coding: utf-8 -*-
# python mimic_dicom_embeddings_augmentation.py

import sys
sys.path.append('/rds/general/user/js3320/home/(2025_summer)embeddings/codes/MedImageInsights')
import os
import pandas as pd
import numpy as np
import pydicom
from PIL import Image, ImageEnhance
import io
import base64
from medimageinsightmodel import MedImageInsight

# === CONFIGURATION ===
LABEL_PATH = '/rds/general/user/js3320/home/(2025_summer)embeddings/data/mimic/atelectasis_2000/atelectasis_labels.csv'
DICOM_DIR = '/rds/general/user/js3320/home/(2025_summer)embeddings/data/mimic/atelectasis_2000'
OUTPUT_PATH = '/rds/general/user/js3320/home/(2025_summer)embeddings/codes/MedImageInsights/ATELECTASIS/mimic/atelectasis_embeddings_with_aug.csv'

# === Load DICOM labels ===
df_labels = pd.read_csv(LABEL_PATH)
df_labels['dicom_id'] = df_labels['dicom_id'].astype(str)
label_map = dict(zip(df_labels['dicom_id'], df_labels['Label']))

# === Match DICOM files ===
dicom_files = [f for f in os.listdir(DICOM_DIR) if f.endswith('.dcm')]
matched = []
for dicom_id in label_map:
    for ext in ['.dcm', '.DCM']:
        fname = f"{dicom_id}{ext}"
        if fname in dicom_files:
            matched.append((dicom_id, fname))
            break

print(f"✅ Matched {len(matched)} DICOMs with labels")

# === Load existing output if exists (resume support) ===
if os.path.exists(OUTPUT_PATH):
    df_existing = pd.read_csv(OUTPUT_PATH)
    processed_pairs = set(zip(df_existing['dicom_id'].astype(str), df_existing['augmentation']))
    print(f"🔁 Resuming from {len(processed_pairs)} already processed (dicom_id, augmentation) pairs")
else:
    df_existing = pd.DataFrame()
    processed_pairs = set()
    print("🆕 No existing file found. Starting fresh.")

# === Load model ===
classifier = MedImageInsight(
    model_dir="/rds/general/user/js3320/home/(2025_summer)embeddings/codes/MedImageInsights/2024.09.27",
    vision_model_name="medimageinsigt-v1.0.0.pt",
    language_model_name="language_model.pth"
)
classifier.load_model()
print("🧠 Model loaded")

# === Helper: Load DICOM as normalized PIL ===
def dicom_to_pil_image(path):
    ds = pydicom.dcmread(path)
    arr = ds.pixel_array
    arr = ((arr - arr.min()) / (arr.max() - arr.min()) * 255).astype(np.uint8)
    return Image.fromarray(arr, mode='L').convert('RGB')

# === Augmentations ===
AUGMENTATIONS = {
    'original': lambda img: img,
    'rot45': lambda img: img.rotate(45, expand=True),
    'rot-45': lambda img: img.rotate(-45, expand=True),
    'bright': lambda img: ImageEnhance.Brightness(img).enhance(1.5),
    'dark': lambda img: ImageEnhance.Brightness(img).enhance(0.7),
    'contrast': lambda img: ImageEnhance.Contrast(img).enhance(1.3),
}

# === Main loop ===
embeddings, dicom_ids, labels, aug_types = [], [], [], []

for i, (dicom_id, fname) in enumerate(matched, 1):
    dicom_path = os.path.join(DICOM_DIR, fname)
    label = label_map[dicom_id]

    try:
        pil_img = dicom_to_pil_image(dicom_path)

        for aug_name, aug_fn in AUGMENTATIONS.items():
            if (dicom_id, aug_name) in processed_pairs:
                continue  # Skip if already done

            aug_img = aug_fn(pil_img)

            buffer = io.BytesIO()
            aug_img.save(buffer, format='PNG')
            b64img = base64.encodebytes(buffer.getvalue()).decode('utf-8')

            emb = classifier.encode(images=[b64img])['image_embeddings'][0]

            embeddings.append(emb)
            dicom_ids.append(dicom_id)
            labels.append(label)
            aug_types.append(aug_name)

        if i % 10 == 0 or i == len(matched):
            print(f"✅ {i}/{len(matched)} images processed ({i * 6} embeddings approx.)")

    except Exception as e:
        print(f"❌ Error processing {dicom_id}: {e}")

# === Save combined CSV ===
if embeddings:
    embedding_dim = len(embeddings[0])
    columns = [f'embedding_{j}' for j in range(embedding_dim)] + ['dicom_id', 'label', 'augmentation']
    rows = [list(e) + [d, l, a] for e, d, l, a in zip(embeddings, dicom_ids, labels, aug_types)]
    df_new = pd.DataFrame(rows, columns=columns)
    df_all = pd.concat([df_existing, df_new], ignore_index=True)
    df_all.to_csv(OUTPUT_PATH, index=False)

    print(f"\n💾 Saved {len(df_new)} new embeddings (Total: {len(df_all)}) to {OUTPUT_PATH}")
    print(f"📊 New label breakdown: {pd.Series(labels).value_counts().to_dict()}")
else:
    print("\n✅ All embeddings already processed. Nothing new to save.")
