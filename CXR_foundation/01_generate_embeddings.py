import sys
sys.path.append('/rds/general/user/js3320/home/(2025_summer)embeddings/codes/cxr_foundation/cxr-foundation')
import os
import glob
import pydicom
import numpy as np
import shutil
import logging
import pandas as pd
from PIL import Image, ImageEnhance
import tempfile
from cxr_foundation.inference import generate_embeddings, InputFileType, OutputFileType, ModelVersion
from cxr_foundation import embeddings_data

LABEL_PATH = '/rds/general/user/js3320/home/(2025_summer)embeddings/data/mimic/atelectasis_2000/atelectasis_labels.csv'
local_dicom_folder = '/rds/general/user/js3320/home/(2025_summer)embeddings/data/mimic/atelectasis_2000'
embeddings_output_dir = '/rds/general/user/js3320/home/(2025_summer)embeddings/codes/cxr_foundation/ATELECTASIS/mimic/mimic_dicom_embeddings'

# === AUGMENTATION CONFIGURATION ===
AUGMENTATIONS = {
    'original': lambda img: img,
    'rot45': lambda img: img.rotate(45, expand=True),
    'rot-45': lambda img: img.rotate(-45, expand=True),
    'bright': lambda img: ImageEnhance.Brightness(img).enhance(1.5),
    'dark': lambda img: ImageEnhance.Brightness(img).enhance(0.7),
    'contrast': lambda img: ImageEnhance.Contrast(img).enhance(1.3),
}

dicom_files = glob.glob(os.path.join(local_dicom_folder, '*.dcm'))
print(f"Found {len(dicom_files)} DICOM files")

# Check if CSV already exists (complete processing)
csv_path = os.path.join(embeddings_output_dir, 'mimic_dicom_embeddings_with_augmentation.csv')
if os.path.exists(csv_path):
    # Check if CSV is complete by verifying row count
    try:
        df_existing = pd.read_csv(csv_path)
        expected_rows = len(glob.glob(os.path.join(local_dicom_folder, '*.dcm'))) * len(AUGMENTATIONS)
        if len(df_existing) >= expected_rows * 0.95:  # Allow for some missing due to invalid files
            print(f"✓ Complete CSV embeddings file already exists at: {csv_path}")
            print(f"  Rows: {len(df_existing)}, Expected: ~{expected_rows}")
            print("Skipping embedding generation. Delete the CSV file if you want to regenerate.")
            exit(0)
        else:
            print(f"⚠️  Incomplete CSV found: {len(df_existing)}/{expected_rows} rows")
            print("Will resume processing from where it left off...")
    except Exception as e:
        print(f"⚠️  Error reading existing CSV: {e}")
        print("Will start fresh processing...")

EMBEDDING_VERSION = 'elixr'
if EMBEDDING_VERSION == 'cxr_foundation':
    MODEL_VERSION = ModelVersion.V1
    TOKEN_NUM = 1
    EMBEDDINGS_SIZE = 1376
elif EMBEDDING_VERSION == 'elixr':
    MODEL_VERSION = ModelVersion.V2
    TOKEN_NUM = 32
    EMBEDDINGS_SIZE = 768
elif EMBEDDING_VERSION == 'elixr_img_contrastive':
    MODEL_VERSION = ModelVersion.V2_CONTRASTIVE
    TOKEN_NUM = 32
    EMBEDDINGS_SIZE = 128

print("Setting up output directories...")
if not os.path.exists(embeddings_output_dir):
    print(f"Creating directory: {embeddings_output_dir}")
    os.makedirs(embeddings_output_dir)
else:
    print(f"Directory already exists: {embeddings_output_dir}")
print("✓ Output directories ready")

logger = logging.getLogger()
logger.setLevel(logging.INFO)

def validate_dicom_file(dicom_path):
    try:
        ds = pydicom.dcmread(dicom_path)
        if not hasattr(ds, 'pixel_array'):
            return False, "No pixel data found"
        pixel_array = ds.pixel_array
        if pixel_array.size == 0:
            return False, "Empty pixel array"
        if len(pixel_array.shape) < 2:
            return False, f"Invalid dimensions: {pixel_array.shape}"
        return True, "Valid"
    except Exception as e:
        return False, str(e)

def dicom_to_pil_image(dicom_path):
    """Convert DICOM to normalized PIL Image"""
    ds = pydicom.dcmread(dicom_path)
    arr = ds.pixel_array
    # Normalize to 0-255
    arr = ((arr - arr.min()) / (arr.max() - arr.min()) * 255).astype(np.uint8)
    # Fix PIL deprecation warning by removing mode parameter
    return Image.fromarray(arr).convert('RGB')

def save_augmented_as_png(dicom_path, augmentation_func, output_path):
    """Create augmented PNG file by applying augmentation to PIL image"""
    try:
        # Convert DICOM to PIL and apply augmentation
        pil_img = dicom_to_pil_image(dicom_path)
        aug_img = augmentation_func(pil_img)
        
        # Save as PNG instead of DICOM to avoid encoding issues
        aug_img.save(output_path, 'PNG')
        return output_path
    except Exception as e:
        raise Exception(f"Failed to create augmented image: {str(e)}")

print("\n📊 Validating DICOM files...")
valid_dicom_files = []
invalid_files = []

for i, dicom_file in enumerate(dicom_files):
    is_valid, error_msg = validate_dicom_file(dicom_file)
    if is_valid:
        valid_dicom_files.append(dicom_file)
    else:
        invalid_files.append((dicom_file, error_msg))
        print(f"❌ Invalid: {os.path.basename(dicom_file)} - {error_msg}")
    if (i + 1) % 50 == 0:
        print(f"Validated {i + 1}/{len(dicom_files)} files...")

print(f"✓ Validation complete:")
print(f"  - Total DICOM files: {len(dicom_files)}")
print(f"  - Valid files: {len(valid_dicom_files)}")
print(f"  - Invalid files: {len(invalid_files)}")

dicom_files_to_process = valid_dicom_files

def print_progress(description, current, total):
    if total == 0:
        return
    percentage = (current / total) * 100
    print(f"{description}: {current}/{total} ({percentage:.1f}%)")

try:
    print("\n📊 Step 1: Generating embeddings with augmentations (temporary processing)...")
    
    # Create temporary directory for augmented images (PNG format)
    temp_dir = os.path.join(embeddings_output_dir, 'temp_augmented')
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    
    # Create temporary directory for TFRecord files (will be deleted after processing)
    temp_tfrecord_dir = os.path.join(embeddings_output_dir, 'temp_tfrecords')
    if not os.path.exists(temp_tfrecord_dir):
        os.makedirs(temp_tfrecord_dir)
    
    # Prepare all augmented files
    all_augmented_files = []
    augmentation_map = {}  # Maps augmented file path to (original_dicom_id, aug_type)
    
    if len(dicom_files_to_process) > 0:
        print(f"Creating augmented versions of {len(dicom_files_to_process)} DICOM files...")
        
        for dicom_file in dicom_files_to_process:
            dicom_id = os.path.splitext(os.path.basename(dicom_file))[0]
            
            # Skip files that consistently fail
            failed_augmentations = 0
            
            for aug_name, aug_func in AUGMENTATIONS.items():
                try:
                    # Create filename for augmented PNG
                    aug_filename = f"{dicom_id}_{aug_name}.png"
                    aug_path = os.path.join(temp_dir, aug_filename)
                    
                    # Create augmented PNG file
                    save_augmented_as_png(dicom_file, aug_func, aug_path)
                    all_augmented_files.append(aug_path)
                    augmentation_map[aug_path] = (dicom_id, aug_name)
                    
                except Exception as e:
                    failed_augmentations += 1
                    print(f"  ❌ Failed to create augmentation {aug_name} for {dicom_id}: {type(e).__name__}")
                    # If all augmentations fail for this file, it's likely corrupted
                    if failed_augmentations >= len(AUGMENTATIONS):
                        print(f"  ⚠️  Skipping {dicom_id} - all augmentations failed (likely corrupted DICOM)")
                        break
        
        print(f"Created {len(all_augmented_files)} augmented PNG files")
        
        # Generate embeddings for all augmented files and convert to CSV immediately
        if len(all_augmented_files) > 0:
            print(f"Processing {len(all_augmented_files)} augmented files to generate embeddings...")
            
            # Load labels for mapping
            df_labels = pd.read_csv(LABEL_PATH)
            df_labels['dicom_id'] = df_labels['dicom_id'].astype(str)
            label_dict = dict(zip(df_labels['dicom_id'], df_labels['Label']))
            
            # Check for existing partial CSV and load it
            rows = []
            processed_files = set()
            start_batch = 0
            
            if os.path.exists(csv_path):
                try:
                    df_existing = pd.read_csv(csv_path)
                    rows = df_existing.values.tolist()
                    # Extract processed dicom_id + augmentation combinations
                    for _, row in df_existing.iterrows():
                        processed_files.add(f"{row['dicom_id']}_{row['augmentation']}")
                    print(f"📊 Resuming: Found {len(df_existing)} existing embeddings")
                    print(f"📊 Processed combinations: {len(processed_files)}")
                except Exception as e:
                    print(f"⚠️  Error loading existing CSV: {e}")
                    print("Starting fresh...")
            
            # Filter out already processed files
            remaining_files = []
            for file_path in all_augmented_files:
                if file_path in augmentation_map:
                    dicom_id, aug_name = augmentation_map[file_path]
                    combo_key = f"{dicom_id}_{aug_name}"
                    if combo_key not in processed_files:
                        remaining_files.append(file_path)
            
            print(f"📊 Files to process: {len(remaining_files)} (skipping {len(all_augmented_files) - len(remaining_files)} already processed)")
            
            if len(remaining_files) == 0:
                print("✓ All files already processed!")
            else:
                try:
                    # Process in batches to avoid memory issues
                    batch_size = 25
                    for i in range(0, len(remaining_files), batch_size):
                        batch_files = remaining_files[i:i+batch_size]
                        batch_num = i//batch_size + 1
                        total_batches = (len(remaining_files) + batch_size - 1)//batch_size
                        print(f"  Processing batch {batch_num}/{total_batches} (remaining files)")
                        
                        # Generate embeddings to temporary TFRecord directory
                        generate_embeddings(
                            input_files=batch_files,
                            output_dir=temp_tfrecord_dir,
                            input_type=InputFileType.PNG,
                            output_type=OutputFileType.TFRECORD,
                            model_version=MODEL_VERSION
                        )
                        
                        # Process each file in the batch immediately
                        batch_rows = []
                        for file_path in batch_files:
                            if file_path in augmentation_map:
                                dicom_id, aug_name = augmentation_map[file_path]
                                
                                # Find the corresponding TFRecord file
                                original_tfrecord = os.path.join(temp_tfrecord_dir, os.path.basename(file_path).replace('.png', '.tfrecord'))
                                
                                if os.path.exists(original_tfrecord):
                                    try:
                                        # Read embedding and add to rows
                                        embedding = embeddings_data.read_tfrecord_values(original_tfrecord)
                                        flattened = embedding.flatten()
                                        label = label_dict.get(dicom_id, "unknown")
                                        row = list(flattened) + [dicom_id, label, aug_name]
                                        batch_rows.append(row)
                                        
                                        # Delete the TFRecord file immediately after reading
                                        os.remove(original_tfrecord)
                                        
                                    except Exception as e:
                                        print(f"❌ Failed reading embedding for {dicom_id}_{aug_name}: {e}")
                                else:
                                    print(f"⚠️  TFRecord file not found for {dicom_id}_{aug_name}")
                        
                        # Add batch results to main rows and save periodically
                        rows.extend(batch_rows)
                        
                        # Save progress every 10 batches to enable resume
                        if batch_num % 10 == 0 or batch_num == total_batches:
                            try:
                                embedding_dim = TOKEN_NUM * EMBEDDINGS_SIZE
                                columns = [f"embedding_{i}" for i in range(embedding_dim)] + ['dicom_id', 'label', 'augmentation']
                                df_progress = pd.DataFrame(rows, columns=columns)
                                df_progress.to_csv(csv_path, index=False)
                                print(f"  💾 Progress saved: {len(rows)} embeddings")
                            except Exception as e:
                                print(f"  ⚠️  Error saving progress: {e}")
                    
                    print(f"✓ Generated embeddings and processed {len(rows)} total samples!")
                    
                except Exception as e:
                    print(f"✗ Error generating embeddings: {str(e)}")
                    # Save whatever we have so far
                    if len(rows) > 0:
                        try:
                            embedding_dim = TOKEN_NUM * EMBEDDINGS_SIZE
                            columns = [f"embedding_{i}" for i in range(embedding_dim)] + ['dicom_id', 'label', 'augmentation']
                            df_partial = pd.DataFrame(rows, columns=columns)
                            df_partial.to_csv(csv_path, index=False)
                            print(f"💾 Saved partial results: {len(rows)} embeddings")
                        except Exception as save_error:
                            print(f"❌ Failed to save partial results: {save_error}")
            
            # Final save of complete dataset
            if len(rows) > 0:
                embedding_dim = TOKEN_NUM * EMBEDDINGS_SIZE
                columns = [f"embedding_{i}" for i in range(embedding_dim)] + ['dicom_id', 'label', 'augmentation']
                df_embeddings = pd.DataFrame(rows, columns=columns)
                df_embeddings.to_csv(csv_path, index=False)
                print(f"💾 Saved final CSV with {len(rows)} embeddings to: {csv_path}")
            else:
                print("⚠️  No embeddings to save")
        
        # Clean up temporary augmented files
        print("🧹 Cleaning up temporary files...")
        for temp_file in all_augmented_files:
            try:
                os.remove(temp_file)
            except:
                pass
        try:
            os.rmdir(temp_dir)
            os.rmdir(temp_tfrecord_dir)
        except:
            pass
        try:
            os.rmdir(temp_dir)
            os.rmdir(temp_tfrecord_dir)
        except:
            pass
        
    else:
        print("✓ No files to process!")

    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    print(f"Total original DICOM files found: {len(dicom_files)}")
    print(f"Invalid/corrupted files skipped: {len(invalid_files)}")
    print(f"Valid files processed: {len(dicom_files) - len(invalid_files)}")
    print(f"Augmentations per file: {len(AUGMENTATIONS)}")
    print(f"Total expected embeddings: {(len(dicom_files) - len(invalid_files)) * len(AUGMENTATIONS)}")
    print(f"CSV embeddings file saved to: {embeddings_output_dir}")
    print(f"Embedding version used: {EMBEDDING_VERSION}")
    print(f"Embedding dimensions: {TOKEN_NUM} x {EMBEDDINGS_SIZE}")
    print(f"Total embedding size per image: {TOKEN_NUM * EMBEDDINGS_SIZE}")
    if len(invalid_files) > 0:
        print(f"⚠️  {len(invalid_files)} files were skipped due to corruption/invalid format")
    
    # Print final statistics if CSV was created successfully
    csv_path = os.path.join(embeddings_output_dir, 'mimic_dicom_embeddings_with_augmentation.csv')
    if os.path.exists(csv_path):
        # Read the CSV to get final statistics
        df_final = pd.read_csv(csv_path)
        
        print(f"\n📊 Final CSV Statistics:")
        aug_counts = df_final['augmentation'].value_counts()
        for aug_type, count in aug_counts.items():
            print(f"  - {aug_type}: {count} embeddings")
        
        print(f"\n📊 Label breakdown:")
        label_counts = df_final['label'].value_counts()
        for label, count in label_counts.items():
            print(f"  - Label {label}: {count} embeddings")
        
        total_unique_images = df_final['dicom_id'].nunique()
        print(f"\n📈 Total unique DICOM images: {total_unique_images}")
        print(f"📈 Total embeddings (with augmentations): {len(df_final)}")
        print(f"📈 Average augmentations per image: {len(df_final) / total_unique_images:.1f}")
        print(f"💾 CSV file saved to: {csv_path}")
    
    print("="*50)

except Exception as e:
    print(f"✗ Error during embedding generation: {str(e)}")
    print("Make sure you have access to the CXR Foundation API and proper authentication.")
