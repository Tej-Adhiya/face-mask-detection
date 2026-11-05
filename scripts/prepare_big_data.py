import os
import io
import sys
import time
import json
import random
import shutil
import zipfile
from PIL import Image
from kaggle.api.kaggle_api_extended import KaggleApi

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TARGET_DIR = os.path.join(PROJECT_ROOT, "big_data_test")
CHECKPOINT_FILE = os.path.join(TARGET_DIR, "progress.json")
TEMP_DIR = os.path.join(TARGET_DIR, "temp_download")

SPLITS = ["train", "val", "test"]
CLASSES = ["mask_proper", "mask_improper", "no_mask"]
TARGET_SIZE = (128, 128)

# Point Kaggle to the workspace .kaggle directory
os.environ["KAGGLE_CONFIG_DIR"] = os.path.join(PROJECT_ROOT, ".kaggle")


def load_progress():
    progress = {
        "completed_datasets": [],
        "counts": {split: {cls: 0 for cls in CLASSES} for split in SPLITS}
    }
    if os.path.exists(CHECKPOINT_FILE):
        try:
            with open(CHECKPOINT_FILE, "r") as f:
                saved = json.load(f)
                progress["completed_datasets"] = saved.get("completed_datasets", [])
        except Exception:
            pass

    # Audit actual files physically on disk for ground truth counts
    for split in SPLITS:
        for cls in CLASSES:
            folder = os.path.join(TARGET_DIR, split, cls)
            if os.path.exists(folder):
                progress["counts"][split][cls] = len(os.listdir(folder))

    return progress


def save_progress(progress):
    os.makedirs(TARGET_DIR, exist_ok=True)
    # Sync disk counts before saving
    for split in SPLITS:
        for cls in CLASSES:
            folder = os.path.join(TARGET_DIR, split, cls)
            if os.path.exists(folder):
                progress["counts"][split][cls] = len(os.listdir(folder))
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(progress, f, indent=2)


def setup_dirs():
    for split in SPLITS:
        for cls in CLASSES:
            os.makedirs(os.path.join(TARGET_DIR, split, cls), exist_ok=True)
    os.makedirs(TEMP_DIR, exist_ok=True)


def clean_temp_zips():
    """Removes any stale zip archives in temp_download to preserve disk space."""
    if os.path.exists(TEMP_DIR):
        for f in os.listdir(TEMP_DIR):
            if f.endswith(".zip"):
                try:
                    os.remove(os.path.join(TEMP_DIR, f))
                except Exception:
                    pass


def find_zip_in_dir(directory):
    if not os.path.exists(directory):
        return None
    zips = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".zip")]
    return zips[0] if zips else None


def split_items(items, train_ratio=0.70, val_ratio=0.15, seed=42):
    random.seed(seed)
    shuffled = list(items)
    random.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    return {
        "train": shuffled[:n_train],
        "val": shuffled[n_train:n_train + n_val],
        "test": shuffled[n_train + n_val:]
    }


def process_and_save_image(img_bytes, dest_path):
    """Resize to 128x128 in-memory and write immediately to keep memory & disk footprint minimal."""
    try:
        with Image.open(io.BytesIO(img_bytes)) as img:
            rgb = img.convert("RGB")
            resized = rgb.resize(TARGET_SIZE, Image.Resampling.LANCZOS)
            resized.save(dest_path, "JPEG", quality=88)
            return True
    except Exception:
        return False


def download_with_retry(api, dataset_slug, download_dir, max_retries=5):
    """Downloads dataset from Kaggle with automatic retries for unstable connections."""
    for attempt in range(1, max_retries + 1):
        try:
            print(f"\n[Download] Pulling '{dataset_slug}' (Attempt {attempt}/{max_retries})...")
            api.dataset_download_files(dataset_slug, path=download_dir, quiet=False, unzip=False)
            print(f"[Download] Successfully completed download for: {dataset_slug}")
            return True
        except Exception as e:
            print(f"[Warning] Download attempt {attempt} failed: {e}")
            if attempt < max_retries:
                wait_s = attempt * 5
                print(f"[Retry] Retrying in {wait_s} seconds (unstable network recovery)...")
                time.sleep(wait_s)
    return False


def process_zip_archive(zip_path, extraction_specs, progress, dataset_slug):
    """
    Extracts, stream-resizes, and routes images into big_data_test/ splits.
    
    extraction_specs: list of tuples (filter_func, target_class, prefix, max_samples)
    """
    with zipfile.ZipFile(zip_path, "r") as z:
        all_members = [f for f in z.namelist() if not f.endswith("/")]

        for filter_func, target_class, prefix, max_samples in extraction_specs:
            source_id = f"{dataset_slug}:{prefix}:{target_class}"
            if source_id in progress.get("completed_datasets", []):
                print(f"  [Checkpoint] '{source_id}' already processed. Skipping.")
                continue

            candidates = [f for f in all_members if filter_func(f) and f.lower().endswith((".png", ".jpg", ".jpeg"))]
            random.seed(42)
            random.shuffle(candidates)
            if max_samples:
                candidates = candidates[:max_samples]

            if not candidates:
                print(f"  [Notice] No matching files found for spec: {source_id}")
                continue

            splits = split_items(candidates)
            print(f"  Processing {len(candidates):,} '{target_class}' images ({prefix}) into (70/15/15)...")

            total_saved = 0
            for split, flist in splits.items():
                out_dir = os.path.join(TARGET_DIR, split, target_class)
                start_idx = len(os.listdir(out_dir))
                for idx, fname in enumerate(flist):
                    dest = os.path.join(out_dir, f"{prefix}_{split}_{start_idx + idx:05d}.jpg")
                    if os.path.exists(dest):
                        continue
                    try:
                        img_bytes = z.read(fname)
                        if process_and_save_image(img_bytes, dest):
                            total_saved += 1
                            if total_saved % 1000 == 0:
                                print(f"    -> Ingested {total_saved:,}/{len(candidates):,} ({prefix})...")
                    except Exception:
                        continue

            progress["completed_datasets"].append(source_id)
            save_progress(progress)
            print(f"  -> Checkpoint saved for '{source_id}' ({total_saved:,} added).")


def print_summary():
    print("\n" + "=" * 65)
    print("           BIG_DATA_TEST CURRENT SPLIT AUDIT")
    print("=" * 65)
    total_imgs = 0
    for split in SPLITS:
        print(f"\nSplit: {split.upper()}")
        subtotal = 0
        for cls in CLASSES:
            folder = os.path.join(TARGET_DIR, split, cls)
            count = len(os.listdir(folder)) if os.path.exists(folder) else 0
            subtotal += count
            total_imgs += count
            print(f"  {cls:<16}: {count:,} images")
        print(f"  Subtotal {split:<10}: {subtotal:,} images")
    print("-" * 65)
    print(f"Grand Total Ingested : {total_imgs:,} images")
    print("=" * 65 + "\n")


def main():
    print("=" * 70)
    print("      BIG DATA TEST: CHECKPOINTED 3-CLASS DATASET INGESTION")
    print("=" * 70)
    setup_dirs()
    progress = load_progress()

    api = KaggleApi()
    api.authenticate()

    # Complete Multi-Source Dataset Ingestion Pipeline
    datasets_pipeline = [
        # --- Batch 1 (Baseline: ~21.6K Images) ---
        {
            "slug": "ashishjangra27/face-mask-12k-images-dataset",
            "zip_name": "face-mask-12k-images-dataset.zip",
            "specs": [
                (lambda f: "WithoutMask" in f, "no_mask", "ashish_nomask", 6000),
                (lambda f: "WithMask" in f, "mask_proper", "ashish_proper", 6000)
            ]
        },
        {
            "slug": "hadjadjib/facemask-recognition-dataset",
            "zip_name": "facemask-recognition-dataset.zip",
            "specs": [
                (lambda f: "dataset/Correct/" in f, "mask_proper", "cmfd_proper", 5000)
            ]
        },
        {
            "slug": "zuhairtarif/facemask-dataset-with-without-incorrect",
            "zip_name": "facemask-dataset-with-without-incorrect.zip",
            "specs": [
                (lambda f: "mask_weared_incorrect" in f, "mask_improper", "imfd_improper", 3500),
                (lambda f: "without_mask" in f, "no_mask", "imfd_nomask", 2500)
            ]
        },
        {
            "slug": "spandanpatnaik09/face-mask-detectormask-not-mask-incorrect-mask",
            "zip_name": "face-mask-detectormask-not-mask-incorrect-mask.zip",
            "specs": [
                (lambda f: "incorrect" in f.lower(), "mask_improper", "spandan_improper", 3000),
                (lambda f: "without" in f.lower() or "not_mask" in f.lower(), "no_mask", "spandan_nomask", 2000),
                (lambda f: "with_mask" in f.lower() or ("mask" in f.lower() and "not" not in f.lower() and "incorrect" not in f.lower()), "mask_proper", "spandan_proper", 2000)
            ]
        },

        # --- Batch 2: Scaling to ~55K-60K Images (Focus on Balancing mask_improper) ---
        # 5. Vijaykumar 3-Class (232 MB): Rich set of incorrect masks + with/without mask
        {
            "slug": "vijaykumar1799/face-mask-detection",
            "zip_name": "face-mask-detection.zip",
            "specs": [
                (lambda f: "mask_weared_incorrect" in f.lower() or "incorrect" in f.lower(), "mask_improper", "vijay_improper", 4000),
                (lambda f: "without_mask" in f.lower(), "no_mask", "vijay_nomask", 3000),
                (lambda f: "with_mask" in f.lower() and "without" not in f.lower() and "incorrect" not in f.lower(), "mask_proper", "vijay_proper", 3000)
            ]
        },
        # 6. Omkar Gurav (171 MB): Clean real-world With/Without Mask faces
        {
            "slug": "omkargurav/face-mask-dataset",
            "zip_name": "face-mask-dataset.zip",
            "specs": [
                (lambda f: "with_mask" in f.lower() and "without" not in f.lower(), "mask_proper", "omkar_proper", 3500),
                (lambda f: "without_mask" in f.lower(), "no_mask", "omkar_nomask", 3500)
            ]
        },
        # 7. Suman Sid FaceMask (120 MB): Clean real-world Mask / Non-Mask
        {
            "slug": "sumansid/facemask-dataset",
            "zip_name": "facemask-dataset.zip",
            "specs": [
                (lambda f: "non" not in f.lower() and "without" not in f.lower() and "mask" in f.lower(), "mask_proper", "suman_proper", 2000),
                (lambda f: "non" in f.lower() or "without" in f.lower(), "no_mask", "suman_nomask", 2000)
            ]
        },
        # 8. Shiekh Burhan (2.5 GB): 14.5K images with dedicated incorrect_mask directory
        {
            "slug": "shiekhburhan/face-mask-dataset",
            "zip_name": "face-mask-dataset.zip",
            "specs": [
                (lambda f: "incorrect" in f.lower(), "mask_improper", "shiekh_improper", 7000),
                (lambda f: "with_mask" in f.lower() and "without" not in f.lower() and "incorrect" not in f.lower(), "mask_proper", "shiekh_proper", 3500),
                (lambda f: "without_mask" in f.lower() or "without" in f.lower(), "no_mask", "shiekh_nomask", 3500)
            ]
        },
        # 9. Kamalesh SP Face Mask Compliance (3.3 GB): Specifically created for 3-class compliance
        {
            "slug": "kamaleshsp/face-mask-compliance-detection-system-dataset-v1",
            "zip_name": "face-mask-compliance-detection-system-dataset-v1.zip",
            "specs": [
                (lambda f: "incorrect" in f.lower(), "mask_improper", "kamalesh_improper", 6000),
                (lambda f: ("proper" in f.lower() or "with_mask" in f.lower()) and "without" not in f.lower() and "incorrect" not in f.lower(), "mask_proper", "kamalesh_proper", 3500),
                (lambda f: "no_mask" in f.lower() or "without" in f.lower(), "no_mask", "kamalesh_nomask", 3500)
            ]
        }
    ]

    for item in datasets_pipeline:
        slug = item["slug"]
        zip_name = item["zip_name"]

        # Check if all specs for this dataset are already in progress
        all_done = all(f"{slug}:{spec[2]}:{spec[1]}" in progress.get("completed_datasets", []) for spec in item["specs"])
        if all_done:
            print(f"\n[Checkpoint] Dataset '{slug}' is completely processed. Skipping.")
            continue

        clean_temp_zips()

        # Download if not present
        zip_path = find_zip_in_dir(TEMP_DIR)
        if not zip_path or not os.path.exists(zip_path):
            success = download_with_retry(api, slug, TEMP_DIR)
            if not success:
                print(f"[Error] Skipping '{slug}' due to network failures. Progress is saved. Re-run anytime to retry.")
                continue
            zip_path = find_zip_in_dir(TEMP_DIR)

        if not zip_path or not os.path.exists(zip_path):
            print(f"[Error] Zip archive not found for '{slug}'. Skipping.")
            continue

        # Process & resize into big_data_test/
        process_zip_archive(zip_path, item["specs"], progress, slug)

        # Immediately delete the zip to save disk space
        clean_temp_zips()
        print(f"  Reclaimed disk space for '{slug}'.")

    # Remove temp dir if empty
    if os.path.exists(TEMP_DIR) and not os.listdir(TEMP_DIR):
        os.rmdir(TEMP_DIR)

    print_summary()


if __name__ == "__main__":
    main()
