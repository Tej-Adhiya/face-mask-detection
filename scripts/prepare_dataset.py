import os
import io
import random
import shutil
import zipfile
from PIL import Image
from kaggle.api.kaggle_api_extended import KaggleApi

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

TARGET_DIR = "data"
SPLITS = ["train", "val", "test"]
CLASSES = ["mask_proper", "mask_improper", "no_mask"]

def setup_dirs():
    for split in SPLITS:
        for cls in CLASSES:
            os.makedirs(os.path.join(TARGET_DIR, split, cls), exist_ok=True)
    os.makedirs(os.path.join(TARGET_DIR, "test_real_world"), exist_ok=True)
    os.makedirs("temp_download", exist_ok=True)

def split_items(items, train_ratio=0.70, val_ratio=0.15):
    """Deterministically partition items into 70/15/15 splits."""
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

def process_and_save_image(img_bytes, dest_path, target_size=(128, 128)):
    """Resize image to 128x128 and save immediately, keeping memory footprint minimal."""
    with Image.open(io.BytesIO(img_bytes)) as img:
        img_rgb = img.convert("RGB")
        resized = img_rgb.resize(target_size, Image.Resampling.LANCZOS)
        resized.save(dest_path, "JPEG", quality=90)

def main():
    print("=== Step 3 & 4: Setting up directories and gathering dataset ===")
    setup_dirs()
    
    api = KaggleApi()
    api.authenticate()

    # -------------------------------------------------------------
    # 1. Ashish Jangra 12K dataset (No Mask: 3000, Mask Proper: 2500)
    # -------------------------------------------------------------
    print("\n[1/4] Downloading ashishjangra27/face-mask-12k-images-dataset...")
    api.dataset_download_files("ashishjangra27/face-mask-12k-images-dataset", path="temp_download", quiet=False)
    zpath_ashish = "temp_download/face-mask-12k-images-dataset.zip"

    with zipfile.ZipFile(zpath_ashish, "r") as z:
        all_files = [f for f in z.namelist() if not f.endswith("/")]
        
        # No Mask: sample 3000
        no_mask_candidates = [f for f in all_files if "WithoutMask" in f and f.lower().endswith((".png", ".jpg", ".jpeg"))]
        random.shuffle(no_mask_candidates)
        no_mask_selected = no_mask_candidates[:3000]
        no_mask_splits = split_items(no_mask_selected)

        print(f"Streaming {len(no_mask_selected)} No Mask images (70/15/15)...")
        for split, flist in no_mask_splits.items():
            for idx, fname in enumerate(flist):
                dest = os.path.join(TARGET_DIR, split, "no_mask", f"no_mask_ashish_{split}_{idx:04d}.jpg")
                img_bytes = z.read(fname)
                process_and_save_image(img_bytes, dest)

        # Mask Proper: sample 2500 from Ashish (we'll add 500 CMFD for total 3000)
        with_mask_candidates = [f for f in all_files if "WithMask" in f and f.lower().endswith((".png", ".jpg", ".jpeg"))]
        random.shuffle(with_mask_candidates)
        proper_ashish_selected = with_mask_candidates[:2500]
        proper_ashish_splits = split_items(proper_ashish_selected)

        print(f"Streaming {len(proper_ashish_selected)} Mask Proper images from Ashish (70/15/15)...")
        for split, flist in proper_ashish_splits.items():
            for idx, fname in enumerate(flist):
                dest = os.path.join(TARGET_DIR, split, "mask_proper", f"proper_ashish_{split}_{idx:04d}.jpg")
                img_bytes = z.read(fname)
                process_and_save_image(img_bytes, dest)

    os.remove(zpath_ashish)
    print("Cleaned up ashishjangra27 zip.")

    # -------------------------------------------------------------
    # 2. CMFD from hadjadjib (Mask Proper supplement: 500 images)
    # -------------------------------------------------------------
    print("\n[2/4] Downloading CMFD supplement from hadjadjib/facemask-recognition-dataset...")
    api.dataset_download_files("hadjadjib/facemask-recognition-dataset", path="temp_download", quiet=False)
    zpath_cmfd = "temp_download/facemask-recognition-dataset.zip"

    with zipfile.ZipFile(zpath_cmfd, "r") as z:
        all_cmfd = [f for f in z.namelist() if "dataset/Correct/" in f and f.lower().endswith((".png", ".jpg", ".jpeg"))]
        random.shuffle(all_cmfd)
        cmfd_selected = all_cmfd[:500]
        cmfd_splits = split_items(cmfd_selected)

        print(f"Streaming {len(cmfd_selected)} CMFD images (70/15/15)...")
        for split, flist in cmfd_splits.items():
            for idx, fname in enumerate(flist):
                dest = os.path.join(TARGET_DIR, split, "mask_proper", f"proper_cmfd_{split}_{idx:04d}.jpg")
                img_bytes = z.read(fname)
                process_and_save_image(img_bytes, dest)

    os.remove(zpath_cmfd)
    print("Cleaned up CMFD zip.")

    # -------------------------------------------------------------
    # 3. IMFD from zuhairtarif (Mask Improper: 3000 images)
    # -------------------------------------------------------------
    print("\n[3/4] Downloading IMFD from zuhairtarif/facemask-dataset-with-without-incorrect...")
    api.dataset_download_files("zuhairtarif/facemask-dataset-with-without-incorrect", path="temp_download", quiet=False)
    zpath_imfd = "temp_download/facemask-dataset-with-without-incorrect.zip"

    with zipfile.ZipFile(zpath_imfd, "r") as z:
        imfd_candidates = [f for f in z.namelist() if "mask_weared_incorrect" in f and f.lower().endswith((".png", ".jpg", ".jpeg"))]
        random.shuffle(imfd_candidates)
        imfd_selected = imfd_candidates[:3000]
        imfd_splits = split_items(imfd_selected)

        print(f"Streaming {len(imfd_selected)} IMFD images (70/15/15)...")
        for split, flist in imfd_splits.items():
            for idx, fname in enumerate(flist):
                dest = os.path.join(TARGET_DIR, split, "mask_improper", f"improper_imfd_{split}_{idx:04d}.jpg")
                img_bytes = z.read(fname)
                process_and_save_image(img_bytes, dest)

    os.remove(zpath_imfd)
    print("Cleaned up IMFD zip.")

    # -------------------------------------------------------------
    # 4. Held-out real-world test set (andrewmvd: full 853 images untouched)
    # -------------------------------------------------------------
    print("\n[4/4] Downloading andrewmvd/face-mask-detection (untouched real-world set)...")
    api.dataset_download_files("andrewmvd/face-mask-detection", path="temp_download", quiet=False)
    zpath_andrew = "temp_download/face-mask-detection.zip"

    with zipfile.ZipFile(zpath_andrew, "r") as z:
        # Extract all files into data/test_real_world preserving structure
        for member in z.infolist():
            if not member.is_dir():
                # Strip leading directory names if any, extract into test_real_world
                fname = os.path.basename(member.filename)
                if fname:
                    dest_file = os.path.join(TARGET_DIR, "test_real_world", fname)
                    with z.open(member) as source, open(dest_file, "wb") as target:
                        shutil.copyfileobj(source, target)

    os.remove(zpath_andrew)
    if os.path.exists("temp_download"):
        os.rmdir("temp_download")
    print("Cleaned up andrewmvd zip and temp folder.")

    # -------------------------------------------------------------
    # Print summary of results
    # -------------------------------------------------------------
    print("\n=== Dataset Ingestion Complete ===")
    total_imgs = 0
    for split in SPLITS:
        print(f"\nSplit: {split}")
        split_total = 0
        for cls in CLASSES:
            folder = os.path.join(TARGET_DIR, split, cls)
            count = len(os.listdir(folder))
            split_total += count
            total_imgs += count
            print(f"  {cls}: {count} images")
        print(f"  Subtotal {split}: {split_total}")

    real_world_folder = os.path.join(TARGET_DIR, "test_real_world")
    rw_files = os.listdir(real_world_folder)
    rw_imgs = [f for f in rw_files if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    rw_xmls = [f for f in rw_files if f.lower().endswith(".xml")]
    print(f"\ntest_real_world (untouched): {len(rw_imgs)} images, {len(rw_xmls)} XML annotations")

if __name__ == "__main__":
    main()
