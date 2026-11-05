import os
import glob
import random
import shutil
import xml.etree.ElementTree as ET
from PIL import Image

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

SOURCE_DIR = os.path.join(DATA_DIR, "test_real_world")
HOLDOUT_DIR = os.path.join(DATA_DIR, "test_real_world_holdout")
TRAIN_DIR = os.path.join(DATA_DIR, "train")

LABEL_MAP = {
    "mask_weared_incorrect": "mask_improper",
    "with_mask": "mask_proper",
    "without_mask": "no_mask"
}

def main():
    print("=== STEP 1: Re-splitting Real-World Dataset at Image Level ===")
    os.makedirs(HOLDOUT_DIR, exist_ok=True)

    # Gather all base image names that have both .png and .xml
    png_files = sorted(glob.glob(os.path.join(SOURCE_DIR, "*.png")))
    valid_bases = []
    for p in png_files:
        base = os.path.splitext(os.path.basename(p))[0]
        xml_path = os.path.join(SOURCE_DIR, base + ".xml")
        if os.path.exists(xml_path):
            valid_bases.append(base)

    print(f"Total verified image-annotation pairs: {len(valid_bases)}")

    # Deterministic shuffle
    random.shuffle(valid_bases)
    n_adapt = 153  # ~18% (153 / 853 = 17.94%)
    adapt_bases = set(valid_bases[:n_adapt])
    holdout_bases = set(valid_bases[n_adapt:])

    print(f"Split allocation (Image Level):")
    print(f"  Domain Adaptation Pool: {len(adapt_bases)} images (~18%)")
    print(f"  Untouched Test Holdout: {len(holdout_bases)} images (~82%)")

    # 1. Populate test_real_world_holdout
    print("\nPopulating data/test_real_world_holdout/ with untouched test scenes...")
    for base in holdout_bases:
        src_png = os.path.join(SOURCE_DIR, base + ".png")
        src_xml = os.path.join(SOURCE_DIR, base + ".xml")
        dst_png = os.path.join(HOLDOUT_DIR, base + ".png")
        dst_xml = os.path.join(HOLDOUT_DIR, base + ".xml")
        shutil.copy2(src_png, dst_png)
        shutil.copy2(src_xml, dst_xml)

    holdout_pngs = len(glob.glob(os.path.join(HOLDOUT_DIR, "*.png")))
    holdout_xmls = len(glob.glob(os.path.join(HOLDOUT_DIR, "*.xml")))
    print(f"  Holdout set ready: {holdout_pngs} PNG images, {holdout_xmls} XML annotations.")

    # 2. Extract crops from domain_adapt pool and add to data/train/<class>/
    print("\nExtracting annotated faces from domain_adapt pool into data/train/<class>/...")
    added_counts = {"mask_proper": 0, "mask_improper": 0, "no_mask": 0}

    for base in adapt_bases:
        img_path = os.path.join(SOURCE_DIR, base + ".png")
        xml_path = os.path.join(SOURCE_DIR, base + ".xml")

        tree = ET.parse(xml_path)
        root = tree.getroot()

        with Image.open(img_path) as img:
            img_w, img_h = img.size
            for obj_idx, obj in enumerate(root.findall("object")):
                tag = obj.find("name").text
                if tag not in LABEL_MAP:
                    continue

                bndbox = obj.find("bndbox")
                if bndbox is None:
                    continue

                xmin = max(0, int(float(bndbox.find("xmin").text)))
                ymin = max(0, int(float(bndbox.find("ymin").text)))
                xmax = min(img_w, int(float(bndbox.find("xmax").text)))
                ymax = min(img_h, int(float(bndbox.find("ymax").text)))

                w = xmax - xmin
                h = ymax - ymin
                # Filter out degenerate boxes under 8px
                if w < 8 or h < 8:
                    continue

                target_class = LABEL_MAP[tag]
                
                # Contextual margin (15%) + square preservation
                cx = (xmin + xmax) / 2.0
                cy = (ymin + ymax) / 2.0
                side = max(w, h) * 1.15
                x1 = max(0, int(cx - side / 2.0))
                y1 = max(0, int(cy - side / 2.0))
                x2 = min(img_w, int(cx + side / 2.0))
                y2 = min(img_h, int(cy + side / 2.0))

                crop = img.crop((x1, y1, x2, y2)).convert("RGB")
                resized_crop = crop.resize((128, 128), Image.Resampling.LANCZOS)

                out_fname = f"real_andrewmvd_domainadapt_train_{base}_{obj_idx:02d}.jpg"
                out_path = os.path.join(TRAIN_DIR, target_class, out_fname)
                resized_crop.save(out_path, "JPEG", quality=90)
                added_counts[target_class] += 1

    print("\n--- Domain Adaptation Ingestion Summary ---")
    print(f"Faces added to data/train/mask_proper:   +{added_counts['mask_proper']} faces")
    print(f"Faces added to data/train/mask_improper: +{added_counts['mask_improper']} faces")
    print(f"Faces added to data/train/no_mask:       +{added_counts['no_mask']} faces")
    total_added = sum(added_counts.values())
    print(f"Total real-world faces added to train:   +{total_added} faces")

    print("\nUpdated Training Set Class Counts:")
    for cls in ["mask_proper", "mask_improper", "no_mask"]:
        p = os.path.join(TRAIN_DIR, cls)
        print(f"  {cls}: {len(os.listdir(p))} images")

if __name__ == "__main__":
    main()
