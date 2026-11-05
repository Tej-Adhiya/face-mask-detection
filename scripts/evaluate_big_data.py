import os
import sys
import time
import json
import glob
import xml.etree.ElementTree as ET
from PIL import Image
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIG_DATA_DIR = os.path.join(PROJECT_ROOT, "big_data_test")
ORIGINAL_DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
EVAL_DIR = os.path.join(PROJECT_ROOT, "evaluation")

IMG_SIZE = (128, 128)
BATCH_SIZE = 64
NUM_WORKERS = 2

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_device():
    if torch.cuda.is_available():
        print(f"Device               : NVIDIA CUDA ({torch.cuda.get_device_name(0)})")
        return torch.device("cuda")
    print("Warning: CUDA not detected, running on CPU.")
    return torch.device("cpu")


def build_resnet18(num_classes=3):
    model = models.resnet18(weights=None)
    model.fc = nn.Sequential(
        nn.Dropout(p=0.25, inplace=False),
        nn.Linear(model.fc.in_features, num_classes)
    )
    return model


@torch.no_grad()
def evaluate_folder(model, folder_path, device, set_name):
    if not os.path.exists(folder_path):
        print(f"  {set_name:<30}: Directory not found!")
        return 0.0, 0, {}

    eval_transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    dataset = datasets.ImageFolder(root=folder_path, transform=eval_transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    correct = 0
    total = 0
    per_class_correct = Counter()
    per_class_total = Counter()

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)

        for p, t in zip(preds.cpu().numpy(), targets.cpu().numpy()):
            c_name = dataset.classes[t]
            per_class_total[c_name] += 1
            if p == t:
                per_class_correct[c_name] += 1

    acc = (correct / total * 100.0) if total > 0 else 0.0
    print(f"  {set_name:<30}: {acc:.2f}% ({correct:,}/{total:,})")
    for cname in dataset.classes:
        c_acc = (per_class_correct[cname] / per_class_total[cname] * 100.0) if per_class_total[cname] > 0 else 0.0
        print(f"    - {cname:<16}: {c_acc:.2f}% ({per_class_correct[cname]:,}/{per_class_total[cname]:,})")

    breakdown = {
        "accuracy": acc,
        "total": total,
        "per_class": {c: {"correct": per_class_correct[c], "total": per_class_total[c]} for c in dataset.classes}
    }
    return acc, total, breakdown


@torch.no_grad()
def evaluate_real_world_holdout(model, device, class_names):
    candidate_paths = [
        os.path.join(ORIGINAL_DATA_DIR, "test_real_world_holdout"),
        os.path.join(ORIGINAL_DATA_DIR, "evaluation_dataset"),
        os.path.join(PROJECT_ROOT, "evaluation_dataset"),
    ]
    holdout_dir = next((p for p in candidate_paths if os.path.exists(p) and glob.glob(os.path.join(p, "*.xml"))), None)
    if not holdout_dir:
        print(f"Evaluation holdout dataset not found in any standard location: {candidate_paths}")
        print("Please place the real-world holdout set in data/test_real_world_holdout/ to run this benchmark.")
        return 0.0, 0, {}


    xml_files = sorted(glob.glob(os.path.join(holdout_dir, "*.xml")))
    label_map = {
        "mask_weared_incorrect": "mask_improper",
        "with_mask": "mask_proper",
        "without_mask": "no_mask"
    }

    eval_transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    correct = 0
    total = 0
    per_class_correct = Counter()
    per_class_total = Counter()
    per_class_pred = Counter()

    for xml_path in xml_files:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        base = os.path.splitext(os.path.basename(xml_path))[0]
        img_path = None
        for ext in [".png", ".jpg", ".jpeg"]:
            candidate = os.path.join(holdout_dir, base + ext)
            if os.path.exists(candidate):
                img_path = candidate
                break

        if not img_path:
            continue

        try:
            with Image.open(img_path) as full_img:
                img_w, img_h = full_img.size

                for obj in root.findall("object"):
                    tag = obj.find("name").text
                    if tag not in label_map:
                        continue

                    target_class = label_map[tag]
                    target_idx = class_to_idx[target_class]

                    bndbox = obj.find("bndbox")
                    if bndbox is None:
                        continue

                    xmin = max(0, int(float(bndbox.find("xmin").text)))
                    ymin = max(0, int(float(bndbox.find("ymin").text)))
                    xmax = min(img_w, int(float(bndbox.find("xmax").text)))
                    ymax = min(img_h, int(float(bndbox.find("ymax").text)))

                    if xmax - xmin < 8 or ymax - ymin < 8:
                        continue

                    crop = full_img.crop((xmin, ymin, xmax, ymax)).convert("RGB")
                    tensor = eval_transform(crop).unsqueeze(0).to(device)

                    output = model(tensor)
                    pred_idx = torch.argmax(output, dim=1).item()
                    pred_class = class_names[pred_idx]

                    total += 1
                    per_class_total[target_class] += 1
                    per_class_pred[pred_class] += 1
                    if pred_idx == target_idx:
                        correct += 1
                        per_class_correct[target_class] += 1
        except Exception:
            continue

    overall_acc = (correct / total * 100.0) if total > 0 else 0.0

    print("\n" + "=" * 75)
    print("      TRUE REAL-WORLD HOLDOUT BENCHMARK (data/test_real_world_holdout)")
    print(f"      Evaluated on {len(xml_files)} scenes ({total:,} ground-truth annotated faces)")
    print("=" * 75)
    print(f"Overall Real-World Holdout Accuracy: {overall_acc:.2f}%\n")

    f1_scores = {}
    for cname in class_names:
        tp = per_class_correct[cname]
        p_total = per_class_pred[cname]
        g_total = per_class_total[cname]

        precision = (tp / p_total * 100.0) if p_total > 0 else 0.0
        recall = (tp / g_total * 100.0) if g_total > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        f1_scores[cname] = f1

        print(f"  {cname:<16}: Recall: {recall:>6.2f}% ({tp:>4}/{g_total:<4}) | Precision: {precision:>6.2f}% | F1: {f1:>6.2f}%")

    print("=" * 75 + "\n")

    breakdown = {
        "accuracy": overall_acc,
        "total_faces": total,
        "per_class": {
            c: {
                "recall": (per_class_correct[c] / per_class_total[c] * 100.0) if per_class_total[c] > 0 else 0.0,
                "precision": (per_class_correct[c] / per_class_pred[c] * 100.0) if per_class_pred[c] > 0 else 0.0,
                "f1": f1_scores[c],
                "correct": per_class_correct[c],
                "ground_truth_total": per_class_total[c]
            } for c in class_names
        }
    }
    return overall_acc, total, breakdown


def main():
    device = get_device()
    checkpoint_path = os.path.join(MODELS_DIR, "best_resnet18_bigdata.pth")
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        return

    print("=" * 75)
    print("      RESNET-18 BIG DATA TEST COMPREHENSIVE BENCHMARK")
    print("=" * 75)
    print(f"Checkpoint Path      : {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    class_names = checkpoint.get("classes", ["mask_improper", "mask_proper", "no_mask"])
    val_acc = checkpoint.get("val_acc", 0.0) * 100.0
    val_loss = checkpoint.get("val_loss", 0.0)
    epoch = checkpoint.get("epoch", 0)

    print(f"Architecture         : ResNet-18")
    print(f"Training Epoch       : {epoch} (Best Validation Checkpoint)")
    print(f"Validation Accuracy  : {val_acc:.2f}% (Val Loss: {val_loss:.4f})")
    print(f"Class Names          : {class_names}")
    print("=" * 75 + "\n")

    model = build_resnet18(num_classes=len(class_names))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    print("--- [1/3] IN-DISTRIBUTION & SYNTHETIC BENCHMARKS ---")
    acc_bigtest, _, data_bigtest = evaluate_folder(model, os.path.join(BIG_DATA_DIR, "test"), device, "big_data_test/test (Clean)")
    acc_synth, _, data_synth = evaluate_folder(model, os.path.join(ORIGINAL_DATA_DIR, "test"), device, "data/test (1,350 Synth)")

    print("\n--- [2/3] TRUE REAL-WORLD SURVEILLANCE HOLDOUT BENCHMARK ---")
    rw_acc, rw_total, rw_data = evaluate_real_world_holdout(model, device, class_names)

    # Save summary JSON
    results = {
        "model_name": "best_resnet18_bigdata",
        "architecture": "resnet18",
        "training_set": "big_data_test_clean (34,622 unique)",
        "best_epoch": epoch,
        "validation_accuracy": val_acc,
        "validation_loss": val_loss,
        "big_data_test_acc": acc_bigtest,
        "synthetic_test_acc": acc_synth,
        "real_world_holdout": rw_data
    }
    json_out = os.path.join(EVAL_DIR, "metrics_summary_resnet18_bigdata.json")
    with open(json_out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Benchmark metrics saved to: {json_out}")

    print("\n" + "=" * 75)
    print("            CHAMPION MODEL BENCHMARK COMPARISON MATRIX")
    print("=" * 75)
    print(f"{'Model Configuration':<36} | {'Real-World Acc':<16} | {'Improper F1':<12} | {'Notes'}")
    print("-" * 75)
    print(f"{'Baseline MobileNetV2 (No Adapt)':<36} | {'65.83%':<16} | {'13.00%':<12} | Severe Domain Gap")
    print(f"{'MobileNetV2 (Domain-Adapted)':<36} | {'89.07%':<16} | {'38.73%':<12} | Initial Baseline")
    print(f"{'EfficientNet-B0':<36} | {'91.38%':<16} | {'52.89%':<12} | High Latency")
    print(f"{'MobileNetV3-Large':<36} | {'92.44%':<16} | {'53.27%':<12} | Strong Edge Baseline")
    print(f"{'ResNet-18 Champion (Domain-Adapted)':<36} | {'94.44%':<16} | {'60.44%':<12} | Target Baseline")
    improper_f1 = rw_data.get("per_class", {}).get("mask_improper", {}).get("f1", 0.0)
    delta = rw_acc - 94.44
    sign = "+" if delta >= 0 else ""
    print(f"{'ResNet-18 (Big Data Clean ~45.5K)':<36} | {f'{rw_acc:.2f}% ({sign}{delta:.2f}%)':<16} | {f'{improper_f1:.2f}%':<12} | Multi-Source Generalization")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    main()
