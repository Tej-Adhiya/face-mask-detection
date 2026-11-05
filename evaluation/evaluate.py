import os
import sys
import glob
import json
import argparse
import xml.etree.ElementTree as ET
from PIL import Image

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

# ---------------------------------------------------------------------------
# Path Configuration
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
EVAL_DIR = os.path.join(PROJECT_ROOT, "evaluation")

CLASS_NAMES = ["mask_improper", "mask_proper", "no_mask"]
CLASS_TO_IDX = {cls_name: i for i, cls_name in enumerate(CLASS_NAMES)}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMG_SIZE = (128, 128)

eval_transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(weights_path, device, arch="mobilenet_v2"):
    """Rebuild model with 3-class head and load trained checkpoint."""
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Model checkpoint not found at '{weights_path}'!")
    
    print(f"Loading {arch} weights from: {weights_path}")
    if arch == "mobilenet_v2":
        model = models.mobilenet_v2(weights=None)
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=False),
            nn.Linear(model.last_channel, len(CLASS_NAMES))
        )
    elif arch == "mobilenet_v3_large":
        model = models.mobilenet_v3_large(weights=None)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, len(CLASS_NAMES))
    elif arch == "efficientnet_b0":
        model = models.efficientnet_b0(weights=None)
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=False),
            nn.Linear(model.classifier[1].in_features, len(CLASS_NAMES))
        )
    elif arch == "resnet18":
        model = models.resnet18(weights=None)
        model.fc = nn.Sequential(
            nn.Dropout(p=0.2, inplace=False),
            nn.Linear(model.fc.in_features, len(CLASS_NAMES))
        )
    else:
        raise ValueError(f"Unsupported architecture: {arch}")

    checkpoint = torch.load(weights_path, map_location=device)
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def plot_confusion_matrix(cm, class_names, title, output_path):
    fig, ax = plt.subplots(figsize=(6, 5))
    cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-9)

    cax = ax.imshow(cm_norm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.set_title(title, fontsize=12, pad=12)
    fig.colorbar(cax)

    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(class_names, rotation=25, ha="right", fontsize=10)
    ax.set_yticklabels(class_names, fontsize=10)

    thresh = cm_norm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm_norm[i, j] > thresh else "black"
            text = f"{cm[i, j]:,}\n({cm_norm[i, j]:.1%})"
            ax.text(j, i, text, ha="center", va="center", color=color, fontsize=9)

    ax.set_ylabel('True Ground Truth', fontsize=11)
    ax.set_xlabel('Predicted Label', fontsize=11)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"  Confusion matrix plot saved: {output_path}")


def evaluate_synthetic_test(model, device, suffix=""):
    test_dir = os.path.join(DATA_DIR, "test")
    if not os.path.exists(test_dir):
        raise FileNotFoundError(f"Test split directory not found at '{test_dir}'!")

    test_dataset = datasets.ImageFolder(root=test_dir, transform=eval_transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)

    print(f"\n=======================================================")
    print(f" [1/2] Evaluating on Synthetic Test Set ({len(test_dataset)} images)")
    print(f"=======================================================")

    all_preds, all_targets = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    acc = np.mean(all_preds == all_targets) * 100.0
    cm = confusion_matrix(all_targets, all_preds)
    report = classification_report(all_targets, all_preds, target_names=CLASS_NAMES, output_dict=True)

    print(f"\nSynthetic Test Accuracy: {acc:.2f}%\n")
    print(classification_report(all_targets, all_preds, target_names=CLASS_NAMES))

    cm_path = os.path.join(EVAL_DIR, f"confusion_matrix_synthetic{suffix}.png")
    plot_confusion_matrix(cm, CLASS_NAMES, f"Synthetic Test Set (Acc: {acc:.2f}%)", cm_path)

    return {
        "accuracy": acc,
        "confusion_matrix": cm.tolist(),
        "report": report
    }


def evaluate_real_world_crops(model, device, use_holdout=True, suffix=""):
    candidate_paths = [
        os.path.join(DATA_DIR, "test_real_world_holdout"),
        os.path.join(DATA_DIR, "evaluation_dataset"),
        os.path.join(PROJECT_ROOT, "evaluation_dataset"),
        os.path.join(DATA_DIR, "test_real_world"),
    ]
    rw_dir = next((p for p in candidate_paths if os.path.exists(p) and glob.glob(os.path.join(p, "*.xml"))), None)
    if rw_dir:
        set_name = f"Real-World Dataset ({os.path.basename(rw_dir)})"
    else:
        rw_dir = os.path.join(DATA_DIR, "test_real_world")
        set_name = "Real-World Set"


    xml_files = sorted(glob.glob(os.path.join(rw_dir, "*.xml")))

    print(f"\n=======================================================")
    print(f" [2/2] Evaluating on {set_name}")
    print(f"       ({len(xml_files)} scenes from andrewmvd/face-mask-detection)")
    print(f"=======================================================")

    label_map = {
        "mask_weared_incorrect": "mask_improper",
        "with_mask": "mask_proper",
        "without_mask": "no_mask"
    }

    crops = []
    ground_truths = []

    for xml_path in xml_files:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        filename_elem = root.find("filename")
        fname = filename_elem.text if filename_elem is not None else None
        img_path = None
        if fname:
            candidate = os.path.join(rw_dir, fname)
            if os.path.exists(candidate):
                img_path = candidate
        if not img_path:
            base = os.path.splitext(os.path.basename(xml_path))[0]
            for ext in [".png", ".jpg", ".jpeg"]:
                candidate = os.path.join(rw_dir, base + ext)
                if os.path.exists(candidate):
                    img_path = candidate
                    break

        if not img_path:
            continue

        try:
            with Image.open(img_path) as img:
                img_w, img_h = img.size

                for obj in root.findall("object"):
                    tag = obj.find("name").text
                    if tag not in label_map:
                        continue

                    bndbox = obj.find("bndbox")
                    if bndbox is None:
                        continue

                    xmin = max(0, int(float(bndbox.find("xmin").text)))
                    ymin = max(0, int(float(bndbox.find("ymin").text)))
                    xmax = min(img_w, int(float(bndbox.find("xmax").text)))
                    ymax = min(img_h, int(float(bndbox.find("ymax").text)))

                    if (xmax - xmin) < 8 or (ymax - ymin) < 8:
                        continue

                    # Contextual margin (15%) + square preservation
                    cx = (xmin + xmax) / 2.0
                    cy = (ymin + ymax) / 2.0
                    side = max(xmax - xmin, ymax - ymin) * 1.15
                    x1 = max(0, int(cx - side / 2.0))
                    y1 = max(0, int(cy - side / 2.0))
                    x2 = min(img_w, int(cx + side / 2.0))
                    y2 = min(img_h, int(cy + side / 2.0))

                    crop = img.crop((x1, y1, x2, y2)).convert("RGB")
                    tensor_crop = eval_transform(crop)
                    crops.append(tensor_crop)
                    ground_truths.append(CLASS_TO_IDX[label_map[tag]])
        except Exception:
            continue

    total_crops = len(crops)
    print(f"Total valid annotated face crops evaluated: {total_crops:,}")

    batch_size = 64
    all_preds = []

    with torch.no_grad():
        for i in range(0, total_crops, batch_size):
            batch = torch.stack(crops[i:i + batch_size]).to(device)
            outputs = model(batch)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(ground_truths)

    acc = np.mean(all_preds == all_targets) * 100.0
    cm = confusion_matrix(all_targets, all_preds)
    report = classification_report(all_targets, all_preds, target_names=CLASS_NAMES, output_dict=True)

    print(f"\nReal-World Holdout Accuracy: {acc:.2f}%\n")
    print(classification_report(all_targets, all_preds, target_names=CLASS_NAMES))

    cm_path = os.path.join(EVAL_DIR, f"confusion_matrix_real_world{suffix}.png")
    plot_confusion_matrix(cm, CLASS_NAMES, f"Real-World Holdout (Acc: {acc:.2f}%)", cm_path)

    return {
        "accuracy": acc,
        "confusion_matrix": cm.tolist(),
        "report": report,
        "total_crops": total_crops
    }


def plot_generalization_gap(synth_metrics, rw_metrics, output_path):
    fig, ax = plt.subplots(figsize=(8, 5))

    classes = CLASS_NAMES + ["Overall Accuracy"]
    synth_f1 = [synth_metrics["report"][c]["f1-score"] * 100.0 for c in CLASS_NAMES] + [synth_metrics["accuracy"]]
    rw_f1 = [rw_metrics["report"][c]["f1-score"] * 100.0 for c in CLASS_NAMES] + [rw_metrics["accuracy"]]

    x = np.arange(len(classes))
    width = 0.35

    rects1 = ax.bar(x - width/2, synth_f1, width, label='Synthetic Test Split', color='#2b5c8f')
    rects2 = ax.bar(x + width/2, rw_f1, width, label='Real-World Holdout Crops', color='#d95f02')

    ax.set_ylabel('Performance Score (%)', fontsize=11)
    ax.set_title('Generalization: Synthetic vs Real-World Holdout', fontsize=12, pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(classes, fontsize=10)
    ax.set_ylim(0, 110)
    ax.grid(True, axis='y', alpha=0.3)
    ax.legend(loc='lower right', fontsize=10)

    for rects in [rects1, rects2]:
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}%',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"  Generalization gap plot saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Multi-Model Evaluation & Domain Adaptation Benchmark")
    parser.add_argument("--arch", type=str, default="mobilenet_v2",
                        choices=["mobilenet_v2", "mobilenet_v3_large", "efficientnet_b0", "resnet18"],
                        help="Model architecture")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to trained checkpoint (.pth)")
    parser.add_argument("--suffix", type=str, default="", help="Suffix for output plots and JSON")
    args = parser.parse_args()

    os.makedirs(EVAL_DIR, exist_ok=True)
    device = get_device()

    model_path = args.model_path
    if model_path is None:
        # Default lookup
        candidates = [
            os.path.join(MODELS_DIR, f"best_{args.arch}_stage2_domainadapt.pth"),
            os.path.join(MODELS_DIR, f"best_{args.arch}_stage2.pth"),
        ]
        if args.arch == "mobilenet_v2":
            candidates.append(os.path.join(MODELS_DIR, "best_model_stage2.pth"))
        for c in candidates:
            if os.path.exists(c):
                model_path = c
                break

    if not model_path or not os.path.exists(model_path):
        raise FileNotFoundError(f"Could not find model checkpoint for {args.arch} in '{MODELS_DIR}'!")

    model = load_model(model_path, device, arch=args.arch)

    # 1. Synthetic Test Split
    synth_results = evaluate_synthetic_test(model, device, suffix=args.suffix)

    # 2. Real-World Holdout Set
    rw_results = evaluate_real_world_crops(model, device, use_holdout=True, suffix=args.suffix)

    # 3. Plot Comparison
    gap_plot_path = os.path.join(EVAL_DIR, f"generalization_gap{args.suffix}.png")
    plot_generalization_gap(synth_results, rw_results, gap_plot_path)

    # 4. Save metrics summary
    summary = {
        "arch": args.arch,
        "model_weights_evaluated": os.path.basename(model_path),
        "synthetic_test": synth_results,
        "real_world_holdout_test": rw_results,
        "generalization_gap_percent": synth_results["accuracy"] - rw_results["accuracy"]
    }
    summary_path = os.path.join(EVAL_DIR, f"metrics_summary{args.suffix}.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=======================================================")
    print(f" Evaluation Complete for {args.arch} ({os.path.basename(model_path)})!")
    print(f"  Synthetic Test Accuracy:       {synth_results['accuracy']:.2f}%")
    print(f"  Real-World Holdout Accuracy:   {rw_results['accuracy']:.2f}%")
    print(f"  Generalization Gap:           {summary['generalization_gap_percent']:.2f}%")
    print(f"  Metrics JSON saved:           {summary_path}")
    print(f"=======================================================")


if __name__ == "__main__":
    main()
