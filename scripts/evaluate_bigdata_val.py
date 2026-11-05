import os
import json
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VAL_DIR = os.path.join(PROJECT_ROOT, "big_data_test", "val")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
EVAL_DIR = os.path.join(PROJECT_ROOT, "evaluation")

IMG_SIZE = (128, 128)
BATCH_SIZE = 128
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def main():
    os.makedirs(EVAL_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    weights_path = os.path.join(MODELS_DIR, "best_resnet18_bigdata.pth")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Checkpoint not found: {weights_path}")

    print(f"Loading checkpoint: {weights_path}")
    checkpoint = torch.load(weights_path, map_location=device)
    classes = checkpoint.get("classes", ["mask_improper", "mask_proper", "no_mask"])

    model = models.resnet18(weights=None)
    model.fc = nn.Sequential(
        nn.Dropout(p=0.25, inplace=False),
        nn.Linear(model.fc.in_features, len(classes))
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    val_transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

    val_dataset = datasets.ImageFolder(root=VAL_DIR, transform=val_transform)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"Evaluating {len(val_dataset):,} samples across classes: {val_dataset.classes}")
    t0 = time.time()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for i, (inputs, targets) in enumerate(val_loader):
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(targets.numpy())
            if (i + 1) % 10 == 0 or (i + 1) == len(val_loader):
                print(f"  Processed [{len(all_preds)}/{len(val_dataset)}] faces...")

    elapsed = time.time() - t0
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # Compute 3x3 Confusion Matrix using numpy
    num_classes = len(val_dataset.classes)
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(all_targets, all_preds):
        cm[t, p] += 1

    acc = (np.trace(cm) / np.sum(cm)) * 100.0

    # Calculate per-class metrics
    report = {}
    for idx, cname in enumerate(val_dataset.classes):
        tp = cm[idx, idx]
        fp = np.sum(cm[:, idx]) - tp
        fn = np.sum(cm[idx, :]) - tp
        support = int(np.sum(cm[idx, :]))

        prec = (tp / (tp + fp)) * 100.0 if (tp + fp) > 0 else 0.0
        rec = (tp / (tp + fn)) * 100.0 if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        report[cname] = {
            "precision": prec / 100.0,
            "recall": rec / 100.0,
            "f1-score": f1 / 100.0,
            "support": support
        }

    print("\n" + "=" * 65)
    print(f"BIG DATA VALIDATION EVALUATION RESULTS ({len(val_dataset):,} Samples)")
    print("=" * 65)
    print(f"Overall Validation Accuracy: {acc:.2f}% (Evaluation Time: {elapsed:.2f}s)")
    for c in val_dataset.classes:
        r = report[c]
        print(f"  {c:<16}: Precision {r['precision']*100:.2f}% | Recall {r['recall']*100:.2f}% | F1 {r['f1-score']*100:.2f}% | Support {r['support']}")
    print("=" * 65)
    print("\nConfusion Matrix:")
    print(cm)

    # Save metrics JSON
    metrics_data = {
        "dataset": "big_data_test/val",
        "total_samples": len(val_dataset),
        "overall_accuracy": acc,
        "evaluation_time_seconds": elapsed,
        "confusion_matrix": cm.tolist(),
        "classes": val_dataset.classes,
        "classification_report": report
    }
    json_path = os.path.join(EVAL_DIR, "metrics_summary_bigdata_val.json")
    with open(json_path, "w") as f:
        json.dump(metrics_data, f, indent=2)
    print(f"\nSaved metrics to: {json_path}")

    # Plot Confusion Matrix Heatmap with pure matplotlib
    display_labels = ["Improper", "Proper", "No Mask"]
    fig, ax = plt.subplots(figsize=(7, 6), dpi=300)
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=11)

    ax.set_xticks(np.arange(num_classes))
    ax.set_yticks(np.arange(num_classes))
    ax.set_xticklabels(display_labels, fontsize=12)
    ax.set_yticklabels(display_labels, fontsize=12)

    ax.set_title(f"Confusion Matrix: ResNet-18 on Big Data Validation Set\n({len(val_dataset):,} Faces | Accuracy: {acc:.2f}%)", fontsize=13, weight="bold", pad=15)
    ax.set_xlabel("Predicted Class", fontsize=12, labelpad=10)
    ax.set_ylabel("Ground Truth Class", fontsize=12, labelpad=10)

    # Annotate numbers with dynamic text color based on cell intensity
    thresh = cm.max() / 2.0
    for i in range(num_classes):
        for j in range(num_classes):
            color = "white" if cm[i, j] > thresh else "black"
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", color=color, fontsize=14, weight="bold")

    plt.tight_layout()
    out_img = os.path.join(EVAL_DIR, "confusion_matrix_bigdata_val.png")
    plt.savefig(out_img, dpi=300)
    plt.close()
    print(f"Saved confusion matrix plot to: {out_img}")

if __name__ == "__main__":
    main()
