import os
import sys
import io
import time
import json
import glob
import random
import argparse
import xml.etree.ElementTree as ET
from PIL import Image
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
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


class RandomJPEGCompression(object):
    """Simulates real-world JPEG compression artifacts by re-encoding in memory."""
    def __init__(self, quality_range=(30, 75), p=0.35):
        self.quality_range = quality_range
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            quality = random.randint(self.quality_range[0], self.quality_range[1])
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=quality)
            buf.seek(0)
            return Image.open(buf).convert("RGB")
        return img


def get_device():
    if torch.cuda.is_available():
        print(f"Hardware Acceleration : NVIDIA CUDA ({torch.cuda.get_device_name(0)})")
        print(f"Compute Capability    : {torch.cuda.get_device_capability(0)}")
        return torch.device("cuda")
    print("Warning: CUDA not detected, falling back to CPU.")
    return torch.device("cpu")


def get_data_loaders(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS):
    train_transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=(3, 5), sigma=(0.1, 1.8))
        ], p=0.30),
        RandomJPEGCompression(quality_range=(35, 75), p=0.35),
        transforms.RandomApply([
            transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25)
        ], p=0.35),
        transforms.RandomApply([
            transforms.RandomAffine(degrees=12, translate=(0.06, 0.06))
        ], p=0.30),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    val_transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    train_path = os.path.join(BIG_DATA_DIR, "train")
    val_path = os.path.join(BIG_DATA_DIR, "val")

    if not os.path.exists(train_path) or not os.path.exists(val_path):
        raise FileNotFoundError(f"Big data split directories not found in '{BIG_DATA_DIR}'!")

    train_dataset = datasets.ImageFolder(root=train_path, transform=train_transform)
    val_dataset = datasets.ImageFolder(root=val_path, transform=val_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )

    # Compute class distribution and dynamic class weights
    targets = [s[1] for s in train_dataset.samples]
    counts = Counter(targets)
    total_samples = len(targets)
    num_classes = len(train_dataset.classes)
    class_weights = [total_samples / (num_classes * counts[i]) for i in range(num_classes)]

    print("\n--- Big Data Loaders Initialized ---")
    print(f"  Training samples   : {len(train_dataset):,} (Batches: {len(train_loader)})")
    print(f"  Validation samples : {len(val_dataset):,} (Batches: {len(val_loader)})")
    print(f"  Classes ({num_classes})         : {train_dataset.classes}")
    for idx, cname in enumerate(train_dataset.classes):
        print(f"    - {cname:<16}: {counts[idx]:,} samples | Loss Weight: {class_weights[idx]:.4f}")

    return train_loader, val_loader, train_dataset.classes, torch.tensor(class_weights, dtype=torch.float)


def build_resnet18(num_classes=3):
    print("\nBuilding ResNet-18 (ImageNet Pretrained)...")
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Sequential(
        nn.Dropout(p=0.25, inplace=False),
        nn.Linear(model.fc.in_features, num_classes)
    )
    return model


def configure_stages(model, stage=1):
    if stage == 1:
        print("[Stage 1] Freezing ResNet-18 feature backbone. Training classification head only.")
        for name, param in model.named_parameters():
            if "fc" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
    elif stage == 2:
        print("[Stage 2] Unfreezing layer4 + layer3 for fine-tuning deep feature representations.")
        for name, param in model.named_parameters():
            if any(k in name for k in ["layer4", "layer3", "fc"]):
                param.requires_grad = True
            else:
                param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable parameters: {trainable:,} / {total:,} ({trainable/total*100:.1f}%)")


def train_epoch(model, dataloader, criterion, optimizer, scaler, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    use_amp = (device.type == "cuda")

    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        optimizer.zero_grad()

        if use_amp:
            with torch.amp.autocast("cuda"):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)

        if (batch_idx + 1) % 150 == 0 or (batch_idx + 1) == len(dataloader):
            print(f"    Batch [{batch_idx+1}/{len(dataloader)}] - Loss: {running_loss/total:.4f} | Acc: {correct/total*100:.2f}%")

    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)

    return running_loss / total, correct / total


@torch.no_grad()
def evaluate_real_world_holdout(model, device, class_names):
    """
    Evaluates model on the 700 untouched real-world scenes in data/test_real_world_holdout
    parsing bounding boxes from Pascal VOC XML annotations.
    """
    holdout_dir = os.path.join(ORIGINAL_DATA_DIR, "test_real_world_holdout")
    if not os.path.exists(holdout_dir):
        print("\n[Holdout] data/test_real_world_holdout not found. Skipping real-world holdout evaluation.")
        return None

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

    for xml_path in xml_files:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # Find matching image
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

                    total += 1
                    per_class_total[target_class] += 1
                    if pred_idx == target_idx:
                        correct += 1
                        per_class_correct[target_class] += 1
        except Exception:
            continue

    overall_acc = (correct / total * 100.0) if total > 0 else 0.0

    print("\n" + "=" * 70)
    print("      TRUE REAL-WORLD HOLDOUT BENCHMARK (data/test_real_world_holdout)")
    print(f"      Evaluated on {len(xml_files)} scenes ({total:,} ground-truth annotated faces)")
    print("=" * 70)
    print(f"Overall Real-World Holdout Accuracy: {overall_acc:.2f}%\n")
    for cname in class_names:
        c_acc = (per_class_correct[cname] / per_class_total[cname] * 100.0) if per_class_total[cname] > 0 else 0.0
        print(f"  {cname:<18}: {c_acc:.2f}% ({per_class_correct[cname]:,}/{per_class_total[cname]:,})")
    print("=" * 70 + "\n")

    return overall_acc


@torch.no_grad()
def evaluate_folder(model, folder_path, device, set_name):
    if not os.path.exists(folder_path):
        return None
    eval_transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    dataset = datasets.ImageFolder(root=folder_path, transform=eval_transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    correct = 0
    total = 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)

    acc = (correct / total * 100.0) if total > 0 else 0.0
    print(f"  {set_name:<30}: {acc:.2f}% ({correct:,}/{total:,})")
    return acc


def main():
    parser = argparse.ArgumentParser(description="Train ResNet-18 on Big Data with Class Weights & Holdout Verification")
    parser.add_argument("--stage1_epochs", type=int, default=3, help="Epochs for Stage 1 (head warmup)")
    parser.add_argument("--stage2_epochs", type=int, default=12, help="Max epochs for Stage 2 (deep fine-tuning)")
    parser.add_argument("--patience", type=int, default=3, help="Early stopping patience for Stage 2")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Batch size")
    parser.add_argument("--lr1", type=float, default=1e-3, help="Stage 1 learning rate")
    parser.add_argument("--lr2", type=float, default=1e-4, help="Stage 2 learning rate")
    args = parser.parse_args()

    os.makedirs(MODELS_DIR, exist_ok=True)
    device = get_device()

    train_loader, val_loader, class_names, class_weights = get_data_loaders(
        batch_size=args.batch_size,
        num_workers=NUM_WORKERS
    )

    model = build_resnet18(num_classes=len(class_names)).to(device)
    # Apply dynamic inverse-frequency class weights to neutralize majority bias
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device), label_smoothing=0.05)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_model_path = os.path.join(MODELS_DIR, "best_resnet18_bigdata.pth")

    # -------------------------------------------------------------
    # STAGE 1: Head Warmup (Frozen Backbone)
    # -------------------------------------------------------------
    print("\n" + "=" * 60)
    print("     STAGE 1: WARMUP TRAINING CLASSIFICATION HEAD")
    print("=" * 60)
    configure_stages(model, stage=1)
    optimizer1 = optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr1, weight_decay=1e-4)
    scheduler1 = optim.lr_scheduler.CosineAnnealingLR(optimizer1, T_max=args.stage1_epochs)

    for epoch in range(1, args.stage1_epochs + 1):
        t0 = time.time()
        print(f"\n--- Stage 1 Epoch [{epoch}/{args.stage1_epochs}] (lr={optimizer1.param_groups[0]['lr']:.6f}) ---")
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer1, scaler, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler1.step()
        elapsed = time.time() - t0

        print(f"Summary: Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}% ({elapsed:.1f}s)")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_acc": val_acc,
                "val_loss": val_loss,
                "classes": class_names,
                "arch": "resnet18",
                "training_dataset": "big_data_test_clean"
            }, best_model_path)
            print(f"  -> Saved Stage 1 best checkpoint (Val Loss: {val_loss:.4f}, Val Acc: {val_acc*100:.2f}%)")

    # -------------------------------------------------------------
    # STAGE 2: Deep Feature Fine-Tuning with Early Stopping
    # -------------------------------------------------------------
    print("\n" + "=" * 60)
    print("     STAGE 2: DEEP FEATURE FINE-TUNING (LAYER3 + LAYER4)")
    print(f"     Max Epochs: {args.stage2_epochs} | Early Stopping Patience: {args.patience}")
    print("=" * 60)
    configure_stages(model, stage=2)
    optimizer2 = optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr2, weight_decay=1e-4)
    scheduler2 = optim.lr_scheduler.CosineAnnealingLR(optimizer2, T_max=args.stage2_epochs)

    patience_counter = 0

    for epoch in range(1, args.stage2_epochs + 1):
        t0 = time.time()
        print(f"\n--- Stage 2 Epoch [{epoch}/{args.stage2_epochs}] (lr={optimizer2.param_groups[0]['lr']:.6f}) ---")
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer2, scaler, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler2.step()
        elapsed = time.time() - t0

        print(f"Summary: Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}% ({elapsed:.1f}s)")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_acc": val_acc,
                "val_loss": val_loss,
                "classes": class_names,
                "arch": "resnet18",
                "training_dataset": "big_data_test_clean"
            }, best_model_path)
            print(f"  -> Saved Stage 2 new best checkpoint (Val Loss: {val_loss:.4f}, Val Acc: {val_acc*100:.2f}%)")
        else:
            patience_counter += 1
            print(f"  -> No val_loss improvement for {patience_counter}/{args.patience} epochs.")
            if patience_counter >= args.patience:
                print(f"\n[Early Stopping Triggered] Validation loss has not improved for {args.patience} consecutive epochs.")
                break

    print(f"\nTraining Complete! Best Validation Accuracy: {best_val_acc*100:.2f}% (Loss: {best_val_loss:.4f})")
    print(f"Checkpoint Saved To: {best_model_path}")

    # -------------------------------------------------------------
    # COMPREHENSIVE MULTI-SET BENCHMARKING
    # -------------------------------------------------------------
    print("\nLoading best model checkpoint for multi-set evaluation...")
    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print("\n" + "=" * 70)
    print("            IN-DISTRIBUTION & SYNTHETIC TEST BENCHMARKS")
    print("=" * 70)
    evaluate_folder(model, os.path.join(BIG_DATA_DIR, "test"), device, "big_data_test/test (Clean)")
    evaluate_folder(model, os.path.join(ORIGINAL_DATA_DIR, "test"), device, "data/test (1,350 Synth)")

    # Crucial: Evaluate true real-world holdout
    rw_acc = evaluate_real_world_holdout(model, device, class_names)

    # Historical Comparison Table
    print("\n" + "=" * 70)
    print("          CHAMPION BENCHMARK COMPARISON MATRIX")
    print("=" * 70)
    print(f"{'Model Configuration':<35} | {'Real-World Holdout':<20} | {'Status'}")
    print("-" * 70)
    print(f"{'Baseline MobileNetV2 (No Adapt)':<35} | {'65.83%':<20} | Domain Gap Failure")
    print(f"{'MobileNetV2 (Domain-Adapted)':<35} | {'89.07%':<20} | Previous Baseline")
    print(f"{'MobileNetV3-Large':<35} | {'92.44%':<20} | Evaluated")
    print(f"{'ResNet-18 (Current Champion)':<35} | {'94.44%':<20} | Target Baseline")
    if rw_acc is not None:
        delta = rw_acc - 94.44
        sign = "+" if delta >= 0 else ""
        print(f"{'ResNet-18 (Big Data Clean ~45.5k)':<35} | {f'{rw_acc:.2f}% ({sign}{delta:.2f}%)':<20} | Evaluated Candidate")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
