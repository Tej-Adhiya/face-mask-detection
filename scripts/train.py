import os
import sys
import io
import time
import json
import random
import argparse
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

# ---------------------------------------------------------------------------
# Path Configuration
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
NUM_WORKERS = 2

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class RandomJPEGCompression(object):
    """Simulates real-world JPEG compression artifacts by re-encoding in memory."""
    def __init__(self, quality_range=(30, 70), p=0.4):
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
    """Detect and return NVIDIA CUDA (Windows/Linux), Apple Silicon MPS (macOS), or CPU fallback."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using Hardware Acceleration: NVIDIA CUDA ({torch.cuda.get_device_name(0)})")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
        print("Using Hardware Acceleration: Apple Silicon MPS (Metal Performance Shaders)")
    else:
        device = torch.device("cpu")
        print("Warning: GPU acceleration not available. Using CPU.")
    return device


def get_data_loaders(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, use_domain_aug=True):
    """
    Build ImageFolder DataLoaders for train and validation.
    Applies domain-gap simulating augmentation to TRAINING pipeline only:
      - Gaussian blur (CCTV/surveillance sensor blur)
      - In-memory random JPEG compression (compression artifacts)
      - ColorJitter (lighting and shadow variations)
      - RandomAffine (slight rotation and translation for off-angle faces)
      - RandomHorizontalFlip
    """
    if use_domain_aug:
        train_transform = transforms.Compose([
            transforms.Resize(IMG_SIZE),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=(3, 5), sigma=(0.1, 2.0))
            ], p=0.35),
            RandomJPEGCompression(quality_range=(30, 70), p=0.4),
            transforms.RandomApply([
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3)
            ], p=0.4),
            transforms.RandomApply([
                transforms.RandomAffine(degrees=15, translate=(0.08, 0.08))
            ], p=0.35),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize(IMG_SIZE),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    val_transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    train_path = os.path.join(DATA_DIR, "train")
    val_path = os.path.join(DATA_DIR, "val")

    if not os.path.exists(train_path) or not os.path.exists(val_path):
        raise FileNotFoundError(f"Training or validation directory not found in '{DATA_DIR}'!")

    train_dataset = datasets.ImageFolder(root=train_path, transform=train_transform)
    val_dataset = datasets.ImageFolder(root=val_path, transform=val_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False
    )

    print(f"DataLoaders initialized (Domain Augmentation: {use_domain_aug}):")
    print(f"  Train samples: {len(train_dataset)} | Batches: {len(train_loader)}")
    print(f"  Val samples:   {len(val_dataset)} | Batches: {len(val_loader)}")
    print(f"  Classes ({len(train_dataset.classes)}): {train_dataset.classes}")

    return train_loader, val_loader, train_dataset.classes


def build_model(arch="mobilenet_v2", num_classes=3):
    """Instantiate model architecture and customize classification head for num_classes."""
    print(f"\nBuilding model: {arch} (ImageNet pretrained)...")
    if arch == "mobilenet_v2":
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=False),
            nn.Linear(model.last_channel, num_classes)
        )
    elif arch == "mobilenet_v3_large":
        model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    elif arch == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=False),
            nn.Linear(model.classifier[1].in_features, num_classes)
        )
    elif arch == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        model.fc = nn.Sequential(
            nn.Dropout(p=0.2, inplace=False),
            nn.Linear(model.fc.in_features, num_classes)
        )
    else:
        raise ValueError(f"Unsupported architecture: {arch}")

    return model


def set_backbone_trainable(model, arch="mobilenet_v2", stage=1):
    """Configure freezing based on architecture and training stage."""
    if stage == 1:
        print(f"\n[Stage 1 Configuration] Freezing backbone for {arch}. Head is trainable.")
        if arch in ["mobilenet_v2", "efficientnet_b0"]:
            for p in model.features.parameters():
                p.requires_grad = False
            for p in model.classifier.parameters():
                p.requires_grad = True
        elif arch == "mobilenet_v3_large":
            for p in model.features.parameters():
                p.requires_grad = False
            for p in model.classifier.parameters():
                p.requires_grad = True
        elif arch == "resnet18":
            for n, p in model.named_parameters():
                if "fc" not in n:
                    p.requires_grad = False
                else:
                    p.requires_grad = True

    elif stage == 2:
        print(f"\n[Stage 2 Configuration] Unfreezing top feature layers for {arch}.")
        if arch == "mobilenet_v2":
            for i in range(14):
                for p in model.features[i].parameters(): p.requires_grad = False
            for i in range(14, len(model.features)):
                for p in model.features[i].parameters(): p.requires_grad = True
            for p in model.classifier.parameters(): p.requires_grad = True
        elif arch == "mobilenet_v3_large":
            for i in range(13):
                for p in model.features[i].parameters(): p.requires_grad = False
            for i in range(13, len(model.features)):
                for p in model.features[i].parameters(): p.requires_grad = True
            for p in model.classifier.parameters(): p.requires_grad = True
        elif arch == "efficientnet_b0":
            for i in range(6):
                for p in model.features[i].parameters(): p.requires_grad = False
            for i in range(6, len(model.features)):
                for p in model.features[i].parameters(): p.requires_grad = True
            for p in model.classifier.parameters(): p.requires_grad = True
        elif arch == "resnet18":
            for n, p in model.named_parameters():
                if "layer4" in n or "fc" in n:
                    p.requires_grad = True
                else:
                    p.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable parameters: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, (correct / total) * 100.0


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, (correct / total) * 100.0


def run_training_stage(model, arch, train_loader, val_loader, device, stage, epochs, lr, suffix=""):
    os.makedirs(MODELS_DIR, exist_ok=True)
    set_backbone_trainable(model, arch=arch, stage=stage)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(trainable_params, lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    scheduler = None
    if stage == 2:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_val_acc = 0.0
    best_val_loss = float("inf")
    patience = 5
    patience_counter = 0

    history = {
        "arch": arch,
        "stage": stage,
        "epochs": epochs,
        "lr": lr,
        "train_loss": [], "val_loss": [],
        "train_acc": [], "val_acc": [],
        "epoch_times": []
    }

    save_name = f"best_{arch}_stage{stage}{suffix}.pth"
    save_path = os.path.join(MODELS_DIR, save_name)
    print(f"\n--- Starting {arch} Stage {stage} ({epochs} epochs, lr={lr}) ---")

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        elapsed = time.time() - t0

        if scheduler:
            scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["epoch_times"].append(elapsed)

        is_best = False
        if val_acc > best_val_acc or (val_acc == best_val_acc and val_loss < best_val_loss):
            best_val_acc = val_acc
            best_val_loss = val_loss
            is_best = True
            torch.save({
                "arch": arch,
                "epoch": epoch,
                "stage": stage,
                "model_state_dict": model.state_dict(),
                "val_acc": val_acc,
                "val_loss": val_loss,
            }, save_path)
            patience_counter = 0
        else:
            patience_counter += 1

        best_flag = " [BEST SAVED]" if is_best else ""
        print(f"Epoch {epoch:02d}/{epochs:02d} [{elapsed:.1f}s] - "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% - "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%{best_flag}")

        if stage == 2 and patience_counter >= patience:
            print(f"\nEarly stopping triggered after {patience} epochs without validation accuracy improvement.")
            break

        if device.type == "mps":
            torch.mps.empty_cache()
        elif device.type == "cuda":
            torch.cuda.empty_cache()

    history_file = os.path.join(MODELS_DIR, f"history_{arch}_stage{stage}{suffix}.json")
    with open(history_file, "w") as f:
        json.dump(history, f, indent=2)
    print(f"Stage {stage} complete! Best Val Acc: {best_val_acc:.2f}%. Saved: {save_path}")

    return history


def plot_curves(histories, output_path=None):
    if output_path is None:
        output_path = os.path.join(MODELS_DIR, "training_curves.png")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    all_train_loss, all_val_loss = [], []
    all_train_acc, all_val_acc = [], []
    stage_boundaries = []
    curr_ep = 0

    for h in histories:
        all_train_loss.extend(h["train_loss"])
        all_val_loss.extend(h["val_loss"])
        all_train_acc.extend(h["train_acc"])
        all_val_acc.extend(h["val_acc"])
        curr_ep += len(h["train_loss"])
        stage_boundaries.append(curr_ep)

    epochs = range(1, len(all_train_loss) + 1)

    ax1.plot(epochs, all_train_loss, "b-o", label="Train Loss", markersize=4)
    ax1.plot(epochs, all_val_loss, "r-s", label="Val Loss", markersize=4)
    for b in stage_boundaries[:-1]:
        ax1.axvline(x=b, color="gray", linestyle="--", label="Stage 2 Transition")
    ax1.set_title("Training & Validation Loss", fontsize=13)
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Loss", fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.plot(epochs, all_train_acc, "b-o", label="Train Accuracy", markersize=4)
    ax2.plot(epochs, all_val_acc, "g-^", label="Val Accuracy", markersize=4)
    ax2.axhline(y=90.0, color="orange", linestyle=":", label="90% Target Baseline")
    for b in stage_boundaries[:-1]:
        ax2.axvline(x=b, color="gray", linestyle="--", label="Stage 2 Transition")
    ax2.set_title("Training & Validation Accuracy (%)", fontsize=13)
    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("Accuracy (%)", fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"Training curves saved to '{output_path}'")


def main():
    parser = argparse.ArgumentParser(description="Multi-Model Mask Classifier Training with Domain Adaptation")
    parser.add_argument("--arch", type=str, default="mobilenet_v2",
                        choices=["mobilenet_v2", "mobilenet_v3_large", "efficientnet_b0", "resnet18"],
                        help="Model architecture to train")
    parser.add_argument("--stage", type=str, default="all", choices=["1", "2", "all"],
                        help="Training stage: 1, 2, or all (default: all)")
    parser.add_argument("--stage1_epochs", type=int, default=8, help="Epochs for Stage 1")
    parser.add_argument("--stage2_epochs", type=int, default=10, help="Epochs for Stage 2")
    parser.add_argument("--stage1_lr", type=float, default=1e-3, help="Stage 1 learning rate")
    parser.add_argument("--stage2_lr", type=float, default=1e-4, help="Stage 2 learning rate")
    parser.add_argument("--suffix", type=str, default="_domainadapt",
                        help="Checkpoint name suffix (default: _domainadapt)")
    parser.add_argument("--no_domain_aug", action="store_true", help="Disable domain-gap augmentation")
    args = parser.parse_args()

    device = get_device()
    train_loader, val_loader, class_names = get_data_loaders(use_domain_aug=not args.no_domain_aug)

    histories = []
    suffix = args.suffix

    if args.stage in ["1", "all"]:
        model = build_model(arch=args.arch, num_classes=len(class_names)).to(device)
        h1 = run_training_stage(
            model=model,
            arch=args.arch,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            stage=1,
            epochs=args.stage1_epochs,
            lr=args.stage1_lr,
            suffix=suffix
        )
        histories.append(h1)

    if args.stage in ["2", "all"]:
        stage1_ckpt = os.path.join(MODELS_DIR, f"best_{args.arch}_stage1{suffix}.pth")
        if not os.path.exists(stage1_ckpt):
            stage1_ckpt = os.path.join(MODELS_DIR, f"best_{args.arch}_stage1.pth")
        if not os.path.exists(stage1_ckpt) and args.arch == "mobilenet_v2":
            stage1_ckpt = os.path.join(MODELS_DIR, "best_model_stage1.pth")
        if not os.path.exists(stage1_ckpt):
            raise FileNotFoundError(f"Stage 1 checkpoint not found for {args.arch} in '{MODELS_DIR}'!")

        print(f"\nLoading Stage 1 checkpoint from '{stage1_ckpt}' for fine-tuning...")
        model = build_model(arch=args.arch, num_classes=len(class_names)).to(device)
        checkpoint = torch.load(stage1_ckpt, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

        h2 = run_training_stage(
            model=model,
            arch=args.arch,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            stage=2,
            epochs=args.stage2_epochs,
            lr=args.stage2_lr,
            suffix=suffix
        )
        histories.append(h2)

    if histories:
        plot_path = os.path.join(MODELS_DIR, f"training_curves_{args.arch}{suffix}.png")
        plot_curves(histories, output_path=plot_path)


if __name__ == "__main__":
    main()
