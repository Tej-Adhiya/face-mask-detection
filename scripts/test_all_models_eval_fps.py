import os
import sys
import time
import json
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
EVAL_DIR = os.path.join(PROJECT_ROOT, "evaluation")

CLASS_NAMES = ["mask_improper", "mask_proper", "no_mask"]
IMG_SIZE = (128, 128)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

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

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def build_model(arch, weights_filename, device):
    weights_path = os.path.join(MODELS_DIR, weights_filename)
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Checkpoint not found: {weights_path}")

    if arch == "mobilenet_v2":
        model = models.mobilenet_v2(weights=None)
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=False),
            nn.Linear(model.last_channel, 3)
        )
    elif arch == "mobilenet_v3_large":
        model = models.mobilenet_v3_large(weights=None)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, 3)
    elif arch == "efficientnet_b0":
        model = models.efficientnet_b0(weights=None)
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=False),
            nn.Linear(model.classifier[1].in_features, 3)
        )
    elif arch == "resnet18":
        model = models.resnet18(weights=None)
        model.fc = nn.Sequential(
            nn.Dropout(p=0.2, inplace=False),
            nn.Linear(model.fc.in_features, 3)
        )
    else:
        raise ValueError(f"Unknown architecture: {arch}")

    try:
        checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(weights_path, map_location=device)

    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    file_size_mb = os.path.getsize(weights_path) / (1024 * 1024)
    params = count_parameters(model)
    return model, file_size_mb, params

@torch.no_grad()
def benchmark_latency_fps(model, device, n_warmup=25, n_runs=150):
    dummy_input = torch.randn(1, 3, 128, 128, device=device)
    
    # Warmup runs
    for _ in range(n_warmup):
        _ = model(dummy_input)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()

    t0 = time.time()
    for _ in range(n_runs):
        _ = model(dummy_input)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()
    elapsed = time.time() - t0

    latency_ms = (elapsed / n_runs) * 1000.0
    fps = n_runs / elapsed
    return latency_ms, fps

@torch.no_grad()
def run_synthetic_eval(model, device):
    test_dir = os.path.join(DATA_DIR, "test")
    if not os.path.exists(test_dir):
        return None, 0

    test_dataset = datasets.ImageFolder(root=test_dir, transform=eval_transform)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2)

    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    acc = (correct / total) * 100.0 if total > 0 else 0.0
    return acc, total

def load_historical_metrics(arch):
    # Match the existing evaluation summary JSONs
    json_path = os.path.join(EVAL_DIR, f"metrics_summary_{arch}.json")
    if not os.path.exists(json_path):
        json_path = os.path.join(EVAL_DIR, "metrics_summary_domainadapt.json")
    
    if os.path.exists(json_path):
        try:
            with open(json_path, "r") as f:
                d = json.load(f)
            rw_acc = d.get("real_world_holdout_test", {}).get("accuracy", 0.0)
            rep = d.get("real_world_holdout_test", {}).get("report", {})
            improper_f1 = rep.get("mask_improper", {}).get("f1-score", 0.0) * 100.0
            return rw_acc, improper_f1
        except Exception:
            pass
    return None, None

def main():
    device = get_device()
    print("=" * 80)
    print("      MASKNET MULTI-MODEL HARDWARE & EVALUATION BENCHMARK RUNNER")
    print("=" * 80)
    print(f"Device Type          : {device.type.upper()}")
    if device.type == "cuda":
        print(f"GPU Name             : {torch.cuda.get_device_name(0)}")
        print(f"CUDA Capability      : {torch.cuda.get_device_capability(0)}")
        total_vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"Total VRAM           : {total_vram:.2f} GB")
    print(f"PyTorch Version      : {torch.__version__}")
    print("=" * 80 + "\n")

    models_config = [
        ("resnet18", "best_resnet18_stage2_domainadapt.pth", "ResNet-18 (Champion)"),
        ("mobilenet_v3_large", "best_mobilenet_v3_large_stage2_domainadapt.pth", "MobileNetV3-Large"),
        ("efficientnet_b0", "best_efficientnet_b0_stage2_domainadapt.pth", "EfficientNet-B0"),
        ("mobilenet_v2", "best_mobilenet_v2_stage2_domainadapt.pth", "MobileNetV2 (Adapted)"),
    ]

    results = []

    for arch, ckpt_name, display_name in models_config:
        print(f"Evaluating [{display_name}]...")
        model, file_size, params = build_model(arch, ckpt_name, device)
        
        # Latency & FPS
        lat_ms, fps = benchmark_latency_fps(model, device)
        
        # Test accuracy verification
        test_acc, test_count = run_synthetic_eval(model, device)
        
        # Historical holdout metrics
        holdout_acc, improper_f1 = load_historical_metrics(arch)
        
        results.append({
            "arch": arch,
            "display_name": display_name,
            "size_mb": file_size,
            "params_m": params / 1e6,
            "latency_ms": lat_ms,
            "fps": fps,
            "test_acc": test_acc,
            "holdout_acc": holdout_acc,
            "improper_f1": improper_f1
        })
        print(f"  -> Latency: {lat_ms:.2f} ms | Throughput: {fps:.1f} FPS | Test Acc: {test_acc:.2f}%\n")

    # Save benchmark JSON for UI & docs
    output_json = os.path.join(EVAL_DIR, "live_benchmark_results.json")
    with open(output_json, "w") as f:
        json.dump({
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
            "results": results
        }, f, indent=2)

    # Print clean benchmark table
    print("\n" + "=" * 105)
    print(f"{'Model Architecture':<24} | {'Size':<8} | {'Params':<7} | {'Latency':<10} | {'FPS':<10} | {'Test Acc':<9} | {'Holdout Acc':<11} | {'Improper F1':<11}")
    print("=" * 105)
    for r in results:
        h_acc_str = f"{r['holdout_acc']:.2f}%" if r['holdout_acc'] else "N/A"
        imp_f1_str = f"{r['improper_f1']:.2f}%" if r['improper_f1'] else "N/A"
        t_acc_str = f"{r['test_acc']:.2f}%" if r['test_acc'] else "N/A"
        print(f"{r['display_name']:<24} | {r['size_mb']:>5.1f} MB | {r['params_m']:>5.1f}M | {r['latency_ms']:>7.2f} ms | {r['fps']:>7.1f} | {t_acc_str:>8} | {h_acc_str:>10} | {imp_f1_str:>10}")
    print("=" * 105)
    print(f"\n[Saved live benchmark metrics to: {output_json}]\n")

if __name__ == "__main__":
    main()
