import os
import time
import torch
import torch.nn as nn
from torchvision import models

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    return torch.device("cpu")

def load_arch(arch, weights_file, device):
    if arch == "mobilenet_v2":
        model = models.mobilenet_v2(weights=None)
        model.classifier = nn.Sequential(nn.Dropout(p=0.2), nn.Linear(model.last_channel, 3))
    elif arch == "mobilenet_v3_large":
        model = models.mobilenet_v3_large(weights=None)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, 3)
    elif arch == "efficientnet_b0":
        model = models.efficientnet_b0(weights=None)
        model.classifier = nn.Sequential(nn.Dropout(p=0.2), nn.Linear(model.classifier[1].in_features, 3))
    elif arch == "resnet18":
        model = models.resnet18(weights=None)
        model.fc = nn.Sequential(nn.Dropout(p=0.2), nn.Linear(model.fc.in_features, 3))

    ckpt = torch.load(os.path.join(MODELS_DIR, weights_file), map_location=device)
    state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model

@torch.no_grad()
def benchmark(model, device, n_warmup=20, n_runs=100):
    dummy_input = torch.randn(1, 3, 128, 128, device=device)
    
    # Warmup
    for _ in range(n_warmup):
        _ = model(dummy_input)
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()

    t0 = time.time()
    for _ in range(n_runs):
        _ = model(dummy_input)
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - t0

    latency_ms = (elapsed / n_runs) * 1000.0
    fps = n_runs / elapsed
    return latency_ms, fps

def main():
    device = get_device()
    print(f"Device: {device}")
    
    configs = [
        ("mobilenet_v2", "best_mobilenet_v2_stage2_domainadapt.pth"),
        ("efficientnet_b0", "best_efficientnet_b0_stage2_domainadapt.pth"),
        ("mobilenet_v3_large", "best_mobilenet_v3_large_stage2_domainadapt.pth"),
        ("resnet18", "best_resnet18_stage2_domainadapt.pth"),
    ]

    print("\n" + "=" * 70)
    print(f"{'Architecture':<20} | {'Latency (ms)':<15} | {'Throughput (FPS)':<18}")
    print("=" * 70)

    for arch, ckpt in configs:
        model = load_arch(arch, ckpt, device)
        lat, fps = benchmark(model, device)
        print(f"{arch:<20} | {lat:>12.2f} ms | {fps:>15.1f} FPS")

    print("=" * 70)

if __name__ == "__main__":
    main()
