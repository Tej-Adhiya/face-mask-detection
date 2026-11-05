import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from src.face_detector import FaceDetector, draw_detection_box, draw_hud_overlay
from src.compliance_scorer import ComplianceScorer, RollingComplianceTracker

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

MODEL_WEIGHTS_MAP = {
    "resnet18_bigdata": "best_resnet18_bigdata.pth",
    "resnet18_baseline": "best_resnet18_stage2_domainadapt.pth",
    "resnet18": "best_resnet18_bigdata.pth",
    "mobilenet_v3_large": "best_mobilenet_v3_large_stage2_domainadapt.pth",
    "efficientnet_b0": "best_efficientnet_b0_stage2_domainadapt.pth",
    "mobilenet_v2": "best_mobilenet_v2_stage2_domainadapt.pth",
}


def load_classifier(arch: str, device: torch.device):
    """Instantiates model architecture and loads checkpoint."""
    weights_name = MODEL_WEIGHTS_MAP.get(arch, "best_resnet18_bigdata.pth")
    weights_path = os.path.join(MODELS_DIR, weights_name)

    if not os.path.exists(weights_path):
        # Fallback check for alternate resnet18 checkpoint
        if "resnet18" in arch:
            alt_path = os.path.join(MODELS_DIR, "best_resnet18_stage2_domainadapt.pth")
            if os.path.exists(alt_path):
                weights_path = alt_path
            else:
                raise FileNotFoundError(f"Model checkpoint not found: {weights_path}")
        else:
            raise FileNotFoundError(f"Model checkpoint not found: {weights_path}")

    if arch in ["resnet18", "resnet18_bigdata", "resnet18_baseline"]:
        model = models.resnet18(weights=None)
        model.fc = nn.Sequential(nn.Dropout(p=0.25, inplace=False), nn.Linear(model.fc.in_features, 3))
    elif arch == "mobilenet_v3_large":
        model = models.mobilenet_v3_large(weights=None)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, 3)
    elif arch == "efficientnet_b0":
        model = models.efficientnet_b0(weights=None)
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=False),
            nn.Linear(model.classifier[1].in_features, 3)
        )
    elif arch == "mobilenet_v2":
        model = models.mobilenet_v2(weights=None)
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=False),
            nn.Linear(model.last_channel, 3)
        )
    else:
        raise ValueError(f"Unknown architecture: {arch}")

    try:
        ckpt = torch.load(weights_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(weights_path, map_location=device)

    state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


class MaskCompliancePipeline:
    """
    Complete End-to-End Face Detection + Classification + Explainable Compliance System.
    """
    def __init__(
        self,
        arch="resnet18_bigdata",
        conf_threshold=0.5,
        min_compliance_conf=0.50,
        detector_backend="auto",
        safe_threshold=75.0,
        min_margin=0.15,
        improper_weight=0.25,
        temperature=1.10,
        min_bbox_size=35
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.arch = arch
        self.classifier = load_classifier(arch, self.device)
        self.detector = FaceDetector(backend=detector_backend, conf_threshold=conf_threshold)
        self.scorer = ComplianceScorer(
            min_confidence=min_compliance_conf,
            safe_threshold=safe_threshold,
            min_bbox_size=min_bbox_size,
            min_margin=min_margin,
            improper_weight=improper_weight,
            temperature=temperature
        )
        self.rolling_tracker = RollingComplianceTracker(window_size=30)

    def set_model(self, arch: str):
        if arch != self.arch:
            self.classifier = load_classifier(arch, self.device)
            self.arch = arch

    def set_detector(self, backend: str):
        if hasattr(self.detector, "set_backend"):
            self.detector.set_backend(backend)

    def set_thresholds(
        self,
        detection_conf: float = None,
        min_compliance_conf: float = None,
        safe_threshold: float = None,
        min_margin: float = None,
        improper_weight: float = None,
        temperature: float = None,
        min_bbox_size: int = None
    ):
        if detection_conf is not None:
            self.detector.set_conf_threshold(float(detection_conf))
        if min_compliance_conf is not None:
            self.scorer.min_confidence = float(min_compliance_conf)
        if safe_threshold is not None:
            self.scorer.safe_threshold = float(safe_threshold)
        if min_margin is not None:
            self.scorer.min_margin = float(min_margin)
        if improper_weight is not None:
            self.scorer.improper_weight = float(improper_weight)
        if temperature is not None:
            self.scorer.temperature = max(0.1, float(temperature))
        if min_bbox_size is not None:
            self.scorer.min_bbox_size = int(min_bbox_size)

    @torch.no_grad()
    def process_frame(self, frame_bgr: np.ndarray, return_annotated=True, draw_hud=False, fps=None):
        """
        Processes single BGR image or video frame:
        1. Detects faces with 15% contextual padding
        2. Classifies each face with the PyTorch model
        3. Computes calibrated probabilities, margins, and compliance score
        4. Draws styled annotations on the frame (and optional HUD banner)
        
        Returns:
            annotated_frame: np.ndarray BGR
            detections: list of scored face dictionaries
            summary: dict with frame-level statistics
        """
        annotated_frame = frame_bgr.copy() if return_annotated else None
        faces = self.detector.detect_faces(frame_bgr)
        scored_faces = []
        frame_scores = []

        safe_count = 0
        violation_count = 0
        uncertain_count = 0

        for idx, f in enumerate(faces):
            face_id = idx + 1
            tensor = f["tensor"].to(self.device)
            logits = self.classifier(tensor)
            logits_np = logits.cpu().numpy()[0]
            probs_np = F.softmax(logits, dim=1).cpu().numpy()[0]

            score_info = self.scorer.score_prediction(probs_np, bbox=f["bbox"], raw_logits=logits_np)
            h_frame, w_frame = frame_bgr.shape[:2]
            pbox = f.get("padded_bbox", f["bbox"])
            vis_box = [max(0, pbox[0]), max(0, pbox[1]), min(w_frame, pbox[2]), min(h_frame, pbox[3])]

            score_info["face_id"] = face_id
            score_info["raw_bbox"] = f["bbox"]
            score_info["bbox"] = vis_box
            score_info["padded_bbox"] = pbox
            score_info["det_confidence"] = f["confidence"]

            status = score_info["status"]
            if status == "SAFE":
                safe_count += 1
            elif status == "VIOLATION":
                violation_count += 1
            else:
                uncertain_count += 1

            frame_scores.append(score_info["compliance_score"])
            scored_faces.append(score_info)

            if return_annotated:
                draw_detection_box(
                    annotated_frame,
                    vis_box,
                    score_info["display_label"],
                    score_info["compliance_score"],
                    score_info["status"],
                    face_id=face_id
                )

        avg_score = float(np.mean(frame_scores)) if frame_scores else 100.0
        rolling_score = self.rolling_tracker.update(frame_scores)

        # Hardware display description
        if self.device.type == "cuda":
            hw_string = "GPU (CUDA Accelerated)"
        else:
            hw_string = "CPU (Host Processor)"

        summary = {
            "total_faces": len(faces),
            "safe_count": safe_count,
            "violation_count": violation_count,
            "uncertain_count": uncertain_count,
            "frame_compliance_score": round(avg_score, 1),
            "rolling_compliance_score": round(rolling_score, 1),
            "hardware": hw_string
        }

        if return_annotated and draw_hud and annotated_frame is not None:
            draw_hud_overlay(annotated_frame, summary, fps=fps)

        return annotated_frame, scored_faces, summary
