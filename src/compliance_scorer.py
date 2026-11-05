import numpy as np
import torch
import torch.nn.functional as F
from collections import deque

CLASS_NAMES = ["mask_improper", "mask_proper", "no_mask"]
DISPLAY_LABELS = {
    "mask_proper": "Mask Worn Properly",
    "mask_improper": "Mask Worn Improperly",
    "no_mask": "No Mask"
}


class ComplianceScorer:
    """
    Intelligent & Explainable Safety Compliance Scoring System
    
    Synthesizes:
      1. Temperature-Scaled Probability Calibration (Guo et al., 2017)
      2. Continuous Expected Protection Score:
           Score = 100 * (p_proper + lambda * p_improper)
      3. Multi-Gate Selective Rejection (Reject Option, Geifman & El-Yaniv):
           - Gate 1 (Resolution): min(W, H) >= min_bbox_size (default: 35px)
           - Gate 2 (Confidence): max(p) >= min_confidence (default: 0.60)
           - Gate 3 (Decision Margin): top1 - top2 >= min_margin (default: 0.20)
      4. Asymmetric Safety Authorization:
           - SAFE strictly requires top_class == 'mask_proper', score >= safe_threshold,
             and margin >= min_margin_safe.
           - UNCERTAIN if any rejection gate fails.
           - VIOLATION for confirmed bare faces or improper mask wearing.
      5. Full Explainability (XAI audit trails with human-readable rationale).
    """
    def __init__(
        self,
        min_confidence: float = 0.50,
        safe_threshold: float = 75.0,
        min_bbox_size: int = 35,
        min_margin: float = 0.15,
        min_margin_safe: float = 0.20,
        improper_weight: float = 0.25,
        temperature: float = 1.10
    ):
        self.min_confidence = min_confidence
        self.safe_threshold = safe_threshold
        self.min_bbox_size = min_bbox_size
        self.min_margin = min_margin
        self.min_margin_safe = min_margin_safe
        self.improper_weight = improper_weight
        self.temperature = max(0.1, temperature)

    def calibrate_logits(self, logits: np.ndarray, temp: float = None) -> np.ndarray:
        """Applies temperature scaling to raw logits and returns calibrated probabilities."""
        t = temp if temp is not None else self.temperature
        scaled = logits / t
        exp_logits = np.exp(scaled - np.max(scaled))  # numerically stable softmax
        return exp_logits / np.sum(exp_logits)

    def score_prediction(self, probs: np.ndarray, bbox=None, raw_logits=None) -> dict:
        """
        Computes safety compliance score, multi-gate status, and explainability audit.

        Args:
            probs: 1D array of 3 probabilities [p_improper, p_proper, p_nomask]
            bbox: Optional [xmin, ymin, xmax, ymax] for resolution evaluation.
            raw_logits: Optional 1D array of 3 raw logits for temperature calibration.

        Returns:
            dict with class labels, calibrated confidences, margins, compliance score,
            status (SAFE, VIOLATION, UNCERTAIN), and explainable audit trail.
        """
        raw_probs = np.array(probs, dtype=float)
        
        # Apply temperature calibration if raw logits are supplied
        if raw_logits is not None:
            calibrated_probs = self.calibrate_logits(np.array(raw_logits, dtype=float))
            eval_probs = calibrated_probs
            is_calibrated = True
        else:
            eval_probs = raw_probs
            calibrated_probs = raw_probs
            is_calibrated = False

        # Extract probabilities for indices [0: improper, 1: proper, 2: no_mask]
        p_improper = float(eval_probs[0])
        p_proper = float(eval_probs[1])
        p_nomask = float(eval_probs[2])

        # Rank hypotheses for decision margin analysis
        sorted_indices = np.argsort(eval_probs)[::-1]
        top1_idx = int(sorted_indices[0])
        top2_idx = int(sorted_indices[1])

        pred_class = CLASS_NAMES[top1_idx]
        runner_up_class = CLASS_NAMES[top2_idx]

        top1_prob = float(eval_probs[top1_idx])
        top2_prob = float(eval_probs[top2_idx])
        margin = float(top1_prob - top2_prob)

        # -------------------------------------------------------------
        # 1. Continuous Expected Protection Score (0 - 100)
        # -------------------------------------------------------------
        # Full protection (1.0) for proper mask, partial protection (lambda) for improper mask,
        # zero protection (0.0) for bare face.
        continuous_score = 100.0 * (p_proper + (self.improper_weight * p_improper))
        continuous_score = max(0.0, min(100.0, continuous_score))

        # -------------------------------------------------------------
        # 2. Multi-Gate Uncertainty Rejection (Reject Option)
        # -------------------------------------------------------------
        box_w, box_h = None, None
        resolution_fail = False
        if bbox is not None:
            box_w = max(0, int(bbox[2] - bbox[0]))
            box_h = max(0, int(bbox[3] - bbox[1]))
            if min(box_w, box_h) < self.min_bbox_size:
                resolution_fail = True

        confidence_fail = top1_prob < self.min_confidence
        margin_fail = margin < self.min_margin

        # Check for exposed nose / chin drop ambiguity:
        # If model predicts proper mask but improper probability is elevated (>= 0.20)
        # or proper margin is narrow (< 0.25), this indicates high risk of exposed nose.
        improper_ambiguity = (
            pred_class == "mask_proper" and (p_improper >= 0.20 or (p_proper - p_improper) < 0.25)
        )

        # -------------------------------------------------------------
        # 3. Asymmetric Safety Status Policy
        # -------------------------------------------------------------
        rejection_gate = None

        if resolution_fail:
            status = "UNCERTAIN"
            rejection_gate = "RESOLUTION_GATE"
            reason = (
                f"Face crop resolution ({min(box_w, box_h)}px) is below spatial reliability threshold "
                f"({self.min_bbox_size}px). Visual detail is degraded by sensor distance/blur."
            )
        elif confidence_fail:
            status = "UNCERTAIN"
            rejection_gate = "CONFIDENCE_GATE"
            if improper_ambiguity:
                reason = (
                    f"Ambiguous Mask Fit: Exposed nose or loose chin strap suspected. Proper coverage "
                    f"({top1_prob * 100.0:.1f}%) lacks decisive separation from improper wear ({p_improper * 100.0:.1f}%)."
                )
            else:
                reason = (
                    f"Classification confidence ({top1_prob * 100.0:.1f}%) is below uncertainty threshold "
                    f"({self.min_confidence * 100.0:.0f}%). Inspection requires human review."
                )
        elif margin_fail:
            status = "UNCERTAIN"
            rejection_gate = "MARGIN_GATE"
            reason = (
                f"Ambiguous decision margin ({margin * 100.0:.1f}%): model is divided between "
                f"'{DISPLAY_LABELS[pred_class]}' ({top1_prob * 100.0:.1f}%) and "
                f"'{DISPLAY_LABELS[runner_up_class]}' ({top2_prob * 100.0:.1f}%)."
            )
        elif improper_ambiguity and continuous_score < 80.0:
            status = "UNCERTAIN"
            rejection_gate = "AMBIGUITY_GATE"
            reason = (
                f"Exposed Nose Advisory: Non-negligible improper mask probability ({p_improper * 100.0:.1f}%). "
                f"Borderline nasal coverage prevents unverified SAFE authorization."
            )
        elif pred_class == "mask_proper" and continuous_score >= self.safe_threshold and margin >= self.min_margin_safe:
            status = "SAFE"
            reason = (
                f"Compliant: Proper mask coverage verified with high compliance score ({continuous_score:.1f}/100) "
                f"and decisive margin ({margin * 100.0:.1f}%)."
            )
        else:
            status = "VIOLATION"
            if pred_class == "no_mask":
                reason = (
                    f"Violation: Confirmed bare face without protective mask "
                    f"(Confidence: {top1_prob * 100.0:.1f}%, Margin: {margin * 100.0:.1f}%)."
                )
            elif pred_class == "mask_improper":
                reason = (
                    f"Violation: Mask detected but worn improperly (exposed nose, mouth, or chin; "
                    f"Confidence: {top1_prob * 100.0:.1f}%). Protocol violation."
                )
            else:
                reason = (
                    f"Safety Bar Failure: Mask detected as proper, but compliance score ({continuous_score:.1f}) "
                    f"or margin ({margin * 100.0:.1f}%) did not satisfy the strict authorization standard."
                )

        # Explainable display label: NEVER label an uncertain borderline face as "Mask Worn Properly"
        display_label = DISPLAY_LABELS[pred_class]
        if status == "UNCERTAIN":
            if improper_ambiguity:
                display_label = "Uncertain (Improper Suspected)"
            elif pred_class == "mask_proper":
                display_label = "Uncertain (Low Confidence)"
            elif pred_class == "mask_improper":
                display_label = "Uncertain (Improper Mask)"
            elif pred_class == "no_mask":
                display_label = "Uncertain (Bare Face Risk)"

        audit_trail = {
            "top1_class": str(pred_class),
            "top1_confidence_pct": float(round(top1_prob * 100.0, 2)),
            "top2_class": str(runner_up_class),
            "top2_confidence_pct": float(round(top2_prob * 100.0, 2)),
            "margin_pct": float(round(margin * 100.0, 2)),
            "crop_dimensions": [int(box_w), int(box_h)] if bbox is not None and box_w is not None else None,
            "rejection_gate": str(rejection_gate) if rejection_gate else None,
            "is_calibrated": bool(is_calibrated),
            "temperature_used": float(self.temperature),
            "improper_credit_weight": float(self.improper_weight),
            "formula_breakdown": (
                f"100 * ({p_proper:.3f} + {self.improper_weight:.2f} * {p_improper:.3f} + 0.0 * {p_nomask:.3f})"
            ),
        }

        return {
            "class_index": int(top1_idx),
            "raw_class": str(pred_class),
            "display_label": str(display_label),
            "confidence": float(round(top1_prob * 100.0, 2)),
            "margin": float(round(margin * 100.0, 2)),
            "compliance_score": float(round(continuous_score, 1)),
            "status": status,
            "rejection_gate": rejection_gate,
            "reason": reason,
            "probabilities": {cls: round(float(p), 4) for cls, p in zip(CLASS_NAMES, eval_probs)},
            "raw_probabilities": {cls: round(float(p), 4) for cls, p in zip(CLASS_NAMES, raw_probs)},
            "audit_trail": audit_trail
        }


class RollingComplianceTracker:
    """
    Maintains a temporal rolling compliance score across video frames.
    Tracks the compliance score over a rolling window of video frames.
    """
    def __init__(self, window_size=30):
        self.window_size = window_size
        self.history = deque(maxlen=window_size)

    def update(self, frame_scores: list) -> float:
        """
        Updates tracker with scores from the current frame.
        If no faces are detected, the frame is not added to history.
        """
        if frame_scores:
            avg_frame_score = sum(frame_scores) / len(frame_scores)
            self.history.append(avg_frame_score)
        
        if not self.history:
            return 100.0  # Default neutral/safe
        return float(np.mean(self.history))

    def reset(self):
        self.history.clear()
