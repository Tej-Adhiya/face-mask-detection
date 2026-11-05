"""
MaskFace Flask Backend: High-Performance Computer Vision & Compliance REST API
Provides endpoints for static image inspection, live webcam streaming, CCTV feed, and telemetry.
"""

import os
import sys
import io
import time
import json
import glob
import base64
import cv2
import numpy as np
import torch
from flask import Flask, request, jsonify, render_template, Response, send_from_directory
from flask_cors import CORS
from flask.json.provider import DefaultJSONProvider

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.pipeline import MaskCompliancePipeline
from src.compliance_scorer import DISPLAY_LABELS, CLASS_NAMES


class NumpyJSONProvider(DefaultJSONProvider):
    """Custom JSON provider that natively handles numpy primitives and arrays."""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def sanitize_for_json(obj):
    """Recursively converts all numpy objects into native Python primitives for JSON encoding."""
    if isinstance(obj, dict):
        return {str(k): sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [sanitize_for_json(v) for v in obj]
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


app = Flask(
    __name__,
    template_folder=os.path.join(PROJECT_ROOT, "templates"),
    static_folder=os.path.join(PROJECT_ROOT, "static")
)
app.json_provider_class = NumpyJSONProvider
app.json = NumpyJSONProvider(app)
CORS(app)

# Global Pipeline Instance (Resident in Memory / VRAM)
print("[Server] Initializing MaskFace Pipeline on GPU/CPU...")
pipeline = MaskCompliancePipeline(
    arch="resnet18_bigdata",
    detector_backend="yunet",
    conf_threshold=0.55,
    min_compliance_conf=0.60,
    safe_threshold=75.0,
    min_margin=0.20,
    improper_weight=0.25,
    temperature=1.35,
    min_bbox_size=35
)

# 5-Frame Persistence Tracker for Violation Debouncing
violation_streak = 0
PERSISTENCE_THRESHOLD = 5

HOLDOUT_DIR = os.path.join(PROJECT_ROOT, "data", "test_real_world_holdout")
SAMPLE_CCTV_PATH = os.path.join(PROJECT_ROOT, "data", "sample_surveillance_cctv.mp4")


def encode_bgr_to_base64(bgr_img: np.ndarray, quality: int = 85) -> str:
    """Encodes BGR image numpy array to base64 JPEG string."""
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    success, buffer = cv2.imencode(".jpg", bgr_img, encode_params)
    if not success:
        return ""
    return base64.b64encode(buffer).decode("utf-8")


def decode_base64_to_bgr(b64_string: str) -> np.ndarray:
    """Decodes a base64 image string (with or without data URI prefix) into BGR numpy array."""
    if "," in b64_string:
        b64_string = b64_string.split(",", 1)[1]
    img_bytes = base64.b64decode(b64_string)
    nparr = np.frombuffer(img_bytes, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)


# -----------------------------------------------------------------------------
# Frontend Route
# -----------------------------------------------------------------------------
@app.route("/")
def index():
    """Serves the modern frontend Single Page Application."""
    return render_template("index.html")


# -----------------------------------------------------------------------------
# API Endpoints
# -----------------------------------------------------------------------------
@app.route("/api/status", methods=["GET"])
def get_status():
    """Returns runtime health and hardware acceleration info."""
    is_cuda = torch.cuda.is_available()
    active_device = "GPU (CUDA Accelerated)" if is_cuda else "CPU (Host Processor)"
    return jsonify({
        "status": "online",
        "is_cuda": is_cuda,
        "active_device": active_device,
        "current_model": pipeline.arch,
        "current_detector": pipeline.detector.active_backend,
        "persistence_threshold": PERSISTENCE_THRESHOLD
    })


@app.route("/api/predict-image", methods=["POST"])
def predict_image():
    """
    Analyzes an uploaded image or preset sample.
    Returns numbered bounding boxes, scores, and explainable audit trails.
    """
    img_bgr = None
    json_data = request.get_json(silent=True) or {}

    if "image" in request.files:
        file = request.files["image"]
        file_bytes = np.frombuffer(file.read(), np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    elif "image_base64" in json_data:
        img_bgr = decode_base64_to_bgr(json_data["image_base64"])
    elif "sample_name" in json_data:
        sample_path = os.path.join(HOLDOUT_DIR, json_data["sample_name"])
        if os.path.exists(sample_path):
            img_bgr = cv2.imread(sample_path)

    if img_bgr is None:
        return jsonify({"error": "No valid image provided"}), 400

    t0 = time.perf_counter()
    annotated_bgr, scored_faces, summary = pipeline.process_frame(img_bgr, return_annotated=True, draw_hud=False)
    infer_latency_ms = (time.perf_counter() - t0) * 1000.0

    annotated_base64 = encode_bgr_to_base64(annotated_bgr)
    orig_base64 = encode_bgr_to_base64(img_bgr)

    return jsonify(sanitize_for_json({
        "summary": summary,
        "latency_ms": round(infer_latency_ms, 2),
        "scored_faces": scored_faces,
        "annotated_image": f"data:image/jpeg;base64,{annotated_base64}",
        "original_image": f"data:image/jpeg;base64,{orig_base64}"
    }))


@app.route("/api/process-frame", methods=["POST"])
def process_frame_api():
    """
    High-throughput endpoint for live browser webcam frames.
    Implements 5-frame persistence buffer for violation debouncing.
    """
    global violation_streak

    data = request.get_json(silent=True)
    if not data or "frame" not in data:
        return jsonify({"error": "Frame payload missing"}), 400

    img_bgr = decode_base64_to_bgr(data["frame"])
    if img_bgr is None:
        return jsonify({"error": "Failed to decode frame"}), 400

    t0 = time.perf_counter()
    annotated_bgr, scored_faces, summary = pipeline.process_frame(
        img_bgr, return_annotated=True, draw_hud=True
    )
    latency_ms = (time.perf_counter() - t0) * 1000.0
    fps = 1000.0 / latency_ms if latency_ms > 0 else 0.0

    # 5-Frame Persistence Hysteresis Debouncing
    # Requires 5 consecutive violation frames to sound alarm.
    # When compliant, recovery is IMMEDIATE (reset streak to 0, stop alarm instantly).
    has_violation = summary["violation_count"] > 0
    if has_violation:
        violation_streak = min(PERSISTENCE_THRESHOLD + 2, violation_streak + 1)
    else:
        violation_streak = 0

    alarm_triggered = violation_streak >= PERSISTENCE_THRESHOLD

    annotated_base64 = encode_bgr_to_base64(annotated_bgr, quality=80)

    return jsonify(sanitize_for_json({
        "annotated_frame": f"data:image/jpeg;base64,{annotated_base64}",
        "summary": summary,
        "latency_ms": round(latency_ms, 1),
        "fps": round(fps, 1),
        "scored_faces": scored_faces,
        "violation_streak": violation_streak,
        "alarm_triggered": alarm_triggered
    }))


def generate_cctv_mjpeg_stream():
    """Generates continuous MJPEG multipart stream from holdout CCTV frames."""
    sample_files = sorted(glob.glob(os.path.join(HOLDOUT_DIR, "maksssksksss*.png")))[:60]
    if not sample_files:
        return

    frame_idx = 0
    while True:
        file_path = sample_files[frame_idx % len(sample_files)]
        frame = cv2.imread(file_path)
        if frame is not None:
            resized = cv2.resize(frame, (640, 480))
            t0 = time.perf_counter()
            annotated_frame, _, summary = pipeline.process_frame(resized, return_annotated=True, draw_hud=True)
            latency = (time.perf_counter() - t0) * 1000.0

            success, buffer = cv2.imencode('.jpg', annotated_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if success:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        frame_idx += 1
        time.sleep(0.12)  # ~8 FPS surveillance rate


@app.route("/api/video-feed")
def video_feed():
    """Provides a native continuous MJPEG stream of surveillance CCTV."""
    return Response(
        generate_cctv_mjpeg_stream(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/api/sample-images", methods=["GET"])
def get_sample_images():
    """Returns catalog of preset surveillance holdout scenes."""
    presets = [
        {"name": "maksssksksss1.png", "title": "Multi-Person Crowd Scene", "desc": "Dense surveillance crowd with multiple masked and unmasked subjects"},
        {"name": "maksssksksss100.png", "title": "Street Surveillance", "desc": "Pedestrians on street at medium optical distance"},
        {"name": "maksssksksss0.png", "title": "Masked Individual", "desc": "Properly masked subject with high resolution"},
        {"name": "maksssksksss110.png", "title": "Dense CCTV Scene", "desc": "Challenging real-world CCTV scene with mixed compliance"}
    ]
    return jsonify({"presets": presets})


@app.route("/api/benchmarks", methods=["GET"])
def get_benchmarks():
    """Returns precomputed hardware benchmark and detector comparison tables."""
    benchmarks_file = os.path.join(PROJECT_ROOT, "evaluation", "live_benchmark_results.json")
    detector_file = os.path.join(PROJECT_ROOT, "evaluation", "end_to_end_scene_benchmark.json")

    bench_data = {}
    if os.path.exists(benchmarks_file):
        with open(benchmarks_file, "r") as f:
            bench_data["hardware_benchmark"] = json.load(f)

    if os.path.exists(detector_file):
        with open(detector_file, "r") as f:
            bench_data["detector_benchmark"] = json.load(f)

    return jsonify(bench_data)


@app.route("/api/config", methods=["POST"])
def update_config():
    """Dynamically tunes pipeline thresholds without re-initializing weights."""
    data = request.get_json(silent=True) or {}
    
    if "model" in data and data["model"]:
        pipeline.set_model(data["model"])
    
    if "detector" in data and data["detector"]:
        pipeline.set_detector(data["detector"])
    
    pipeline.set_thresholds(
        detection_conf=data.get("detection_conf"),
        min_compliance_conf=data.get("min_compliance_conf"),
        safe_threshold=data.get("safe_threshold"),
        min_margin=data.get("min_margin"),
        improper_weight=data.get("improper_weight"),
        temperature=data.get("temperature"),
        min_bbox_size=data.get("min_bbox_size")
    )

    print(f"[Server Config] Updated pipeline config: detector={pipeline.detector.active_backend}, "
          f"det_conf={pipeline.detector.conf_threshold}, min_conf={pipeline.scorer.min_confidence}, "
          f"min_margin={pipeline.scorer.min_margin}, min_bbox={pipeline.scorer.min_bbox_size}, "
          f"temp={pipeline.scorer.temperature}, weight={pipeline.scorer.improper_weight}")

    return jsonify(sanitize_for_json({
        "status": "success",
        "message": "Configuration updated successfully",
        "active_config": {
            "model": pipeline.arch,
            "detector": pipeline.detector.active_backend,
            "detection_conf": pipeline.detector.conf_threshold,
            "min_compliance_conf": pipeline.scorer.min_confidence,
            "min_margin": pipeline.scorer.min_margin,
            "min_bbox_size": pipeline.scorer.min_bbox_size,
            "temperature": pipeline.scorer.temperature,
            "improper_weight": pipeline.scorer.improper_weight
        }
    }))


if __name__ == "__main__":
    print("[Server] Starting MaskFace Flask Web Server on 0.0.0.0:5000...")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
