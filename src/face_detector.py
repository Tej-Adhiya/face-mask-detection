import os
import urllib.request
import numpy as np
import cv2
import torch
from torchvision import transforms
from PIL import Image

# Directory Configuration
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# Pretrained Weights Locations & URLs
YUNET_MODEL_NAME = "face_detection_yunet_2023mar.onnx"
YUNET_PATH = os.path.join(MODELS_DIR, YUNET_MODEL_NAME)
YUNET_URL = (
    "https://media.githubusercontent.com/media/opencv/opencv_zoo/"
    "47534e27c9851bb1128ccc0102f1145e27f23f98/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
)

CAFFE_PROTO_NAME = "deploy.prototxt"
CAFFE_MODEL_NAME = "res10_300x300_ssd_iter_140000.caffemodel"
CAFFE_PROTO_PATH = os.path.join(MODELS_DIR, CAFFE_PROTO_NAME)
CAFFE_MODEL_PATH = os.path.join(MODELS_DIR, CAFFE_MODEL_NAME)
CAFFE_PROTO_URL = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
CAFFE_MODEL_URL = "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"

# Normalization constants (Matches ImageNet & Model Training)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMG_SIZE = (128, 128)

crop_transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


def download_file(url: str, dest_path: str, description: str):
    """Download model weight file if not present on disk."""
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 1000:
        return True
    print(f"[FaceDetector] Downloading {description} (~{url.split('/')[-1]})...")
    try:
        opener = urllib.request.build_opener()
        opener.addheaders = [("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64)")]
        urllib.request.install_opener(opener)
        urllib.request.urlretrieve(url, dest_path)
        print(f"[FaceDetector] Successfully downloaded: {dest_path}")
        return True
    except Exception as e:
        print(f"[FaceDetector] Warning: Failed to download {description} from {url}: {e}")
        if os.path.exists(dest_path):
            os.remove(dest_path)
        return False


class FaceDetector:
    """
    Robust Face Detector supporting:
      1. YuNet (cv2.FaceDetectorYN) - Fast, lightweight (~330KB) modern ONNX detector.
      2. OpenCV DNN ResNet-10 SSD (Caffe) - Accurate deep SSD model.
      3. Haar Cascades - Offline local CPU fallback.
    
    Includes 15% contextual margin expansion to preserve mask and chin contours.
    """
    def __init__(self, backend="auto", conf_threshold=0.5):
        self.conf_threshold = conf_threshold
        self.backend = backend
        self.detector = None
        self._init_detector()

    def _init_detector(self):
        # 1. Try YuNet (Primary Recommended)
        if self.backend in ["auto", "yunet"]:
            if not os.path.exists(YUNET_PATH):
                download_file(YUNET_URL, YUNET_PATH, "YuNet ONNX Model")
            
            if os.path.exists(YUNET_PATH) and hasattr(cv2, "FaceDetectorYN"):
                try:
                    self.detector = cv2.FaceDetectorYN.create(
                        model=YUNET_PATH,
                        config="",
                        input_size=(320, 320),
                        score_threshold=self.conf_threshold,
                        nms_threshold=0.3,
                        top_k=5000
                    )
                    self.active_backend = "yunet"
                    print("[FaceDetector] Initialized primary backend: OpenCV FaceDetectorYN (YuNet ONNX)")
                    return
                except Exception as e:
                    print(f"[FaceDetector] YuNet initialization failed: {e}. Falling back...")

        # 2. Try OpenCV DNN Caffe ResNet-10 SSD
        if self.backend in ["auto", "caffe_ssd"]:
            download_file(CAFFE_PROTO_URL, CAFFE_PROTO_PATH, "Caffe Prototxt")
            download_file(CAFFE_MODEL_URL, CAFFE_MODEL_PATH, "Caffe Model Weights")

            if os.path.exists(CAFFE_PROTO_PATH) and os.path.exists(CAFFE_MODEL_PATH):
                try:
                    if hasattr(cv2.dnn, "readNetFromCaffe"):
                        self.detector = cv2.dnn.readNetFromCaffe(CAFFE_PROTO_PATH, CAFFE_MODEL_PATH)
                    elif hasattr(cv2.dnn, "readNet"):
                        # Note: cv2.dnn.readNet takes (model_weights, config_prototxt)
                        self.detector = cv2.dnn.readNet(CAFFE_MODEL_PATH, CAFFE_PROTO_PATH)
                    else:
                        raise AttributeError("Neither readNetFromCaffe nor readNet is available in cv2.dnn")

                    self.active_backend = "caffe_ssd"
                    print("[FaceDetector] Initialized backend: OpenCV DNN ResNet-10 SSD")
                    return
                except Exception as e:
                    print(f"[FaceDetector] Caffe SSD initialization failed: {e}. Falling back...")

        # 3. Try Haar Cascade (Built into OpenCV)
        haar_cls = getattr(cv2, "CascadeClassifier", None)
        if haar_cls is None and hasattr(cv2, "objdetect"):
            haar_cls = getattr(cv2.objdetect, "CascadeClassifier", None)

        if haar_cls is not None:
            haar_dir = getattr(cv2.data, "haarcascades", "") if hasattr(cv2, "data") else ""
            haar_path = os.path.join(haar_dir, "haarcascade_frontalface_default.xml")
            if os.path.exists(haar_path):
                self.detector = haar_cls(haar_path)
                self.active_backend = "haar"
                print("[FaceDetector] Initialized offline fallback backend: OpenCV Haar Cascade")
                return

        # 4. Final Safety Fallback: Use YuNet ONNX (Always reliable on modern OpenCV)
        if not os.path.exists(YUNET_PATH):
            download_file(YUNET_URL, YUNET_PATH, "YuNet ONNX Model")

        if os.path.exists(YUNET_PATH) and hasattr(cv2, "FaceDetectorYN"):
            print("[FaceDetector] Falling back to proven OpenCV FaceDetectorYN (YuNet ONNX)")
            self.detector = cv2.FaceDetectorYN.create(
                model=YUNET_PATH,
                config="",
                input_size=(320, 320),
                score_threshold=self.conf_threshold,
                nms_threshold=0.3,
                top_k=5000
            )
            self.active_backend = "yunet"
            return
        
        raise RuntimeError("No compatible face detection backend could be initialized in OpenCV.")

    def set_conf_threshold(self, conf: float):
        """Dynamically tunes face detection confidence threshold."""
        self.conf_threshold = float(conf)
        if self.active_backend == "yunet" and hasattr(self.detector, "setScoreThreshold"):
            self.detector.setScoreThreshold(float(conf))

    def set_backend(self, backend: str):
        """Dynamically switches face detector backend between YuNet and Caffe SSD."""
        if backend and backend != self.backend:
            self.backend = backend
            self._init_detector()

    def detect_faces(self, image_bgr: np.ndarray):
        """
        Detects faces in BGR image.
        
        Returns:
            list of dicts, each containing:
              - 'bbox': [xmin, ymin, xmax, ymax]
              - 'confidence': float [0.0 - 1.0]
              - 'cropped_bgr': np.ndarray of face crop with 15% contextual padding
              - 'tensor': torch.Tensor of shape (1, 3, 128, 128) normalized for classification
        """
        if image_bgr is None or image_bgr.size == 0:
            return []

        h, w = image_bgr.shape[:2]
        detections = []

        if self.active_backend == "yunet":
            if hasattr(self.detector, "setScoreThreshold"):
                self.detector.setScoreThreshold(float(self.conf_threshold))
            self.detector.setInputSize((w, h))
            _, faces = self.detector.detect(image_bgr)
            if faces is not None:
                for face in faces:
                    x, y, bw, bh = face[0:4].astype(int)
                    conf = float(face[14])
                    if conf >= self.conf_threshold:
                        detections.append(([x, y, x + bw, y + bh], conf))

        elif self.active_backend == "caffe_ssd":
            blob = cv2.dnn.blobFromImage(
                cv2.resize(image_bgr, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0)
            )
            self.detector.setInput(blob)
            preds = self.detector.forward()
            for i in range(preds.shape[2]):
                conf = float(preds[0, 0, i, 2])
                if conf >= self.conf_threshold:
                    box = preds[0, 0, i, 3:7] * np.array([w, h, w, h])
                    xmin, ymin, xmax, ymax = box.astype(int)
                    detections.append(([xmin, ymin, xmax, ymax], conf))

        elif self.active_backend == "haar":
            gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
            faces = self.detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            for (x, y, bw, bh) in faces:
                detections.append(([x, y, x + bw, y + bh], 0.85))

        # Process detections with 15% contextual margin and square crop
        results = []
        for (xmin, ymin, xmax, ymax), conf in detections:
            # Validate box
            xmin, ymin = max(0, xmin), max(0, ymin)
            xmax, ymax = min(w, xmax), min(h, ymax)
            box_w = xmax - xmin
            box_h = ymax - ymin
            if box_w < 10 or box_h < 10:
                continue

            # 25% Contextual Margin Expansion + Square Aspect Ratio (Preserves chin strap & exposed nose geometry)
            cx = (xmin + xmax) / 2.0
            cy = (ymin + ymax) / 2.0
            side = max(box_w, box_h) * 1.25

            x1 = int(round(cx - side / 2.0))
            y1 = int(round(cy - side / 2.0))
            x2 = int(round(cx + side / 2.0))
            y2 = int(round(cy + side / 2.0))

            # Crop with border padding if square exceeds image frame
            pad_left = max(0, -x1)
            pad_top = max(0, -y1)
            pad_right = max(0, x2 - w)
            pad_bottom = max(0, y2 - h)

            crop_x1 = max(0, x1)
            crop_y1 = max(0, y1)
            crop_x2 = min(w, x2)
            crop_y2 = min(h, y2)

            face_crop = image_bgr[crop_y1:crop_y2, crop_x1:crop_x2]
            if pad_left > 0 or pad_top > 0 or pad_right > 0 or pad_bottom > 0:
                face_crop = cv2.copyMakeBorder(
                    face_crop, pad_top, pad_bottom, pad_left, pad_right,
                    borderType=cv2.BORDER_REFLECT_101
                )

            if face_crop.size == 0:
                continue

            # Convert to RGB PIL Image and normalized Tensor for PyTorch Classifier
            rgb_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb_crop)
            tensor_crop = crop_transform(pil_img).unsqueeze(0)  # Shape: (1, 3, 128, 128)

            results.append({
                "bbox": [int(xmin), int(ymin), int(xmax), int(ymax)],
                "padded_bbox": [int(x1), int(y1), int(x2), int(y2)],
                "confidence": float(conf),
                "cropped_bgr": face_crop,
                "tensor": tensor_crop
            })

        return results


def draw_detection_box(frame, bbox, label, score, status, face_id=None):
    """
    Renders styled compliance bounding box and numbered label banner on frame.
    Colors:
      - SAFE      -> Green  (0, 200, 0)
      - VIOLATION -> Red    (0, 0, 220)
      - UNCERTAIN -> Amber  (0, 180, 240)
    """
    color_map = {
        "SAFE": (34, 180, 34),        # Green
        "VIOLATION": (40, 40, 220),    # Red
        "UNCERTAIN": (0, 190, 245)     # Amber/Yellow
    }
    color = color_map.get(status, (200, 200, 200))
    xmin, ymin, xmax, ymax = bbox

    # Draw rounded-like box
    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2, cv2.LINE_AA)

    # Format text with face_id if present
    id_tag = f"[#{face_id}] " if face_id is not None else ""
    text = f"{id_tag}{label} | {int(score)}% ({status})"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.52
    thickness = 1
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)

    banner_ymin = max(0, ymin - th - 10)
    cv2.rectangle(frame, (xmin, banner_ymin), (xmin + tw + 8, banner_ymin + th + 8), color, -1)
    cv2.putText(frame, text, (xmin + 4, banner_ymin + th + 3), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return frame


def draw_hud_overlay(frame, summary, fps=None):
    """
    Renders a sleek top HUD banner on video stream frames showing facility safety metrics.
    """
    h, w = frame.shape[:2]
    hud_h = 36
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, hud_h), (15, 23, 42), -1)
    cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)

    roll = summary.get("rolling_compliance_score", 100.0)
    safe = summary.get("safe_count", 0)
    viol = summary.get("violation_count", 0)
    unc = summary.get("uncertain_count", 0)

    color = (34, 180, 34) if roll >= 75 else ((0, 190, 245) if roll >= 60 else (40, 40, 220))
    status_text = "SAFE" if roll >= 75 else ("ALERT" if viol > 0 else "CAUTION")

    text = f"Facility Safety: {roll:.0f}% [{status_text}]  |  Compliant: {safe}  |  Violations: {viol}  |  Uncertain: {unc}"
    if fps:
        text += f"  |  {fps:.1f} FPS"

    cv2.putText(frame, text, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.circle(frame, (w - 18, 18), 6, color, -1)
    return frame

