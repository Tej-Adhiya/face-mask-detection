import os
import sys
import glob
import time
import json
import argparse
import xml.etree.ElementTree as ET
import numpy as np
import cv2
import torch
import torch.nn.functional as F

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from src.pipeline import MaskCompliancePipeline

LABEL_MAP = {
    "mask_weared_incorrect": "mask_improper",
    "with_mask": "mask_proper",
    "without_mask": "no_mask"
}

CLASS_NAMES = ["mask_improper", "mask_proper", "no_mask"]


def compute_iou(box1, box2):
    """Calculates Intersection over Union (IoU) between [xmin, ymin, xmax, ymax]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    inter_area = inter_w * inter_h

    area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])
    union_area = area1 + area2 - inter_area

    if union_area <= 0:
        return 0.0
    return inter_area / union_area


def parse_xml_annotations(xml_path):
    """Parses Pascal VOC ground truth faces and labels."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    annotations = []

    for obj in root.findall("object"):
        raw_name = obj.find("name").text
        if raw_name not in LABEL_MAP:
            continue
        cls_name = LABEL_MAP[raw_name]

        bndbox = obj.find("bndbox")
        if bndbox is None:
            continue

        xmin = int(float(bndbox.find("xmin").text))
        ymin = int(float(bndbox.find("ymin").text))
        xmax = int(float(bndbox.find("xmax").text))
        ymax = int(float(bndbox.find("ymax").text))

        if (xmax - xmin) < 8 or (ymax - ymin) < 8:
            continue

        annotations.append({
            "bbox": [xmin, ymin, xmax, ymax],
            "label": cls_name
        })
    return annotations


def run_pipeline_eval(pipeline, arch_name, xml_files, iou_thresh=0.45):
    """Runs complete detection + classification on full scenes and compares against ground truth XML."""
    print(f"\n---> Evaluating Pipeline [{arch_name.upper()}] on {len(xml_files)} scenes...")
    pipeline.set_model(arch_name)

    total_gt_faces = 0
    total_detected_faces = 0
    true_positive_detections = 0
    correct_classifications = 0
    
    iou_scores = []
    latencies = []

    class_gt = {cls: 0 for cls in CLASS_NAMES}
    class_tp = {cls: 0 for cls in CLASS_NAMES}
    class_pred = {cls: 0 for cls in CLASS_NAMES}

    for xml_path in xml_files:
        gt_boxes = parse_xml_annotations(xml_path)
        if not gt_boxes:
            continue

        base = os.path.splitext(xml_path)[0]
        img_path = None
        for ext in [".png", ".jpg", ".jpeg"]:
            candidate = base + ext
            if os.path.exists(candidate):
                img_path = candidate
                break

        if not img_path:
            continue

        img = cv2.imread(img_path)
        if img is None:
            continue

        total_gt_faces += len(gt_boxes)
        for gt in gt_boxes:
            class_gt[gt["label"]] += 1

        # Time full frame processing on GPU
        t0 = time.time()
        _, scored_faces, _ = pipeline.process_frame(img, return_annotated=False)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed_ms = (time.time() - t0) * 1000.0
        latencies.append(elapsed_ms)

        total_detected_faces += len(scored_faces)
        for sf in scored_faces:
            class_pred[sf["raw_class"]] += 1

        # Greedy IoU Matching
        matched_dets = set()
        for gt in gt_boxes:
            best_iou = 0.0
            best_idx = -1
            for d_idx, det in enumerate(scored_faces):
                if d_idx in matched_dets:
                    continue
                iou = compute_iou(gt["bbox"], det["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_idx = d_idx

            if best_iou >= iou_thresh and best_idx != -1:
                true_positive_detections += 1
                matched_dets.add(best_idx)
                iou_scores.append(best_iou)

                pred_class = scored_faces[best_idx]["raw_class"]
                if pred_class == gt["label"]:
                    correct_classifications += 1
                    class_tp[gt["label"]] += 1

    detection_recall = (true_positive_detections / total_gt_faces * 100.0) if total_gt_faces > 0 else 0.0
    mean_iou = float(np.mean(iou_scores)) if iou_scores else 0.0
    e2e_acc = (correct_classifications / true_positive_detections * 100.0) if true_positive_detections > 0 else 0.0
    avg_latency = float(np.mean(latencies)) if latencies else 0.0
    fps = 1000.0 / avg_latency if avg_latency > 0 else 0.0

    per_class_f1 = {}
    for cls in CLASS_NAMES:
        tp = class_tp[cls]
        prec = tp / class_pred[cls] if class_pred[cls] > 0 else 0.0
        rec = tp / class_gt[cls] if class_gt[cls] > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) * 100.0 if (prec + rec) > 0 else 0.0
        per_class_f1[cls] = f1

    return {
        "arch": arch_name,
        "detector": pipeline.detector.active_backend,
        "scenes_evaluated": len(xml_files),
        "gt_faces": total_gt_faces,
        "detected_faces": total_detected_faces,
        "matched_faces": true_positive_detections,
        "detection_recall": detection_recall,
        "mean_iou": mean_iou,
        "e2e_classification_acc": e2e_acc,
        "proper_f1": per_class_f1["mask_proper"],
        "improper_f1": per_class_f1["mask_improper"],
        "nomask_f1": per_class_f1["no_mask"],
        "avg_latency_ms": avg_latency,
        "fps": fps
    }


def main():
    parser = argparse.ArgumentParser(description="End-to-End Pipeline Multi-Model & Detection Accuracy Benchmark")
    parser.add_argument(
        "--num-samples",
        type=int,
        default=50,
        help="Number of scenes to evaluate (default: 50 for quick run, 0 for all 700)"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test_real_world_holdout",
        help="Dataset split directory in data/"
    )
    args = parser.parse_args()

    data_dir = os.path.join(PROJECT_ROOT, "data", args.split)
    xml_files = sorted(glob.glob(os.path.join(data_dir, "*.xml")))

    if not xml_files:
        print(f"Error: No XML files found in {data_dir}")
        return

    if args.num_samples > 0 and args.num_samples < len(xml_files):
        indices = np.linspace(0, len(xml_files) - 1, args.num_samples, dtype=int)
        test_xmls = [xml_files[i] for i in indices]
    else:
        test_xmls = xml_files

    print("=" * 95)
    print("      END-TO-END PIPELINE: DETECTION ACCURACY (IoU) & CLASSIFIER BENCHMARK")
    print("=" * 95)
    print(f"Dataset Split         : {args.split}")
    print(f"Scenes Selected       : {len(test_xmls)}")
    print(f"IoU Matching Target   : >= 0.45")
    print(f"Hardware Engine       : {'CUDA (RTX 5070)' if torch.cuda.is_available() else 'CPU'}")
    print("=" * 95)

    configs_to_test = [
        ("resnet18", "yunet", "ResNet-18 + YuNet (330KB)"),
        ("resnet18", "caffe_ssd", "ResNet-18 + Caffe SSD (10MB)"),
        ("mobilenet_v3_large", "yunet", "MobileNetV3 + YuNet"),
        ("efficientnet_b0", "yunet", "EfficientNet-B0 + YuNet"),
        ("mobilenet_v2", "yunet", "MobileNetV2 + YuNet")
    ]

    results = []
    for arch, detector_backend, display_name in configs_to_test:
        pipeline = MaskCompliancePipeline(arch=arch, conf_threshold=0.45, detector_backend=detector_backend)
        res = run_pipeline_eval(pipeline, arch, test_xmls)
        res["display_name"] = display_name
        results.append(res)

    print("\n" + "=" * 115)
    print("                END-TO-END PIPELINE REAL-WORLD SCENE BENCHMARK RESULTS")
    print("=" * 115)
    header = (
        f"{'Model Architecture':<24} | {'Det Recall':<10} | {'Mean IoU':<9} | {'E2E Acc':<9} | "
        f"{'Proper F1':<10} | {'Improper F1':<12} | {'Latency (ms)':<13} | {'FPS':<8}"
    )
    print(header)
    print("-" * 115)

    for r in results:
        row = (
            f"{r['display_name']:<24} | "
            f"{r['detection_recall']:>9.2f}% | "
            f"{r['mean_iou']:>8.3f} | "
            f"{r['e2e_classification_acc']:>8.2f}% | "
            f"{r['proper_f1']:>9.2f}% | "
            f"{r['improper_f1']:>11.2f}% | "
            f"{r['avg_latency_ms']:>10.2f} ms | "
            f"{r['fps']:>7.1f}"
        )
        print(row)
    print("=" * 115 + "\n")

    out_json = os.path.join(PROJECT_ROOT, "evaluation", "end_to_end_scene_benchmark.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[Results saved to: {out_json}]\n")


if __name__ == "__main__":
    main()
