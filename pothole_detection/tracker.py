import numpy as np


def iou(boxA, boxB):
    """IoU between two boxes in xyxy format."""
    xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
    xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = max(0, boxA[2] - boxA[0]) * max(0, boxA[3] - boxA[1])
    areaB = max(0, boxB[2] - boxB[0]) * max(0, boxB[3] - boxB[1])
    denom = areaA + areaB - inter
    return inter / denom if denom > 1e-8 else 0.0


def match_track_to_detection(track_tlwh, detections_xyxy):
    x, y, w, h = track_tlwh
    track_box = [x, y, x + w, y + h]
    best_iou, best_idx = 0.0, -1
    for i, det in enumerate(detections_xyxy):
        val = iou(track_box, det[:4].tolist())
        if val > best_iou:
            best_iou, best_idx = val, i
    return best_idx, best_iou