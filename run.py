import os
import argparse
import csv
import json
import numpy as np
import cv2
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from pothole_detection.config import load_config
from pothole_detection.gps import parse_gpx, nearest_gps
from pothole_detection.tracker import match_track_to_detection

try:
    from yolox.tracker.byte_tracker import BYTETracker
except ImportError:
    raise ImportError("ByteTrack not found. See README for setup instructions.")

from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Pothole Detection & Tracking")
    parser.add_argument("--config", default="config/default_config.yaml")
    return parser.parse_args()


def setup_tracker(trk_cfg, fps):
    bt_args = SimpleNamespace(
        track_thresh=trk_cfg["track_thresh"],
        match_thresh=trk_cfg["match_thresh"],
        track_buffer=trk_cfg["track_buffer"],
        mot20=False
    )
    return BYTETracker(bt_args, frame_rate=max(1, int(round(fps))))


def setup_output(paths, fps, W, H):
    os.makedirs("outputs", exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vid_writer = cv2.VideoWriter(paths["annotated_video"], fourcc, fps, (W, H))
    csv_file = open(paths["csv_log"], "w", newline="", encoding="utf-8")
    csv_writer = csv.DictWriter(
        csv_file,
        fieldnames=["Frame", "Type", "ByteTrack ID", "Timestamp", "GPS Coordinates"]
    )
    csv_writer.writeheader()
    return vid_writer, csv_file, csv_writer


def process_video(cfg):
    paths = cfg["paths"]
    det_cfg = cfg["detection"]
    trk_cfg = cfg["tracking"]
    class_names = {k: v for k, v in cfg["classes"].items() if isinstance(k, int)}
    poi_ids = cfg["classes"]["poi_class_ids"]

    for key in ("model", "video", "gpx"):
        if not os.path.exists(paths[key]):
            raise FileNotFoundError(f"Required file not found: {paths[key]}")

    gps_points = parse_gpx(paths["gpx"])
    start_time = gps_points[0]["time"] if gps_points else datetime.now(timezone.utc)

    model = YOLO(paths["model"])
    cap = cv2.VideoCapture(paths["video"])
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {paths['video']}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    line_y = H // 2
    cap.release()

    tracker = setup_tracker(trk_cfg, fps)
    vid_writer, csv_file, csv_writer = setup_output(paths, fps, W, H)

    FONT = cv2.FONT_HERSHEY_SIMPLEX
    prev_centers = {}
    crossed_ids = set()
    geo_features = []

    stream = model(
        paths["video"],
        stream=True,
        conf=det_cfg["confidence_threshold"],
        iou=det_cfg["iou_threshold"],
        imgsz=det_cfg["image_size"]
    )

    print("Processing video — press Ctrl+C to stop early.\n")

    for frame_id, result in enumerate(stream, start=1):
        frame = result.orig_img
        if frame is None:
            continue

        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            tracker.update(np.zeros((0, 5), dtype=np.float32), [H, W], [H, W])
            vid_writer.write(frame)
            continue

        xyxy = boxes.xyxy.cpu().numpy()
        scores = boxes.conf.cpu().numpy()
        classes = boxes.cls.cpu().numpy().astype(int)

        detections = np.zeros((len(xyxy), 6), dtype=np.float32)
        detections[:, 0:4] = xyxy
        detections[:, 4] = scores
        detections[:, 5] = classes

        online_targets = tracker.update(
            detections[:, :5].astype(np.float32), [H, W], [H, W]
        )

        video_time = (start_time + timedelta(seconds=(frame_id - 1) / fps)).astimezone(timezone.utc)
        gps = nearest_gps(gps_points, video_time)
        gps_str = f"{gps['lat']}, {gps['lon']}" if gps else "N/A"

        cv2.line(frame, (0, line_y), (W, line_y), (0, 255, 255), 2)

        for b, s, c in zip(xyxy, scores, classes):
            x1, y1, x2, y2 = map(int, b)
            label = f"{class_names.get(int(c), str(c))} {s:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
            cv2.putText(frame, label, (x1, max(10, y1 - 6)), FONT, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

        for t in online_targets:
            tid = getattr(t, "track_id", None) or getattr(t, "trackID", None)
            tlwh = getattr(t, "tlwh", None)
            if tid is None or tlwh is None:
                try:
                    arr = np.asarray(t)
                    tid = int(arr[5])
                    tlwh = arr[:4].tolist()
                except Exception:
                    continue

            x, y, w, h = map(int, tlwh)
            center_x = x + w // 2
            center_y = y + h // 2

            best_idx, best_iou_val = match_track_to_detection(tlwh, detections[:, :4])
            cls_id = int(detections[best_idx, 5]) if best_idx >= 0 and best_iou_val > 0.01 else None
            cls_name = class_names.get(cls_id, "Unknown") if cls_id is not None else "Unknown"

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 128, 255), 2)
            cv2.putText(frame, f"ID {int(tid)} {cls_name}", (x, y - 8), FONT, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.circle(frame, (center_x, center_y), 3, (0, 0, 255), -1)

            if tid in prev_centers:
                if prev_centers[tid] < line_y and center_y >= line_y and tid not in crossed_ids:
                    if cls_id is None or cls_id in poi_ids:
                        crossed_ids.add(tid)
                        csv_writer.writerow({
                            "Frame": frame_id,
                            "Type": cls_name,
                            "ByteTrack ID": int(tid),
                            "Timestamp": video_time.isoformat(),
                            "GPS Coordinates": gps_str
                        })
                        if gps:
                            geo_features.append({
                                "type": "Feature",
                                "properties": {
                                    "frame": frame_id,
                                    "id": int(tid),
                                    "type": cls_name,
                                    "time": video_time.isoformat()
                                },
                                "geometry": {
                                    "type": "Point",
                                    "coordinates": [gps["lon"], gps["lat"]]
                                }
                            })

            prev_centers[tid] = center_y

        vid_writer.write(frame)

    csv_file.close()
    vid_writer.release()

    with open(paths["geojson"], "w", encoding="utf-8") as f:
        json.dump({"type": "FeatureCollection", "features": geo_features}, f, indent=2)

    print(f"\nDone!")
    print(f"  CSV     -> {paths['csv_log']}")
    print(f"  Video   -> {paths['annotated_video']}")
    print(f"  GeoJSON -> {paths['geojson']}")


def main():
    args = parse_args()
    cfg = load_config(args.config)
    process_video(cfg)


if __name__ == "__main__":
    main()