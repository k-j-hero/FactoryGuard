import csv
import math
from pathlib import Path

import cv2


def open_writer(path, fps, size):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    if not writer.isOpened():
        writer.release()
        raise RuntimeError(f"Cannot create MP4: {path}. Check OpenCV codec support.")
    return writer


def extract_clips(video, events, output_dir, pre_sec=2.0, post_sec=2.0):
    """Read annotated video, one event at a time; constant memory, no ffmpeg needed."""
    if not all(math.isfinite(v) and v >= 0 for v in (pre_sec, post_sec)):
        raise ValueError("Clip padding must be finite and non-negative")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / "clips.csv"
    if manifest.exists():
        raise FileExistsError(f"Clip manifest already exists: {manifest}")
    rows = []
    cap = cv2.VideoCapture(str(video))
    try:
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if not math.isfinite(fps) or fps <= 0 or total <= 0:
            raise ValueError("Video must report positive FPS and frame count")
        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        for event in events:
            event = event.to_dict() if hasattr(event, "to_dict") else event
            event_id = str(event["event_id"])
            if not event_id.startswith("event_") or not event_id[6:].isdigit():
                raise ValueError(f"Invalid event ID: {event_id}")
            start = max(0, int(event["start_frame"]) - math.ceil(pre_sec * fps))
            end = min(total, int(event["end_frame"]) + 1 + math.ceil(post_sec * fps))
            if not 0 <= start < end <= total:
                raise ValueError(f"Invalid event interval: {event_id}")
            path = output / f"{event_id}.mp4"
            if path.exists():
                raise FileExistsError(path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, start)
            writer = open_writer(path, fps, size)
            count = 0
            try:
                for _ in range(start, end):
                    ok, frame = cap.read()
                    if not ok:
                        raise RuntimeError(f"Unexpected decode failure extracting {event_id}")
                    writer.write(frame)
                    count += 1
            finally:
                writer.release()
            rows.append({"event_id": event_id, "clip": path.name,
                         "start_frame": start, "end_frame_exclusive": end,
                         "start_sec": start / fps, "end_sec": end / fps,
                         "frames": count})
    finally:
        cap.release()
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        fields = ["event_id", "clip", "start_frame", "end_frame_exclusive", "start_sec", "end_sec", "frames"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return rows
