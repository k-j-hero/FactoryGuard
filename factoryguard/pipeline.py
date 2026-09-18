import csv
from dataclasses import fields
import json
import logging
import math
from pathlib import Path
import time

import cv2

from .clips import extract_clips, open_writer
from .config import resolve_classes
from .proximity import Event, ProximityEngine, Track

LOG = logging.getLogger(__name__)


def result_tracks(result, class_mapping):
    boxes = result.boxes
    if boxes is None or boxes.id is None:
        return []
    tracks = []
    for bbox, track_id, class_id, confidence in zip(
        boxes.xyxy.cpu().tolist(), boxes.id.cpu().tolist(),
        boxes.cls.cpu().tolist(), boxes.conf.cpu().tolist()
    ):
        label = class_mapping.get(int(class_id))
        if label:
            tracks.append(Track(int(track_id), label, tuple(bbox), float(confidence)))
    return tracks


def annotate(frame, result, tracks, distances, active, frame_index, fps, enabled, enter,
             anchor_mode):
    # plot() also shows detections that have not received a track ID yet.
    frame = result.plot(img=frame, labels=True, conf=True)
    by_id = {track.track_id: track for track in tracks}
    for pair, distance in distances.items():
        if distance > enter and pair not in active:
            continue
        a, b = [tuple(round(v) for v in by_id[tid].anchor_for(anchor_mode)) for tid in pair]
        color = (20, 30, 240) if pair in active else (0, 190, 255)
        cv2.line(frame, a, b, color, 2)
        cv2.putText(frame, f"NP={distance:.3f}", a, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    status = "CANDIDATE" if active else "MONITORING"
    if not enabled:
        status = "DETECTION ONLY: no forklift class"
    lines = [f"FactoryGuard | {frame_index / fps:.2f}s | {status}",
             "Normalized proximity (NP): image space only",
             "Potential near-miss candidate" if active else "Review aid; no physical distance estimate"]
    for line_index, line in enumerate(lines):
        y = 23 + line_index * 23
        cv2.putText(frame, line, (9, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 3)
        cv2.putText(frame, line, (9, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)
    return frame


def run(source, output_dir, config, max_frames=None, model=None):
    """Inject a model for deterministic integration tests; normal runs load YOLO."""
    source = Path(source).resolve()
    output = Path(output_dir).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Input MP4 not found: {source}")
    if output.exists():
        raise FileExistsError(f"Use a new output directory to preserve existing results: {output}")
    if max_frames is not None and max_frames <= 0:
        raise ValueError("max_frames must be positive")
    if model is None:
        from ultralytics import YOLO
        model = YOLO(config["model"])
    if getattr(model, "task", "detect") != "detect":
        raise ValueError("Use detection model weights, not segmentation, pose, or classification")
    mapping = resolve_classes(model.names, config["person_names"], config["forklift_names"],
                              config["require_forklift"])
    enabled = "forklift" in mapping.values()
    if not enabled:
        LOG.warning("DETECTION ONLY: model has no forklift class. Proximity events are disabled. "
                    "Use a person/forklift-trained best.pt; car/truck are never mapped to forklift.")
    cap = cv2.VideoCapture(str(source))
    writer = None
    events = []
    frame_index = 0
    started = time.perf_counter()
    try:
        if not cap.isOpened():
            raise ValueError(f"Cannot decode input video: {source}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        expected_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if not math.isfinite(fps) or fps <= 0 or width <= 0 or height <= 0:
            raise ValueError("Video metadata must include valid FPS and dimensions")
        # mp4v may silently truncate odd frame dimensions. Refuse that ambiguity.
        if width % 2 or height % 2:
            raise ValueError("MP4 output requires even width and height. Resize the input first.")
        output.mkdir(parents=True, exist_ok=False)
        (output / "config.resolved.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
        writer = open_writer(output / "annotated.mp4", fps, (width, height))
        engine = ProximityEngine(fps, **config["proximity"])
        with (output / "events.jsonl").open("w", encoding="utf-8") as event_log, \
             (output / "tracks.csv").open("w", newline="", encoding="utf-8") as track_log:
            track_csv = csv.writer(track_log)
            track_csv.writerow(["frame", "time_sec", "track_id", "class", "confidence", "x1", "y1", "x2", "y2"])

            def record(closed):
                for event in closed:
                    events.append(event)
                    event_log.write(json.dumps(event.to_dict()) + "\n")
                    event_log.flush()

            while max_frames is None or frame_index < max_frames:
                ok, frame = cap.read()
                if not ok:
                    break
                if frame.shape[:2] != (height, width):
                    raise ValueError("Input video changes dimensions mid-stream")
                result = model.track(frame, persist=True, tracker=config["tracker"],
                                     classes=list(mapping), conf=config["conf"], iou=config["iou"],
                                     imgsz=config["imgsz"], device=config["device"], verbose=False)[0]
                tracks = result_tracks(result, mapping)
                for track in tracks:
                    track_csv.writerow([frame_index, frame_index / fps, track.track_id,
                                        track.label, track.confidence, *track.bbox])
                distances, active, closed = engine.update(frame_index, tracks, width, height)
                record(closed)
                rendered = annotate(frame, result, tracks, distances, active, frame_index,
                                    fps, enabled, engine.enter, engine.anchor_mode)
                writer.write(rendered)
                frame_index += 1
                if frame_index % 100 == 0:
                    LOG.info("Processed %d frames; %d closed candidates", frame_index, len(events))
            limited = max_frames is not None and frame_index >= max_frames
            if frame_index == 0:
                raise ValueError("No frames decoded")
            if not limited and expected_frames > 0 and frame_index < expected_frames:
                raise RuntimeError(f"Premature decode end: {frame_index}/{expected_frames} frames")
            record(engine.finish("frame_limit" if limited else "end_of_video"))
    except BaseException as exc:
        if output.is_dir():
            (output / "FAILED.txt").write_text(f"Partial results only: {type(exc).__name__}: {exc}\n", encoding="utf-8")
        raise
    finally:
        cap.release()
        if writer is not None:
            writer.release()
    try:
        with (output / "events.csv").open("w", newline="", encoding="utf-8") as handle:
            event_csv = csv.DictWriter(handle, fieldnames=[field.name for field in fields(Event)])
            event_csv.writeheader()
            event_csv.writerows(event.to_dict() for event in events)
        # Verify encoding actually produced a readable video before reporting success.
        check = cv2.VideoCapture(str(output / "annotated.mp4"))
        try:
            ok, _ = check.read()
            encoded = int(check.get(cv2.CAP_PROP_FRAME_COUNT))
            if not ok or encoded != frame_index:
                raise RuntimeError(f"MP4 validation failed: {encoded}/{frame_index} frames")
        finally:
            check.release()
        clips = []
        if config["clips"]["enabled"]:
            clips = extract_clips(output / "annotated.mp4", events, output / "clips",
                                  config["clips"]["pre_sec"], config["clips"]["post_sec"])
        summary = {
            "status": "complete", "source": str(source), "model": config["model"],
            "model_classes": model.names, "class_mapping": mapping,
            "proximity_enabled": enabled, "mode": "proximity" if enabled else "detection_only",
            "processed_frames": frame_index, "fps": fps, "frame_width": width, "frame_height": height,
            "frame_limit_reached": limited,
            "duration_sec": frame_index / fps, "event_count": len(events),
            "clip_count": len(clips), "elapsed_sec": time.perf_counter() - started,
            "metric": f"{engine.anchor_mode.replace('_', '-')} Euclidean distance / frame diagonal; lower is closer",
            "limitations": "Image-space candidate only. No physical distance or accident probability.",
        }
        (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    except BaseException as exc:
        (output / "FAILED.txt").write_text(f"Output finalization failed: {exc}\n", encoding="utf-8")
        raise
    LOG.info("Completed: %s", output)
    return summary
