from copy import deepcopy
import math
from pathlib import Path

import yaml


DEFAULTS = {
    "model": "yolo26s.pt", "device": "cpu", "imgsz": 640,
    "conf": 0.1, "iou": 0.7, "tracker": "configs/bytetrack.yaml",
    "require_forklift": False, "person_names": ["person"],
    "forklift_names": ["forklift"],
    "proximity": {"enter_threshold": 0.08, "exit_threshold": 0.11,
                  "min_duration_sec": 0.5, "lost_tolerance_sec": 0.3,
                  "anchor_mode": "bottom_center"},
    "clips": {"enabled": True, "pre_sec": 2.0, "post_sec": 2.0},
}


def read_yaml(path):
    with Path(path).open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a YAML mapping: {path}")
    return value


def load_config(path):
    config = deepcopy(DEFAULTS)
    for key, value in read_yaml(path).items():
        if key not in config:
            raise ValueError(f"Unknown configuration key: {key}")
        if isinstance(config[key], dict):
            if not isinstance(value, dict) or set(value) - set(config[key]):
                raise ValueError(f"Invalid configuration section: {key}")
            config[key].update(value)
        else:
            config[key] = value
    p = config["proximity"]
    numeric = {key: p[key] for key in (
        "enter_threshold", "exit_threshold", "min_duration_sec", "lost_tolerance_sec")}
    numeric.update(pre_sec=config["clips"]["pre_sec"], post_sec=config["clips"]["post_sec"])
    for key, value in numeric.items():
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{key} must be a finite non-negative number")
    if not 0 < p["enter_threshold"] < p["exit_threshold"] <= 1:
        raise ValueError("Require 0 < enter_threshold < exit_threshold <= 1")
    if p["min_duration_sec"] <= 0:
        raise ValueError("min_duration_sec must be positive")
    if p["anchor_mode"] not in {"bottom_center", "center"}:
        raise ValueError("anchor_mode must be 'bottom_center' or 'center'")
    for key in ("conf", "iou"):
        if not isinstance(config[key], (int, float)) or not 0 < config[key] <= 1:
            raise ValueError(f"{key} must be in (0, 1]")
    if type(config["imgsz"]) is not int or config["imgsz"] < 32:
        raise ValueError("imgsz must be an integer >= 32")
    if type(config["require_forklift"]) is not bool or type(config["clips"]["enabled"]) is not bool:
        raise ValueError("require_forklift and clips.enabled must be booleans")
    aliases = []
    for key in ("person_names", "forklift_names"):
        names = config[key]
        if not isinstance(names, list) or not names or any(not isinstance(n, str) or not n.strip() for n in names):
            raise ValueError(f"{key} must be a non-empty list of class names")
        config[key] = [name.strip().lower() for name in names]
        aliases.append(set(config[key]))
    if aliases[0] & aliases[1]:
        raise ValueError("person and forklift class aliases cannot overlap")
    tracker = read_yaml(config["tracker"])
    if tracker.get("tracker_type") != "bytetrack":
        raise ValueError("FactoryGuard MVP requires a ByteTrack configuration")
    if config["conf"] > tracker.get("track_low_thresh", 0.1):
        raise ValueError("conf must be <= ByteTrack track_low_thresh")
    return config


def resolve_classes(names, person_names, forklift_names, require_forklift=False):
    names = dict(enumerate(names)) if isinstance(names, list) else names
    mapping = {}
    for class_id, name in names.items():
        normalized = str(name).strip().lower()
        if normalized in person_names:
            mapping[int(class_id)] = "person"
        elif normalized in forklift_names:
            mapping[int(class_id)] = "forklift"
    if "person" not in mapping.values():
        raise ValueError(f"Model has no person class. Model classes: {names}")
    if require_forklift and "forklift" not in mapping.values():
        raise ValueError("Model has no forklift class. Use a person/forklift-trained best.pt.")
    return mapping
