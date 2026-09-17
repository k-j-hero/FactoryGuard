import argparse
from pathlib import Path

import yaml

from factoryguard.config import read_yaml


def prepare_data(path, destination):
    """Resolve dataset paths explicitly, independent of Ultralytics global datasets_dir."""
    path = Path(path).resolve()
    data = read_yaml(path)
    names = data.get("names")
    if isinstance(names, list):
        names = dict(enumerate(names))
    if not isinstance(names, dict) or set(names) != {0, 1}:
        raise ValueError("Dataset must have exactly two class IDs: 0 and 1")
    if {str(name).strip().lower() for name in names.values()} != {"person", "forklift"}:
        raise ValueError("Dataset names must be person and forklift in actual label-ID order")
    root = (path.parent / data.get("path", ".")).resolve()
    normalized = {"path": str(root), "names": names, "nc": 2}
    for split in ("train", "val", "test"):
        value = data.get(split)
        if not value and split == "test":
            continue
        if not isinstance(value, str) or not value:
            raise ValueError(f"Expected a directory path for {split}")
        split_path = (root / value).resolve()
        if not split_path.is_dir():
            raise ValueError(f"Missing {split} image directory: {split_path}. "
                             "Correct path/train/val/test relative to data.yaml; see README.")
        if not any(p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"} for p in split_path.rglob("*")):
            raise ValueError(f"No images in {split_path}")
        label_dir = split_path.parent / "labels"
        if not label_dir.is_dir() or not any(label_dir.rglob("*.txt")):
            raise ValueError(f"Missing YOLO labels in {label_dir}")
        normalized[split] = str(split_path)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(yaml.safe_dump(normalized, sort_keys=False), encoding="utf-8")
    return destination.resolve()


def main():
    parser = argparse.ArgumentParser(description="Fine-tune YOLO26s on a person/forklift YOLO detection dataset")
    parser.add_argument("--data", required=True)
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--device")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch", type=int)
    parser.add_argument("--check-only", action="store_true", help="Validate dataset layout without loading YOLO")
    args = parser.parse_args()
    config = read_yaml(args.config)
    for key in ("device", "epochs", "batch"):
        if getattr(args, key) is not None:
            config[key] = getattr(args, key)
    model_name = config.pop("model", "yolo26s.pt")
    # Ultralytics may prepend its default runs/detect directory to a relative
    # project path. Resolve it here so every run lands exactly where configured.
    project = Path(config.get("project", "runs/train")).resolve()
    config["project"] = str(project)
    data = prepare_data(args.data, project / "data.resolved.yaml")
    print(f"Resolved dataset: {data}")
    if args.check_only:
        return
    from ultralytics import YOLO
    model = YOLO(model_name)
    model.train(data=str(data), **config)
    print(f"Training outputs: {model.trainer.save_dir}")
    print(f"Best checkpoint: {Path(model.trainer.save_dir) / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    main()
