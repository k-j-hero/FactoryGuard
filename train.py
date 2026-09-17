import argparse
import os
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
        values = [value] if isinstance(value, str) else value
        if not isinstance(values, list) or not values or not all(isinstance(item, str) and item for item in values):
            raise ValueError(f"Expected one or more directory paths for {split}")
        split_paths = []
        for item in values:
            split_path = (root / item).resolve()
            if not split_path.is_dir():
                raise ValueError(f"Missing {split} image directory: {split_path}. "
                                 "Correct path/train/val/test relative to data.yaml; see README.")
            if not any(p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"} for p in split_path.rglob("*")):
                raise ValueError(f"No images in {split_path}")
            label_dir = split_path.parent / "labels"
            if not label_dir.is_dir() or not any(label_dir.rglob("*.txt")):
                raise ValueError(f"Missing YOLO labels in {label_dir}")
            split_paths.append(str(split_path))
        normalized[split] = split_paths[0] if isinstance(value, str) else split_paths
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
    parser.add_argument("--wandb", action="store_true", help="Log training metrics and best.pt to Weights & Biases")
    parser.add_argument("--wandb-project", default="FactoryGuard")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-mode", choices=("online", "offline"), default="online")
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
    wandb_run = None
    if args.wandb:
        ultralytics_settings = project / "ultralytics-settings"
        ultralytics_settings.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("YOLO_CONFIG_DIR", str(ultralytics_settings))
        wandb_support = project / "wandb-support"
        os.environ.setdefault("WANDB_CACHE_DIR", str(wandb_support / "cache"))
        os.environ.setdefault("WANDB_CONFIG_DIR", str(wandb_support / "config"))
        os.environ.setdefault("WANDB_DATA_DIR", str(wandb_support / "data"))
        import wandb
        from ultralytics import settings

        settings.update({"wandb": True})
        wandb_config = {"dataset": str(data), "model": model_name, **config}
        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=str(config.get("name", "train")),
            mode=args.wandb_mode,
            job_type="train",
            config=wandb_config,
            dir=str(project),
        )
    from ultralytics import YOLO
    model = YOLO(model_name)
    failed = False
    try:
        model.train(data=str(data), **config)
        print(f"Training outputs: {model.trainer.save_dir}")
        print(f"Best checkpoint: {Path(model.trainer.save_dir) / 'weights' / 'best.pt'}")
    except Exception:
        failed = True
        raise
    finally:
        if wandb_run is not None:
            import wandb

            if wandb.run is not None:
                wandb.finish(exit_code=1 if failed else 0)


if __name__ == "__main__":
    main()
