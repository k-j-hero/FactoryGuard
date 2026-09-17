"""Import an existing Ultralytics training run into Weights & Biases."""

import argparse
import csv
import json
import os
from pathlib import Path

import yaml


def read_metrics(path):
    rows = []
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        for raw in csv.DictReader(handle):
            row = {}
            for key, value in raw.items():
                key = key.strip()
                value = value.strip()
                if not value:
                    continue
                row[key] = int(float(value)) if key == "epoch" else float(value)
            rows.append(row)
    if not rows:
        raise ValueError(f"No metric rows in {path}")
    return rows


def evaluation_rows(report):
    rows = []
    for evaluation_name, evaluation in report.get("evaluations", {}).items():
        for scope, metrics in [("all", evaluation["all"]), *evaluation.get("per_class", {}).items()]:
            rows.append({"evaluation": evaluation_name, "scope": scope, **metrics})
    return rows


def main():
    parser = argparse.ArgumentParser(description="Log a completed YOLO training run to Weights & Biases")
    parser.add_argument("--run-dir", required=True, help="Ultralytics run containing results.csv")
    parser.add_argument("--project", default="FactoryGuard")
    parser.add_argument("--entity")
    parser.add_argument("--name")
    parser.add_argument("--mode", choices=("online", "offline"), default="offline")
    parser.add_argument("--evaluation", help="Optional evaluation JSON produced for this model")
    parser.add_argument("--upload-model", action="store_true", help="Upload weights/best.pt as a model artifact")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    metrics_path = run_dir / "results.csv"
    if not metrics_path.is_file():
        raise FileNotFoundError(f"Missing {metrics_path}")
    metrics = read_metrics(metrics_path)

    config_path = run_dir / "args.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) if config_path.is_file() else {}
    config.update({"source_run_dir": run_dir.as_posix(), "imported_history": True})

    wandb_support = run_dir / "wandb-support"
    os.environ.setdefault("WANDB_CACHE_DIR", str(wandb_support / "cache"))
    os.environ.setdefault("WANDB_CONFIG_DIR", str(wandb_support / "config"))
    os.environ.setdefault("WANDB_DATA_DIR", str(wandb_support / "data"))

    import wandb

    run = wandb.init(
        project=args.project,
        entity=args.entity,
        name=args.name or run_dir.name,
        mode=args.mode,
        job_type="training-import",
        config=config,
        dir=str(run_dir),
        tags=["yolo26s", "person-forklift", "imported"],
    )
    run.define_metric("epoch")
    run.define_metric("*", step_metric="epoch")
    for row in metrics:
        run.log(row)

    best = max(metrics, key=lambda row: row.get("metrics/mAP50-95(B)", float("-inf")))
    run.summary.update({f"best/{key}": value for key, value in best.items()})

    images = {}
    image_keys = {
        "results.png": "plots",
        "confusion_matrix_normalized.png": "cm",
        "PR_curve.png": "pr",
        "F1_curve.png": "f1",
    }
    for filename, key in image_keys.items():
        path = run_dir / filename
        if path.is_file():
            images[key] = wandb.Image(str(path))
    if images:
        run.log(images)

    history_artifact = wandb.Artifact(f"{run.name}-history", type="training-history")
    history_artifact.add_file(str(metrics_path))
    if config_path.is_file():
        history_artifact.add_file(str(config_path))
    run.log_artifact(history_artifact)

    if args.evaluation:
        evaluation_path = Path(args.evaluation).resolve()
        report = json.loads(evaluation_path.read_text(encoding="utf-8"))
        rows = evaluation_rows(report)
        if rows:
            columns = ["evaluation", "scope", "precision", "recall", "map50", "map50_95"]
            run.log({"eval": wandb.Table(columns=columns, data=[[row[col] for col in columns] for row in rows])})
        evaluation_artifact = wandb.Artifact(f"{run.name}-evaluation", type="evaluation")
        evaluation_artifact.add_file(str(evaluation_path))
        run.log_artifact(evaluation_artifact)

    best_weights = run_dir / "weights" / "best.pt"
    if args.upload_model:
        if not best_weights.is_file():
            raise FileNotFoundError(f"Missing {best_weights}")
        model_artifact = wandb.Artifact(f"{run.name}-model", type="model")
        model_artifact.add_file(str(best_weights))
        run.log_artifact(model_artifact, aliases=["best"])

    run.finish()
    print(f"W&B mode: {args.mode}")
    print(f"Imported epochs: {len(metrics)}")
    print(f"Best epoch: {best['epoch']} | mAP50-95: {best['metrics/mAP50-95(B)']:.5f}")


if __name__ == "__main__":
    main()
