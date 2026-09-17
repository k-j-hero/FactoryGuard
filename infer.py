"""Run from the project root: python infer.py --source video.mp4 --output runs/demo"""
import argparse
import logging

from factoryguard.config import load_config
from factoryguard.pipeline import run


def main():
    parser = argparse.ArgumentParser(description="FactoryGuard video detection, tracking and close approach candidates")
    parser.add_argument("--source", required=True, help="Local industrial MP4")
    parser.add_argument("--output", required=True, help="New output directory")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--model", help="yolo26s.pt or a trained best.pt")
    parser.add_argument("--device", help="cpu or CUDA GPU index, e.g. 0")
    parser.add_argument("--max-frames", type=int, help="Optional short smoke test")
    parser.add_argument("--require-forklift", action="store_true", help="Fail if model has no forklift class")
    args = parser.parse_args()
    config = load_config(args.config)
    for key in ("model", "device"):
        if getattr(args, key) is not None:
            config[key] = getattr(args, key)
    if args.require_forklift:
        config["require_forklift"] = True
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run(args.source, args.output, config, args.max_frames)


if __name__ == "__main__":
    main()
