r"""Run in a user terminal: .venv\Scripts\python.exe scripts/first_demo.py --setup.

Installation is opt-in. All packages go into this project's .venv. Downloads
and generated outputs stay inside the project and are excluded from Git.
"""
import argparse
from datetime import datetime
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import urllib.request
import uuid

ROOT = Path(__file__).resolve().parents[1]
SOURCE_PAGE = "https://www.pexels.com/video/man-walking-towards-the-forklift-inside-the-warehouse-4294434/"
SOURCE_MP4 = "https://videos.pexels.com/video-files/4294434/4294434-uhd_3840_2160_25fps.mp4"


def execute(*args, stdout=None):
    subprocess.run([sys.executable, *map(str, args)], cwd=ROOT, check=True, stdout=stdout)


def download_sample(destination):
    if destination.is_file():
        return
    print("Downloading Pexels sample (Tiger Lily, video 4294434)...", flush=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Incomplete downloads are never reused as MP4 inputs.
    partial = destination.with_suffix(".mp4.part")
    try:
        request = urllib.request.Request(SOURCE_MP4, headers={"User-Agent": "FactoryGuard/0.1"})
        with urllib.request.urlopen(request, timeout=60) as response, partial.open("wb") as handle:
            total = int(response.headers.get("Content-Length", 0))
            received = 0
            next_report = 0
            while block := response.read(1024 * 1024):
                handle.write(block)
                received += len(block)
                if received >= next_report:
                    print(f"Downloaded {received / 1048576:.0f} MB" +
                          (f" / {total / 1048576:.0f} MB" if total else ""), flush=True)
                    next_report = received + 25 * 1048576
        if received == 0 or (total and received != total):
            raise RuntimeError("Sample download is incomplete")
        partial.replace(destination)
    except Exception as error:
        raise RuntimeError(f"Sample download failed. Download the MP4 manually from {SOURCE_PAGE} "
                           "and rerun with --source followed by its local file path.") from error


def make_preview(source, destination, max_frames):
    import cv2
    import math
    cap = cv2.VideoCapture(str(source))
    writer = None
    count = 0
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        width, height = (int(cap.get(prop)) for prop in
                         (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT))
        if not cap.isOpened() or not math.isfinite(fps) or fps <= 0 or min(width, height) < 2:
            raise RuntimeError(f"Cannot read valid video metadata: {source}")
        scale = min(1.0, 1280 / width, 720 / height)
        size = (max(2, int(width * scale) // 2 * 2), max(2, int(height * scale) // 2 * 2))
        writer = cv2.VideoWriter(str(destination), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
        if not writer.isOpened():
            raise RuntimeError("OpenCV cannot create the preview MP4")
        while count < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            writer.write(cv2.resize(frame, size, interpolation=cv2.INTER_AREA))
            count += 1
        if not count:
            raise RuntimeError("No frames decoded from input")
    finally:
        cap.release()
        if writer is not None:
            writer.release()
    return {"source_width": width, "source_height": height, "fps": fps,
            "preview_width": size[0], "preview_height": size[1], "preview_frames": count,
            "preview_duration_sec": count / fps, "audio": False}


def main():
    parser = argparse.ArgumentParser(description="Install optional dependencies and run the first YOLO26s video demo")
    parser.add_argument("--setup", action="store_true", help="Install packages into the project .venv (several GB)")
    parser.add_argument("--source", type=Path, help="Use your own MP4 instead of downloading the Pexels sample")
    parser.add_argument("--frames", type=int, default=300, help="Process at most this many original frames")
    parser.add_argument("--device", choices=["0", "cpu"], default="0")
    args = parser.parse_args()
    if args.frames < 1:
        parser.error("--frames must be positive")
    if Path(sys.prefix).resolve() != (ROOT / ".venv").resolve():
        parser.error("Use this project's .venv Python; global Python is not modified")
    if args.source and not args.source.is_file():
        parser.error(f"Input MP4 does not exist: {args.source}")
    job = ROOT / "runs" / ("first_demo_" + datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6])
    job.mkdir(parents=True)
    print(f"Run folder: {job}", flush=True)
    try:
        if args.setup:
            print("[1/4] Installing PyTorch and FactoryGuard requirements...", flush=True)
            index = "cu130" if args.device == "0" else "cpu"
            # Official matching Windows/Linux binary versions, documented in FIRST_DEMO.md.
            execute("-m", "pip", "install", "torch==2.12.1", "torchvision==0.27.1",
                    "--index-url", f"https://download.pytorch.org/whl/{index}")
            execute("-m", "pip", "install", "-r", ROOT / "requirements.txt")
            execute("-m", "pip", "check")
        print("[2/4] Checking environment and GPU execution...", flush=True)
        import torch
        import cv2
        import ultralytics
        import yaml
        if args.device == "0":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA unavailable. Check the PyTorch installation, or use --device cpu.")
            sample = torch.ones((32, 32), device="cuda:0")
            if (sample @ sample).sum().item() != 32768:
                raise RuntimeError("GPU computation check failed")
            torch.cuda.synchronize()
        env = {"python": sys.version, "torch": torch.__version__, "opencv": cv2.__version__,
               "ultralytics": ultralytics.__version__, "yaml": yaml.__version__,
               "lap": importlib.metadata.version("lap"), "device": args.device,
               "cuda": torch.version.cuda,
               "gpu": torch.cuda.get_device_name(0) if args.device == "0" else None}
        (job / "environment.json").write_text(json.dumps(env, indent=2), encoding="utf-8")
        with (job / "requirements.freeze.txt").open("w", encoding="utf-8") as handle:
            execute("-m", "pip", "freeze", stdout=handle)
        print("[3/4] Preparing a short preview video...", flush=True)
        source = args.source.resolve() if args.source else ROOT / "data/videos/pexels_4294434.mp4"
        if not args.source:
            download_sample(source)
        preview = job / "input_preview.mp4"
        metadata = make_preview(source, preview, args.frames)
        metadata.update({"original_file": str(source), "source_page": SOURCE_PAGE if not args.source else None,
                         "creator": "Tiger Lily" if not args.source else None})
        (job / "input.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print("[4/4] YOLO26s + ByteTrack: person detection and tracking...", flush=True)
        print("COCO has no forklift class: no proximity events are expected.", flush=True)
        execute(ROOT / "infer.py", "--source", preview, "--output", job / "result",
                "--model", "yolo26s.pt", "--device", args.device)
        print(f"\nDONE. Play this video:\n{job / 'result/annotated.mp4'}", flush=True)
        print(f"Tracks: {job / 'result/tracks.csv'}\nRun summary: {job / 'result/summary.json'}", flush=True)
    except Exception as error:
        (job / "SETUP_OR_DEMO_FAILED.txt").write_text(f"{type(error).__name__}: {error}\n", encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
