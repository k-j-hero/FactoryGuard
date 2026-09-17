"""Codec + output integration test with scripted detections; not a model accuracy test."""
import copy
import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

HAS_DEPS = all(importlib.util.find_spec(name) for name in ("cv2", "yaml", "numpy"))


@unittest.skipUnless(HAS_DEPS, "Install requirements.txt for MP4 integration tests")
class PipelineTests(unittest.TestCase):
    def test_annotated_video_event_and_clip(self):
        import cv2
        import numpy as np
        from factoryguard.clips import open_writer
        from factoryguard.config import DEFAULTS
        from factoryguard.pipeline import run

        class Tensor:
            def __init__(self, data):
                self.data = data

            def cpu(self):
                return self

            def tolist(self):
                return self.data

        class Model:
            names = {0: "forklift", 1: "person"}  # deliberately reverse COCO order
            task = "detect"
            frame = 0

            def track(self, image, **kwargs):
                assert kwargs["persist"] is True
                x = 70 if self.frame < 8 else 250
                boxes = SimpleNamespace(xyxy=Tensor([[40, 60, 60, 120], [x, 60, x + 20, 120]]),
                                        id=Tensor([1, 2]), cls=Tensor([1, 0]), conf=Tensor([0.9, 0.9]))
                self.frame += 1
                return [SimpleNamespace(boxes=boxes, plot=lambda img, **kw: img.copy())]

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "input.mp4"
            writer = open_writer(video, 10, (320, 240))
            for _ in range(20):
                writer.write(np.zeros((240, 320, 3), dtype=np.uint8))
            writer.release()
            config = copy.deepcopy(DEFAULTS)
            summary = run(video, root / "result", config, model=Model())
            self.assertEqual((summary["processed_frames"], summary["event_count"], summary["clip_count"]), (20, 1, 1))
            with (root / "result/events.csv").open() as handle:
                event = list(csv.DictReader(handle))[0]
            self.assertEqual((event["start_frame"], event["end_frame"]), ("0", "7"))
            cap = cv2.VideoCapture(str(root / "result/clips/event_00001.mp4"))
            self.assertTrue(cap.read()[0])
            self.assertEqual(cap.get(cv2.CAP_PROP_FRAME_COUNT), 20)
            cap.release()
            with self.assertRaises(FileExistsError):
                run(video, root / "result", config, model=Model())


if __name__ == "__main__":
    unittest.main()
