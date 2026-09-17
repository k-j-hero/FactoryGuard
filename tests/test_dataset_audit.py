import json
from pathlib import Path
import tempfile
import unittest

from audit_dataset import audit_dataset


class DatasetAuditTests(unittest.TestCase):
    def make_dataset(self, root):
        root = Path(root)
        (root / "data.yaml").write_text(
            "path: .\ntrain: train/images\nval: valid/images\nnames:\n  0: forklift\n  1: person\n",
            encoding="utf-8",
        )
        for split in ("train", "valid"):
            (root / split / "images").mkdir(parents=True)
            (root / split / "labels").mkdir(parents=True)
        return root / "data.yaml"

    def test_valid_dataset_counts_classes_and_empty_background(self):
        with tempfile.TemporaryDirectory() as directory:
            data = self.make_dataset(directory)
            (data.parent / "train/images/a.jpg").write_bytes(b"train-image")
            (data.parent / "train/labels/a.txt").write_text("0 0.5 0.5 0.2 0.2\n1 0.2 0.3 0.1 0.2\n")
            (data.parent / "valid/images/b.jpg").write_bytes(b"valid-image")
            (data.parent / "valid/labels/b.txt").write_text("")

            report = audit_dataset(data)

            self.assertEqual(report["status"], "pass")
            self.assertEqual(report["totals"]["images"], 2)
            self.assertEqual(report["totals"]["class_instances"], {"0": 1, "1": 1})
            self.assertEqual(report["splits"]["val"]["empty_labels"], 1)

    def test_invalid_box_missing_label_and_split_duplicate_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            data = self.make_dataset(directory)
            duplicate = b"same-image"
            (data.parent / "train/images/frame_1.jpg").write_bytes(duplicate)
            (data.parent / "train/labels/frame_1.txt").write_text("0 0.95 0.5 0.2 0.2\n")
            (data.parent / "valid/images/frame_1.jpg").write_bytes(duplicate)

            report = audit_dataset(data)
            codes = {item["code"] for item in report["issues"]}

            self.assertEqual(report["status"], "fail")
            self.assertIn("box_crosses_image", codes)
            self.assertIn("missing_label", codes)
            self.assertIn("exact_duplicate_across_splits", codes)

    def test_multiple_directories_in_one_split_are_aggregated(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for source in ("source_a", "source_b"):
                for split in ("train", "valid"):
                    (root / source / split / "images").mkdir(parents=True)
                    (root / source / split / "labels").mkdir(parents=True)
                    (root / source / split / "images" / f"{source}.jpg").write_bytes(
                        f"{source}-{split}".encode()
                    )
                    (root / source / split / "labels" / f"{source}.txt").write_text(
                        "0 0.5 0.5 0.2 0.2\n", encoding="utf-8"
                    )
            data = root / "data.yaml"
            data.write_text(
                "path: .\n"
                "train: [source_a/train/images, source_b/train/images]\n"
                "val: [source_a/valid/images, source_b/valid/images]\n"
                "names: {0: forklift, 1: person}\n",
                encoding="utf-8",
            )

            report = audit_dataset(data)

            self.assertEqual(report["status"], "pass")
            self.assertEqual(report["splits"]["train"]["images"], 2)
            self.assertEqual(report["splits"]["val"]["paired"], 2)


if __name__ == "__main__":
    unittest.main()
