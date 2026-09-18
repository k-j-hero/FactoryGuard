import csv
import json
from pathlib import Path
import tempfile
import unittest

from prepare_curated_dataset import prepare_curated_dataset


class CuratedDatasetTests(unittest.TestCase):
    def test_excludes_reviewed_image_and_applies_line_fix(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            for split in ("train", "valid"):
                (source / split / "images").mkdir(parents=True)
                (source / split / "labels").mkdir(parents=True)
            (source / "data.yaml").write_text(
                "path: .\ntrain: train/images\nval: valid/images\nnames: {0: forklift, 1: person}\n"
            )
            for name in ("keep", "drop"):
                (source / "train/images" / f"{name}.jpg").write_bytes(name.encode())
                (source / "train/labels" / f"{name}.txt").write_text(
                    "0 0.5 0.5 0.4 0.4\n1 0.5 0.5 0.2 0.2\n"
                )
            (source / "valid/images/val.jpg").write_bytes(b"val")
            (source / "valid/labels/val.txt").write_text("0 0.5 0.5 0.2 0.2\n")
            review = root / "review.json"
            review.write_text(json.dumps({"items": [{
                "label": (source / "train/labels/drop.txt").as_posix(),
                "issue_codes": ["large_box"],
            }]}))
            fixes = root / "fixes.csv"
            with fixes.open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["label", "remove_lines", "reason"])
                writer.writerow([(source / "train/labels/keep.txt").as_posix(), "1", "test"])

            summary = prepare_curated_dataset(
                source / "data.yaml", review, root / "output", ["large_box"], fixes
            )

            self.assertEqual(summary["splits"]["train"], {"kept": 1, "excluded": 1})
            self.assertFalse((root / "output/train/images/drop.jpg").exists())
            self.assertEqual((root / "output/train/labels/keep.txt").read_text().strip(),
                             "1 0.5 0.5 0.2 0.2")


if __name__ == "__main__":
    unittest.main()
