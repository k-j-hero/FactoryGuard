from pathlib import Path
import tempfile
import unittest

import yaml

from train import prepare_data


class TrainDataPreparationTests(unittest.TestCase):
    def test_multiple_source_directories_are_resolved_for_ultralytics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for source in ("source_a", "source_b"):
                for split in ("train", "valid"):
                    image_dir = root / source / split / "images"
                    label_dir = root / source / split / "labels"
                    image_dir.mkdir(parents=True)
                    label_dir.mkdir(parents=True)
                    (image_dir / "sample.jpg").write_bytes(b"image")
                    (label_dir / "sample.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")

            source_yaml = root / "data.yaml"
            source_yaml.write_text(
                "path: .\n"
                "train: [source_a/train/images, source_b/train/images]\n"
                "val: [source_a/valid/images, source_b/valid/images]\n"
                "names: {0: forklift, 1: person}\n",
                encoding="utf-8",
            )
            output_yaml = root / "resolved.yaml"

            prepare_data(source_yaml, output_yaml)
            resolved = yaml.safe_load(output_yaml.read_text(encoding="utf-8"))

            self.assertEqual(len(resolved["train"]), 2)
            self.assertEqual(len(resolved["val"]), 2)
            self.assertTrue(all(Path(path).is_absolute() for path in resolved["train"] + resolved["val"]))
            self.assertEqual(resolved["names"], {0: "forklift", 1: "person"})


if __name__ == "__main__":
    unittest.main()
