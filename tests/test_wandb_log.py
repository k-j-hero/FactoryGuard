import json
from pathlib import Path
import tempfile
import unittest

from log_wandb import evaluation_rows, read_metrics


class WandbImportTests(unittest.TestCase):
    def test_reads_ultralytics_csv_and_selects_numeric_values(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.csv"
            path.write_text(
                "                  epoch,   metrics/precision(B), metrics/mAP50-95(B)\n"
                "                      1,                 0.75,               0.50\n",
                encoding="utf-8",
            )
            self.assertEqual(
                read_metrics(path),
                [{"epoch": 1, "metrics/precision(B)": 0.75, "metrics/mAP50-95(B)": 0.5}],
            )

    def test_flattens_evaluation_scopes_for_wandb_table(self):
        report = {
            "evaluations": {
                "ground": {
                    "all": {"precision": 0.8, "recall": 0.7, "map50": 0.75, "map50_95": 0.5},
                    "per_class": {
                        "forklift": {"precision": 0.9, "recall": 0.8, "map50": 0.85, "map50_95": 0.6}
                    },
                }
            }
        }
        rows = evaluation_rows(report)
        self.assertEqual([row["scope"] for row in rows], ["all", "forklift"])
        self.assertEqual(rows[1]["map50_95"], 0.6)


if __name__ == "__main__":
    unittest.main()
