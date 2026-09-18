"""Build a non-destructive curated copy of a YOLO detection dataset."""

import argparse
import csv
import json
from pathlib import Path
import shutil

import yaml

from audit_dataset import IMAGE_SUFFIXES, display_path, resolve_dataset


def load_exclusions(review_path, excluded_codes):
    report = json.loads(Path(review_path).read_text(encoding="utf-8"))
    excluded_codes = set(excluded_codes)
    return {
        Path(item["label"]).as_posix()
        for item in report.get("items", [])
        if excluded_codes.intersection(item.get("issue_codes", []))
    }


def load_fixes(path):
    if not path:
        return {}
    fixes = {}
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            lines = {int(value) for value in row["remove_lines"].split("|") if value.strip()}
            fixes[Path(row["label"]).as_posix()] = {"remove_lines": lines, "reason": row.get("reason", "")}
    return fixes


def write_clean_label(source, destination, fix):
    lines = source.read_text(encoding="utf-8-sig").splitlines()
    removed = fix["remove_lines"] if fix else set()
    destination.write_text("\n".join(line for number, line in enumerate(lines, 1) if number not in removed)
                           + ("\n" if lines else ""), encoding="utf-8")


def prepare_curated_dataset(data_yaml, review_path, output_dir, excluded_codes, fixes_path=None):
    _, names, splits = resolve_dataset(data_yaml)
    output_dir = Path(output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    exclusions = load_exclusions(review_path, excluded_codes)
    fixes = load_fixes(fixes_path)
    summary = {"source": display_path(data_yaml), "review": display_path(review_path),
               "excluded_codes": list(excluded_codes), "splits": {}, "applied_fixes": []}

    for split, image_dirs in splits.items():
        output_split = "valid" if split == "val" else split
        image_output = output_dir / output_split / "images"
        label_output = output_dir / output_split / "labels"
        image_output.mkdir(parents=True, exist_ok=True)
        label_output.mkdir(parents=True, exist_ok=True)
        kept = excluded = 0
        for source_index, image_dir in enumerate(image_dirs):
            label_dir = image_dir.parent / "labels"
            for image in sorted(path for path in image_dir.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES):
                relative = image.relative_to(image_dir)
                label = label_dir / relative.with_suffix(".txt")
                if not label.is_file():
                    continue
                label_key = Path(display_path(label)).as_posix()
                if label_key in exclusions:
                    excluded += 1
                    continue
                prefix = f"source{source_index}_" if len(image_dirs) > 1 else ""
                target_image = image_output / f"{prefix}{image.name}"
                target_label = label_output / f"{prefix}{label.name}"
                shutil.copy2(image, target_image)
                fix = fixes.get(label_key)
                write_clean_label(label, target_label, fix)
                if fix:
                    summary["applied_fixes"].append({"label": label_key,
                                                     "remove_lines": sorted(fix["remove_lines"]),
                                                     "reason": fix["reason"]})
                kept += 1
        summary["splits"][split] = {"kept": kept, "excluded": excluded}

    data = {
        "path": ".",
        "train": "train/images",
        "val": "valid/images",
        "names": {class_id: name for class_id, name in sorted(names.items())},
    }
    if "test" in splits:
        data["test"] = "test/images"
    (output_dir / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    (output_dir / "curation-summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary


def main():
    parser = argparse.ArgumentParser(description="Copy a YOLO dataset while excluding reviewed images")
    parser.add_argument("--data", required=True)
    parser.add_argument("--review", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--exclude-code", action="append", default=["large_box"],
                        help="Review issue code to exclude; repeat for multiple codes")
    parser.add_argument("--fixes", help="Optional CSV with label, remove_lines, reason")
    args = parser.parse_args()
    summary = prepare_curated_dataset(args.data, args.review, args.output, args.exclude_code, args.fixes)
    for split, values in summary["splits"].items():
        print(f"{split}: kept={values['kept']} excluded={values['excluded']}")
    print(f"Output: {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
