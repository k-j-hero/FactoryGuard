import argparse
import csv

from factoryguard.clips import extract_clips


def main():
    parser = argparse.ArgumentParser(description="Extract event clips from an annotated MP4 and events.csv")
    parser.add_argument("--video", required=True)
    parser.add_argument("--events", required=True)
    parser.add_argument("--output", required=True, help="New clips directory")
    parser.add_argument("--pre-sec", type=float, default=2.0)
    parser.add_argument("--post-sec", type=float, default=2.0)
    args = parser.parse_args()
    with open(args.events, newline="", encoding="utf-8") as handle:
        events = list(csv.DictReader(handle))
    rows = extract_clips(args.video, events, args.output, args.pre_sec, args.post_sec)
    print(f"Extracted {len(rows)} clips to {args.output}")


if __name__ == "__main__":
    main()
