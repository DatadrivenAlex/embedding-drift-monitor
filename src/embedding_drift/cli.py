import argparse
import json
from pathlib import Path

from embedding_drift.monitor import compare


def main(argv=None):
    p = argparse.ArgumentParser(description="Compare two JSON embedding matrices")
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--current", type=Path, required=True)
    p.add_argument("--threshold", type=float, default=0.1)
    p.add_argument("--bins", type=int, default=10)
    args = p.parse_args(argv)
    try:
        reference = json.loads(args.reference.read_text(encoding="utf-8"))
        current = json.loads(args.current.read_text(encoding="utf-8"))
        report = compare(reference, current, threshold=args.threshold, bins=args.bins)
        print(json.dumps(report.to_dict(), allow_nan=False))
        return 1 if report.alert else 0
    except (ValueError, OSError) as exc:
        print(json.dumps({"error": str(exc)}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
