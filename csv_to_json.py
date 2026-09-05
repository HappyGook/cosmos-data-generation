import csv
import json
import sys
from pathlib import Path

INT_FIELDS = ("expected_length_frames", "fps")


def row_to_record(row: dict) -> dict:
    record = dict(row)
    for field in INT_FIELDS:
        raw = record.get(field, "")
        if raw not in (None, ""):
            try:
                record[field] = int(raw)
            except ValueError:
                pass
    return record


def main() -> None:
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("resources/prompts.csv")
    dst = Path(sys.argv[2]) if len(sys.argv) > 2 else src.with_suffix(".json")

    with src.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        records = [row_to_record(row) for row in reader]

    dst.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {len(records)} records -> {dst}")


if __name__ == "__main__":
    main()