from pathlib import Path
import csv
import pandas as pd


def export_table(frame, path, required=(), omit=()):
    
    path = Path(path)
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(f"Cannot export {path}: missing columns {sorted(missing)}")
    empty = frame.replace(r"^\s*$", pd.NA, regex=True).isna().all()
    remove = (set(empty.index[empty]) | set(omit)) - set(required)
    frame = frame.drop(columns=list(remove & set(frame.columns)))
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def append_record(row, path, omit=()):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {key: value for key, value in row.items() if key not in omit
           and value is not None and not pd.isna(value)
           and not (isinstance(value, str) and not value.strip())}
    fields = []
    if path.exists() and path.stat().st_size:
        with path.open(newline="", encoding="utf-8") as stream:
            fields = next(csv.reader(stream))
    extra = [key for key in row if key not in fields]
    if fields and extra:
        temporary = path.with_suffix(path.suffix + ".tmp")
        with path.open(newline="", encoding="utf-8") as source, temporary.open("w", newline="", encoding="utf-8") as target:
            reader = csv.DictReader(source)
            writer = csv.DictWriter(target, fieldnames=fields + extra)
            writer.writeheader()
            writer.writerows(reader)
            writer.writerow(row)
        temporary.replace(path)
    else:
        with path.open("a", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields or list(row))
            if not fields:
                writer.writeheader()
            writer.writerow(row)

def read_table(path, required):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Input data not found: {path}")
    frame = pd.read_csv(path)
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    return frame
