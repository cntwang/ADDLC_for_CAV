from pathlib import Path
import pandas as pd


def export_table(frame, path, required=()):
    """Call this from training/evaluation to persist a plotting contract."""
    path = Path(path)
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(f"Cannot export {path}: missing columns {sorted(missing)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)

def read_table(path, required):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Input data not found: {path}")
    frame = pd.read_csv(path)
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    return frame
