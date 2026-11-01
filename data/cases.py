

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

FIELDS = ["init_car", "init_cav_ratio", "input_supply", "input_cav_supply",
          "ramp_inflow", "ramp_cav_ratio"]
MANIFEST_FIELDS = ["case_id", "traffic_seed", "p", "main", "ramp", "date", "rep",
                   "boundary_scale", "composition_scale", "scenario_sha256"]


def read_manifest(path):
    table = pd.read_csv(path)
    for old, new in [("case", "case_id"), ("flow", "traffic_seed")]:
        if new not in table and old in table:
            table = table.rename(columns={old: new})
    required = {"case_id", "traffic_seed", "p"}
    if "date" not in table:
        required.update(["main", "ramp"])
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"Missing scenario fields: {sorted(missing)}")
    table = table[[name for name in MANIFEST_FIELDS if name in table]].copy()
    for name in ["case_id", "traffic_seed", "main", "ramp", "rep"]:
        if name in table:
            value = pd.to_numeric(table[name], errors="raise")
            if not np.isfinite(value).all() or (value % 1 != 0).any():
                raise ValueError(f"{name} must contain finite integers")
            table[name] = value.astype("int64")
    table["p"] = pd.to_numeric(table["p"], errors="raise")
    if not table["p"].between(0, 1).all():
        raise ValueError("p must be between 0 and 1")
    if table.case_id.duplicated().any():
        raise ValueError("case_id must be unique within a scenario manifest")
    for name in ["boundary_scale", "composition_scale"]:
        if name in table:
            table[name] = pd.to_numeric(table[name], errors="raise")
            if not np.isfinite(table[name]).all() or (table[name] <= 0).any():
                raise ValueError(f"{name} must be positive and finite")
    return table


def fingerprint(data, fields=FIELDS):
    digest = hashlib.sha256()
    for key in fields:
        digest.update(np.asarray(data[key], dtype="<f8").tobytes())
    return digest.hexdigest()


def save_case(data, folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    arrays = {key: value for key, value in data.items() if isinstance(value, np.ndarray)}
    meta = {key: value.item() if isinstance(value, np.generic) else value
            for key, value in data.items() if key not in arrays}
    meta["arrays_sha256"] = fingerprint(data)
    np.savez_compressed(folder / "input_arrays.npz", **arrays)
    (folder / "reset_config.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def load_case(folder):
    folder = Path(folder)
    data = json.loads((folder / "reset_config.json").read_text(encoding="utf-8"))
    with np.load(folder / "input_arrays.npz", allow_pickle=False) as archive:
        data.update({key: archive[key].copy() for key in archive.files})
    digest = fingerprint(data)
    if "arrays_sha256" in data:
        expected = data["arrays_sha256"]
    else:
        
        expected = data["sha256"]
        other = FIELDS[:3] + ["ramp_inflow", "input_cav_supply", "ramp_cav_ratio"]
        if digest != expected and fingerprint(data, other) == expected:
            digest = expected
    if digest != expected:
        raise ValueError(f"Scenario fingerprint mismatch: {folder}")
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="CSV: case_id,traffic_seed,p,main,ramp or date")
    parser.add_argument("--output", default="cases")
    parser.add_argument("--pems-data", help="Generate PeMS cases using manifest date column")
    parser.add_argument("--lanes", type=int, choices=[3, 4, 5], default=3)
    parser.add_argument("--pems-scale", type=float, help="Explicit boundary scaling")
    args = parser.parse_args()
    rows = read_manifest(args.manifest)
    output = Path(args.output)
    from .synthetic import MODES, scene
    if args.pems_data:
        from .pems import PeMSDemand
        demand = PeMSDemand(args.pems_data, args.lanes, scale=args.pems_scale)
    for row in rows.itertuples():
        if args.pems_data:
            if args.pems_scale is None and hasattr(row, "boundary_scale"):
                demand.scale = float(row.boundary_scale)
            data = demand.scene(int(row.traffic_seed), float(row.p), row.date,
                                composition_scale=getattr(row, "composition_scale", None))
            data["boundary_scale"] = float(demand.scale)
            if hasattr(row, "composition_scale"):
                data["composition_scale"] = float(row.composition_scale)
        else:
            data = scene(int(row.traffic_seed), float(row.p), MODES[int(row.main)], MODES[int(row.ramp)])
        case_id = int(row.case_id)
        data["case_id"] = case_id
        if hasattr(row, "rep"):
            data["rep"] = int(row.rep)
        expected = getattr(row, "scenario_sha256", None)
        if expected is not None and data["sha256"] != expected:
            raise ValueError(f"Seed regeneration differs from saved case {case_id}; use exact arrays")
        save_case(data, output / f"case_{int(case_id):05d}")
    print(f"Saved {len(rows)} exact demand cases to {output}")


if __name__ == "__main__":
    main()
