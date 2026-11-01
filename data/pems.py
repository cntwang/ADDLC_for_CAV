

import hashlib
import json
from pathlib import Path

import numpy as np

from .synthetic import EpisodeInputGenerator, quota15, weighted_quota


class PeMSDemand:
    def __init__(self, directory, lanes=3, split="train", scale=None):
        directory = Path(directory)
        self.manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        self.archive = np.load(directory / "daily_flows.npz", allow_pickle=False)
        self.days = self.manifest["date_splits"][split]
        self.scale = lanes / 5 if scale is None else scale
        if not 0 < self.scale <= 1:
            raise ValueError("The archived PeMS protocol requires boundary scale in (0, 1]")

    def scene(self, seed, p, day, composition_scale=None):
        saved = np.random.get_state()
        try:
            np.random.seed(seed)
            data = EpisodeInputGenerator().generate_test_reset_config(seed, 10600, 6500, [p])
        finally:
            np.random.set_state(saved)
        q = self.archive[day.replace("-", "")][:, 36:162]
        main, ramp = np.repeat(q / 20, 20, axis=1)
        original_main, original_ramp = main, ramp
        main, ramp = main * self.scale, ramp * self.scale
        if composition_scale is None:
            cm, cr = quota15(main, ramp, p, seed)
        else:
            
            cm, cr = quota15(original_main * composition_scale,
                            original_ramp * composition_scale, p, seed)
        data.update(input_supply=main.copy(), ramp_inflow=ramp.copy(),
                    input_cav_supply=cm, ramp_cav_ratio=cr)
        data["init_cav_ratio"] = weighted_quota(data["init_car"], data["init_cav_ratio"], p)
        digest = hashlib.sha256()
        for key in ["init_car", "init_cav_ratio", "input_supply", "ramp_inflow",
                    "input_cav_supply", "ramp_cav_ratio"]:
            digest.update(np.asarray(data[key], dtype="<f8").tobytes())
        data.update(p=float(p), seed=int(seed), date=day, sha256=digest.hexdigest(),
                    offered_arrivals=float(main.sum() + ramp.sum()),
                    arrival_quota_error=float(main @ cm + ramp @ cr - p * (main.sum() + ramp.sum())))
        return data
