# PeMS demand data

- `daily_flows.npz`: paired mainline and on-ramp vehicle counts for 2025-09-24 through 2025-10-07. Each date key (`YYYYMMDD`) contains a (2, 288) array: row 0 is mainline station 717772, row 1 is on-ramp station 716699. Values are vehicles per five-minute interval, rather than vehicles/hour.
- `manifest.json`: source archive hashes, station identifiers, date splits, and data-quality information.

The demand loader reads intervals `[36:162]` (03:00-13:30, America/Los_Angeles). Each five-minute count is divided by 20 and repeated over twenty 15-second simulation steps. Boundary-flow scale is 0.6, 0.8, or 1.0 for three, four, or five lanes, respectively. CAV composition is generated separately using the scenario seed.

Usage with the existing loader: `PeMSDemand("assets/pems", lanes=4)`.

These are the extracted flow inputs used by the experiments, not the full regional PeMS archive. Official source: https://pems.dot.ca.gov/ . PeMS-supplied estimated values are retained; see the manifest for quality and repeated-profile details.
