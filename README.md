# ADDLC for CAV

CTM simulation, PPO/DQN/discrete SAC training, saved-policy evaluation, and CSV-based figures.

## Files

- `envs/`, `agents/`: traffic environments and agents.
- `training.py`, `experiments.py`, `evaluation.py`: training and paired simulation tests.
- `data/`, `plots/`: demand/CSV generation and CSV-only plotting.
- `configs/`, `assets/pems/`: saved scenario seeds and PeMS inputs.
- `models/`: 33 trained policies with their paired normalizers; `index.json` identifies and verifies them.
- `records/`: simulation records and training logs used to reproduce the figures.
- `code/`, `examples/`: figure replay/export and four-lane retraining entry points.
- `tests/`: consistency checks.

## Install

Python 3.11 or 3.12:

```sh
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
```

## Reproduce

Rebuild plotting data from archived records, then export manuscript filenames:

```sh
python generate_data.py all --from-records records --data generated
python code/export_paper_figures.py --data generated --output outputs/paper
```

Rerun Figure 5 or Figure 15 with the saved policies and scenarios:

```sh
python code/reproduce_figure05.py
python code/reproduce_figure15.py --workers 4
```

Rerun the reward ablation:

```sh
python generate_data.py 4 --output results/ablation --data generated --workers 4
```

Training settings are in `defaults.json`. Use `cav-dl-train --help`; `code/lane4_study.py` contains the later four-lane training/refinement procedure. Training curves come from the supplied historical logs.

PeMS boundary-flow scales are 0.6/0.8/1.0 for 3/4/5 lanes. Figure 5 is a dedicated 60-minute correlated-composition example with 15-second random arrivals and overall CAV fraction 0.4; main training/ablation arrival generation is unchanged. Figure 15 uses the later four-lane models on the saved 100-case cohort and the unchanged five-lane 100-case cohort. Policy and normalizer files must be used together.

See `records/README.md` for record and cohort definitions.
