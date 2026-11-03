# ADDLC for CAV

CTM environments, PPO/DQN/discrete SAC agents, saved models, and CSV-based plotting.

## Structure

- `envs/`: three-, four-, and five-lane environments and reward functions.
- `agents/`: policies, discrete SAC, and learning schedules.
- `training.py`, `evaluation.py`, `experiments.py`: training and simulation evaluation.
- `data/`: demand generation and CSV export code; `plots/`: CSV-only plotting.
- `assets/pems/`: extracted PeMS flow inputs and source manifest.
- `configs/`: scenario seeds, demand parameters, and reward/training settings.
- `models/`: trained policies and their paired `normalizer_final.pkl`; `index.json` records file hashes.
- `examples/figure05/`: Figure 5 scenario configuration and data-generation entry point.

## Installation

Use Python 3.11 or 3.12. From this directory:

```sh
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
```

The flat source directory is installed as the `cav_dl` package. Console entry points are defined in `pyproject.toml`.

## Figure 5

```sh
python examples/figure05/generate.py --output outputs/figure05_data
cav-dl-plot 5 --data outputs/figure05_data --output outputs/figures
```

The generator runs the saved scenario with PPO and FS1 and exports CSV files; the plotter reads those files. Each trained policy must use the normalizer stored in the same model directory.

## PeMS

Default boundary-flow scales are 0.6, 0.8, and 1.0 for three, four, and five lanes, respectively. See `assets/pems/README.md` for units and time windows.

Scenario files are in `configs/`. Figure 5 uses the seed in `examples/figure05/config.json`. Model training seeds are recorded in `models/index.json`.

## Rebuild data and figures

`records/` contains archived simulation outputs and training logs. `data/fig*.csv` contains plotting data generated from these records.

```sh
python generate_data.py all --from-records records --data generated
cav-dl-plot all --data generated --output outputs/figures
```

To rerun the saved scenarios rather than aggregate archived records:

```sh
python generate_data.py 4 --output results/ablation --data generated --workers 4
```

Figure 3 and Figure 4(a) use one million RL control decisions per training seed. The Figure 4 test Full score is 10.640; its final-100k-decision training mean is 10.648. These are different statistics.

The lane-width comparison uses the later retrained four-lane model on its saved 100-case cohort; five-lane results use their original 100-case cohort unchanged. Its four-lane model was trained at boundary scale 0.8. `code/lane4_study.py` preserves the corresponding study driver. See `docs/manuscript_alignment.md` for remaining discrepancies in the supplied PDF.

To export the manuscript filenames, including `reward_p05_week.pdf` and the combined compliance figure:

```sh
python code/export_paper_figures.py --data data --output outputs/paper
```

The source PDF still requires the corrections listed in `docs/manuscript_alignment.md`; replacement paragraphs are in `docs/manuscript_replacements.tex`.

## Figure 15: later four-lane retraining

Recompute the figure using the saved 100-case manifests per width and paired model/normalizer files:

```sh
python code/reproduce_figure15.py --workers 4
```

Or rebuild the same CSV and figures from the supplied raw records:

```sh
python code/reproduce_figure15.py --from-records records
```

`code/lane4_study.py` is the matching four-lane training/refinement driver. `experiments.py` performs policy replay, `data/figure_data.py` aggregates case/seed results, and `plots/lanes.py` reads the CSV. The builder checks demand fingerprints and matched training seeds before generating Figure 15.

Figure 5 is a dedicated correlated-arrival stress example: a 60-minute block determines the local CAV-composition regime, with random composition at each 15-second CTM step and a vehicle-weighted episode target of 0.4. Main training and ablation demand generation is unchanged. Its scenario settings are in `examples/figure05/config.json`; explored outcomes are retained in `docs/figure05_block_tests.csv`.

## Figure 5: correlated composition

```sh
python code/reproduce_figure05.py
```

This runs PPO and FS1 on the same saved scenario, writes CSVs to `outputs/figure05/data`, and creates the six manuscript panels in `outputs/figure05/figures`. Use `--format all` to export PDF, PNG, and SVG together. Demand and seed settings are read from `examples/figure05/config.json`; the generator is `data/illustration.py`, and the CSV-only renderer is `plots/timeseries.py`.
