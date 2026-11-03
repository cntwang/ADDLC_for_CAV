# Experiment records

Each series contains raw simulation outputs. Training curves use the logs in `training/`; plotting CSVs are rebuilt by `data/figure_data.py`.

The synthetic 100-case demonstration cohort includes five cases at each endpoint and ten at each interior penetration level. It was retained after examining outcomes; it is not an unbiased independent random test set. The separately supplied `confirmation/` records and `configs/confirmation_cases_3lane.csv` contain the independent 90-case check. The PeMS penetration and lane-width comparisons use saved illustrative cohorts; the four- and five-lane manifests each contain 100 cases with nonuniform penetration counts.

Figure 5 is a separate correlated-composition stress example, defined by `examples/figure05/config.json`. Other block tests, prior figure versions, and detailed audit reports are preserved outside this compact repository in the cleanup backup. The favorable example is not a population-wide performance estimate.

Full's 10.648 training return and 10.640 fixed-window test reward use different horizons and scenario distributions. TTS includes the road and both entrance queues until complete clearance. Lane-count variation sums absolute changes in executed DL counts.
