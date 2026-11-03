# Manuscript alignment

Reference: `Manuscript_TRC_26_01508 (2).pdf`, supplied 2026-11-03.

## Implemented consistency fixes

- Figure 3 and ablation: shared five-term reward; main training budget is 1,000,000 RL control decisions. The callback now stops at the requested budget, rather than the next rollout boundary.
- Training return Full 10.648 and fixed-window test reward Full 10.640 are separate statistics.
- Figures 6--8 include p=0.1--0.9; Figure 9 retains the supplied paper's p=0.3--0.9 presentation range.
- Compliance evaluation exports the nine interior penetration levels and all six compliance rates.
- Figure 12 dates panel uses the ten weekdays; its day-of-week panel includes all seven days, with Saturday and Sunday labels.
- SAC/DQN replay tasks use their own saved algorithms and models.
- Model/normalizer files and scenario manifests are matched to the checked experiment archive.
- Figure 5 reads the supplied JSON scenario configuration, uses the same CTM as the main experiments, and generates CSV before plotting.
- Four-lane PeMS boundary scale is 0.8, three-lane 0.6, five-lane 1.0.

## PDF items that still require manuscript replacement

These are inconsistencies in the supplied PDF, not values to force into the simulator.

1. Table 1: native episode storage/progress horizon is 5000 CTM steps, not 4000; density incentive width is 2.0, not 0.5. Termination flags can stop an episode before this storage horizon.
2. Table 2: the native PeMS runs use the same 5000-step storage horizon, 10.5-h input duration, time-penalty growth weight 13, timeout flag at step 3920, outflow target 65000, and main training budget 1000000 RL decisions. The old 8520 / 18 h / 11 / 8400 / 150000 / 5000000 table does not describe these runs. Physical evaluation separately continues to clearance (safety cap 28800 steps).
3. Figure 5 uses a dedicated correlated-composition stress scenario: 60-minute inlet-specific latent blocks with random arrivals every 15 seconds, while the whole-demand vehicle-weighted fraction is 0.4. It retains the main linear-headway CTM, current trained PPO, reward, and initial-state generator. It is an illustrative stress scenario, not the independent-step composition distribution used by the main training/ablation. The previous independent-step replay had equal TTS; that result is preserved in the audit backup. All explored block tests, including ties and losses, are retained in docs/figure05_block_tests.csv.
4. Figure 15: use the later four-lane retraining on the same saved 100-case four-lane cohort, with the original 100-case five-lane cohort unchanged. Four-lane transfer/retrained TTS is 132519.280293 / 131365.763234 vehicle-hours: a 0.87% reduction, not the earlier 4.98% increase and not 9.1%. Five-lane TTS falls 15.88%. The corresponding entry-wait reductions are 2.75% and 13.64%, and downstream-throughput increases are 0.25% and 4.29%. These are separate saved cohorts per width, not the independent 90-case audit cohorts; four-lane penetration groups are not uniformly distributed. Replace the supplied PDF four-lane bars and paragraph with the matched retrained results.
5. Figure 12 text says five weekday groups, but the updated panel has seven day-of-week groups. Specify ten weekday dates in panel (c), seven groups based on fourteen dates in panel (d), and 30 paired scenarios per date.
6. The threshold-rule implementation matches the numerical thresholds, but the supplied PDF's claim that validation selected them needs supporting validation records. Without those records, describe them as fixed thresholds.
7. DQN in the code uses the SB3 Huber loss with terminal masking; discrete SAC uses a three-step discounted target. The PDF's squared-error DQN loss and single-step SAC target are schematic and should be updated for implementation-level correspondence.
8. The implemented reallocation searches discrete trial counts; it is not the closed-form allocation algorithm printed in the supplied PDF. State the implemented search rather than claiming equivalence.

No policy was retrained or outcome value edited to match the PDF. The corrections preserve the verified models, reward, and case manifests. Archived records are provided for reproducing training curves without pretending that a fresh stochastic training run is bitwise identical.

Additional compatibility fix: arbitrary new training seeds are mapped into the NumPy legacy seed range. The original three training seeds retain their exact demand streams.
