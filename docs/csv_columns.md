# CSV字段与单位

所有CSV为UTF-8、无行索引。原始计算保留双精度，显示时不回写四舍五入值。
缺失车辆暴露的速度是空值，绘图显示白色，不改成零速。
来源文件路径、SHA256和各表行数写在生成目录的generated_manifest.json；包内快照见data/manifest.json。

场景清单位于configs/。公共字段为case_id、traffic_seed、p、scenario_sha256。
合成需求另含main、ramp（0–5）；PeMS需求另含date、boundary_scale，可选rep、composition_scale。
编号和种子均为整数。case_id标识案例，traffic_seed控制需求随机生成。

| 表 | 绘图必需字段及单位 |
|---|---|
| fig02 | mode、hours（h）、main_veh_h、ramp_veh_h（veh/h） |
| fig03 | algorithm、metric（reward5/controls）、steps、mean、seed_sd |
| fig04_training | arm、steps、mean、seed_sd；三种子均值、总体SD（ddof=0），与原曲线一致 |
| fig04_test | metric、arm、mean、low、high、scale；已经转换为显示单位，不能再除scale |
| fig05 | time_h、main/ramp_CAV_share、Full/FS1_CAV_share、Full/FS1_density、Full/FS1_critical_density、Full/FS1_speed_kmh、Full_minus_FS1_outflow |
| fig05_input | time_h、main_CAV_share、ramp_CAV_share；15秒到达比例 |
| fig05_lanes | time_h、unit（1–4）、lanes（0–2）；已执行配置 |
| fig06 | p、arm、kind（DL/GL/CAV）、cell（1–120）、speed_kmh；附distance_km/vehicle_hours便于核对 |
| fig07 | p、arm、unit（0–5）、density（veh/km/lane）、speed（km/h） |
| fig08 | p、arm、density_std_units、critical_density_RMSE（均veh/km/lane） |
| fig09 | p、arm、source_total_fuel_kg；同p的10例总和，PPO先按案例平均三个种子，再相加 |
| fig10 | p、compliance、PPO_vs_FS1_TTS_pct；正值表示PPO完整TTS较低 |
| fig11 | date、time_h、inlet、flow_veh_h；5分钟计数乘12转小时流率 |
| fig12_training | steps、mean、seed_sd；每种子2k步分箱，再跨种子均值与样本SD |
| fig12_penetration | p、arm、mean、lower95、upper95、n；保存100例的条件bootstrap区间 |
| fig12_dates | scope（date/weekday）、key、arm、mean、lower95、upper95、n；p=0.5日期比较 |
| fig13 | arm、time_h（当天小时）、density、cumulative_outflow |
| fig13_lanes | time_h、unit（1–4）、lanes |
| fig13_difference | time_h、outflow_difference（veh） |
| fig14 | arm、time_h、cell、density（veh/km/lane）、flow（veh/h）、speed（km/h） |
| fig15 | lanes（4/5）、arm（Transfer/Retrained/FS1）、native_episode_reward、complete_tts（veh h）、mean_entry_wait_h（h）、fixed_departures（veh） |

## 口径

- 图4测试：共同五项评分、固定控制窗口；TTS和入口等待包括完整排空。
- 图6–8：同一保存案例清单的90例p=0.1–0.9，CTM步[120,2520)，完全服从；速度为距离/车辆时间。
- 图9：60秒聚合，含入口怠速和完整排空，来源系数为156、3.54、-0.0388、0.000776；显示单位为10³kg。
- 图10：与FS1配对、计入两入口排队的完整TTS，保留负收益；p=0和1端点诊断不进0.1–0.9曲线。
- 图13/14：固定PeMS日期2025-09-29，p=0.5，双入口流量缩放0.6；显示共同窗口。
- 图15：两种路宽各自90例预先固定的新案例，每种p=0.1–0.9各10例；四车道双入口需求缩放0.8，五车道1.0；下游throughput指需求窗口出口车辆数，不含后续排空。

## 新实验如何接入

实验入口会自动保存原始记录并汇总，不需要手工填写绘图数据：

```bash
python generate_data.py 8 --output results/figure08 --data generated
cav-dl-plot 8 --data generated --output figures
```

| 原始记录 | 内容 |
|---|---|
| `<实验组>/raw.csv` | 每次回放的job_id、case_id、traffic_seed、scenario_sha256、arm、training_seed、道路车道数、服从率及各项指标 |
| `<实验组>/details/*_cells.csv` | 每个cell的DL/GL/CAV行驶距离、车辆时间和速度 |
| `<实验组>/details/*_units.csv` | 每个路段的密度、临界密度和速度 |
| `pems_spatial/details/*_states.csv` | 每分钟、每个cell的密度、流量和速度 |
| `pems_spatial/details/*_trace.csv` | 检测位置密度与下游累计出口车辆数 |
| `pems_spatial/details/*_controls.csv` | 完整40个周期、4个受控路段的实际DL数；road_lanes是道路车道数 |
| `training/<组>/<算法或变体>_seed<种子>.csv` | steps、episode、controls、common_reward；逐回合记录 |

包内records/将同类空间记录合并为cells.csv、units.csv等，字段相同。
新训练的episodes.csv也可放在`<组>/<算法或变体>_seed<种子>/`，通过`--training`指定根目录。

图3算法比较与图4共享100例；algorithm_summary.csv保存其共同固定窗口评分及完整排空指标。另有algorithm_interior90.csv与ablation_interior90.csv专门报告0.1–0.9。图12日期表包括14天、七类星期。
