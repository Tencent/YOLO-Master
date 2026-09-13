| 实验目录（均在 `A2OR/runs/`） | CSV进度 / 当前预算 | batch / workers | 最后 mAP50-95 | 面积评估覆盖 |
|---|---:|---:|---:|---|
| `baseline_fixedk10_vd10pct_s42_20e_b4_w1` | 50 / 50 | 4 / 1 | 2.444 | checkpoint_area_metrics.json: 20轮，末轮20；checkpoint_area_metrics_50e.json: 50轮，末轮50 |
| `dtk_lambda0p55_vd10pct_s42_20e_b4_w1` | 21 / 50 | 4 / 1 | 1.412 | checkpoint_area_metrics.json: 20轮，末轮20 |
| `dtk_lambda0p5_vd10pct_s42_20e_b4_w1` | 50 / 50 | 4 / 1 | 2.586 | checkpoint_area_metrics.json: 20轮，末轮20；checkpoint_area_metrics_50e.json: 50轮，末轮50 |
| `dtk_lambda0p65_vd10pct_s42_20e_b4_w1` | 20 / 20 | 4 / 1 | 1.388 | checkpoint_area_metrics.json: 20轮，末轮20 |
| `dtk_lambda0p6_vd10pct_s42_20e_b4_w1` | 20 / 20 | 4 / 1 | 1.374 | checkpoint_area_metrics.json: 20轮，末轮20 |
| `dtk_lambda0p75_vd10pct_s42_20e_b4_w1` | 20 / 20 | 4 / 1 | 1.504 | checkpoint_area_metrics.json: 20轮，末轮20 |
| `dtk_lambda0p7_vd10pct_s42_20e_b4_w1` | 20 / 20 | 4 / 1 | 1.336 | checkpoint_area_metrics.json: 20轮，末轮20 |
| `dtk_lambda0p80_vd10pct_s42_20e_w1` | 20 / 20 | 4 / 1 | 1.401 | 无面积 JSON |
| `full_baseline_fixedk10_vd100pct_s0_10e_b4_w3` | 10 / 10 | 4 / 3 | 5.164 | checkpoint_area_metrics.json: 10轮，末轮10 |
| `full_dtk_bounded_l0p80_kmin3_kmax10_vd100pct_s0_10e_b4_w3` | 10 / 10 | 4 / 3 | 5.292 | checkpoint_area_metrics.json: 10轮，末轮10 |
| `full_dtk_l080_kmin3_kmax10-2` | 30 / 30 | 16 / 8 | 10.860 | 无面积 JSON |
| `full_dtk_lambda0p5_vd100pct_s0_20e_b4_w4` | 15 / 20 | 4 / 4 | 7.568 | checkpoint_area_metrics_1to15.json: 15轮，末轮15 |
| `full_dtk_lambda0p5_vd100pct_s0_20e_b4_w6` | 0 / 20 | 4 / 6 | — | 无面积 JSON |