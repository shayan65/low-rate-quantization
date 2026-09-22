# Qwen task-aware pairing screen

BF16 evaluation NLL: 3.421625. Calibration and evaluation windows are disjoint. Each method changes one tensor independently.

| Layer | Method | Selected candidate | Calibration ΔNLL | Evaluation ΔNLL |
|---:|---|---|---:|---:|
| 0 | task_real4 | 0.95 | +0.003935 | +0.008660 |
| 0 | task_polar3_5 | reverse_half | -0.001638 | +0.004083 |
| 8 | task_real4 | 0.85 | -0.004529 | +0.001316 |
| 8 | task_polar3_5 | reverse_half | -0.005207 | -0.000176 |
| 16 | task_real4 | 0.85 | -0.001939 | +0.004793 |
| 16 | task_polar3_5 | split_half | -0.000756 | -0.000703 |

{"complete": true, "jobs": 6, "seconds": 21.895133018959314, "polar_wins": 3, "layers": 3, "promote": true, "action": "No cumulative or training run launched."}
