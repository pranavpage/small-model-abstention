| Model | Dev balanced | Dev halluc. | Dev over-refusal | Dev accuracy | Dev P(IDK) AUC | AUC vs SFT [95% CI] | Test balanced | Test halluc. | Test over-refusal | Test accuracy |
|---|---|---|---|---|---|---|---|---|---|---|
| Baseline 1.7B | 59.3 | 54.7 | 8.0 | 73.3 | 0.797 | -0.140 [-0.187, -0.096] | 60.2 | 50.8 | 10.8 | 71.2 |
| SFT | 83.7 | 12.7 | 12.0 | 80.0 | 0.937 | — | 84.8 | 11.6 | 14.0 | 81.2 |
| DPO v1 | 82.7 | 8.7 | 19.3 | 74.0 | 0.939 | +0.002 [-0.001, +0.004] | 83.2 | 5.6 | 24.0 | 72.0 |
| DPO v2 | 84.3 | 12.0 | 10.7 | 80.7 | 0.940 | +0.002 [-0.002, +0.007] | 82.2 | 12.4 | 17.2 | 76.8 |
| G-default best (step 200) | 84.3 | 8.0 | 17.3 | 76.7 | 0.941 | +0.004 [-0.000, +0.008] | not run | not run | not run | not run |
| G-equal best (step 200) | 85.0 | 11.3 | 12.7 | 81.3 | 0.941 | +0.004 [-0.006, +0.014] | not run | not run | not run | not run |
| Twin DPO input-side (lr5e-06 e2) | 85.0 | 11.3 | 12.0 | 81.3 | 0.940 | +0.002 [-0.005, +0.010] | not run | not run | not run | not run |
| Twin DPO output-side (lr5e-06 e2) | 85.7 | 11.3 | 10.0 | 82.7 | 0.942 | +0.004 [-0.002, +0.011] | 83.6 | 12.4 | 14.4 | 79.6 |
| 8B ceiling | 76.7 | 36.7 | 2.0 | 90.0 | 0.892 | -0.046 [-0.082, -0.010] | 72.6 | 44.0 | 2.0 | 89.2 |
