# HREM vs HRM Performance Comparison (hrem_optimization-smoke)

## Summary
**Best HREM**: Final loss = N/A
**HRM**: Final loss = N/A

## Best HREM Parameters
| Parameter | Value |
|---|---|
| m_loc | 128 |
| d_mem | 108 |
| top_k | 4 |
| H_layers | 2 |
| L_layers | 2 |
| H_cycles | 1 |
| L_cycles | 1 |
| hidden_size | 16 |

## Final Metrics Comparison
| Metric | HRM | Best HREM |
|---|---|
| all/accuracy | 0.1300 | 0.0900 |
| all/exact_accuracy | 0.0000 | 0.0000 |
| all/lm_loss | 2.4838 | 2.5342 |
| all/q_halt_accuracy | 1.0000 | 1.0000 |
| all/q_halt_loss | 0.0067 | 0.0067 |
| all/steps | 16.0000 | 16.0000 |
| step | 10.0000 | 10.0000 |