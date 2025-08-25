# HREM vs HRM Performance Comparison (hrem_optimization-smoke)

## Summary
**Best HREM**: Final loss = 2.6159
**HRM**: Final loss = 2.6356

## Best HREM Parameters
| Parameter | Value |
|---|---|
| m_loc | 120 |
| d_mem | 118 |
| top_k | 4 |
| H_layers | 2 |
| L_layers | 1 |
| H_cycles | 2 |
| L_cycles | 2 |
| hidden_size | 32 |

## Final Metrics Comparison
| Metric | HRM | Best HREM |
|---|---|
| all/accuracy | 0.0400 | 0.0900 |
| all/exact_accuracy | 0.0000 | 0.0000 |
| all/lm_loss | 2.6356 | 2.6159 |
| all/q_halt_accuracy | 1.0000 | 1.0000 |
| all/q_halt_loss | 0.0067 | 0.0067 |
| all/steps | 16.0000 | 16.0000 |
| step | 10.0000 | 10.0000 |