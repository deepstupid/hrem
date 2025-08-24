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
|---|---|---|
| all | {'accuracy': 0.1300000101327896, 'exact_accuracy': 0.0, 'lm_loss': 2.5769829750061035, 'q_halt_accuracy': 1.0, 'q_halt_loss': 0.006715297698974609, 'steps': 16.0} | {'accuracy': 0.10999999940395355, 'exact_accuracy': 0.0, 'lm_loss': 2.534660577774048, 'q_halt_accuracy': 1.0, 'q_halt_loss': 0.006715297698974609, 'steps': 16.0} |