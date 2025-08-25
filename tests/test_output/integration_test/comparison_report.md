# Model Comparison Report: integration_test

## Summary
**HRM**: Final loss = 2.6685
**HREM**: Final loss = 2.7128

## HREM Parameters
| Parameter | Value |
|---|---|
| m_loc | 128 |
| d_mem | 128 |
| top_k | 4 |
| H_layers | 2 |
| L_layers | 2 |
| H_cycles | 2 |
| L_cycles | 8 |
| hidden_size | 256 |

## Final Metrics Comparison
| Metric | HRM | HREM |
|---|---|---|
| all/accuracy | 0.0800 | 0.0300 |
| all/exact_accuracy | 0.0000 | 0.0000 |
| all/lm_loss | 2.6685 | 2.7128 |
| all/q_halt_accuracy | 1.0000 | 1.0000 |
| all/q_halt_loss | 187.2579 | 187.2579 |
| all/steps | 16.0000 | 12.0000 |
| step | 0.0000 | 0.0000 |