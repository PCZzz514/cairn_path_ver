# GPU3 packed-attention five-epoch benchmark: complete

Exit code 0; epochs recorded [1, 2, 3, 4, 5] of cap 5 (KL schedule from epochs 100). Summarized 2026-10-12T11:45:38+0800.

| epoch | train s | val s | train+val s | val macroF1(85) | val acc | train CE | train loss | historical/this |
|---|---|---|---|---|---|---|---|---|
| 1 | 101.4 | 8.7 | 110.1 | 0.2184 | 0.6638 | 1.5200 | 1.5691 | 13.61-13.93x |
| 2 | 100.7 | 9.3 | 110.0 | 0.3434 | 0.7012 | 0.6512 | 0.6960 | 13.62-13.94x |
| 3 | 100.5 | 9.0 | 109.6 | 0.4412 | 0.7256 | 0.4440 | 0.4854 | 13.67-14.00x |
| 4 | 104.8 | 13.9 | 118.6 | 0.5019 | 0.7310 | 0.3402 | 0.3781 | 12.63-12.93x |
| 5 | 112.9 | 9.2 | 122.1 | 0.5242 | 0.7324 | 0.2778 | 0.3127 | 12.27-12.56x |

Mean train+validation epoch 114.1 s; historical original 1498-1534 s -> ratio 13.13-13.45x (different GPU/time; same-GPU step evidence below).

Same-GPU3 OLD vs NEW step medians (seconds):

- breakdown.transfer: OLD 0.0033, NEW 0.0009, OLD/NEW 3.71
- breakdown.forward: OLD 0.6458, NEW 0.0379, OLD/NEW 17.03
- breakdown.backward: OLD 0.4794, NEW 0.0370, OLD/NEW 12.97
- breakdown.optimizer: OLD 0.0051, NEW 0.0028, OLD/NEW 1.81
- breakdown.fullstep: OLD 1.1342, NEW 0.0786, OLD/NEW 14.42
- fullstep.fullstep: OLD 1.1456, NEW 0.0784, OLD/NEW 14.62
- eval.transfer: OLD 0.0032, NEW 0.0017, OLD/NEW 1.93
- eval.forward: OLD 0.5824, NEW 0.0809, OLD/NEW 7.19
- eval.fullstep: OLD 0.5858, NEW 0.0828, OLD/NEW 7.07
