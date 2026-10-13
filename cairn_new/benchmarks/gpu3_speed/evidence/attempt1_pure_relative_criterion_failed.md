# GPU3 OLD (per-pair) vs NEW (packed) DSN inter-graph attention evidence

Finished 2026-10-12T11:15:33+0800; 213 s. GPU: NVIDIA A100-PCIE-40GB f9b84e89-f742-29a6-2257-ec4a4b242fe5 (nvidia-smi GPU3 GPU-f9b84e89-f742-29a6-2257-ec4a4b242fe5), torch 2.8.0+cu128 CUDA 12.8, PyG 2.7.0, driver 570.133.20, TF32 matmul False.

Relative discrepancy: per tensor ||new - old||_inf / ||old||_inf (max-norm relative); worst = maximum over tensors, batches and parameter sets.

**All numerical checks passed: False**

## Checks

- gpu3_verified: True
- initialization_identical: True
- structure_all_batches: True
- global_diagonal_edges_present_and_kept: True
- fresh_seed42_init.attention_forward: True
- fresh_seed42_init.attention_gradients: True
- fresh_seed42_init.attention_fp64_equivalence: True
- fresh_seed42_init.attention_edges_and_alpha: True
- fresh_seed42_init.cross_pair_isolation: True
- fresh_seed42_init.model_forward: True
- fresh_seed42_init.model_parameter_gradients: False
- fresh_seed42_init.model_input_gradients: True
- fresh_seed42_init.path_mask_and_grad_sets: True
- trained_production_best.attention_forward: True
- trained_production_best.attention_gradients: True
- trained_production_best.attention_fp64_equivalence: True
- trained_production_best.attention_edges_and_alpha: True
- trained_production_best.cross_pair_isolation: True
- trained_production_best.model_forward: True
- trained_production_best.model_parameter_gradients: False
- trained_production_best.model_input_gradients: True
- trained_production_best.path_mask_and_grad_sets: True

## Structure (all timing batches)

- batches 35, pairs nh<nt 2273, nh>nt 2059, nh=nt 148; local loops 90583
- global row==col non-loop edges kept: 10158; a default-loop GAT on the packed graph would wrongly add 101821 cross-pair loops

## Numerical worst discrepancies

| parameters | comparison | quantity | worst rel | tensor | worst abs | tensor |
|---|---|---|---|---|---|---|
| fresh_seed42_init | attention new32_vs_old32 | forward | 7.409e-07 | 2.1.out.t_rep | 1.550e-06 | 2.1.out.t_rep |
| fresh_seed42_init | attention new32_vs_old32 | gradients | 6.409e-06 | 1.1.param.inter.att_src | 1.831e-04 | 2.0.param.inter.lin_src.weight |
| fresh_seed42_init | attention old32_vs_old64 | forward | 7.128e-07 | 2.1.out.t_rep | 1.491e-06 | 2.1.out.t_rep |
| fresh_seed42_init | attention old32_vs_old64 | gradients | 3.519e-06 | 1.1.param.inter.att_src | 1.557e-04 | 2.1.param.inter.lin_src.weight |
| fresh_seed42_init | attention new32_vs_old64 | forward | 5.988e-07 | 2.1.out.t_rep | 1.253e-06 | 2.1.out.t_rep |
| fresh_seed42_init | attention new32_vs_old64 | gradients | 2.889e-06 | 1.1.param.inter.att_src | 9.476e-05 | 1.0.param.inter.att_dst |
| fresh_seed42_init | attention new64_vs_old64 | forward | 1.274e-15 | 2.1.out.t_rep | 2.665e-15 | 2.1.out.t_rep |
| fresh_seed42_init | attention new64_vs_old64 | gradients | 4.475e-15 | 0.0.param.inter.att_dst | 3.411e-13 | 2.1.param.inter.lin_src.weight |
| fresh_seed42_init | model NEW vs OLD | forward | 4.850e-07 | eval.logits | 3.278e-07 | logits |
| fresh_seed42_init | model NEW vs OLD | parameter | 4.429e+00 | cse.blocks.1.readout.gnn.lin_rel.bias | 2.586e-05 | path.linear.2.bias |
| fresh_seed42_init | model NEW vs OLD | input | 8.151e-07 | input.graph_a.edge_attr | 3.492e-10 | input.graph_b.x |
| fresh_seed42_init | model OLD repeat vs OLD | forward | 4.972e-07 | logits | 3.576e-07 | logits |
| fresh_seed42_init | model OLD repeat vs OLD | parameter | 7.500e+00 | cse.blocks.1.readout.gnn.lin_rel.bias | 2.587e-05 | path.linear.2.bias |
| fresh_seed42_init | model OLD repeat vs OLD | input | 8.567e-06 | input.graph_b.x | 5.724e-09 | input.graph_b.x |
| fresh_seed42_init | alpha per pair | max abs | | | 1.192e-07 | global-diag edges 1450, min alpha 5.974e-03 |
| trained_production_best | attention new32_vs_old32 | forward | 6.000e-07 | 0.1.out.h_rep | 2.146e-06 | 0.1.out.h_rep |
| trained_production_best | attention new32_vs_old32 | gradients | 1.843e-06 | 2.1.param.inter.att_dst | 1.678e-04 | 2.0.param.inter.lin_src.weight |
| trained_production_best | attention old32_vs_old64 | forward | 5.566e-07 | 1.0.out.h_rep | 1.930e-06 | 0.1.out.h_rep |
| trained_production_best | attention old32_vs_old64 | gradients | 2.715e-06 | 2.1.param.inter.att_dst | 1.708e-04 | 2.0.param.inter.lin_src.weight |
| trained_production_best | attention new32_vs_old64 | forward | 7.948e-07 | 2.1.out.t_rep | 2.380e-06 | 2.1.out.t_rep |
| trained_production_best | attention new32_vs_old64 | gradients | 1.524e-06 | 2.1.param.inter.att_dst | 1.382e-04 | 0.1.param.inter.att_src |
| trained_production_best | attention new64_vs_old64 | forward | 1.753e-15 | 0.0.out.h_rep | 4.441e-15 | 0.0.out.h_rep |
| trained_production_best | attention new64_vs_old64 | gradients | 3.606e-15 | 2.1.param.inter.att_dst | 3.411e-13 | 2.0.param.inter.att_src |
| trained_production_best | model NEW vs OLD | forward | 2.426e-07 | ce | 7.629e-06 | logits |
| trained_production_best | model NEW vs OLD | parameter | 9.000e+00 | cse.blocks.0.readout.gnn.lin_rel.bias | 6.444e-05 | path.linear.3.bias |
| trained_production_best | model NEW vs OLD | input | 1.964e-06 | input.graph_b.edge_attr | 2.049e-08 | input.graph_b.x |
| trained_production_best | model OLD repeat vs OLD | forward | 3.190e-07 | eval.logits | 1.144e-05 | eval.logits |
| trained_production_best | model OLD repeat vs OLD | parameter | 4.333e+00 | cse.blocks.0.readout.gnn.lin_rel.bias | 6.440e-05 | path.linear.3.bias |
| trained_production_best | model OLD repeat vs OLD | input | 2.185e-06 | input.graph_a.edge_attr | 1.583e-08 | input.graph_b.x |
| trained_production_best | alpha per pair | max abs | | | 5.960e-07 | global-diag edges 1450, min alpha 3.475e-05 |

## Timing (median of block medians, seconds)

GPU3 only; same 35 real batches and identical fresh seed42 parameters for OLD and NEW; 5 warmup + 30 measured steps per block; ABBA block order; CUDA synchronized around each timed region; transfer = pinned host batch to GPU (non_blocking, as training); fullstep = transfer + zero_grad + forward + loss + backward + AdamW step; breakdown blocks synchronize between phases, fullstep blocks only at step boundaries; eval = no_grad eval forward. Collation (CPU, DataLoader workers in training) is reported separately.

| measurement | OLD | NEW | OLD/NEW |
|---|---|---|---|
| breakdown.transfer | 0.0033 | 0.0009 | 3.56 |
| breakdown.forward | 0.6592 | 0.0379 | 17.39 |
| breakdown.backward | 0.4755 | 0.0366 | 13.00 |
| breakdown.optimizer | 0.0052 | 0.0028 | 1.87 |
| breakdown.fullstep | 1.1447 | 0.0782 | 14.64 |
| fullstep.fullstep | 1.1323 | 0.0828 | 13.67 |
| eval.transfer | 0.0033 | 0.0008 | 3.97 |
| eval.forward | 0.5809 | 0.0325 | 17.89 |
| eval.fullstep | 0.5842 | 0.0334 | 17.47 |

| block | mode | peak activation MiB | peak reserved MiB |
|---|---|---|---|
| old | breakdown | 1434 | 2656 |
| new | breakdown | 1428 | 3032 |
| new | fullstep | 1273 | 2876 |
| old | fullstep | 1278 | 2798 |
| old | eval | 121 | 1016 |
| new | eval | 121 | 1016 |
| new | eval | 121 | 1016 |
| old | eval | 121 | 1016 |

Collate (single process, CPU) mean seconds: OLD 0.0228, NEW 0.0394.

## Contention

- before timing: {"time": "2026-10-12T11:13:02+0800", "loadavg": [6.51123046875, 6.46435546875, 6.576171875], "cpu_count": 80, "this_process_affinity_cpus": 80, "gpu3": "3, GPU-f9b84e89-f742-29a6-2257-ec4a4b242fe5, 45 %, 2875 MiB, 40960 MiB", "gpu3_compute_apps": ["GPU-f9b84e89-f742-29a6-2257-ec4a4b242fe5, 2923319, 2866 MiB"], "original_cairn_dsn_train_processes": ["2560540 2560536  101    11:55:19 /data/interns/zhangpc/cairn_dev/.conda/cairn/bin/python -m cairn_dsn.train --config /data/interns/zhangpc/cairn_new/configs/path_single_drugout.yaml", "2560541 2560537  101    11:55:19 /data/interns/zhangpc/cairn_dev/.conda/cairn/bin/python -m cairn_dsn.train --config /data/interns/zhangpc/cairn_new/configs/path_double_drugout.yaml", "2560542 2560538  101    11:55:19 /data/interns/zhangpc/cairn_dev/.conda/cairn/bin/python -m cairn_dsn.train --config /data/interns/zhangpc/cairn_new/configs/no_path_single_drugout.yaml", "2560543 2560539  101    11:55:19 /data/interns/zhangpc/cairn_dev/.conda/cairn/bin/python -m cairn_dsn.train --config /data/interns/zhangpc/cairn_new/configs/no_path_double_drugout.yaml", "2906862 2560542  2.6       19:35 /data/interns/zhangpc/cairn_dev/.conda/cairn/bin/python -m cairn_dsn.train --config /data/interns/zhangpc/cairn_new/configs/no_path_single_drugout.yaml", "2906864 2560542  2.8       19:35 /data/interns/zhangpc/cairn_dev/.conda/cairn/bin/python -m cairn_dsn.train --config /data/interns/zhangpc/cairn_new/configs/no_path_single_drugout.yaml", "2910016 2560543  2.4       16:15 /data/interns/zhangpc/cairn_dev/.conda/cairn/bin/python -m cairn_dsn.train --config /data/interns/zhangpc/cairn_new/configs/no_path_double_drugout.yaml", "2910018 2560543  2.3       16:15 /data/interns/zhangpc/cairn_dev/.conda/cairn/bin/python -m cairn_dsn.train --config /data/interns/zhangpc/cairn_new/configs/no_path_double_drugout.yaml", "2920091 2560541  3.3       05:08 /data/interns/zhangpc/cairn_dev/.conda/cairn/bin/python -m cairn_dsn.train --config /data/interns/zhangpc/cairn_new/configs/path_double_drugout.yaml", "2920095 2560541  3.1       05:08 /data/interns/zhangpc/cairn_dev/.conda/cairn/bin/python -m cairn_dsn.train --config /data/interns/zhangpc/cairn_new/configs/path_double_drugout.yaml", "2920109 2560540  2.9       05:08 /data/interns/zhangpc/cairn_dev/.conda/cairn/bin/python -m cairn_dsn.train --config /data/interns/zhangpc/cairn_new/configs/path_single_drugout.yaml", "2920111 2560540  2.8       05:08 /data/interns/zhangpc/cairn_dev/.conda/cairn/bin/python -m cairn_dsn.train --config /data/interns/zhangpc/cairn_new/configs/path_single_drugout.yaml"]}
- after timing: {"time": "2026-10-12T11:15:33+0800", "loadavg": [7.67431640625, 7.03564453125, 6.77978515625], "cpu_count": 80, "this_process_affinity_cpus": 80, "gpu3": "3, GPU-f9b84e89-f742-29a6-2257-ec4a4b242fe5, 29 %, 1535 MiB, 40960 MiB", "gpu3_compute_apps": ["GPU-f9b84e89-f742-29a6-2257-ec4a4b242fe5, 2923319, 1526 MiB"], "original_cairn_dsn_train_processes": ["2560540 2560536  101    11:57:51 /data/interns/zhangpc/cairn_dev/.conda/cairn/bin/python -m cairn_dsn.train --config /data/interns/zhangpc/cairn_new/configs/path_single_drugout.yaml", "2560541 2560537  101    11:57:51 /data/interns/zhangpc/cairn_dev/.conda/cairn/bin/python -m cairn_dsn.train --config /data/interns/zhangpc/cairn_new/configs/path_double_drugout.yaml", "2560542 2560538  101    11:57:51 /data/interns/zhangpc/cairn_dev/.conda/cairn/bin/python -m cairn_dsn.train --config /data/interns/zhangpc/cairn_new/configs/no_path_single_drugout.yaml", "2560543 2560539  101    11:57:51 /data/interns/zhangpc/cairn_dev/.conda/cairn/bin/python -m cairn_dsn.train --config /data/interns/zhangpc/cairn_new/configs/no_path_double_drugout.yaml", "2906862 2560542  2.5       22:07 /data/interns/zhangpc/cairn_dev/.conda/cairn/bin/python -m cairn_dsn.train --config /data/interns/zhangpc/cairn_new/configs/no_path_single_drugout.yaml", "2906864 2560542  2.7       22:07 /data/interns/zhangpc/cairn_dev/.conda/cairn/bin/python -m cairn_dsn.train --config /data/interns/zhangpc/cairn_new/configs/no_path_single_drugout.yaml", "2910016 2560543  2.4       18:47 /data/interns/zhangpc/cairn_dev/.conda/cairn/bin/python -m cairn_dsn.train --config /data/interns/zhangpc/cairn_new/configs/no_path_double_drugout.yaml", "2910018 2560543  2.3       18:47 /data/interns/zhangpc/cairn_dev/.conda/cairn/bin/python -m cairn_dsn.train --config /data/interns/zhangpc/cairn_new/configs/no_path_double_drugout.yaml", "2920091 2560541  3.2       07:40 /data/interns/zhangpc/cairn_dev/.conda/cairn/bin/python -m cairn_dsn.train --config /data/interns/zhangpc/cairn_new/configs/path_double_drugout.yaml", "2920095 2560541  3.1       07:40 /data/interns/zhangpc/cairn_dev/.conda/cairn/bin/python -m cairn_dsn.train --config /data/interns/zhangpc/cairn_new/configs/path_double_drugout.yaml", "2920109 2560540  2.9       07:40 /data/interns/zhangpc/cairn_dev/.conda/cairn/bin/python -m cairn_dsn.train --config /data/interns/zhangpc/cairn_new/configs/path_single_drugout.yaml", "2920111 2560540  2.8       07:40 /data/interns/zhangpc/cairn_dev/.conda/cairn/bin/python -m cairn_dsn.train --config /data/interns/zhangpc/cairn_new/configs/path_single_drugout.yaml"]}
