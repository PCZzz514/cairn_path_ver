# GPU3 OLD (per-pair) vs NEW (packed) DSN inter-graph attention evidence

Finished 2026-10-12T11:34:45+0800; 199 s. GPU: NVIDIA A100-PCIE-40GB f9b84e89-f742-29a6-2257-ec4a4b242fe5 (nvidia-smi GPU3 GPU-f9b84e89-f742-29a6-2257-ec4a4b242fe5), torch 2.8.0+cu128 CUDA 12.8, PyG 2.7.0, driver 570.133.20, TF32 matmul False.

Relative discrepancy: per tensor ||new - old||_inf / ||old||_inf (max-norm relative); worst = maximum over tensors, batches and parameter sets.

**All numerical checks passed: True**

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
- fresh_seed42_init.model_parameter_gradients: True
- fresh_seed42_init.model_input_gradients: True
- fresh_seed42_init.path_mask_and_grad_sets: True
- trained_production_best.attention_forward: True
- trained_production_best.attention_gradients: True
- trained_production_best.attention_fp64_equivalence: True
- trained_production_best.attention_edges_and_alpha: True
- trained_production_best.cross_pair_isolation: True
- trained_production_best.model_forward: True
- trained_production_best.model_parameter_gradients: True
- trained_production_best.model_input_gradients: True
- trained_production_best.path_mask_and_grad_sets: True

## Structure (all timing batches)

- batches 35, pairs nh<nt 2273, nh>nt 2059, nh=nt 148; local loops 90583
- global row==col non-loop edges kept: 10158; a default-loop GAT on the packed graph would wrongly add 101821 cross-pair loops

## Numerical worst discrepancies

| parameters | comparison | quantity | worst rel | tensor | worst abs | tensor |
|---|---|---|---|---|---|---|
| fresh_seed42_init | attention new32_vs_old32 | forward | 8.164e-07 | batch0.block1.out.h_rep | 1.669e-06 | batch0.block1.out.h_rep |
| fresh_seed42_init | attention new32_vs_old32 | gradients | 3.752e-06 | batch1.block1.param.inter.lin_dst.weight | 2.441e-04 | batch2.block0.param.inter.lin_src.weight |
| fresh_seed42_init | attention old32_vs_old64 | forward | 6.071e-07 | batch2.block1.out.t_rep | 1.270e-06 | batch2.block1.out.t_rep |
| fresh_seed42_init | attention old32_vs_old64 | gradients | 4.432e-06 | batch1.block1.param.inter.att_src | 1.969e-04 | batch2.block0.param.inter.lin_src.weight |
| fresh_seed42_init | attention new32_vs_old64 | forward | 7.776e-07 | batch2.block1.out.t_rep | 1.627e-06 | batch2.block1.out.t_rep |
| fresh_seed42_init | attention new32_vs_old64 | gradients | 2.713e-06 | batch2.block1.param.inter.att_dst | 8.444e-05 | batch0.block0.param.inter.lin_src.weight |
| fresh_seed42_init | attention new64_vs_old64 | forward | 1.224e-15 | batch1.block1.out.h_rep | 2.442e-15 | batch0.block1.out.h_rep |
| fresh_seed42_init | attention new64_vs_old64 | gradients | 3.891e-15 | batch0.block0.param.inter.att_dst | 5.116e-13 | batch2.block0.param.inter.lin_src.weight |
| fresh_seed42_init | model NEW vs OLD | forward | 5.419e-07 | batch0.logits | 4.768e-07 | batch2.loss |
| fresh_seed42_init | model NEW vs OLD | parameter | 1.300e+01 | batch0.cse.blocks.0.readout.gnn.lin_rel.bias | 5.960e-08 | batch0.classifier.4.weight |
| fresh_seed42_init | model NEW vs OLD | input | 8.920e-07 | batch2.input.graph_a.edge_attr | 4.802e-10 | batch1.input.graph_b.x |
| fresh_seed42_init | model OLD repeat vs OLD | forward | 5.543e-07 | batch1.eval.logits | 4.768e-07 | batch2.loss |
| fresh_seed42_init | model OLD repeat vs OLD | parameter | 2.100e+01 | batch0.cse.blocks.0.readout.gnn.lin_rel.bias | 5.960e-08 | batch0.classifier.4.weight |
| fresh_seed42_init | model OLD repeat vs OLD | input | 7.703e-07 | batch2.input.graph_a.edge_attr | 6.403e-10 | batch2.input.graph_a.x |
| fresh_seed42_init | model NEW vs OLD | parameter grads above atol (396 tensor-batches) | 1.364e-06 | batch1.cse.blocks.0.interAtt.inter.att_dst | 5.960e-08 | batch0.classifier.4.weight |
| fresh_seed42_init | gradient gate batch 0 | passed True | | atol 3.609e-08 | G 3.609e-01 | failures 0 |
| fresh_seed42_init | below-atol grad batch 0 | cse.blocks.0.feature_conv.lin_key.bias | new-old abs 1.273e-11 | ref 1.273e-11 | OLD repeat abs 1.296e-11 | |
| fresh_seed42_init | below-atol grad batch 0 | cse.blocks.0.feature_conv2.lin_key.bias | new-old abs 1.364e-11 | ref 1.819e-11 | OLD repeat abs 1.330e-11 | |
| fresh_seed42_init | below-atol grad batch 0 | cse.blocks.0.readout.gnn.lin_rel.bias | new-old abs 4.729e-11 | ref 3.638e-12 | OLD repeat abs 7.640e-11 | |
| fresh_seed42_init | below-atol grad batch 0 | cse.blocks.1.feature_conv.lin_key.bias | new-old abs 5.286e-12 | ref 7.617e-12 | OLD repeat abs 4.547e-12 | |
| fresh_seed42_init | below-atol grad batch 0 | cse.blocks.1.feature_conv2.lin_key.bias | new-old abs 4.889e-12 | ref 5.912e-12 | OLD repeat abs 3.752e-12 | |
| fresh_seed42_init | below-atol grad batch 0 | cse.blocks.1.readout.gnn.lin_rel.bias | new-old abs 5.821e-11 | ref 1.492e-10 | OLD repeat abs 1.091e-10 | |
| fresh_seed42_init | gradient gate batch 1 | passed True | | atol 4.365e-08 | G 4.365e-01 | failures 0 |
| fresh_seed42_init | below-atol grad batch 1 | cse.blocks.0.feature_conv.lin_key.bias | new-old abs 2.387e-11 | ref 2.046e-11 | OLD repeat abs 1.364e-11 | |
| fresh_seed42_init | below-atol grad batch 1 | cse.blocks.0.feature_conv2.lin_key.bias | new-old abs 2.569e-11 | ref 1.228e-11 | OLD repeat abs 1.592e-11 | |
| fresh_seed42_init | below-atol grad batch 1 | cse.blocks.0.readout.gnn.lin_rel.bias | new-old abs 1.455e-10 | ref 5.821e-11 | OLD repeat abs 0.000e+00 | |
| fresh_seed42_init | below-atol grad batch 1 | cse.blocks.1.feature_conv.lin_key.bias | new-old abs 8.981e-12 | ref 1.069e-11 | OLD repeat abs 1.387e-11 | |
| fresh_seed42_init | below-atol grad batch 1 | cse.blocks.1.feature_conv2.lin_key.bias | new-old abs 5.684e-12 | ref 4.462e-12 | OLD repeat abs 5.741e-12 | |
| fresh_seed42_init | below-atol grad batch 1 | cse.blocks.1.readout.gnn.lin_rel.bias | new-old abs 6.548e-11 | ref 1.091e-10 | OLD repeat abs 1.091e-10 | |
| fresh_seed42_init | gradient gate batch 2 | passed True | | atol 4.670e-08 | G 4.670e-01 | failures 0 |
| fresh_seed42_init | below-atol grad batch 2 | cse.blocks.0.feature_conv.lin_key.bias | new-old abs 1.637e-11 | ref 1.637e-11 | OLD repeat abs 1.637e-11 | |
| fresh_seed42_init | below-atol grad batch 2 | cse.blocks.0.feature_conv2.lin_key.bias | new-old abs 2.092e-11 | ref 3.820e-11 | OLD repeat abs 3.047e-11 | |
| fresh_seed42_init | below-atol grad batch 2 | cse.blocks.0.readout.gnn.lin_rel.bias | new-old abs 4.366e-11 | ref 5.821e-11 | OLD repeat abs 6.548e-11 | |
| fresh_seed42_init | below-atol grad batch 2 | cse.blocks.1.feature_conv.lin_key.bias | new-old abs 9.095e-12 | ref 8.185e-12 | OLD repeat abs 1.091e-11 | |
| fresh_seed42_init | below-atol grad batch 2 | cse.blocks.1.feature_conv2.lin_key.bias | new-old abs 4.320e-12 | ref 4.547e-12 | OLD repeat abs 4.889e-12 | |
| fresh_seed42_init | below-atol grad batch 2 | cse.blocks.1.readout.gnn.lin_rel.bias | new-old abs 1.382e-10 | ref 1.455e-11 | OLD repeat abs 1.019e-10 | |
| fresh_seed42_init | alpha per pair | max abs | | | 1.192e-07 | global-diag edges 1450, min alpha 5.974e-03 |
| trained_production_best | attention new32_vs_old32 | forward | 6.666e-07 | batch0.block1.out.h_rep | 2.384e-06 | batch0.block1.out.h_rep |
| trained_production_best | attention new32_vs_old32 | gradients | 1.936e-06 | batch2.block1.param.inter.att_dst | 2.136e-04 | batch2.block0.param.inter.att_src |
| trained_production_best | attention old32_vs_old64 | forward | 6.635e-07 | batch0.block1.out.h_rep | 2.373e-06 | batch0.block1.out.h_rep |
| trained_production_best | attention old32_vs_old64 | gradients | 2.011e-06 | batch2.block1.param.inter.att_dst | 2.697e-04 | batch2.block0.param.inter.att_src |
| trained_production_best | attention new32_vs_old64 | forward | 9.255e-07 | batch2.block1.out.t_rep | 2.771e-06 | batch2.block1.out.t_rep |
| trained_production_best | attention new32_vs_old64 | gradients | 1.491e-06 | batch1.block1.param.inter.att_dst | 1.077e-04 | batch1.block1.param.inter.att_dst |
| trained_production_best | attention new64_vs_old64 | forward | 1.577e-15 | batch0.block0.out.h_rep | 3.997e-15 | batch0.block0.out.h_rep |
| trained_production_best | attention new64_vs_old64 | gradients | 4.123e-15 | batch2.block0.param.inter.lin_dst.weight | 4.832e-13 | batch1.block0.param.inter.lin_src.weight |
| trained_production_best | model NEW vs OLD | forward | 7.277e-07 | batch2.ce | 9.537e-06 | batch2.logits |
| trained_production_best | model NEW vs OLD | parameter | 5.250e+00 | batch0.cse.blocks.1.readout.gnn.lin_rel.bias | 6.434e-05 | batch1.path.linear.3.bias |
| trained_production_best | model NEW vs OLD | input | 1.874e-06 | batch0.input.graph_b.edge_attr | 2.980e-08 | batch2.input.graph_b.x |
| trained_production_best | model OLD repeat vs OLD | forward | 3.440e-07 | batch0.ce | 1.144e-05 | batch2.eval.logits |
| trained_production_best | model OLD repeat vs OLD | parameter | 2.143e+00 | batch2.cse.blocks.0.feature_conv.lin_key.bias | 3.874e-07 | batch0.path.linear.3.bias |
| trained_production_best | model OLD repeat vs OLD | input | 2.400e-06 | batch0.input.graph_a.x | 2.235e-08 | batch2.input.graph_b.x |
| trained_production_best | model NEW vs OLD | parameter grads above atol (396 tensor-batches) | 2.270e-04 | batch1.path.linear.3.bias | 6.434e-05 | batch1.path.linear.3.bias |
| trained_production_best | gradient gate batch 0 | passed True | | atol 2.306e-08 | G 2.306e-01 | failures 0 |
| trained_production_best | below-atol grad batch 0 | cse.blocks.0.feature_conv.lin_key.bias | new-old abs 1.637e-10 | ref 3.783e-10 | OLD repeat abs 3.693e-10 | |
| trained_production_best | below-atol grad batch 0 | cse.blocks.0.feature_conv2.lin_key.bias | new-old abs 4.220e-10 | ref 3.929e-10 | OLD repeat abs 2.965e-10 | |
| trained_production_best | below-atol grad batch 0 | cse.blocks.0.readout.gnn.lin_rel.bias | new-old abs 1.979e-09 | ref 1.630e-09 | OLD repeat abs 1.164e-09 | |
| trained_production_best | below-atol grad batch 0 | cse.blocks.1.feature_conv.lin_key.bias | new-old abs 1.965e-10 | ref 1.746e-10 | OLD repeat abs 2.001e-10 | |
| trained_production_best | below-atol grad batch 0 | cse.blocks.1.feature_conv2.lin_key.bias | new-old abs 1.237e-10 | ref 1.983e-10 | OLD repeat abs 2.401e-10 | |
| trained_production_best | below-atol grad batch 0 | cse.blocks.1.readout.gnn.lin_rel.bias | new-old abs 3.056e-10 | ref 5.821e-11 | OLD repeat abs 1.164e-10 | |
| trained_production_best | gradient gate batch 1 | passed True | | atol 2.835e-08 | G 2.835e-01 | failures 0 |
| trained_production_best | below-atol grad batch 1 | cse.blocks.0.feature_conv.lin_key.bias | new-old abs 3.056e-10 | ref 3.492e-10 | OLD repeat abs 2.037e-10 | |
| trained_production_best | below-atol grad batch 1 | cse.blocks.0.feature_conv2.lin_key.bias | new-old abs 2.619e-10 | ref 2.456e-10 | OLD repeat abs 2.619e-10 | |
| trained_production_best | below-atol grad batch 1 | cse.blocks.0.readout.gnn.lin_rel.bias | new-old abs 5.093e-10 | ref 2.285e-09 | OLD repeat abs 4.948e-10 | |
| trained_production_best | below-atol grad batch 1 | cse.blocks.1.feature_conv.lin_key.bias | new-old abs 2.983e-10 | ref 2.365e-10 | OLD repeat abs 2.947e-10 | |
| trained_production_best | below-atol grad batch 1 | cse.blocks.1.feature_conv2.lin_key.bias | new-old abs 6.203e-10 | ref 2.019e-10 | OLD repeat abs 2.965e-10 | |
| trained_production_best | below-atol grad batch 1 | cse.blocks.1.readout.gnn.lin_rel.bias | new-old abs 8.731e-11 | ref 1.164e-10 | OLD repeat abs 7.276e-11 | |
| trained_production_best | gradient gate batch 2 | passed True | | atol 2.121e-08 | G 2.121e-01 | failures 0 |
| trained_production_best | below-atol grad batch 2 | cse.blocks.0.feature_conv.lin_key.bias | new-old abs 2.328e-10 | ref 1.528e-10 | OLD repeat abs 3.274e-10 | |
| trained_production_best | below-atol grad batch 2 | cse.blocks.0.feature_conv2.lin_key.bias | new-old abs 2.819e-10 | ref 3.783e-10 | OLD repeat abs 3.201e-10 | |
| trained_production_best | below-atol grad batch 2 | cse.blocks.0.readout.gnn.lin_rel.bias | new-old abs 1.397e-09 | ref 1.281e-09 | OLD repeat abs 1.455e-09 | |
| trained_production_best | below-atol grad batch 2 | cse.blocks.1.feature_conv.lin_key.bias | new-old abs 3.238e-10 | ref 3.020e-10 | OLD repeat abs 2.219e-10 | |
| trained_production_best | below-atol grad batch 2 | cse.blocks.1.feature_conv2.lin_key.bias | new-old abs 2.619e-10 | ref 1.601e-10 | OLD repeat abs 2.838e-10 | |
| trained_production_best | below-atol grad batch 2 | cse.blocks.1.readout.gnn.lin_rel.bias | new-old abs 3.420e-10 | ref 9.459e-11 | OLD repeat abs 2.910e-11 | |
| trained_production_best | alpha per pair | max abs | | | 5.960e-07 | global-diag edges 1450, min alpha 3.475e-05 |

## Timing (median of block medians, seconds)

GPU3 only; same 35 real batches in the same row order; EVERY block restarts from the identical fresh seed42 parameters, a fresh AdamW and torch.manual_seed(42); 5 warmup + 30 measured steps per block; ABBA block order; CUDA synchronized around each timed region; transfer = pinned host batch to GPU (non_blocking, as training); fullstep = transfer + zero_grad + forward + loss + backward + AdamW step; breakdown blocks synchronize between phases, fullstep blocks only at step boundaries; eval = no_grad eval forward. Collation (CPU, DataLoader workers in training) is reported separately.

| measurement | OLD | NEW | OLD/NEW |
|---|---|---|---|
| breakdown.transfer | 0.0033 | 0.0009 | 3.71 |
| breakdown.forward | 0.6458 | 0.0379 | 17.03 |
| breakdown.backward | 0.4794 | 0.0370 | 12.97 |
| breakdown.optimizer | 0.0051 | 0.0028 | 1.81 |
| breakdown.fullstep | 1.1342 | 0.0786 | 14.42 |
| fullstep.fullstep | 1.1456 | 0.0784 | 14.62 |
| eval.transfer | 0.0032 | 0.0017 | 1.93 |
| eval.forward | 0.5824 | 0.0809 | 7.19 |
| eval.fullstep | 0.5858 | 0.0828 | 7.07 |

| block (ABBA order) | mode | fullstep median | min | p90 | max | peak activation MiB | peak reserved MiB |
|---|---|---|---|---|---|---|---|
| old | breakdown | 1.1342 | 1.1257 | 1.3881 | 4.1252 | 1434 | 2656 |
| new | breakdown | 0.0786 | 0.0766 | 0.0800 | 0.0815 | 1427 | 2670 |
| new | fullstep | 0.0784 | 0.0759 | 0.1073 | 0.1097 | 1427 | 2670 |
| old | fullstep | 1.1456 | 1.1335 | 1.4098 | 4.0594 | 1434 | 2656 |
| old | eval | 0.5860 | 0.5813 | 0.5915 | 0.6133 | 121 | 640 |
| new | eval | 0.0355 | 0.0333 | 0.0493 | 0.0928 | 122 | 640 |
| new | eval | 0.1301 | 0.0660 | 0.1602 | 0.1751 | 122 | 640 |
| old | eval | 0.5855 | 0.5800 | 0.5874 | 0.5897 | 121 | 640 |

Collate (single process, CPU) mean seconds: OLD 0.0227, NEW 0.0382.

## Contention

- before timing: {"time": "2026-10-12T11:32:18+0800", "loadavg": [6.7275390625, 6.69287109375, 6.74658203125], "cpu_count": 80, "this_process_affinity_cpus": 80, "gpu3": "3, GPU-f9b84e89-f742-29a6-2257-ec4a4b242fe5, 30 %, 2875 MiB, 40960 MiB", "gpu3_compute_apps": ["GPU-f9b84e89-f742-29a6-2257-ec4a4b242fe5, 2938561, 2866 MiB"], "original_cairn_dsn_train_processes": ["2560540 2560536  101    12:14:35 /data/interns/zhangpc/cairn_dev/.conda/cairn/bin/python -m cairn_dsn.train --config /data/interns/zhangpc/cairn_new/configs/path_single_drugout.yaml", "2560541 2560537  101    12:14:35 /data/interns/zhangpc/cairn_dev/.conda/cairn/bin/python -m cairn_dsn.train --config /data/interns/zhangpc/cairn_new/configs/path_double_drugout.yaml", "2560542 2560538  101    12:14:35 /data/interns/zhangpc/cairn_dev/.conda/cairn/bin/python -m cairn_dsn.train --config /data/interns/zhangpc/cairn_new/configs/no_path_single_drugout.yaml", "2560543 2560539  101    12:14:35 /data/interns/zhangpc/cairn_dev/.conda/cairn/bin/python -m cairn_dsn.train --config /data/interns/zhangpc/cairn_new/configs/no_path_double_drugout.yaml", "2928827 2560542  2.3       12:58 /data/interns/zhangpc/cairn_dev/.conda/cairn/bin/python -m cairn_dsn.train --config /data/interns/zhangpc/cairn_new/configs/no_path_single_drugout.yaml", "2928829 2560542  2.2       12:58 /data/interns/zhangpc/cairn_dev/.conda/cairn/bin/python -m cairn_dsn.train --config /data/interns/zhangpc/cairn_new/configs/no_path_single_drugout.yaml", "2931091 2560543  2.2       10:36 /data/interns/zhangpc/cairn_dev/.conda/cairn/bin/python -m cairn_dsn.train --config /data/interns/zhangpc/cairn_new/configs/no_path_double_drugout.yaml", "2931093 2560543  2.2       10:35 /data/interns/zhangpc/cairn_dev/.conda/cairn/bin/python -m cairn_dsn.train --config /data/interns/zhangpc/cairn_new/configs/no_path_double_drugout.yaml", "2938877 2560541  3.6       00:41 /data/interns/zhangpc/cairn_dev/.conda/cairn/bin/python -m cairn_dsn.train --config /data/interns/zhangpc/cairn_new/configs/path_double_drugout.yaml", "2938879 2560541  3.5       00:41 /data/interns/zhangpc/cairn_dev/.conda/cairn/bin/python -m cairn_dsn.train --config /data/interns/zhangpc/cairn_new/configs/path_double_drugout.yaml", "2939320 2560540  9.0       00:08 /data/interns/zhangpc/cairn_dev/.conda/cairn/bin/python -m cairn_dsn.train --config /data/interns/zhangpc/cairn_new/configs/path_single_drugout.yaml", "2939322 2560540  8.5       00:08 /data/interns/zhangpc/cairn_dev/.conda/cairn/bin/python -m cairn_dsn.train --config /data/interns/zhangpc/cairn_new/configs/path_single_drugout.yaml"]}
- after timing: {"time": "2026-10-12T11:34:45+0800", "loadavg": [7.02490234375, 6.7978515625, 6.775390625], "cpu_count": 80, "this_process_affinity_cpus": 80, "gpu3": "3, GPU-f9b84e89-f742-29a6-2257-ec4a4b242fe5, 28 %, 1159 MiB, 40960 MiB", "gpu3_compute_apps": ["GPU-f9b84e89-f742-29a6-2257-ec4a4b242fe5, 2938561, 1150 MiB"], "original_cairn_dsn_train_processes": ["2560540 2560536  101    12:17:03 /data/interns/zhangpc/cairn_dev/.conda/cairn/bin/python -m cairn_dsn.train --config /data/interns/zhangpc/cairn_new/configs/path_single_drugout.yaml", "2560541 2560537  101    12:17:03 /data/interns/zhangpc/cairn_dev/.conda/cairn/bin/python -m cairn_dsn.train --config /data/interns/zhangpc/cairn_new/configs/path_double_drugout.yaml", "2560542 2560538  101    12:17:03 /data/interns/zhangpc/cairn_dev/.conda/cairn/bin/python -m cairn_dsn.train --config /data/interns/zhangpc/cairn_new/configs/no_path_single_drugout.yaml", "2560543 2560539  101    12:17:03 /data/interns/zhangpc/cairn_dev/.conda/cairn/bin/python -m cairn_dsn.train --config /data/interns/zhangpc/cairn_new/configs/no_path_double_drugout.yaml", "2928827 2560542  2.2       15:26 /data/interns/zhangpc/cairn_dev/.conda/cairn/bin/python -m cairn_dsn.train --config /data/interns/zhangpc/cairn_new/configs/no_path_single_drugout.yaml", "2928829 2560542  2.2       15:26 /data/interns/zhangpc/cairn_dev/.conda/cairn/bin/python -m cairn_dsn.train --config /data/interns/zhangpc/cairn_new/configs/no_path_single_drugout.yaml", "2931091 2560543  2.2       13:04 /data/interns/zhangpc/cairn_dev/.conda/cairn/bin/python -m cairn_dsn.train --config /data/interns/zhangpc/cairn_new/configs/no_path_double_drugout.yaml", "2931093 2560543  2.2       13:03 /data/interns/zhangpc/cairn_dev/.conda/cairn/bin/python -m cairn_dsn.train --config /data/interns/zhangpc/cairn_new/configs/no_path_double_drugout.yaml", "2938877 2560541  2.8       03:09 /data/interns/zhangpc/cairn_dev/.conda/cairn/bin/python -m cairn_dsn.train --config /data/interns/zhangpc/cairn_new/configs/path_double_drugout.yaml", "2938879 2560541  2.8       03:09 /data/interns/zhangpc/cairn_dev/.conda/cairn/bin/python -m cairn_dsn.train --config /data/interns/zhangpc/cairn_new/configs/path_double_drugout.yaml", "2940481 2560540  3.5       00:45 /data/interns/zhangpc/cairn_dev/.conda/cairn/bin/python -m cairn_dsn.train --config /data/interns/zhangpc/cairn_new/configs/path_single_drugout.yaml", "2940483 2560540  3.8       00:45 /data/interns/zhangpc/cairn_dev/.conda/cairn/bin/python -m cairn_dsn.train --config /data/interns/zhangpc/cairn_new/configs/path_single_drugout.yaml"]}
