# accelerated_four resume/compatibility evidence

Summarized 2026-10-12T12:04:31+0800. **All four passed: True**

States are the snapshot last.pt/best.pt the new jobs resume (hard links of the immutable `snapshot/` files). OLD = original `cairn_dsn` (read-only), NEW = `accelerated_four.dsn`.

| arm | cutoff epoch | update | best epoch | best val macroF1(85) | bad epochs | will train | first-batch rows | evidence s |
|---|---|---|---|---|---|---|---|---|
| path_single_drugout | 29 | 33785 | 20 | 0.5696 | 9/15 | True | [7350, 82683, 149163]... | 9 |
| path_double_drugout | 33 | 38445 | 28 | 0.1237 | 5/15 | True | [32356, 111145, 135315]... | 6 |
| no_path_single_drugout | 30 | 34950 | 16 | 0.5668 | 14/15 | True | [154213, 6188, 143171]... | 35 |
| no_path_double_drugout | 32 | 37280 | 24 | 0.1260 | 8/15 | True | [20874, 14723, 67]... | 36 |

## Checks

- **path_single_drugout** passed True: snapshot_provenance_passed=True, resumed_files_are_immutable_snapshot=True, config_only_whitelist_changed=True, strict_resume_config=True, state_dict_layout=True, construction_and_rng=True, strict_model_adamw_load_exact=True, adamw_hyperparameters_and_steps=True, rng_restore_and_permutation=True, epoch_update_schedule=True, history_and_selection=True, collator_exact=True
- **path_double_drugout** passed True: snapshot_provenance_passed=True, resumed_files_are_immutable_snapshot=True, config_only_whitelist_changed=True, strict_resume_config=True, state_dict_layout=True, construction_and_rng=True, strict_model_adamw_load_exact=True, adamw_hyperparameters_and_steps=True, rng_restore_and_permutation=True, epoch_update_schedule=True, history_and_selection=True, collator_exact=True
- **no_path_single_drugout** passed True: snapshot_provenance_passed=True, resumed_files_are_immutable_snapshot=True, config_only_whitelist_changed=True, strict_resume_config=True, state_dict_layout=True, construction_and_rng=True, strict_model_adamw_load_exact=True, adamw_hyperparameters_and_steps=True, rng_restore_and_permutation=True, epoch_update_schedule=True, history_and_selection=True, collator_exact=True, no_path_gpu_numerical=True
- **no_path_double_drugout** passed True: snapshot_provenance_passed=True, resumed_files_are_immutable_snapshot=True, config_only_whitelist_changed=True, strict_resume_config=True, state_dict_layout=True, construction_and_rng=True, strict_model_adamw_load_exact=True, adamw_hyperparameters_and_steps=True, rng_restore_and_permutation=True, epoch_update_schedule=True, history_and_selection=True, collator_exact=True, no_path_gpu_numerical=True

## Collator structure (real next-epoch batches, both directions vs PyG default loop rule)

| arm | batches | pairs nh<nt | nh>nt | nh=nt | local loops | global-diagonal edges kept | DataLoader batch == collate |
|---|---|---|---|---|---|---|---|
| path_single_drugout | 3 | 200 | 171 | 13 | 7740 | 737 | True |
| path_double_drugout | 3 | 187 | 185 | 12 | 7766 | 786 | True |
| no_path_single_drugout | 3 | 200 | 164 | 20 | 7571 | 524 | True |
| no_path_double_drugout | 3 | 194 | 184 | 6 | 7630 | 929 | True |

## No-path GPU numerical (OLD per-pair vs NEW packed, same seed, train mode, full loss)

Gate (GPU3-reviewed): forward rel <= 1e-4; input gradients rel <= 1e-3; every parameter-gradient tensor max_abs <= 1e-3 * ref + 1e-7 * G (G = largest parameter-gradient magnitude of that batch).

| arm | GPU | parameters | batch | loss OLD | loss NEW | forward worst rel | forward worst abs | param grad worst rel (above atol) | atol | below-atol tensors | gate |
|---|---|---|---|---|---|---|---|---|---|---|---|
| no_path_single_drugout | 5fc36a30b066 | snapshot_last_resumed | 0 | 0.05273396 | 0.05273397 | 2.946e-07 | 1.335e-05 | 2.950e-06 (cse.blocks.1.interAtt.inter.lin_dst.weight) | 7.126e-09 | 6 | True |
| no_path_single_drugout | 5fc36a30b066 | snapshot_last_resumed | 1 | 0.07032658 | 0.07032660 | 4.236e-07 | 1.144e-05 | 4.140e-06 (cse.blocks.1.interAtt.inter.att_dst) | 1.291e-08 | 6 | True |
| no_path_single_drugout | 5fc36a30b066 | snapshot_last_resumed | 2 | 0.08655381 | 0.08655380 | 2.668e-07 | 1.144e-05 | 3.197e-06 (cse.blocks.0.feature_conv2.lin_query.weight) | 2.488e-08 | 6 | True |
| no_path_single_drugout | 5fc36a30b066 | snapshot_best_exported | 0 | 0.13972884 | 0.13972883 | 2.999e-07 | 9.537e-06 | 1.110e-05 (cse.blocks.1.interAtt.inter.att_dst) | 1.697e-08 | 6 | True |
| no_path_single_drugout | 5fc36a30b066 | snapshot_best_exported | 1 | 0.15055242 | 0.15055239 | 3.726e-07 | 1.240e-05 | 4.136e-06 (cse.blocks.0.feature_conv.lin_query.bias) | 1.380e-08 | 6 | True |
| no_path_single_drugout | 5fc36a30b066 | snapshot_best_exported | 2 | 0.10927439 | 0.10927442 | 4.667e-07 | 1.144e-05 | 3.061e-06 (cse.blocks.1.interAtt.inter.lin_dst.weight) | 1.804e-08 | 6 | True |
- no_path_single_drugout snapshot_last_resumed: OLD-repeat vs OLD noise floor worst abs 1.144e-05 (batch0.forward.eval.logits); NEW vs OLD worst abs 1.335e-05 (batch0.forward.logits)
- no_path_single_drugout snapshot_best_exported: OLD-repeat vs OLD noise floor worst abs 9.537e-06 (batch2.forward.logits); NEW vs OLD worst abs 1.240e-05 (batch1.forward.logits)
- no_path_single_drugout GPU: {"cuda_visible_devices": "6", "cuda_device_order": "PCI_BUS_ID", "torch_device_uuid": "9a4c3035-f289-b3d1-b6df-5fc36a30b066", "nvidia_smi_uuid": "GPU-9a4c3035-f289-b3d1-b6df-5fc36a30b066", "physical_gpu_verified": true, "gpu_name": "NVIDIA A100-PCIE-40GB", "compute_apps_on_this_gpu": ["GPU-9a4c3035-f289-b3d1-b6df-5fc36a30b066, 2560542, 1490 MiB", "GPU-9a4c3035-f289-b3d1-b6df-5fc36a30b066, 2962014, 414 MiB"], "tf32_matmul": false, "torch": "2.8.0+cu128", "utilization": "23 %, 1919 MiB, 40960 MiB", "loadavg": [7.94580078125, 6.9287109375, 6.9033203125]}
| no_path_double_drugout | defd79ee32fc | snapshot_last_resumed | 0 | 0.06088518 | 0.06088519 | 4.291e-07 | 1.526e-05 | 3.734e-06 (cse.blocks.1.interAtt.inter.att_dst) | 1.098e-08 | 6 | True |
| no_path_double_drugout | defd79ee32fc | snapshot_last_resumed | 1 | 0.05545098 | 0.05545098 | 2.786e-07 | 1.144e-05 | 3.132e-06 (cse.blocks.1.interAtt.inter.att_dst) | 1.353e-08 | 6 | True |
| no_path_double_drugout | defd79ee32fc | snapshot_last_resumed | 2 | 0.07249908 | 0.07249911 | 6.544e-07 | 1.144e-05 | 2.539e-06 (cse.blocks.1.interAtt.inter.att_dst) | 1.391e-08 | 6 | True |
| no_path_double_drugout | defd79ee32fc | snapshot_best_exported | 0 | 0.07159454 | 0.07159453 | 3.128e-07 | 1.144e-05 | 2.910e-06 (cse.blocks.1.re_shape_e.weight) | 1.074e-08 | 6 | True |
| no_path_double_drugout | defd79ee32fc | snapshot_best_exported | 1 | 0.08569004 | 0.08569004 | 2.918e-07 | 1.144e-05 | 3.422e-06 (cse.blocks.1.interAtt.inter.att_dst) | 9.711e-09 | 6 | True |
| no_path_double_drugout | defd79ee32fc | snapshot_best_exported | 2 | 0.07514481 | 0.07514481 | 3.479e-07 | 1.335e-05 | 2.896e-06 (cse.blocks.1.interAtt.inter.lin_dst.weight) | 9.765e-09 | 6 | True |
- no_path_double_drugout snapshot_last_resumed: OLD-repeat vs OLD noise floor worst abs 1.335e-05 (batch0.forward.eval.logits); NEW vs OLD worst abs 1.526e-05 (batch0.forward.eval.logits)
- no_path_double_drugout snapshot_best_exported: OLD-repeat vs OLD noise floor worst abs 1.335e-05 (batch1.forward.eval.logits); NEW vs OLD worst abs 1.335e-05 (batch2.forward.eval.logits)
- no_path_double_drugout GPU: {"cuda_visible_devices": "7", "cuda_device_order": "PCI_BUS_ID", "torch_device_uuid": "33ef51ee-980e-6d67-10c7-defd79ee32fc", "nvidia_smi_uuid": "GPU-33ef51ee-980e-6d67-10c7-defd79ee32fc", "physical_gpu_verified": true, "gpu_name": "NVIDIA A100-PCIE-40GB", "compute_apps_on_this_gpu": ["GPU-33ef51ee-980e-6d67-10c7-defd79ee32fc, 2560543, 1492 MiB", "GPU-33ef51ee-980e-6d67-10c7-defd79ee32fc, 2962015, 414 MiB"], "tf32_matmul": false, "torch": "2.8.0+cu128", "utilization": "33 %, 1921 MiB, 40960 MiB", "loadavg": [7.94580078125, 6.9287109375, 6.9033203125]}

Below-atol tensors are the analytically zero-gradient biases disclosed in the GPU3 review (TransformerConv lin_key.bias, SAGPooling lin_rel.bias); their raw relative values are roundoff residue and the OLD code is not reproducible on them either (OLD-repeat). Path arms reuse the GPU3-reviewed attention/path numerical evidence; their resume/shuffle/collator evidence is CPU.
