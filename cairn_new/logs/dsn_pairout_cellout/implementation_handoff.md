# dsn_pairout_cellout — implementation ready for Codex review (NOT launched)
Author: Claude claude-opus-5-5 (default effort), 2026-10-12. Codex writes the review/approval prose and does the launch.
No job, monitor, CUDA work, model forward, benchmark, cache build, preparation or audit rerun was performed.
Existing trainers and monitors (pure RareDDIE GPU6/7, full-path GPU4/5, everything else) were only read, never signalled.

## Scope
DSN-DDI+RareDDIE NO PATH control: `DSNRareDDI(with_path=False)` from the unchanged accelerated_four files. It keeps
the DSN InterGraphAttention (GATConv 2x32, `add_self_loops=False`), the 192->128 inter_projection, and the packed
per-pair local self loops built by the collator. The path module is absent (`path_module: NONE`, 0 path parameters,
the collator has no `paths` attribute, so no path arrays are loaded). PVR input is 256, presence_dim 0, and
deterministic mu (128) feeds the same LayerNorm128->256 GELU dropout .2->85 classifier. FIN is the same pretrained
asset and stays trainable.
PairOut on GPU6 and CellOut on GPU7, in parallel with the LIVE pure controls on the same cards, so contention is
expected. Both arms are independent fresh seed42 replicates on the SAME frozen PairOut DDI split (the CellOut DDI
frame is identical), NOT true cell-cold DDI generalization. That note is in the configs, the saved
config.yaml/checkpoints, test_metrics.json, test_complete.json, launch.json and the monitor prompt, and the
verifier checks it.

## Files (namespace `dsn_pairout_cellout/`; diffs in `review/implementation.diff`, `review/new_files.diff`)
- `dsn/model.py`, `dsn/data.py`: byte-for-byte copies of `accelerated_four/dsn/` (cmp output is at the top of
  implementation.diff, followed by empty `diff -u`). Those files also equal `benchmarks/gpu3_speed/dsn_gpu3/`
  (GPU3 `all_numerical_checks_passed`), and accelerated_four was approved. `dsn/__init__.py` is new (docstring only).
- `inventory.py` (new): disjoint GROUPS, `group_of`, `UNUSED_TRAINABLE` (last block norm2, which the original
  forward never consumes), and `parameter_inventory(model, collator)`. The inventory records counts per group,
  frozen tensors, the DSN GATConvs, inter_projection, with_path, path_module NONE, path parameters, whether the
  collator loaded path arrays, FIN tables and trainability, PVR dims, the mu->classifier route, and every tensor.
- `train.py`: the rareddie_pairout_cellout trainer with only these changes:
  - imports `.dsn.model.DSNRareDDI`, `.dsn.data` and `.inventory`;
  - namespace, outputs, generator and ablation text;
  - refuses any config with `with_path` not false;
  - `DSNRareDDI(assets, config)` and `Collator(assets, prepared, config['with_path'])`;
  - `first_update.json` records `cross_edges` (count, expected >0), `path_mask_shape` ([128, 0]) and per-group
    gradient norms over the disjoint groups (no path group).

  Seeding, AdamW, loss (CE + .1 rec + warmed .001 KL), the 150/15 schedule assert, 1184 batches, progress lines
  (first update and every 100), validation accuracy/MacroF1, epoch seconds, strict fixed85 selection, test after
  the stop only, every export, and resume/RNG are all unchanged.
- `configs/{pairout,cellout}.yaml`: same as the pure configs plus `with_path: false`, with outputs
  `dsn_pairout_cellout/outputs/<arm>`.
- `jobs.py`: rareddie controller with a namespace swap. The launch gate changes:
  - model/data byte-identity is checked against `accelerated_four/dsn`;
  - prior evidence is the GPU3 `all_numerical_checks_passed`, plus the gpu3_speed and accelerated_four approvals;
  - the idle-GPU refusal becomes ">= 8192 MiB free on the shared GPU6/7", because the pure controls live there.
    The coexisting compute apps are recorded in status.json at launch.

  Verifier, bounded recovery (max 3, own checkpoint, exact config, --resume / --test-only, never --fresh again)
  and status are otherwise unchanged.
- `confirm_launch.py`: the first-update check requires:
  - finite, nonzero gradient norms for all 8 groups (incl. dsn_inter_attention and inter_projection);
  - no gradient only on last-block norm2;
  - cross edges present, no path graph.

  It does not require a PATH gradient, and it does not require gradients on the frozen readout.select weights.
  The inventory check requires DSN GAT x2 without auto self loops, inter_projection present, path NONE, no path
  arrays, PVR 256/0/128, 4 TransformerConvs, and counts/groups/frozen equal to the evidence's CPU inventory.
  launch.json carries the contention + same-split note.
- `monitor/*`: identical mechanism to rareddie_pairout_cellout (namespace swap only). The prompt scope adds:
  rareddie_pairout_cellout is someone else's, and GPU6/7 are shared with its LIVE trainers, which must never be
  signalled. Own lock/PID/log files are under `logs/dsn_pairout_cellout/monitor/`. CODEX_HOME is the existing
  `.codex_runtime`, the proxy is 127.0.0.1:7890, and the model is gpt-6.1-sol LOW every 2400 s, with an immediate
  first check.
- `env.sh`, `launch.sh`, `status.py`: namespace swap plus comments.
- `tools/cohort_evidence.py`: the rareddie narrow cohort checks plus the copy record and the CPU inventory.
  `tools/make_review.sh` regenerates the diffs.

## Evidence (`dsn_pairout_cellout/evidence/cohort_copy_evidence.json`, passed=true)
- Source PairOut == CellOut frames, prepared == source, pair indices equal.
- Counts 151508/18939/18938, classes 85/82/83, 1666 drugs all with graph and NAI row, event names exact.
- 1184/177600/35520 (30 epochs).
- Inventory: 13,206,361 parameters (13,206,359 trainable; frozen = the 2 readout.select weights).
  - dsn_inter_attention 33,152 and inter_projection 49,408.
  - cse 340,356, nai 98,432, pvr 230,400, classifier 55,125, FIN 12,386,048 + 13,440: each equal to the live pure
    control inventory.
  - The total equals the accelerated_four no-path inventory.
- Dry gate (no launch): the only refusals were the missing approval and the not-yet-generated review diff.
  GPU6/7 had ~39 GiB free beside the pure trainers (1364 MiB each).

## Codex next steps (Claude did NOT launch)
1. Review the diffs and evidence, then write `logs/dsn_pairout_cellout/review_approval.json`
   (`approved: true`, reviewer containing "Codex"). It must be newer than all code, the evidence and the diff.
2. `bash dsn_pairout_cellout/launch.sh --launch-reviewed`. This launches detached wrappers (PPID 1) and waits up to
   1800 s for both arms' actual first update plus a 100-update line, then writes `logs/dsn_pairout_cellout/launch.json`.
3. `bash dsn_pairout_cellout/monitor/start_monitor.sh`. This waits for the first successful check and records
   next_check in status.json.

Timing: no promise. The accelerated_four no-path DSN ran ~1470 s/epoch, and here the CPU is shared (nproc 2) and
the GPUs are shared with the pure controls.
