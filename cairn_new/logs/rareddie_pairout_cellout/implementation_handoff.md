# rareddie_pairout_cellout — implementation ready for Codex review (NOT launched)
Author: Claude claude-opus-5-5 (default effort), 2026-10-12. Codex authors review/approval prose.
No job, monitor, CUDA work, model forward, benchmark, preparation or audit rerun was performed.

## Scope
Pure RareDDIE control (rareddie_control model/data, unchanged) on PairOut (GPU6) and CellOut (GPU7).
Both arms are independent fresh seed42 replicates on the SAME frozen PairOut DDI split (CellOut DDI frame is
identical), NOT true cell-cold DDI generalization. This note is in configs, saved config.yaml/checkpoints,
test_config.yaml, test_metrics.json and test_complete.json, and is checked by the verifier.

## Files (namespace `rareddie_pairout_cellout/`; actual diffs in `review/implementation.diff`, `review/new_files.diff`)
- `model.py`, `data.py` — byte-for-byte copies of `rareddie_control/{model,data}.py` (cmp-verified, shown in the
  diff header). Bond-aware original CSE, molecular TransformerConv x4, NAI/PairVAE imported unchanged from
  accelerated_four.dsn.model, mu classifier. No DSN InterGraphAttention/inter_projection/cross edges/path.
- `train.py` — rareddie_control trainer, narrowly adapted: GPU map pairout=6/cellout=7, namespace/outputs,
  `preparation_gate` reusing the existing read-only `pairout_cellout/prepared` audit (passed, frames identical,
  per-protocol rows/columns/dtypes/event names/drug ids/pair index exact; NAI cap 30; no cold-drug or path checks),
  authorized SCHEDULE in every mode (150 epochs, patience 15, 1184 train batches asserted, 177600 planned,
  35520 KL warmup = 30 epochs), `ddi_split_note` required and `cohort_source_identity` recorded in config.
  first_update.json additionally records logits finiteness, actual batch/output keys and
  `cross_edges_present`/`path_graph_present` (must be false). Test metrics add protocol, split note,
  parameter count and training_seconds; marker adds protocol and split note. Loss, state keys, seeding,
  AdamW, resume/RNG restoration, selection (strict fixed85 val MacroF1 only), and export unchanged.
- `configs/{pairout,cellout}.yaml` — seed42, epochs150, patience15, batch128, AdamW 1e-4/wd 1e-4, rec .1,
  kl .001, warmup fraction .2, NAI cap 30, cpu_threads 2, num_workers 2 (persistent_workers false in code),
  prepared `pairout_cellout/prepared`, outputs `rareddie_pairout_cellout/outputs/<arm>`.
- `jobs.py` — rareddie_control controller: arms pairout/cellout; verifier extended with authorized schedule,
  1184 updates/epoch, seed/batch/with_path=false, split note and protocol in metrics/marker (same frozen cohort
  rows from prepared/<arm>/ddi.parquet); resume refuses any other schedule. Launch gate: Codex approval newer than
  code/diff/evidence, `evidence/cohort_copy_evidence.json` passed, model.py/data.py still byte-identical to
  rareddie_control, prior `rareddie_control/evidence/cse_equivalence.json` passed and
  `logs/rareddie_control/review_approval.json` approved, idle GPU6/7, no records/outputs.
- `confirm_launch.py` — also requires start line fresh/seed42/with_path false/1184 batches/151508 rows/150/15/
  177600/35520/batch128, config.yaml schedule, KL beta of first and 100-update lines on the 35520 warmup,
  every GROUP finite and nonzero, no cross edges/path graph in the actual batch, and inventory without DSN/path
  modules with parameters/trainable/groups/frozen equal to the verified rareddie_control inventory (read-only).
- `status.py`, `env.sh`, `launch.sh`, `monitor/*` — renamed copies. Monitor prompt: ONLY the two new arms,
  150 epochs/1184 updates per epoch; all other jobs/namespaces untouched. monitor_loop inherits the launching
  shell's proxy (fallback to the existing 127.0.0.1:7890 service only if unset), CODEX_HOME=.codex_runtime,
  gpt-6.1-sol low, immediate first check then every 2400 s, setsid -f (PPID 1), until both exports verified.
- `tools/cohort_evidence.py` (new) — small exact checks; `tools/make_review.sh` regenerates the review diffs.

## Evidence (already produced)
`rareddie_pairout_cellout/evidence/cohort_copy_evidence.json`: passed=true. Original sources
`../cairn_dev/data/processed/ddi.parquet` and `../cairn_dev/data/experiments/reference85/cellout/ddi.parquet`
(symlink to the same file) equal each other and both prepared frames in columns/order/dtypes/index/values;
pair_indices equal across arms, one per row; counts train151508/val18939/test18938; classes 85/82/83; all 1666
drugs have graphs and NAI rows; event ids 0..84 and event names exact; schedule 1184/177600/35520 (30 epochs);
model/data byte-identical copies; prior CSE evidence (forward and gradients bitwise equal) and prior approval.
Static checks: ast parse of all .py, `bash -n` of all .sh, imports + both configs through `preparation_gate`.
Launch-gate dry run (`list(jobs.launch_refusals())`) refuses ONLY for the missing Codex approval (GPU6/7 idle).

## Commands (from /data/interns/zhangpc/cairn_new)
1. Review: `rareddie_pairout_cellout/review/implementation.diff`, `review/new_files.diff`, evidence JSON above.
   Regenerate diffs after any edit: `bash rareddie_pairout_cellout/tools/make_review.sh`
   (re-run evidence if needed: `source rareddie_pairout_cellout/env.sh; CUDA_VISIBLE_DEVICES='' "$PYTHON_BIN" -m rareddie_pairout_cellout.tools.cohort_evidence`).
2. Codex writes `logs/rareddie_pairout_cellout/review_approval.json` with `"approved": true` and a reviewer
   string containing "Codex" (must be newer than code, review diff and evidence).
3. Launch both detached fresh jobs and confirm start/first update/update100 (bounded 1800 s wait):
   `bash rareddie_pairout_cellout/launch.sh --launch-reviewed` -> `logs/rareddie_pairout_cellout/launch.json`.
4. Start the LOW monitor (waits for the first successful check):
   `bash rareddie_pairout_cellout/monitor/start_monitor.sh` -> `logs/rareddie_pairout_cellout/monitor/`.
- Status anytime: `source rareddie_pairout_cellout/env.sh; CUDA_VISIBLE_DEVICES='' "$PYTHON_BIN" rareddie_pairout_cellout/jobs.py --status`.
- Recovery is only via monitor requests drained by the host daemon (`jobs.py --recover <arm> --reason ...`,
  bounded 3, own checkpoint, exact config).

## Runtime / params
Expected parameters 13123801 (trainable 13123799), equal to the rareddie_control inventory (checked at launch
confirmation). Per-epoch seconds in history.jsonl; total training_seconds and parameters in test_metrics.json.
