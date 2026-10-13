# rareddie_control — implementation ready for Codex review (NOT launched)
Author: Claude claude-opus-5-5 (default effort), 2026-10-12. Codex authors review/approval prose.

## What was built (isolated namespace `rareddie_control/`)
- `model.py` — RareDDIEControl = accelerated_four no-path model minus DSN `interAtt`, `inter_projection`
  (both blocks) and cross edges. CSE block is the original MVN_DDI_Block order; feature_conv2 128-d atom
  states go untouched into SAGPooling. Molecular TransformerConv x4 kept. NAI, PairVAE(256) imported unchanged
  from accelerated_four.dsn.model; classifier on deterministic mu (LayerNorm128->256 GELU Dropout0.2 ->85).
  `parameter_inventory()` lists every tensor/group.
- `data.py` — molecule-only Collator (no cross_edges, no path); DDIDataset/to_device reused.
- `train.py` — accelerated_four trainer adapted: explicit `--fresh|--resume|--test-only`; fresh refuses a
  non-empty output; resume/test-only load ONLY own checkpoints (namespace+output asserted). Writes
  `initial_rng.pt` (seeded state before model construction), `model_inventory.json`, `config.yaml`,
  `first_update.json` (actual CE/rec/KL/beta/loss, per-group grad norms, finiteness, params without grad).
  Same AdamW 1e-4/1e-4, batch 128, CE + 0.1 rec + beta KL (warmup ceil(0.2*100*1165)=23300), patience 15 on
  strict fixed85 val MacroF1, val accuracy, progress line every 100 updates, test only after stop with
  best.pt; exports parquet/npz/per-class/confusion/metrics/test_config/test_complete + report-only top5_accuracy.
- `configs/{single_drugout,double_drugout}.yaml` — comparator configs minus path keys; GPU6 / GPU7.
- `jobs.py`, `launch.sh`, `confirm_launch.py`, `status.py`, `env.sh`, `monitor/*` — adapted copies
  (accelerated_four originals untouched). Recovery: genuine dead+unlocked+unverified only, bounded 3,
  own-state --resume/--test-only with exact recorded config; --fresh only for the initial launch.
  Verifier recomputes metrics and Top-5 from saved probabilities over the 85-class universe.
- Monitor: `codex exec --dangerously-bypass-approvals-and-sandbox --ignore-user-config --ephemeral
  --skip-git-repo-check -c model_reasoning_effort="low" -m gpt-6.1-sol`, CODEX_HOME=.codex_runtime, existing
  proxy, first check immediate then every 2400 s, setsid -f daemon (PPID 1), no notifications.

## Evidence
- `rareddie_control/evidence/cse_equivalence.json` (log: logs/rareddie_control/cse_equivalence.log): unmodified
  official MVN_DDI.forward (cairn_dev retained source, read-only) vs new CSE, shared params, 60 real
  single_drugout train pairs, 1..208 atoms incl. 1-atom self-loop molecules: forward bitwise equal (0.0),
  all shared-param gradients bitwise equal (0.0), identical no-grad set (block1.norm2, frozen select scalars);
  negative control DSN CSE differs by 0.406. Full model: 14 removed names = interAtt/inter_projection only,
  none added, FIN/NAI/PVR/classifier names+shapes equal to comparator; logits [60,85]; no forbidden names.
- `rareddie_control/review/implementation.diff`, `new_files.diff` — actual diffs vs accelerated_four.
- Launch gate currently refuses ONLY: missing logs/rareddie_control/review_approval.json.

## Exact commands (run from /data/interns/zhangpc/cairn_new, non-sandboxed host shell)
1. Approval (Codex writes; must be newer than code/evidence/diff):
   logs/rareddie_control/review_approval.json  {"approved": true, "reviewer": "Codex gpt-6.1-sol medium", ...}
2. Launch + bounded confirmation (waits <=1800 s for first update + 100-update line on both arms):
   bash rareddie_control/launch.sh --launch-reviewed
3. Re-confirm standalone (read-only):
   bash -c 'source rareddie_control/env.sh; CUDA_VISIBLE_DEVICES= "$PYTHON_BIN" rareddie_control/confirm_launch.py'
4. Monitor (waits <=1500 s for first successful check: Codex exit 0, after.json, next_check.txt):
   bash rareddie_control/monitor/start_monitor.sh
5. Status any time:
   bash -c 'source rareddie_control/env.sh; CUDA_VISIBLE_DEVICES= "$PYTHON_BIN" rareddie_control/jobs.py --status'
Outputs: rareddie_control/outputs/{single_drugout,double_drugout}; jobs: logs/rareddie_control/jobs;
monitor: logs/rareddie_control/monitor; launch record: logs/rareddie_control/launch.json; status.json.

## Notes for reviewer
- A run dying before its first epoch checkpoint is NOT auto-recoverable (fresh only at initial launch).
- Trajectory differs from the no-path comparator (different init RNG consumption); no paired bitwise claim.
