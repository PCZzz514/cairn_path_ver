# pairout_cellout implementation handoff (Claude -> Codex review)

Author: claude --model claude-opus-5-5, DEFAULT effort (all code). State:
**ready for review, NOT launched** (no training, monitor, forward or GPU use).

**150-epoch schedule (logs/pairout_cellout/user_update_150.md; overrides every 100-epoch
instruction for these jobs): max 150 epochs, patience 15, KL warmup fraction 0.2.**
151508 train rows / batch 128 = **1184 batches** (final batch 84; verified with the real
DataLoader for both arms) -> **177600 planned updates**, **KL warmup 35520** (30 epochs).
Superseded history only: 100 epochs / 118400 / 23680.

1. Claude PID 3230783 (initial implementation) read the update before finishing and applied
   150: `configs/{pairout,cellout}.yaml` (epochs 150), `dsn/train.py` (`SCHEDULE`/`TRAIN_BATCHES`,
   comment, refusal of any other schedule in fresh/resume/test-only), `audit.py` (expected
   config + derived schedule), `jobs.py`, `confirm_launch.py`, `monitor/prompt.md`, `REVIEW.md`.
2. **Authorized 150 follow-up (this Claude, claude-opus-5-5 DEFAULT)** verified that change and
   strengthened schedule validation (delta: `pairout_cellout/review/followup_150_delta.diff`):
   - `jobs.py`: literal `AUTHORIZED_SCHEDULE` = 150/15/0.2/177600/35520 and
     `AUTHORIZED_TRAIN_BATCHES` = 1184, asserted equal to the trainer constants. Completion
     (`verify_exports`) requires it in best/last/test configs and the run's `config.yaml`,
     `planned_updates == epochs * 1184`, and `update == epoch * 1184` for every history line,
     last and best. Recovery (`resume_point`) refuses any other schedule.
   - `confirm_launch.py`: new `schedule_ok` (part of `progress_confirmed`) requires the run's
     written `config.yaml` schedule, first progress line at update start+1, and the ACTUAL KL
     beta of the first and 100-update lines == .001 * update / 35520. A 23680 warmup would fail.
   - `audit.py`: configs must have exactly the reviewed key set, so no planned/warmup
     override key can be added.
   - Docs/metadata: `review/REVIEW.md`, regenerated `review/implementation.diff` and
     `new_files.diff`, `status.json` (`implementation.followup_150`), this handoff, and a
     SUPERSEDED note appended to `source_identity_note.md`. That file's Codex-written 100-epoch
     line is kept as history and is not edited.
   - Unchanged: architecture, `dsn/train.py`, configs, the model/data/metrics/paths copies
     (still byte-identical, checked with `cmp`), losses, metrics, preparation (not rerun),
     data rows/order/events/split, monitor scripts.

**Verification evidence (real, CPU only, after the last Python edit):**
- `bash -n` on all 5 shell scripts and `ast.parse` on all Python files pass, and the CPU
  imports work.
- Full real audit reran in 50 s and passed (`logs/pairout_cellout/audit.log`,
  `prepared/audit_report.json`, newer than all `.py`). Its schedule section is
  150/15/0.2/1184/177600/35520.
- Real DataLoader: pairout and cellout each have 151508 rows, 1184 batches, final batch 84.
  The configs derive 177600/35520 (30.0 warmup epochs) == `SCHEDULE` == `AUTHORIZED_SCHEDULE`.
  The trainer `preparation_gate` passes for both configs.
- The old 100/118400/23680 schedule fails the `AUTHORIZED_SCHEDULE` check. A 23680-warmup beta
  at update 1 (4.22e-8) is not close to the required 2.82e-8.
- `jobs.py --status`: both arms `not_launched`. The only launch-gate refusal is the missing
  Codex review approval, so GPU6/7 were idle and outputs/records empty at check time.
- Stale-100 review of `pairout_cellout/` code, configs, prompt and docs: the only `100` left
  is the progress interval (`PROGRESS_EVERY = 100`, `progress_100`, every-100 progress
  lines). 118400/23680 appear only as explicitly superseded history.
  `preparation_report.json` `fin.completed_epoch: 100` is the pre-existing FIN pretraining
  record. In current logs metadata, `plan.md`, `request.md` (appended scope note),
  `status.json` and this handoff state 150. `claude_request.md` is the historical original
  request, and `source_identity_note.md` now carries the superseded marker.

Read: `pairout_cellout/review/REVIEW.md`, `pairout_cellout/review/followup_150_delta.diff`,
`pairout_cellout/review/implementation.diff`,
`pairout_cellout/review/new_files.diff`, `pairout_cellout/prepared/preparation_report.json`,
`pairout_cellout/prepared/audit_report.json`, `pairout_cellout/prepared/class_support.csv`,
`logs/pairout_cellout/prepare.log`, `logs/pairout_cellout/audit.log`.

Inspected before adapting: `cairn_dsn/{prepare,paths,audit}.py`,
`accelerated_four/dsn/{model,data,train,metrics}.py` and review/approval,
`benchmarks/gpu3_speed` evidence (model/data identical to GPU3-reviewed copies;
grad set excludes only `cse.blocks.1.norm2.*`), `rareddie_control` controller/monitor
and approval, real assets (`../cairn_dev/data/processed/{ddi.parquet,bundle.pt,fin.pt,
drkg_triples.pt,drugs.parquet,ddi_events.csv}`, reference85/cellout, old `data/prepared`).

## Commands (from /data/interns/zhangpc/cairn_new)

Already run (CPU):
- prepare: `bash -c 'source pairout_cellout/env.sh; CUDA_VISIBLE_DEVICES= "$PYTHON_BIN" -m pairout_cellout.prepare'`
- audit: `bash -c 'source pairout_cellout/env.sh; CUDA_VISIBLE_DEVICES= "$PYTHON_BIN" -m pairout_cellout.audit'`
- review diffs: `bash pairout_cellout/tools/make_review.sh`
- status: `bash -c 'source pairout_cellout/env.sh; CUDA_VISIBLE_DEVICES= "$PYTHON_BIN" pairout_cellout/jobs.py --status'`

After Codex writes `logs/pairout_cellout/review_approval.json` (`approved: true`,
reviewer containing "Codex", written AFTER all code, `review/implementation.diff` and
`prepared/audit_report.json`; any code change requires rerunning audit and make_review):
1. launch both arms + bounded confirmation: `bash pairout_cellout/launch.sh --launch-reviewed`
   (refuses without approval / with stale audit / busy GPU6-7 / existing records or outputs;
   then `confirm_launch.py --wait 1800` writes `logs/pairout_cellout/launch.json`). Per arm,
   `progress_confirmed` needs: fresh seed42 path start line with 1184 batches / 150 epochs /
   patience 15 / 177600 / 35520 / batch 128; `first_update.json` finite, nonzero gradients in
   ALL groups incl. path (only `cse.blocks.1.norm2.*` without gradient), path batch edges > 0;
   13522485/13522483 parameter inventory; first progress line and a 100-update line;
   `schedule_ok` (run config.yaml schedule, actual beta == .001*update/35520). Wrapper PPID 1
   and trainer liveness are recorded in launch.json.
2. re-check anytime: `bash -c 'source pairout_cellout/env.sh; CUDA_VISIBLE_DEVICES= "$PYTHON_BIN" pairout_cellout/confirm_launch.py'`
3. monitor (only after both arms confirmed): `bash pairout_cellout/monitor/start_monitor.sh`
   (detached setsid daemon PPID 1, CODEX_HOME=.codex_runtime, existing proxy, gpt-6.1-sol LOW,
   first check immediate, waits for its first successful check, then every 2400 s until both
   selected-best test exports verify).

Current launch-gate refusals (expected before review): missing Codex review approval.

## Notes

- PairOut and CellOut DDI frames are byte-identical (CellOut ddi.parquet symlinks to the
  processed PairOut frame). Two independent fresh seed42 same-DDI-split replicates; not DDI
  cell-cold generalization. Same inputs/config except GPU/name/output; near-identical results
  expected, GPU roundoff can differ.
- Schedule: 151508 / 128 = 1184 batches/epoch (final batch 84), 150 epochs = 177600 planned
  updates, KL warmup ceil(0.2 * 177600) = 35520 (30 epochs), patience 15, strict fixed85 val
  macroF1 selection. (Superseded historical plan: 100 epochs / 118400 / 23680.)
- Outputs: `pairout_cellout/outputs/{pairout,cellout}`; job state `logs/pairout_cellout/jobs`.
- Original GPU4/5 path jobs (2560540/2560541) and monitor 2568426 are live and untouched;
  their own `outputs/path_double_drugout` writes are theirs.
