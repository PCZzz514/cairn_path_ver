# accelerated_four: review packet (for Codex)

Author of all code here: claude-opus-5-5 (default effort). Codex owns review
approval (`logs/accelerated_four/review_approval.json`); nothing here writes it.
Nothing has been launched. Original `cairn_dsn/`, `configs/`, `outputs/`,
`scripts/`, `docs/`, the original jobs (PIDs 2560540-3) and monitor (2568426)
were only read (no signals). All writes: `accelerated_four/`, `logs/accelerated_four/`.

Diffs (actual `diff -u`, regenerate with `bash accelerated_four/tools/make_review.sh`):
`review/implementation.diff` (copies vs originals and vs the GPU3-reviewed
model/data), `review/new_files.diff` (all new files).

## Files

| path | origin | change |
|---|---|---|
| `dsn/model.py`, `dsn/data.py` | `benchmarks/gpu3_speed/dsn_gpu3/` (GPU3 reviewed) | none (diff empty) |
| `dsn/metrics.py`, `dsn/__init__.py` | `cairn_dsn/` | none |
| `dsn/train.py` | `cairn_dsn/train.py` (production: resume, test-only, early stop, full test export) | see below |
| `configs/<arm>.yaml` | `configs/<arm>.yaml` | only `output: accelerated_four/outputs/<arm>` |
| `jobs.py` | `scripts/server_jobs.py` | isolated paths, resume-only modes, crash/history handling, launch gate, no batch reduction |
| `monitor/{monitor_loop.sh,start_monitor.sh,drain_recovery_requests.py,prompt.md}` | `scripts/…`, `docs/server_monitor_prompt.md` | isolated state/scope, notifier, progress-gated start |
| `launch_four.sh` | `scripts/launch_four.sh` | + bounded progress confirmation (`tools/confirm_launch.py`) |
| `env.sh`, `status.py`, `monitor/notify.py`, `tools/*` | new | runtime, status merge, notes, snapshot, evidence, review |

### `dsn/train.py` vs production (complete list)

1. `ROOT = parents[2]`; output must lie inside `accelerated_four/outputs` (containment).
2. Whitelisted metadata in the resolved config: `generator` (new string), new keys
   `namespace: accelerated_four`, `optimization: <packed attention note>`. Constants
   `GENERATOR/NAMESPACE/OPTIMIZATION` are imported by the snapshot tool so the
   migrated checkpoint config equals what the trainer rebuilds; the original strict
   `saved['config'] != config` resume check is unchanged.
3. A `{"phase": "start", ...}` line after construction/load/RNG restore (mode,
   start epoch, update, best metric, bad epochs, will_train, schedule, batches,
   device UUID, PID). Print + device query only, no RNG.
4. `train_progress` line at the first update of the process and every 100 updates
   (flushed; from already-synchronized running totals; no RNG).
5. Validation `accuracy` added to the epoch record from the probabilities that
   `evaluate` already returns (inside its RNG fork; no RNG effect). Test export and
   `evaluate` itself are unchanged, so `test_metrics.json` has the original layout.

Unchanged: seed 42, construction order, AdamW, loss, KL `beta = kl_weight*min(1,(update+1)/warmup)`,
`planned_updates = epochs*len(train loader)` (116500) / warmup 23300 taken from the
checkpoint on resume, batch 128, workers 2 nonpersistent, cpu_threads 2, pin_memory,
`restore_rng` AFTER model/optimizer/collator/datasets/loaders, before iteration;
`if not test_only and bad_epochs < patience` loop guard (a snapshot already at
patience skips training and exports the historical best); `epochs` 100 (no GPU3 cap);
strict `>` fixed85 val macroF1 selection; best.pt-before-last.pt atomic publication;
best-checkpoint test export (predictions, 85 probabilities, per-class, confusion,
metrics, test_config, test_complete). No AMP, no hyperparameter/layer/formula change.

## Snapshots (`tools/snapshot.py`, run 2026-10-12 11:59, CPU only)

Per arm, opens `outputs/<arm>/last.pt` and `best.pt` once and `torch.load`s from
that pinned inode into CPU (the original trainer publishes by atomic rename), then
reads `history.jsonl` (newline-terminated records only) and `config.yaml`.
Accepted only if coherent at cutoff = last.pt epoch: history epochs exactly
1..cutoff (later lines dropped), last.pt `metrics` == history record at cutoff,
strict-selection replay of the history reproduces every `improved/bad_epochs/
best_validation` field and gives best.pt's epoch, record, update, `best_metric`
(== last.pt), last.pt `bad_epochs`; config.yaml == last.pt config == best.pt
config; consistent with `configs/<arm>.yaml` and GPU mapping. Otherwise (e.g.
best.pt newer than last.pt during publication) it waits 20 s and reads the next
state (bounded 1 h); processes are never signalled. All four were coherent on the
first read.

Migration: copy of last/best with `config` replaced by the migrated config
(whitelist `output, generator, namespace, optimization`; `prepared` stays the
same absolute path, `source_provenance` unchanged); the new trainer's resume
config is recomputed and must equal it. After writing, both files are reloaded
and compared to the in-memory originals recursively and exactly (every model,
AdamW state/param_group, RNG tensor/array/value; dtype, shape, device, values) —
no mismatches. Output per arm `accelerated_four/outputs/<arm>/`:
`snapshot/{last.pt,best.pt,history.jsonl,config.yaml,original_config.yaml,
original_first_update_gradients.json,original_model_inventory.json,provenance.json}`
(0444, immutable), working `last.pt`/`best.pt` = hard links (the trainer replaces
by rename, so the snapshot inode survives), `history.jsonl`/`config.yaml` copies.

| arm | GPU | cutoff epoch | update | best epoch (update) | best val macroF1(85) | bad/patience |
|---|---|---|---|---|---|---|
| path_single_drugout | 4 | 29 | 33785 | 20 (23300) | 0.569643 | 9/15 |
| path_double_drugout | 5 | 33 | 38445 | 28 (32620) | 0.123709 | 5/15 |
| no_path_single_drugout | 6 | 30 | 34950 | 16 (18640) | 0.566757 | 14/15 |
| no_path_double_drugout | 7 | 32 | 37280 | 24 (27960) | 0.125956 | 8/15 |

All four resume with `--resume` (none had reached patience); no_path_single
early-stops after epoch 31 unless it improves. The originals keep running from
their own states; the replicas are independent continuations (numerically
equivalent within the reviewed roundoff, not a bitwise-identical trajectory).

## Evidence (`tools/run_evidence.sh` -> `evidence/<arm>.json`, `evidence/summary.{json,md}`; log `logs/accelerated_four/evidence_run.log`)

Uses the working `last.pt`/`best.pt` that the jobs resume (same inode as the
immutable snapshot, checked). **All four passed.**

CPU, all four arms (OLD = original `cairn_dsn`, NEW = `accelerated_four.dsn`):
- config: vs original `outputs/<arm>/config.yaml` only the whitelist differs;
  strict resume config equality using the trainer's own reconstruction;
- trainer order replayed for OLD and NEW: seed 42, model construction (bitwise
  equal state, equal CPU RNG afterwards), AdamW, collator, datasets, real
  DataLoaders (workers 2, pinned, nonpersistent), schedule recomputed
  (116500/23300 == checkpoint), strict model + AdamW load (state_dict()/optimizer
  state_dict() after loading exactly equal to the checkpoint; 110 state entries,
  names/shapes identical OLD/NEW/checkpoint incl. FIN entity/relation tables, NAI,
  PVR, classifier), AdamW lr/wd/betas and every `step` == update, best.pt strict load;
- RNG restored after construction (CPU streams equal saved; on GPU6/7 the CUDA
  stream too), then the next-epoch permutation: OLD == NEW, a full permutation,
  and the REAL worker DataLoader's first batch == its head; update == epoch x 1165;
- history: last record == checkpoint metrics, best record == best.pt metrics;
- collator on the first 3 real next-epoch batches: the real DataLoader batch
  exactly equals the NEW collate; labels/rows/graphs (and path graph) OLD == NEW;
  per-pair edges in both directions exactly PyG's `remove_self_loops` +
  `add_self_loops(min(nh,nt))` with loops last, all edges inside their own pair,
  unequal pairs both ways (nh<nt and nh>nt), global-diagonal non-loop edges kept.

GPU, no-path arms only (no_path_single on physical GPU6, no_path_double on GPU7;
UUID checked against nvidia-smi, original PID present on the same UUID): OLD
per-pair vs NEW packed with identical seeds per pass, train mode (dropout and PVR
sampling live), loss = CE + 0.1 rec + beta KL with the real resumed beta (0.001),
for BOTH the snapshot last.pt (resumed) and best.pt (exported) parameters, on the
3 real next-epoch batches: logits/CE/rec/KL/loss, eval logits/rec, every parameter
gradient, graph input gradients, OLD-repeat noise floor. GPU3-reviewed thresholds
unchanged: forward rel <= 1e-4, input grads rel <= 1e-3, every parameter-gradient
tensor `max_abs <= 1e-3*ref + 1e-7*G`. Worst forward rel 6.5e-7 (abs 1.5e-5, same
order as OLD-repeat 1.3e-5); worst parameter-gradient rel above atol 1.1e-5.
The 6 below-atol tensors per batch are the analytically zero-gradient biases
disclosed in the GPU3 review (TransformerConv `lin_key.bias`, SAGPooling
`lin_rel.bias`), refs ~1e-10, OLD-repeat differences of the same size.

Path arms: the GPU3-reviewed attention/path numerical evidence is reused (same
model/data files); their resume/RNG/shuffle/collator evidence above is CPU.

Speed: no new timing campaign. Reviewed GPU3 figures: same-GPU step 14.62x
(1.1456 s -> 0.0784 s), five real epochs mean 114.09 s train+val vs historical
1498-1534 s. Replicas share GPU4-7 with the original jobs (GPU 22-39 % util,
1.5-3 GiB of 40 GiB) and the host CPUs (80 cores, load ~6-8; each replica adds
2 torch threads + 2 loader workers), so per-epoch time here may be slower and the
originals may slow somewhat; nothing is paused.

## Controller (`jobs.py`, state `logs/accelerated_four/jobs/<arm>/`)

Same structure as `scripts/server_jobs.py`: per-arm detached wrapper
(`start_new_session`, orphaned when the controller exits -> PPID 1), `job.lock`
held by the wrapper and inherited by the trainer (`pass_fds`), process-start
identity, attempt logs/exit files never overwritten, registry lock.
Differences: isolated ROOT-relative STATE/OUTPUTS/CONFIGS; trainer
`python -m accelerated_four.dsn.train` with `CUDA_VISIBLE_DEVICES=<gpu>`,
`CUDA_DEVICE_ORDER=PCI_BUS_ID`, `PYTHONPATH=ROOT`; modes restricted to
`--resume` / `--test-only` (no from-scratch path, no `--reduce-batch`);
`--launch-reviewed` refuses unless Codex approval exists, names Codex, is newer
than all accelerated_four code/configs/prompt, `evidence/summary.json` and
`review/implementation.diff`; evidence passed and newer than `dsn/`; each arm's
working last/best are still the snapshot inodes, history equals the snapshot, no
test exports and no job record. Initial mode per arm from the snapshot
(`--test-only` if epoch limit/patience reached, else `--resume`).
Verifier = original checks (stopping condition, selected best, frozen rows, 85
probabilities, predictions, metrics, per-class, confusion, test_config, marker)
plus: namespace/output, best.epoch <= last.epoch and equal best_metric, history
epochs exactly 1..last, last/best records equal history, best is the first strict
maximum. Recovery (<= 3 per arm, failures included; dead + unlocked + unverified
only): archive (`recovery_archive/<n>/`, hard links for .pt, copies for history/
config/test files); if best.pt is newer than last.pt (crash between best and last
publication) best.pt — a complete newer checkpoint — becomes last.pt by atomic
link-replace (the old last stays archived, the historical best is never lost);
history trimmed to the resumed epoch (removes a torn/duplicate line), refusing if
incoherent; relaunch with the exact checkpoint config `--resume`, or
`--test-only` when stopped. Recovery was not executed (no smoke runs).

## Monitor (`accelerated_four/monitor/`, state `logs/accelerated_four/monitor/`)

Copy of the existing pattern (existing monitor untouched): `start_monitor.sh`
refuses until all four job records exist AND each arm's log shows the start line,
first step and a 100-update progress line (or finished); launches
`monitor_loop.sh` detached (`start_new_session`, survives CLI exit), confirms the
startup check, writes `monitor.pid`. Loop: daemon lock, check lock, deterministic
`jobs.py --status` before/after (status errors recorded, never stop the loop),
one `codex exec -m gpt-6.1-sol` reasoning low with `CODEX_HOME=.codex_runtime`
(never read) and the same inherited proxy settings, 1200 s timeout affecting only
the CLI; then the host drains recovery requests (`{"arm","reason"}` only) through
the bounded controller outside the CLI lifetime; `last_check.txt`,
`next_check.txt` (2400 s between check starts), `state.txt`,
`snapshot_summary.json`, `checks/<id>/{before,after}.json, codex.log,
codex_final.txt`; `notify.py` writes `check_note.json` every check (quiet) and
appends to `notifications.log` only on state/attempt changes or terminal outcome.
Exits on verified all-complete (`final_summary.json`) or all complete/exhausted.
The LOW prompt allows only reading, `--status`, and recovery requests (no code/
model/data edits, no launching, no `--block`).

## Commands (Codex, after review)

```bash
cd /data/interns/zhangpc/cairn_new
bash accelerated_four/tools/make_review.sh            # if code changed
# write logs/accelerated_four/review_approval.json, e.g.
# {"approved": true, "reviewer": "Codex gpt-6.1-sol", "approved_at": "...", "notes": "..."}
bash accelerated_four/launch_four.sh --launch-reviewed  # launch + wait (<=30 min) for real progress -> logs/accelerated_four/launch.json
source accelerated_four/env.sh; CUDA_VISIBLE_DEVICES='' "$PYTHON_BIN" accelerated_four/jobs.py --status
bash accelerated_four/monitor/start_monitor.sh          # only after all four progress-confirmed
cat logs/accelerated_four/monitor/monitor.pid logs/accelerated_four/monitor/next_check.txt
```
Evidence re-run (foreground, no training): `bash accelerated_four/tools/run_evidence.sh`.
Snapshots exist; re-snapshotting requires removing `accelerated_four/outputs/<arm>`
first (the tool refuses to overwrite).

## Limitations

- Replicas resume the 11:59 snapshots; the originals continue independently, so
  the two trajectories diverge (roundoff + independent continuation).
- Resume is numerically equivalent within the reviewed roundoff, not bitwise.
- History records before the cutoff (from the original) have no validation
  `accuracy`; new epochs do.
- Recovery/verifier paths are reviewed code, not exercised (no smoke runs).
