# Reviewed launch and server handoff

Run from `/data/interns/zhangpc/cairn_new` after coordinator code review. These
scripts have been prepared only; training and monitoring have not been started.
Preparation is already complete. Do not rerun it: `data/prepared/audit_report.json`
confirms all 64,533,910 cached real edges match source facts and multiplicities,
and both depth-4 reachability directions match the explicit masks. The CSR
ownership fix (`copy=True`) is already present.

```bash
cd /data/interns/zhangpc/cairn_new
bash scripts/launch_four.sh --launch-reviewed
source scripts/runtime.sh
CUDA_VISIBLE_DEVICES='' "$PYTHON_BIN" scripts/server_jobs.py --status
# After coordinator confirms all four jobs started:
bash scripts/start_monitor.sh
cat logs/server_monitor/monitor.pid
cat logs/server_monitor/next_check.txt
```

The initial monitor check runs immediately, then every 2400 seconds measured
between check starts. Allow a few seconds for its PID/next-check files. Once
the coordinator confirms the four jobs and detached monitor, the main chat/goal
ends. The server owns completion; it needs no SSH session or app automation.
`start_monitor.sh` is safe to repeat while the daemon is live. Initial training
launch deliberately refuses existing run records/checkpoints; recovery is separate.

| Job | Physical GPU | Output |
| --- | --- | --- |
| path single_drugout | 4 | outputs/path_single_drugout |
| path double_drugout | 5 | outputs/path_double_drugout |
| no_path single_drugout | 6 | outputs/no_path_single_drugout |
| no_path double_drugout | 7 | outputs/no_path_double_drugout |

Each job is a separate detached session. `logs/server_jobs/<arm>/job.pid` and
`record.json` identify its wrapper; `attempt_N_process.json` identifies the
training process. `attempt_N.log` has normal unbuffered output, and
`attempt_N_exit.json` records exit status. A lock inherited by the training child
prevents relaunch even if its wrapper dies. Missing/slow epoch logs never justify
a restart. DataLoader workers are nonpersistent and transforms are deterministic;
resume restores RNG after model/loader construction, retaining the shuffle order.
Early-stopped checkpoints skip additional epochs and proceed to best-test.

PVR applies affine-free layer normalization independently to each 128-dimensional
NAI/path branch, then zeros absent path directions. Its encoder can see the two
presence flags; its decoder produces only 256/512 continuous values. Reconstruction
is each branch's mean squared error against a detached normalized target, summed
over valid branches and divided by that sample's valid branch count, then averaged
over samples. Classification uses `mu`, the decoder uses standard sampled `z`, and
KL remains the elementwise mean. FIN remains trainable. The first real training
update records gradient norms in `first_update_gradients.json` for review.

The monitor uses the existing private `.codex_runtime` and the same networked
workspace-write settings as `run_codegen.sh`, with **gpt-6.1-sol / low**. It never
reads credentials. Authentication/runtime, logs, outputs and caches remain
excluded by `.gitignore`. Temporary/cache writes stay in this workspace, and the
existing `cairn_dev` Python environment is read-only. GPUs 0–3 are never assigned.
The prompt is `docs/server_monitor_prompt.md`. No training starts from reading it.

Daemon and per-check locks prevent overlap. `logs/server_monitor/daemon.log`
captures ordinary server output; timestamped check directories contain the
before/after state, Codex output and final local note. `last_check.txt`,
`next_check.txt`, `state.txt`, `check_note.json` and `actions.jsonl` retain progress.
Unchanged checks are quiet. The CLI has a 1200-second timeout affecting only the
monitor invocation. Three consecutive CLI failures stop monitoring with a recorded
blocked state; they never stop training. Restart the daemon after resolving the
recorded CLI/network issue; job records and checkpoint schedules are retained.

Completion requires **all four** stopped training schedules (epoch limit or
patience), selected `best.pt` test metrics, `test_complete.json`, matching test
configuration, and every frozen test row in predictions and the N×85 probability
export. The verifier checks probabilities, predictions, labels/row identities,
aggregate/per-class metrics and the full 85×85 confusion matrix. Exit status alone
is insufficient. Success writes `logs/server_jobs/final_summary.json` and
`logs/server_monitor/final_summary.json`, sets `state.txt=complete` and exits.

Only dead, unlocked, incomplete jobs may be recovered. The controller allows at
most three recovery attempts per arm, including recovery preparation failures.
It archives previous checkpoints/configs, appends new attempt logs, resumes exact
`last.pt` config/optimizer/update/RNG/schedule, and uses `--test-only` when training
has finished. An interrupted best-before-last publication resumes that preserved
epoch. Checkpoint writes are atomic. For a recorded OOM before **any** checkpoint,
the monitor may halve only that run's batch with `--reduce-batch`; the actual config
is recorded. A checkpointed schedule/batch cannot be reset. Irreparable failures
are recorded through `--block`, with no fabricated completion. When all remaining
runs exhaust recovery, the daemon records a failed summary and exits nonzero.

For read-only progress (no job launch):

```bash
source scripts/runtime.sh
CUDA_VISIBLE_DEVICES='' "$PYTHON_BIN" scripts/server_jobs.py --status
tail -n 5 logs/server_jobs/path_single_drugout/attempt_1.log
cat logs/server_monitor/state.txt
```

Do not call `--recover`, `--block`, or single-arm launch commands while a run is
live. The monitor uses the bounded recovery controller after inspecting actual
errors, without requesting this chat's help.
