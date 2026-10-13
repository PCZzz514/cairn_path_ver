# Claude implementation handoff to Codex (accelerated_four)

Author: claude-opus-5-5, default effort. NOTHING LAUNCHED. No approval written.
Originals (PIDs 2560540-2560543, monitor 2568426) live and untouched; original
cairn_dsn/, configs/, outputs/, scripts/, docs/ only read.

Review packet: `accelerated_four/review/REVIEW.md` (all files, every trainer
change, snapshot procedure, evidence, controller, monitor, limitations).
Diffs: `accelerated_four/review/implementation.diff`, `accelerated_four/review/new_files.diff`
(`bash accelerated_four/tools/make_review.sh` regenerates).

Code: `accelerated_four/dsn/{train,model,data,metrics,__init__}.py`
(model/data = GPU3-reviewed copies unchanged; metrics unchanged; train = production
trainer + containment, whitelisted metadata, start/first-step/100-update lines,
validation accuracy), `accelerated_four/configs/<arm>.yaml` (output only),
`accelerated_four/jobs.py`, `accelerated_four/launch_four.sh`, `accelerated_four/env.sh`,
`accelerated_four/status.py`, `accelerated_four/tools/{snapshot,evidence,summarize_evidence,confirm_launch}.py`,
`accelerated_four/tools/{run_evidence,make_review}.sh`,
`accelerated_four/monitor/{monitor_loop.sh,start_monitor.sh,drain_recovery_requests.py,notify.py,prompt.md}`.

Snapshots (run 11:59, `accelerated_four/outputs/<arm>/snapshot/provenance.json`; all coherent
on first read, exact state equivalence, config diff only output/generator/namespace/optimization):

| arm | GPU | resume from epoch | update | best epoch | best val macroF1(85) | bad/patience |
|---|---|---|---|---|---|---|
| path_single_drugout | 4 | 29 -> 30 | 33785 | 20 | 0.569643 | 9/15 |
| path_double_drugout | 5 | 33 -> 34 | 38445 | 28 | 0.123709 | 5/15 |
| no_path_single_drugout | 6 | 30 -> 31 | 34950 | 16 | 0.566757 | 14/15 |
| no_path_double_drugout | 7 | 32 -> 33 | 37280 | 24 | 0.125956 | 8/15 |

Evidence: `accelerated_four/evidence/summary.md` (+ `<arm>.json`, log
`logs/accelerated_four/evidence_run.log`): ALL FOUR PASSED. CPU (all arms): strict
config/model/AdamW load exact, OLD/NEW construction + RNG equal, schedule 116500/23300,
steps == update, RNG restore after construction, next-epoch permutation OLD == NEW and
real worker DataLoader first batch == head, collator exact both directions/local loops/
unequal pairs. GPU6 (no_path single) / GPU7 (no_path double), snapshot last.pt and best.pt,
3 real next-epoch batches: forward worst rel 6.5e-7 (abs 1.5e-5 ~ OLD-repeat 1.3e-5),
parameter-gradient gate (GPU3 thresholds) passed for every tensor, worst rel above atol 1.1e-5.
No new timing campaign (reviewed GPU3: 14.62x step, 114.09 s/epoch; contention caveat).

Commands (after Codex writes `logs/accelerated_four/review_approval.json` with
"approved": true and a reviewer naming Codex; the gate needs it newer than code/diff/evidence):

```bash
cd /data/interns/zhangpc/cairn_new
bash accelerated_four/launch_four.sh --launch-reviewed   # 4 detached PPID1 jobs; waits <=30 min for start/first-step/100-update lines -> logs/accelerated_four/launch.json
source accelerated_four/env.sh; CUDA_VISIBLE_DEVICES='' "$PYTHON_BIN" accelerated_four/jobs.py --status
CUDA_VISIBLE_DEVICES='' "$PYTHON_BIN" accelerated_four/tools/confirm_launch.py   # re-check progress + originals
bash accelerated_four/monitor/start_monitor.sh           # refuses until all four show real progress
cat logs/accelerated_four/monitor/monitor.pid logs/accelerated_four/monitor/next_check.txt
```

State: jobs `logs/accelerated_four/jobs/<arm>/{record.json,attempt_N.log,attempt_N_process.json,attempt_N_exit.json}`;
monitor `logs/accelerated_four/monitor/{monitor.pid,last_check.txt,next_check.txt,state.txt,check_note.json,notifications.log,checks/}`.
