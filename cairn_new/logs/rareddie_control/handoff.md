Codex MEDIUM handoff — ACTIVE TRAINING / DURABLE LOW MONITOR, 2026-10-12
Implementation: Claude exact claude-opus-5-5 DEFAULT effort. Independent review/approval: Codex gpt-6.1-sol MEDIUM. Review passed; source CSE outputs/gradients match with shared parameters on real ragged molecules. No DSN inter attention/projection/path registered; four molecular TransformerConv retained, mu same MLP. Frozen splits exact.

Both fresh trainings actually started, seed42 plus same pretrained trainable FIN, full batch128 first updates finite CE4.515107/loss4.618603 and all gradient tensors finite; group gradients positive. Both passed 100 updates. Expected original last-CSE-block norm2 receives no gradient because its post-readout output is unused. RNG and inventory saved. Planned116500 updates/warmup23300. Fixed85 strict validation MacroF1 selection; 100epoch/patience15, test only after stop.

GPU6 single_drugout: trainer3096029, durable wrapper3096026 PPID1.
GPU7 double_drugout: trainer3096030, durable wrapper3096027 PPID1.
Outputs: rareddie_control/outputs/single_drugout and rareddie_control/outputs/double_drugout.
Attempt logs: logs/rareddie_control/jobs/{single_drugout,double_drugout}/attempt_1.log.

LOW gpt-6.1-sol monitor: daemon3097388 PPID1, separate session.
First check20261012T073142Z_3097388: Codex exit0, after.json persisted, both live, no recovery.
Next check2026-10-12T08:11:42Z (16:11:42 Asia/Shanghai), interval2400sec.
Monitor artifacts: logs/rareddie_control/monitor/.
Runs independently after MEDIUM/desktop conversation ends. Monitor CLI termination does not kill training. Genuine exited failures only; at most3 own-state exact-config recoveries, no fresh restart, never live/stale-log restarts.
Monitor verifier requires selected-best fixed85 checkpoint consistent with completed history and all frozen test rows/85 probabilities/parquet/per-class/confusion/metrics/test_config/completion marker; recomputes Top5 from saved probabilities. Final summary written only on verified completion, or explicit bounded-recovery exhaustion.

No other experiment/source/output/controller modified or signaled. GPU4/5 existing jobs/monitor remain untouched. Environment/cairn_dev read-only. All writes stay cairn_new; existing auth reused without copying/exposure. Training/test completion is pending; MEDIUM coordinator stops here as requested.
