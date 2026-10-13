# dsn_pairout_cellout — plan (author: Claude claude-opus-5-5, default effort, 2026-10-12)

Goal: DSN-DDI+RareDDIE **NO PATH** control on PairOut (GPU6) and CellOut (GPU7). This is the middle ablation
between the pure RareDDIE control (rareddie_pairout_cellout, LIVE on GPU6/7) and the full-path model
(pairout_cellout, LIVE on GPU4/5): DSN InterGraphAttention + inter_projection with packed per-pair local self loops
are KEPT; the EmerGNN path module is absent and no path arrays are loaded.

Both arms are independent fresh seed42 replicates on the SAME frozen PairOut DDI split (the CellOut DDI frame is
identical), NOT true cell-cold DDI generalization.

## Steps
1. Clone `accelerated_four/dsn/{model,data}.py` byte-for-byte to `dsn_pairout_cellout/dsn/` (+ `__init__.py`).
   cmp-verified, and transitively equal to the GPU3-verified `benchmarks/gpu3_speed/dsn_gpu3/` files. DONE
2. Put GROUPS/parameter_inventory in a new helper `inventory.py`. The cloned files are not edited. Groups are disjoint:
   dsn_inter_attention, inter_projection, cse (rest), fin_entities, fin_relations, nai, pvr, classifier. No path group. DONE
3. Adapt the latest 150-epoch rareddie_pairout_cellout train/jobs/confirm/status/env/launch/configs/monitor by
   namespace and the DSN model/data only. Configs set `with_path: false`. DONE
4. Run narrow CPU evidence (`tools/cohort_evidence.py`): source/prepared frame identity, counts 151508/18939/18938,
   classes 85/82/83, pair indices, schedule arithmetic, byte copies, prior GPU3 evidence and approvals, and a CPU-built
   inventory with no forward pass. No cache, hash, smoke run or benchmark. DONE, passed
5. Generate review diffs (`tools/make_review.sh`). DONE
6. Codex: review, then write `logs/dsn_pairout_cellout/review_approval.json`, then run
   `bash dsn_pairout_cellout/launch.sh --launch-reviewed`. That launches both arms detached and waits for the
   actual first update plus 100 updates on both. After that, run `bash dsn_pairout_cellout/monitor/start_monitor.sh`
   (gpt-6.1-sol LOW, every 2400 s, first check immediate). **Not done by Claude.**
