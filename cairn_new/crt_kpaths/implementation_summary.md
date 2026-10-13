# crt_kpaths implementation summary (Claude, 2026-10-13) — for desktop review

Plan: `docs/crt_kpaths_implementation_plan.md`. Review round 1: `docs/crt_kpaths_review_round1.md` (items 1–10 addressed, see §5).
**No full training has been launched.** GPU use so far: Step1 parent extraction checks on GPU4 only (test split, seconds, no training). K-Paths cache preparation ran on CPU, as the plan allows.

## 1. Sources read before editing
- cRT: `references/cRT.txt` §3–5 (cRT = re-initialize and re-train W,b with class-balanced sampling for a few epochs; SGD .9, batch 512, cosine, 10 epochs). Official source f162adef: `data/ClassAwareSampler.py`, `models/DotProductClassifier.py` (plain `nn.Linear`, default init), `config/ImageNet_LT/cls_crt.yaml` (lr .2, wd 5e-4, `num_samples_cls: 4`, `num_epochs: 10`, `coslr`, `endlr 0`), `run_networks.py` (`epoch_steps = int(N/batch)`, feature model fixed, scheduler stepped at epoch start, best by validation).
- K-Paths: `references/K-Paths.txt` §3.3–3.5 and Algorithm 1 (dedupe by whole relation sequence, Yen shortest-first, one-end-absent neighbors, both-absent empty). Official bf25d24: `Kpaths_utils.py` (`find_Kpaths`, `one_node_in_graph`, `two_nodes_in_graph`: `nx.DiGraph`, `nx.shortest_simple_paths`, disconnected -> both neighbor lists, connected but >L -> []), `augmented_network_utils.py`, `subgraph_utils.py`, `get-subgraph.py`. The "no repeated relation" comment does not match the code: only whole sequences are deduplicated. The code is followed.
- Installed networkx 3.6.1 `shortest_simple_paths` (Yen + bidirectional BFS; equal-length paths in discovery order).
- Production code reused unchanged: `accelerated_four/dsn/{model,data,metrics,train}.py`, configs, `data/prepared`. Old route extractor `cairn_dsn/{paths,prepare}.py` was read for comparison only.

## 2. Exact change list
New files only, all under `crt_kpaths/` (plus logs under `logs/crt_kpaths/`). No existing file was modified. `accelerated_four`, `cairn_dsn`, `configs/`, `data/`, `outputs/`, old monitors and `../cairn_dev` are untouched (mtime scan; the only other changes during the session were `docs/crt_kpaths_review_round1.md` and `references/classifier-balancing`, both made by the reviewer, not by Claude).

| file | content |
|---|---|
| `__init__.py` | package marker |
| `env.sh` | existing read-only Python (`../cairn_dev/.conda/cairn`), caches/tmp in crt_kpaths |
| `sampler.py` | official RandomCycleIter / class_aware_sample_generator verbatim; ClassAwareSampler over actual sparse class IDs, one seeded `random.Random` |
| `crt.py` | Step1/Step3 (see §3) |
| `kpaths.py` | `Background` (DRKG + inverses, atomic facts with multiplicity, per-node-pair signatures, components, distance balls), `full_graph()`, `diverse_paths()` (official loop + 1000 budget), `Retriever` (official fallbacks, atomic assembly, graph budget, metadata) |
| `prepare.py` | cache builder (fork pool, full graph built once), `--sample` speed probe |
| `train.py` | Step2 trainer: original `main()` + authorized differences only; `review/step2_train_vs_original.diff` |
| `check_kpaths.py`, `check_sampler.py` | real-data method-consistency checks vs official code |
| `jobs.py` | status, detached dispatcher, bounded resume-only recovery, launch confirmation, Codex request drain |
| `launch.sh`, `monitor/{monitor_loop.sh,start_monitor.sh,prompt.md}` | host launch and 2400 s Codex monitor |
| `configs/step{1,3}_{single,double}_drugout.yaml`, `configs/step2_kpaths_{single,double}_drugout.yaml` | job configs |
| `cache/kpaths_k10_l4/` | K-Paths cache (`nodes/edges/node_ptr/edge_ptr/endpoints/path_mask.npy`, same layout as `data/prepared`; `metadata.parquet`, `report.json`; `sample_*` = 2000-pair speed probe) |
| `evidence/*.json`, `review/*.diff`, `README.md`, this file | review evidence |

Step2 config vs original `configs/path_*.yaml`: gpu 5/6, output, `path_cache`, and `kpaths_k/max_length/candidate_budget` replace `path_meeting_cap/arm_cap`. Every hyperparameter is identical (epochs 100, patience 15, batch 128, AdamW 1e-4/1e-4, rec .1, KL .001 warmup .2, depth 4, NAI 30, seed 42, threads/workers 2).

## 3. cRT (Step1 / Step3) — `crt.py`, `sampler.py`
- Parent: `torch.load(best.pt)`, `DSNRareDDI(assets, config)`, `load_state_dict(strict=True)`, `.eval()`, every parameter `requires_grad=False`. The model is never optimized; features are cached in `features.pt` (train/val/test `g` 256-d, parent logits, labels, frozen rows, ordered drug_a/drug_b, pair ids) together with parent checkpoint path/epoch/update/best metric/config/prepared/path cache. No hashes.
- g_ab comes from a forward_pre_hook on `model.classifier[4]` during the unmodified forward (classifier on deterministic mu; eval mode turns off dropout). Every batch asserts `allclose(last(g), logits, atol 1e-6, rtol 1e-5)` and exact argmax. The test softmax must match the parent's `test_probabilities.npz`: same rows and labels, max abs error ≤1e-5, and every top-1 identical (review item 3).
- Head: new `nn.Linear(256,85)`, default init from `torch.manual_seed(head_seed)`, saved as `heads/init_seed{s}.pt` and loaded by both arms (and every lr). CE only. SGD momentum .9, wd 5e-4, batch 512, 291 steps/epoch for both arms, 10 epochs, cosine stepped after each epoch (review item 2). cRT arm: class-aware sampler, 4 per class. Natural arm: seeded `randperm`, first 291×512. Validation fixed85 macro-F1 every epoch, strict-improvement best, no early stop.
- Step1 lr: [.01,.05,.2] at head seed 42. Choose the max of the mean of the two arms' best validation F1, ties in candidate order, validation only (`lr_selection.json`). Then seeds 43/44 at that lr (seed 42 reused). Step3 copies the Step1 `lr_selection.json` of the same protocol and never reselects.
- Test, after all selection: for each arm×seed and the parent's own head (`parent_original_head`), export metrics (Acc, Top5, fixed85 macro-F1, supported-class macro AUROC with class count, ...), 85-row per-class, `*_test_class_counts.csv` (train/val/test support, test predicted, P/R/F1; review item 6), confusion, full probabilities with frozen row IDs and drug IDs. Also a diagnostic macro-F1 over classes positive in both train and test (IDs and count listed); the primary metric stays fixed85. `summary.json` gives mean/std over 3 head seeds.
- Resume: `features.pt`, `lr_selection.json` and finished head JSONs are reused. An interrupted head reruns from its saved init with the same seeded randomness.

## 4. K-Paths (Step2 topology) — `kpaths.py`, `prepare.py`, `train.py`
- Background: 4,488,504 non-DDI DRKG facts (asserted: no Compound–Compound), each with an explicit inverse (r+105) -> 8,977,002 atomic directed relations (3 duplicate facts kept as multiplicity 2), 6,035,852 directed node pairs, 13,511 composite signatures, 96,766 nodes, 310 components. Nodes and edges are inserted in global ID order. `nx.DiGraph` only; edge `relation` = full sorted signature.
- Composite relations: official preprocessing (`augmented_network_utils.create_augmentedKG` -> `merge_duplicate_relations`) merges a node pair's parallel relations into one composite relation ID before building the DiGraph, and `get-subgraph.py` (`subgraph_utils.expand_relations`) splits selected composites back to atomic relations. Our signature tuple keeps that composite semantics, adapted to the existing DRKG atomic relation IDs (tuples formed after adding inverses), and expands selected hops back to atomic DRKG facts. This is an adaptation of the official pipeline, not a correction of it.
- Retrieval = official `two_nodes_in_graph` on the complete graph (built once, shared by 32 forked workers), plus the explicit 1000-candidate budget. After a budget stop, one peek decides exactness. Fallbacks follow the official code. "Disconnected" and "overlength" come from full-graph connected components and exact shortest distance. Selected hops expand to all atomic facts (with multiplicity) and their inverses once, plus one self loop per node. No induced edges. `path_mask` is true only when connecting paths were selected.
- Per-pair metadata (`metadata.parquet`): drug ids, FIN endpoints, in-graph flags, true shortest distance, status, fallback, candidate count, retrieved/selected counts, length counts, `is_exact`, `budget_truncated`, graph-budget reason, graph nodes/edges, selected node paths with full signature sequences, 2-hop common neighbors by type (diagnostic only), seconds, old-cache mask.
- Step2 `train.py`: the only differences from `accelerated_four/dsn/train.py:main` are in `review/step2_train_vs_original.diff`: GPU table, output root, `Collator(assets, path_cache)`, K-Paths preparation asserts in place of meeting/arm-cap asserts, `initial_state.pt` saved right after the seed-42 construction (no extra RNG draws), the test additions below, metadata.
- Step2 test additions (`crt.step2_test_additions`, after the unchanged `export_test`, never used for selection): from the exported `test_probabilities.npz`, `test_metrics.json` gains `top5_accuracy` and `diagnostic_train_and_test_positive_classes` (same `crt.evaluation` convention as the heads). Every existing field (`loss_terms`, `selected_epoch`, `selection`, fixed85 metrics, ...) is kept unchanged. `test_class_counts.csv` (85 rows) is written too. Optimization order, loss, RNG consumption, epoch plan, selection and test export are imported unchanged. No original checkpoint is loaded. The original accelerated_four path models stay the single-seed (42) end-to-end reference; we do not claim their initial state was recorded.

## 5. Review round 1 — responses
1. `log(..., lr=lr, **record)` duplicate key: fixed. Records carry `epoch_lr`; the log passes `base_lr`. Other log calls rechecked, no collisions.
2. Cosine now uses the modern order (step after each epoch); the warning filter is removed; README documents the adaptation; both arms share the schedule.
3. Parent test check now also requires every top-1 to match. Verified on the full test split (below).
4. Production uses the complete DiGraph, built once and shared by forked workers. The search view is deleted. Components and distances are kept for fallbacks and metadata.
5. The same 2000-pair sample was re-measured on the full graph. Official check: 24/24 connected pairs identical, all edge cases identical (below). The search-view success claims are withdrawn. Superseded search-view evidence: `logs/crt_kpaths/kpaths_sample_2000_searchview_superseded.log` (0.61 s/pair mean, 1/2000 truncated). The earlier check JSON was overwritten by the rerun; its result was 23/24 identical, with pair 179372 differing only in equal-length order.
6. 85-row class-count tables are added for every cRT head, the parent head, and Step2.
7. Dependencies re-verified (`jobs.py --status`): Step1 ×2 GPU4 concurrent; Step2 GPU5/GPU6; Step3 GPU7 serial, each waiting for `step1_<p>` and `step2_kpaths_<p>`. Step3 loads its own Step2 `best.pt` and that protocol's Step1 lr. Codex low every 2400 s.
8. The monitor uses `--dangerously-bypass-approvals-and-sandbox` (flag present in the installed CLI). `start_monitor.sh` confirms `codex_exit.txt == 0`, `after.json` and `next_check.txt`.
9. Step2 `test_metrics.json` now also has `top5_accuracy` and the train-and-test-positive-class diagnostic, computed from the exported probabilities with `crt.evaluation`, with existing fields preserved (§4). Checked on the real baseline exports (§6).
10. Docs and docstrings (`kpaths.py`, `README.md`, §4) now state that official preprocessing already merges parallel relations into composite relations before the DiGraph. Our tuple is described as an adaptation that keeps the composite semantics and expands to existing DRKG atomic relation IDs.

## 6. Real-data evidence (`crt_kpaths/evidence/`)
- `extraction_check_step1_{single,double}_drugout.json`, full test split through the real parents on GPU4: last-Linear recomputation error 0.0. Single: 16,978/16,978 top-1 identical, max prob error 3.5e-6. Double: 534/534, 2.9e-6. Rows and labels equal. Test extraction took 9.1 s / 1.0 s.
- `sampler_official_check.json`: official sampler (rank-relabeled real train labels, global `random.seed(42)`) vs port (sparse IDs). Index stream identical for 10 epochs × 291×512 on both protocols. Nominal length 4,091,485 for both. Every 4-group is one class. Absent train classes [47, 74]. 17,948–17,952 rows per present class over 10 epochs.
- `kpaths_official_check.json`, official functions AST-loaded and called on the complete DiGraph (signature tuples). 24/24 random connected pairs identical (node paths and signature sequences); 400 multi-relation hops; 4 paths with a repeated signature; inverse signatures reversible. One-end-absent ×2, both-absent ×1, overlength ×3 and disconnected ×2 identical. No frozen pair is disconnected, so the disconnected cases pair a real drug with a real node of another component; they yield up to 2K. Atomic expansion equals raw DRKG facts in every case. Fallback `path_mask` is always false.
- `kpaths_cache_check.json`: on all 188,105 pairs, `path_mask` == connected status ∧ 4-layer reachability in both directions; 0 mismatches, 0 fallback masks.
- `kpaths_split_coverage.json` and `cache/kpaths_k10_l4/report.json`:
  - Full build: 192 s wall on 32 workers (setup 64 s); per pair mean 0.027 s, p99 0.21 s, max 7.1 s.
  - Status: connected 187,409 (exactly the old cache's masked pair set), one-end-absent 681, overlength 14 (true distance 5 ×11, 6 ×3), both-absent 1.
  - 187,630 pairs have 10 paths. Connected pairs with <K: 170. Of these, 131 are exact (fewer than 10 distinct signature sequences within 4 hops; 127 at distance 4) and 39 hit the budget (35 at distance 4).
  - Budget-truncated rows per split: single test 2, single val 3, train 34; double val/test 0. This is not widespread (0.02%), so per the plan it is reported rather than escalated; flags are saved per pair.
  - Graph budget never reached. Max 28 nodes / 212 edges, mean 12.4 / 94.3 (the old cache averaged ~69 nodes / ~412 edges, max 130/695).
- `step2_cache_collator_check.json`: Step2 preparation asserts pass for both configs. The unchanged Collator packs a real 128-row batch from the cache (relations 0..210, multiplicity ≥1, endpoints valid).
- `step2_test_additions_check.json`: `crt.step2_test_additions` run read-only on the real baseline exports (`accelerated_four/outputs/path_*`; nothing written there). Existing fields are preserved and only the 2 keys are added. Top-5 equals an independent recomputation: single 0.92196, double 0.78277. The diagnostic matches the per-class CSV: single 71 classes, macro-F1 0.63602; double 27 classes, 0.32897. The 85-row class counts sum to the split sizes. Fixed85 stays 0.53126 / 0.10449.
- `review/step2_train_vs_original.diff`: trainer differences.

## 7. Points for desktop attention
- cRT cosine order follows review item 2 (modern order), not the literal official epoch-start call.
- Old-run initialization cannot be proven identical to Step2's replayed seed-42 initialization. Step2 saves `initial_state.pt`; both Step2 runs construct with the same seed and architecture.
- 39 budget-truncated and 131 genuinely <K pairs are flagged in metadata; nothing is truncated silently.
- Step2 recovery refuses to restart from scratch when no `last.pt` exists (crash in epoch 1). The monitor then notes it for the desktop.
- Expected time: Step1 ≈ 10–15 min per protocol (train extraction ~1.5 min, 10 heads). Step2 ≲ 100 epochs at ≤ 2 min/epoch (K-Paths graphs are ~5× smaller than the old cache). Step3 ≈ Step1 time after each Step2.

## 8. Launch (desktop to run after approval; nothing launched yet)
```
cd /data/interns/zhangpc/cairn_new
setsid nohup bash crt_kpaths/launch.sh > logs/crt_kpaths/launch.log 2>&1 < /dev/null &
```
`launch.sh` starts the detached dispatcher (`jobs.py --launch`; Step1 ×2 on GPU4 and Step2 on GPU5/6 start at once, Step3 later on GPU7). It then waits up to 2400 s until each launched job has a real update (`head_epoch` / `train_progress`) and writes `logs/crt_kpaths/launch_confirmation.json`. Finally it starts the monitor and waits for the first Codex check (`codex_exit=0`, `after.json`, `next_check.txt`, then every 2400 s). Follow-up:
```
cat logs/crt_kpaths/launch.log logs/crt_kpaths/launch_confirmation.json
source crt_kpaths/env.sh; $PYTHON_BIN crt_kpaths/jobs.py --status
cat logs/crt_kpaths/monitor/next_check.txt; ls logs/crt_kpaths/monitor/checks/*/
```
