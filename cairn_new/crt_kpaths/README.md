# crt_kpaths — DDI cRT / K-Paths (plan: docs/crt_kpaths_implementation_plan.md)

Isolated namespace. Code, configs, cache, evidence and outputs live here; logs in
`logs/crt_kpaths/`. `accelerated_four`, `cairn_dsn`, `data/prepared` and
`../cairn_dev` are read only. Model/data/metrics are imported unchanged from
`accelerated_four.dsn` (no copy of the model body).

| file | role |
|---|---|
| `sampler.py` | port of official cRT `ClassAwareSampler` (sparse actual class IDs, seeded `random.Random`) |
| `crt.py` | Step1/Step3: parent feature extraction (forward_pre_hook on last Linear), natural/cRT heads, lr selection, test export |
| `kpaths.py` | K-Paths retrieval on the complete non-DDI DRKG DiGraph (full relation signatures, atomic expansion) |
| `prepare.py` | CPU-parallel K-Paths cache for all 188,105 ordered pairs (`cache/kpaths_k10_l4`), `--sample N` speed probe |
| `train.py` | Step2 trainer = original `main()` + K-Paths cache (diff: `review/step2_train_vs_original.diff`) |
| `check_kpaths.py`, `check_sampler.py` | method-consistency checks vs official code on real data (`evidence/`) |
| `jobs.py` | status / detached dependency dispatcher / bounded resume-only recovery |
| `launch.sh`, `monitor/` | host launch; 2400 s Codex (gpt-6.1-sol, low) monitor |

## Groups
- Step1 (GPU4, single+double concurrent): frozen original `accelerated_four/outputs/path_*/best.pt`; new Linear(256,85); cRT arm vs natural-sampling arm with identical init and budget.
- Step2 (GPU5 single, GPU6 double): end-to-end retrain on K-Paths topology from the replayed seed-42 initialization (no original checkpoint loaded), 100 epochs max, patience 15.
- Step3 (GPU7, serial): Step1 procedure on each Step2 `best.pt`, own extracted features, fresh heads, Step1's lr of the same protocol.

## Declared adaptations
- **cRT cosine order.** `run_networks.train` calls `scheduler.step()` at the start of each epoch. Its effect depends on the PyTorch version (on current PyTorch the 1st epoch already runs below base lr and the 10th at lr 0). Here CosineAnnealingLR(T_max=10, eta_min=0) is stepped after each epoch's updates (modern order): epoch e trains at `lr*(1+cos(pi*(e-1)/10))/2`, and all 10 epochs update. The natural and cRT arms use the identical schedule.
- **cRT epoch budget.** floor(149048/512)=291 steps of 512 per epoch for both arms (official `epoch_steps`). The cRT sampler's persistent cycle iterators continue across epochs. DataLoader prefetch consumption beyond the used batches is not replicated.
- **Sampler class IDs.** The official sampler assumes labels 0..C-1. Train has 83 present classes (47 and 74 absent), so the class cycle runs over the actual IDs without renumbering, and the head still outputs 85 logits. On rank-relabeled labels the official sampler gives an identical index stream (`evidence/sampler_official_check.json`).
- **lr selection.** One common lr from [.01,.05,.2] (head seed 42, validation only, mean of the two arms' best fixed85 F1, ties by candidate order). Seeds 43/44 repeat only head randomness on one end-to-end parent.
- **K-Paths L.** K=10, L=4 (official default L=3) to match the existing 4-layer path branch.
- **K-Paths relations.** Official preprocessing (`augmented_network_utils.merge_duplicate_relations`) merges a node pair's parallel relations into one composite relation before building the `nx.DiGraph`, and `get-subgraph.py` (`expand_relations`) splits selected composites back to atomic relations. We keep that composite semantics, adapted to the existing DRKG relation IDs: each directed node pair carries the sorted tuple of its distinct relations, formed after adding inverses (inverse pair = inverse tuple). Selected hops are expanded back to every atomic fact (with multiplicity) plus its inverse once, plus one self loop per node. No induced edges and no learned composite embeddings.
- **K-Paths candidate budget.** At most 1000 examined <=L node paths per pair. A budget stop is flagged (`budget_truncated`, `is_exact=False`) and never reported as "no path". Measured: 39 of 188,105 pairs.
- **K-Paths graph budget.** Old cache maxima (130 nodes / 695 edges). Whole paths only, in retrieval order. Never reached (max 28 nodes / 212 edges).
- **K-Paths fallbacks.** Official fallbacks kept: both absent -> empty; one absent -> its neighbors; disconnected -> both neighbor lists; connected but >L -> empty. Fallback topology never sets `path_mask`.

## Commands
```
bash crt_kpaths/launch.sh                                      # after desktop review
source crt_kpaths/env.sh; $PYTHON_BIN crt_kpaths/jobs.py --status
$PYTHON_BIN crt_kpaths/jobs.py --recover <job> --reason '<observed failure>'
```
