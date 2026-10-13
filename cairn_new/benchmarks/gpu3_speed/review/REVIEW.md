# GPU3 packed cross-pair attention: review packet (for Codex)

Author of all code here: claude-opus-5-5. Codex owns review approval, status
review fields and `ready_for_host_launch`; nothing here sets them.

## What changed (see `implementation.diff`, actual `diff -u` against originals)

1. `dsn_gpu3/data.py` Collator (CPU, DataLoader workers): per pair, build the
   original `meshgrid(indexing='xy')` local edges, drop the LOCAL diagonal
   (`row != col`), append `min(n_h, n_t)` LOCAL loops `arange(m).repeat(2,1)`,
   then add the pair's head/tail node offsets; concatenate once
   (`cross_edges` becomes one `[2, sum n_h*n_t]` tensor). This is exactly what
   the installed PyG 2.7 `GATConv.forward` does per call
   (`remove_self_loops` then `add_self_loops(num_nodes=min(n_src, n_dst))`,
   loops concatenated at the end). Offsetting happens AFTER the loop rule, so
   global `row == col` edges of different pairs are ordinary edges and stay.
2. `dsn_gpu3/model.py`: `GATConv((128,128), 32, 2, add_self_loops=False)` in
   the copy only (no parameter/buffer change; state-dict names and shapes are
   identical, strict cross-loading is exercised in the evidence), and
   `CSEBlock.inter_pair_batch` makes ONE `interAtt` call on the packed batch.
   `InterGraphAttention.forward` is unchanged: elu, `t_rep` from
   `(h, t), edge_index` first, then `h_rep` from `(t, h), edge_index[[1, 0]]`.
   Since local loops `(k, k)` are symmetric, `edge_index[[1, 0]]` of the packed
   list is exactly the per-pair default rule applied to the swapped edges.
   No `ptr.tolist()` host sync and one H2D edge transfer per batch.
   GATConv dropout stays its default 0.0 (no RNG consumed, as before).
3. `dsn_gpu3/train.py`: GPU3-only guard; output restricted to
   `benchmarks/gpu3_speed/outputs`; `epoch_cap: 5` stops the loop while
   `planned_updates = epochs(100) * len(train loader)` and
   `kl_warmup_updates = ceil(0.2 * planned)` are unchanged (116500 / 23300 for
   1165 batches); no early-stop break inside the cap; no test loader, test
   export or `--test-only`; per-epoch `train_seconds`, `validation_seconds`,
   `seconds` (sum), validation accuracy, ratio vs historical 1498/1534 s;
   flushed `train_progress` line + status every 100 updates and a
   `validation` phase line (from already-synchronized running totals, no RNG). Seed, model construction and loader order are
   unchanged, so RNG/shuffle consumption is the original one.
4. Config copy: `gpu: 3`, independent `output`, `epoch_cap: 5`; everything
   else identical (diff included).

Unchanged: FIN/CSE (TransformerConvs, SAGPooling, norms)/NAI/path/PVR/
classifier/loss/data rows/split/batch 128/AdamW/FP32 (no AMP; PyTorch defaults:
TF32 matmul off; cuDNN TF32 flag default but no cuDNN convolutions; recorded in evidence).

## Evidence (`../evidence/gpu3_evidence.{json,md}`; `logs/gpu3_speed/evidence_run.log`)

`logs/gpu3_speed/run_evidence.sh` (foreground). Real seed-42 epoch-1 train
batches (first 35 x 128 rows; row indices recorded), copied parameters
(fresh seed-42 init AND the read-only production `outputs/path_single_drugout/best.pt`):

- structure on all 35 batches vs PyG's own `remove_self_loops`/`add_self_loops`
  per pair, both directions, exact order; counts of nh<nt / nh>nt pairs;
  global-diagonal valid edges kept; what a naive default GAT would break;
- attention level per CSE block on captured real inputs: OLD per-pair
  `inter_pair_batch` vs NEW packed (FP32 and FP64), outputs + input and
  parameter gradients; per-pair processed edge lists and attention
  coefficients for both directions; min alpha on global-diagonal edges;
- cross-pair isolation: gradient of each pair's outputs w.r.t. every other
  pair's inputs is exactly zero (all 128 pairs);
- whole model, train mode, identical `torch.manual_seed` before each forward
  (NAI/classifier dropout and PVR sampling live), loss = CE + 0.1 rec + beta KL:
  logits/CE/rec/KL/loss, every parameter gradient, input gradients of both
  graphs' node and bond features; eval-mode logits; OLD-repeat vs OLD as
  nondeterminism floor; seed-42 init bitwise equality and equal CPU RNG state
  after construction;
- timing on GPU3: same 35 batches in the same row order; EVERY block restarts
  from the identical fresh seed-42 parameters, a fresh AdamW and
  `torch.manual_seed(42)` (seed and update counts recorded); ABBA blocks of
  5 warmup + 30 measured steps, synchronized phases (transfer / forward /
  backward / optimizer / fullstep), separate unsynchronized fullstep blocks,
  eval forward, peak memory per block, raw per-step series, GPU3 clocks /
  power / throttle reasons and load average per block, single-process collate
  times, and contention snapshots (the four original jobs, GPU3 apps).
- gradient gate evaluated for EVERY tensor on EVERY numerical batch (G per
  batch); worst abs and worst rel aggregated independently over all
  batch x tensor entries (Codex finding 2).

Earlier evidence runs (all in `logs/gpu3_speed/evidence_run.log`): two
crashed in the evidence script itself (output width; PyG `Batch.to` mutating
stored pinned batches, fixed with a non-mutating copy); attempt 1 failed the
pure-relative gradient criterion (kept, see below); two further runs passed
all numerical checks but each had ONE NEW training block with a bimodal
contended median (0.185 s, 0.209 s; min 0.076-0.080 s) while the other NEW
block was 0.077-0.078 s. The final run (per-step series recorded) shows no
such block; isolated 2-4 s spikes hit single OLD steps instead. Treat per-step
ratios as measured under shared-host conditions.

Tolerances (max-norm relative `||new-old||_inf / ||old||_inf` per tensor):
attention forward 1e-5, attention gradients 1e-4, alpha absolute 1e-6,
FP64 packed vs per-pair 1e-10, model forward 1e-4. Model parameter
gradients: `max_abs <= 1e-3 * ||g_old||_inf + 1e-7 * G`, G = largest
parameter-gradient magnitude in the model (atol ~3e-8 here).
Rationale: FP32 unit roundoff u = 6e-8. Packing changes only summation order
(cuBLAS GEMM over different row counts with K = 128, atomic scatter of
softmax denominators / messages over up to ~100 neighbours, parameter
gradient reductions over ~2e5 packed edges instead of 128 pieces); typical
error ~sqrt(n) u gives 1e-7..3e-5 before propagation through two CSE blocks,
LayerNorm, pooling, NAI, path and PVR. The FP64 check isolates semantics from
roundoff: packed and per-pair must agree to double precision (observed ~4e-15).

Criterion revision (disclosed): attempt 1
(`evidence/attempt1_pure_relative_criterion_failed.{json,md}`) used a pure
per-tensor relative 1e-3 for model gradients and FAILED only on six tensors
whose gradients are analytically zero: TransformerConv `lin_key.bias` (a
constant key shift cancels in every query's softmax) and SAGPooling
`readout.gnn.lin_rel.bias` (softmax score with `min_score` is
shift-invariant). Their "reference" is 1e-12..5e-10 roundoff residue against
G ~0.3, and the ORIGINAL code is not reproducible on them either (OLD repeat
vs OLD rel 7.5 / 4.3). All other gradients were within 6.7e-4. The final run
reports these tensors explicitly with the OLD-repeat noise floor and applies
the atol above; everything else is unchanged. Largest remaining gradient
discrepancy: path branch (`path.linear.3.bias` 2.3e-4 rel, trained parameters),
where four multiplicative gated ReLU layers amplify 1e-7 CSE output roundoff.

## Launch (Codex, after review)

1. `bash benchmarks/gpu3_speed/tools/make_review.sh` (refresh diffs if code changed).
2. Write `logs/gpu3_speed/review_approval.json`, e.g.
   ```json
   {"approved": true, "reviewer": "codex gpt-6.1-sol",
    "reviewed_diff": "/data/interns/zhangpc/cairn_new/benchmarks/gpu3_speed/review/implementation.diff",
    "reviewed_evidence": "/data/interns/zhangpc/cairn_new/benchmarks/gpu3_speed/evidence/gpu3_evidence.json",
    "approved_at": "<ISO time>", "notes": "<review findings>"}
   ```
   The gate also requires the approval to be newer than code, diff and evidence.
3. `bash logs/gpu3_speed/launch_benchmark.sh` — refuses without approval or
   passing current GPU3 evidence, non-fresh output, a live runner, or another
   process on GPU3; otherwise starts `setsid nohup run_benchmark.sh` from an
   exiting subshell (orphaned, PPID 1 unless a subreaper exists — recorded and
   warned), writes `logs/gpu3_speed/launch.json`, waits for the training start
   line, and returns. Durable log: `logs/gpu3_speed/benchmark.log`.
4. After the five epochs (or any failure) the runner executes
   `tools/summarize.py`, writing `five_epoch_summary.{json,md}` into the
   output dir and `logs/gpu3_speed/`, and sets status `complete` or `failed`.
