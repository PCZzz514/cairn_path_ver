# pairout_cellout: review packet (for Codex)

Author of all code here: claude-opus-5-5 (default effort). Codex owns approval
(`logs/pairout_cellout/review_approval.json`); nothing here writes it. Nothing has
been launched (no training, no monitor, no model forward). All writes:
`pairout_cellout/`, `logs/pairout_cellout/`. Originals (`cairn_dsn/`, `data/prepared/`,
`accelerated_four/`, `rareddie_control/`, `configs/`, `outputs/`, `scripts/`,
`benchmarks/`) and `../cairn_dev` were only read; live original GPU4/5 jobs
(2560540/2560541) and monitor 2568426 were not signalled.

**Schedule (latest human update, overrides every earlier 100-epoch instruction for
these jobs): maximum 150 epochs, patience 15, KL warmup fraction 0.2; 151508 / 128 =
1184 batches/epoch -> 177600 planned updates, KL warmup ceil(0.2 * 177600) = 35520
(30 epochs).** Single source `dsn/train.py: SCHEDULE`; enforced by both configs, the
audit, the trainer in every mode, completion verification, recovery and launch
confirmation, so a 100-epoch state can never verify or launch.

Diffs: `review/implementation.diff` (actual `diff -u` of every file vs the file it
copies/adapts; regenerate with `bash pairout_cellout/tools/make_review.sh`),
`review/new_files.diff` (only `__init__.py` is new).

## Cohort and source identity

PairOut source `../cairn_dev/data/processed/ddi.parquet`; CellOut source
`../cairn_dev/data/experiments/reference85/cellout/ddi.parquet` is a symlink to the
same file: byte-identical, DataFrame-equal (10 columns, row order, drugs, events,
split). train 151508 / val 18939 / test 18938; 189385 rows = 189385 unique ordered
pairs. Both arms are independent fresh seed42 **same-DDI-split replicates**, not DDI
cell-cold generalization; no cell/synergy/LINCS/DepMap features. Same input and
config apart from GPU/name/output: trajectories can differ only by GPU roundoff /
nondeterministic kernels.

Leakage (both protocols): ordered, reversed, unordered pair and `pair_group`
overlap across train/val/test = 0. Drugs are shared across splits (PairOut by
design: 1509 train/val, 1508 train/test). Class support: train 85 classes (min 4),
val 82 (zero: 11, 26, 65), test 83 (zero: 50, 52); full table
`prepared/class_support.csv`.

## Preparation (`prepare.py` vs `cairn_dsn/prepare.py`)

- Frames copied byte-identically; every row: event template == original event name
  by event_id, `source_drugbank_*` contained in `drugs.parquet` source_ids of the
  global drug ID (= `bundle['drug_ids']` order). No renumbering.
- Old `data/prepared` covered 188105 pairs (single/double DrugOut union); all are
  in the new cohort. New cohort pair index = sorted unique ordered pairs (original
  rule). 188105 reused topology segments copied verbatim (1215 contiguous runs; local
  indices are pair-relative, only global offsets repacked). 1280 missing pairs
  extracted with the unchanged `RouteExtractor` (`paths.py` byte-identical to
  `cairn_dsn/paths.py`), full 4488504 non-DDI DRKG triples, meeting cap 32, arm cap 2.
  2048 evenly spaced reused pairs were re-extracted: exactly equal to old segments.
- Path coverage: 188689 / 189385 pairs with a real <=4-hop path; missing 696 (0.367%;
  the same 696 as old, all 1280 new pairs have paths). Row missing-path rates: train
  0.374%, val 0.306%, test 0.380%.
- Molecular graphs: all 1666 active drugs taken unchanged from the read-only bundle
  (55 atom / 17 bond features), all equal to the old `data/prepared` graphs (0 added).
- FIN tables and drug->FIN mapping reused (== fin.pt/bundle); six-type NAI cap30
  table re-derived with the original selection == old global table and parquet.
- `data/prepared` file sizes/mtimes snapshotted before and verified after.

## Audit (`audit.py` vs `cairn_dsn/audit.py`), all passed (`prepared/audit_report.json`)

Original every-pair checks unchanged on the new cache: real edges exist in source
DRKG with exact multiplicity, every real edge has its inverse, one explicit self
relation per node of present subgraphs, missing paths have no edges, endpoints map
to drug FIN IDs, both-direction depth-4 reachability == mask for all 189385 pairs.
Added: no Compound-Compound background facts; every reused segment (nodes, edges,
endpoints, mask, sizes) exactly equals its old segment; new pairs absent from old;
frames/columns/dtypes/order/events/global IDs/pair indices exact for both
protocols; split leakage 0; graphs == bundle; FIN/NAI equal; configs (exact key set, so no
planned/warmup override can be added); schedule
151508/128 -> 1184 batches, 150 epochs -> 177600 updates, KL warmup ceil(0.2*177600) = 35520;
unchanged copies byte-identical; original `data/prepared` unchanged.

## Trainer (`dsn/train.py` vs `accelerated_four/dsn/train.py`)

model.py/data.py/metrics.py byte-identical to the reviewed GPU3/accelerated_four
copies (packed attention, six NAI cap30, trainable pretrained FIN, EmerGNN 4-hop,
CSE 128/64 x2, PVR 128 mu classifier, loss CE + .1 rec + .001 KL warm 20%, AdamW
1e-4/1e-4, batch 128, threads 2, workers 2 nonpersistent). Changes only:
1. Namespace/output containment, `GPU = {pairout: 6, cellout: 7}`, `with_path` must be true.
2. `--fresh | --resume | --test-only` (rareddie_control pattern); no batch override;
   `--fresh` needs empty output; non-fresh loads only this namespace's own checkpoint;
   `initial_rng.pt` on fresh.
3. `preparation_gate`: prepared must be `pairout_cellout/prepared`; same namespace;
   every audit check true; audit newer than every prepared file; per-protocol exactness;
   caps/depth equal; loader batches == audited 1184; fresh planned/warmup == audited;
   every mode refuses a config whose epochs/patience/KL fraction/planned/warmup differ
   from `SCHEDULE` (150/15/0.2/177600/35520).
4. `first_update.json` (actual first full-training update): CE/rec/KL/beta/loss,
   logits finite, gradient norms of fin_entities/fin_relations/cse/nai/path/pvr/classifier,
   all-finite flag, trainable params without gradient (expected only
   `cse.blocks.1.norm2.*`, per GPU3 evidence), path batch nodes/edges/present directions.
5. Report-only `top5_accuracy` in test metrics; inventory adds groups, state_dict keys, frozen.
Unchanged: seed/RNG order, loop, strict `>` fixed85 val macroF1 selection, patience 15,
val accuracy, flushed first/every-100 progress with epoch_seconds,
best.pt-before-last.pt, test only after selection, all exports.

## Controller/monitor (vs `rareddie_control`)

Paths/arms/namespace substituted. `verify_exports` additionally requires
with_path/batch128/seed42, the exact 150-epoch schedule as a controller literal
`AUTHORIZED_SCHEDULE` (150/15/0.2/177600/35520, also asserted equal to the trainer's
`SCHEDULE`) in best/last/test configs and the run's `config.yaml`, every history
epoch/last/best `update == epoch * 1184`, and per-row path-presence columns == npz
path_mask; recovery refuses a checkpoint (or trainer constant) with any other schedule.
A 100-epoch (118400/23680) state therefore can never verify as complete.
Launch gate: Codex approval newer than all code/diff/audit, audit passed and newer
than all `.py`, review diff present, GPU6/7 idle, no records/outputs. Recovery:
<=3, only exited failures with own checkpoint, `--resume`/`--test-only`, no fresh
restart, no batch change. `confirm_launch` requires fresh seed42 path start line with
1184 batches, 150 epochs, patience 15, 177600/35520, first-update all-group finite nonzero gradients incl. path,
13522485/13522483 parameters (accelerated_four path inventory) and packed/path keys,
plus a 100-update progress line; `schedule_ok` additionally requires the run's written
`config.yaml` schedule and the ACTUAL KL beta of the first and 100-update lines to equal
.001 * update / 35520 (a 23680 warmup would fail). Monitor: detached (PPID 1), first check immediate,
then every 2400 s, gpt-6.1-sol LOW, until both selected-best exports verify; no
notifications.

## 150-epoch follow-up verification (claude-opus-5-5, separate authorized follow-up)

Verified the 150 change already applied, then strengthened schedule validation
(`jobs.py`, `confirm_launch.py`, `audit.py`; see `review/implementation.diff`). No
architecture/data/preparation/loss/metric change; preparation not rerun; full real audit
rerun after the last Python edit (passed). The only remaining `100` in current code is the
progress interval (`PROGRESS_EVERY = 100`, `progress_100`); 118400/23680 appear only as
explicitly superseded history. `fin.completed_epoch: 100` in `preparation_report.json` is
the pre-existing FIN pretraining record, not this schedule.
