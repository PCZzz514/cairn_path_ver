# DSN + RareDDIE implementation plan
Status: source/data inspection in progress; code generation and training have NOT started.
Workspace: /data/interns/zhangpc/cairn_new only. cairn_dev read-only.
Code generator requested: remote Codex gpt-6.1-sol, high.
User explicitly authorized copied authentication runtime on 2026-10-11. Codex gpt-6.1-sol high is running with workspace-write sandbox; auth must remain private and excluded from git.

## Approved conversation design
Retain RareDDIE CSE bond-aware TransformerConv, atom/bond pooling, transplant original DSN bidirectional InterGraphAttention; atom128 + inter64 -> projection128. NAI six relation types, local max30, difference weak relation, padding mask.
PVR mu/logvar, standard Gaussian sampling for reconstruction; mu for deterministic classification, MLP LayerNorm ->256 GELU dropout -> event logits.
Objective CE +0.1 detached-target mean reconstruction +0.001 mean KL, warmup first20% planned updates. Train FIN with downstream gradients.
Path variant adds original EmerGNN-inspired bidirectional pair-conditioned relation-gated propagation, bounded subgraphs from full cleaned DRKG independently of NAI cap30; depth4 to cover gene/pathway links, preserve relation types/inverses, explicit missing-path masks.
Read all original source before porting. No smoke tests, hashes, or excessive defensive code.

## Verified existing resources
cairn_dev/references/RareDDIE_source/RareDDIE-f658cae053af1e93eecc88ad77dc3af2e33cf527/RareDDIE/models_t_struc.py: CSE uses bond-aware TransformerConv and atom/bond pooled multiplication, combines first/last blocks 0.6/0.4.
cairn_dev/data/processed/prepare_drkg_report.json: 4,488,504 retained triples, 96,766 entities, 105 relations. 1,385,757 Compound-Compound removed; published embeddings_used=false. Still inspect FIN training provenance and actual serialized mappings.
Read-only environment candidate cairn_dev/.conda/cairn/bin/python.
Frozen splits under cairn_dev/data/experiments/reference85/{single_drugout,double_drugout}/ddi.parquet.
Dataset actual event universe85 (discussion used86); preserve actual IDs and catalog. Counts from documentation: train149048 each; single val20785/test16978; double val760/test534. Verify parquet.
User confirmed ONLY DDI SingleDrugOut and DoubleDrugOut. Do not add LINCS or synergy tasks.

## Steps
- [ ] Inspect upstream DSN-DDI and EmerGNN source; copy/clone references into this directory.
- [ ] Inspect FIN provenance, graph mappings, frozen parquet and molecular feature code.
- [ ] Generate shared model/data/train/eval implementation with remote requested model.
- [ ] Review original formula fidelity, pair batching, real gradient flow and leakage isolation.
- [ ] Full train/validation then selected-checkpoint test, no toy substitute.
- [ ] Export configs, best/last checkpoints, history, predictions, per-class and aggregate metrics.
GPU assignment: path SingleDrugOut4/DoubleDrugOut5; no-path SingleDrugOut6/DoubleDrugOut7. Never use0-3. Validation macroF1 selects checkpoint; test never selects model.
