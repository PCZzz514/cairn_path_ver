"""CPU check: rareddie_control CSE == ORIGINAL RareDDIE MVN_DDI on real ragged molecules.

The original MVN_DDI class (models_t_struc.py, retained official source under
cairn_dev, imported read-only by cairn_dev/cairn/rareddie_official85/source.py)
runs its own unmodified forward, including the discarded MergeFD line (zero
fingerprint/descriptor placeholders) and the official per-molecule bond Data.
Both CSEs share every active parameter. Compared: both pooled 0.6/0.4 outputs
and the gradients of every shared parameter. Negative control: the
accelerated_four DSN CSE with the same shared parameters must differ.
Also records the full control model inventory, its parameter names against the
accelerated_four no-path comparator, and the shapes of one real forward.
No training step, no GPU.
"""
import importlib.util
import json
from pathlib import Path
import sys
import pandas as pd
import torch
from torch_geometric.data import Batch, Data

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from accelerated_four.dsn.data import Collator as DSNCollator  # noqa: E402
from accelerated_four.dsn.model import CSE as DSNCSE, DSNRareDDI  # noqa: E402
from rareddie_control.data import Collator  # noqa: E402
from rareddie_control.model import CSE, RareDDIEControl, parameter_inventory  # noqa: E402

OFFICIAL85 = ROOT.parent / 'cairn_dev/cairn/rareddie_official85/source.py'
# EmbedMatcher.__init__: MVN_DDI([n_atom_feats, 2048, 200], 17, kge_dim, kge_dim, rel_total, [64, 64], [2, 2], 64, 0.0)
MVN_ARGS = ([55, 2048, 200], 17, 128, 128, 0, [64, 64], [2, 2], 64, 0.0)
EVIDENCE = ROOT / 'rareddie_control/evidence/cse_equivalence.json'


def official_mvn_class():
    spec = importlib.util.spec_from_file_location('rareddie_official85_source', OFFICIAL85)
    source = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(source)
    matcher = source.load_official()
    return matcher.MVN_DDI, source.official_files()


def real_rows(assets):
    """Real frozen train pairs: the first 48 rows plus rows with the smallest
    (1 atom, self-loop bond) and largest molecules in the train split."""
    frame = pd.read_parquet(ROOT / 'data/prepared/single_drugout/ddi.parquet')
    train = frame[frame.split.eq('train')]
    size = {d: assets['graphs'][d].num_nodes for d in set(train.drug_a) | set(train.drug_b)}
    smallest, largest = min(size, key=size.get), max(size, key=size.get)
    extra = train[train.drug_a.isin([smallest, largest]) | train.drug_b.isin([smallest, largest])]
    picks = pd.concat([train.iloc[:48], extra.drop_duplicates(['drug_a', 'drug_b']).iloc[:16]])
    return [(int(r.drug_a), int(r.drug_b), int(r.event_id), int(i), 0) for i, r in picks.iterrows()], size


def main():
    torch.set_num_threads(2)
    assets = torch.load(ROOT / 'data/prepared/assets.pt', map_location='cpu', weights_only=False)
    rows, size = real_rows(assets)
    graphs = assets['graphs']
    MVN_DDI, files = official_mvn_class()

    torch.manual_seed(0)
    official = MVN_DDI(*MVN_ARGS)
    for block in official.blocks:
        # Same PyG2.7 adaptation as the trained model (and official85): no select scalar in PyG2.0.4.
        block.readout.select.weight.data.fill_(1.)
        block.readout.select.weight.requires_grad_(False)
    ours = CSE()
    shared = {k.replace('block0.', 'blocks.0.').replace('block1.', 'blocks.1.'): v
              for k, v in official.state_dict().items()
              if not k.startswith('fdmer.') and '.re_shape.' not in k}
    ours.load_state_dict(shared, strict=True)  # identical active parameter sets
    official.train()
    ours.train()

    batch = Collator(assets)(rows)
    edge_data = [Batch.from_data_list([Data(x=graphs[i].edge_attr, edge_index=torch.LongTensor()) for i in batch[f'drug_{s}'].tolist()])
                 for s in ('a', 'b')]
    n = len(rows)
    triples = (batch['graph_a'], torch.zeros(n, 2048), torch.zeros(n, 200),
               batch['graph_b'], torch.zeros(n, 2048), torch.zeros(n, 200),
               torch.zeros(n, dtype=torch.long), edge_data[0], edge_data[1])
    weights = torch.randn(2, n, 128, generator=torch.Generator().manual_seed(1))

    def run(model, inputs):
        model.zero_grad(set_to_none=True)
        h, t = model(inputs)
        ((h * weights[0]).sum() + (t * weights[1]).sum()).backward()
        return h.detach(), t.detach()

    oh, ot = run(official, triples)
    ch, ct = run(ours, batch)
    official_grads = {k.replace('block0.', 'blocks.0.').replace('block1.', 'blocks.1.'): p.grad
                      for k, p in official.named_parameters() if not k.startswith('fdmer.') and '.re_shape.' not in k}
    our_grads = dict((k, p.grad) for k, p in ours.named_parameters())
    assert official_grads.keys() == our_grads.keys()
    grad_none_match = all((official_grads[k] is None) == (our_grads[k] is None) for k in our_grads)
    grad_diff = {k: float((official_grads[k] - our_grads[k]).abs().max()) for k in our_grads
                 if our_grads[k] is not None and official_grads[k] is not None}
    grad_equal = all(torch.equal(official_grads[k], our_grads[k]) for k in grad_diff)

    # Negative control: the DSN-conditioned CSE with the same shared parameters.
    torch.manual_seed(0)
    dsn = DSNCSE()
    missing, unexpected = dsn.load_state_dict(shared, strict=False)
    assert not unexpected and all('interAtt' in k or 'inter_projection' in k for k in missing)
    dsn.train()
    with torch.no_grad():
        dh, dt = dsn(DSNCollator(assets, ROOT / 'data/prepared', False)(rows))

    # Full model: inventory, names against the comparator, shapes of one real forward.
    torch.manual_seed(42)
    model = RareDDIEControl(assets)
    inventory = parameter_inventory(model)
    torch.manual_seed(42)
    comparator = DSNRareDDI(assets, {'with_path': False})
    ours_names = {k for k, _ in model.named_parameters()}
    comparator_names = {k for k, _ in comparator.named_parameters()}
    removed = sorted(comparator_names - ours_names)
    same_shapes_nai_pvr_classifier = all(
        dict(model.named_parameters())[k].shape == dict(comparator.named_parameters())[k].shape
        for k in ours_names if not k.startswith('cse.'))
    model.train()
    with torch.no_grad():
        head, tail = model.cse(batch)
        out = model(batch)

    sizes = [size[int(d)] for d in batch['drug_a'].tolist() + batch['drug_b'].tolist()]
    result = {
        'official_source_files': files,
        'reference_copy_identical_to_official': all(
            (ROOT / 'references/RareDDIE' / f).read_bytes()
            == (OFFICIAL85.parents[2] / 'references/RareDDIE_source/RareDDIE-f658cae053af1e93eecc88ad77dc3af2e33cf527/RareDDIE' / f).read_bytes()
            for f in ['models_t_struc.py', 'matcher_structure_acc_fp_neigh_VAE_struc.py']),
        'pairs': n, 'protocol_split': 'single_drugout train (real frozen rows)',
        'atoms_per_molecule': {'min': min(sizes), 'max': max(sizes)},
        'directed_bonds': {'a': int(batch['graph_a'].edge_index.shape[1]), 'b': int(batch['graph_b'].edge_index.shape[1])},
        'shared_parameter_tensors': len(shared),
        'official_inactive_not_built': sorted(k for k in official.state_dict() if k.startswith('fdmer.') or '.re_shape.' in k),
        'forward_bitwise_equal': torch.equal(oh, ch) and torch.equal(ot, ct),
        'forward_max_abs_diff': max(float((oh - ch).abs().max()), float((ot - ct).abs().max())),
        'gradient_none_pattern_equal': grad_none_match,
        'parameters_without_gradient': sorted(k for k, g in our_grads.items() if g is None),
        'gradients_bitwise_equal': grad_equal,
        'gradient_max_abs_diff': max(grad_diff.values()),
        'negative_control_dsn_cse_max_abs_diff': max(float((oh - dh).abs().max()), float((ot - dt).abs().max())),
        'model_parameter_names_removed_vs_no_path_comparator': removed,
        'model_parameter_names_added_vs_no_path_comparator': sorted(ours_names - comparator_names),
        'nai_pvr_classifier_fin_same_names_and_shapes': same_shapes_nai_pvr_classifier and all(
            k.startswith(('cse.blocks.0.interAtt', 'cse.blocks.1.interAtt', 'cse.blocks.0.inter_projection',
                          'cse.blocks.1.inter_projection')) for k in removed),
        'forward_shapes': {'cse_head': list(head.shape), 'cse_tail': list(tail.shape),
                           'logits': list(out['logits'].shape), 'rec': list(out['rec'].shape), 'kl': list(out['kl'].shape)},
        'inventory': {k: v for k, v in inventory.items() if k != 'tensors'},
    }
    result['passed'] = bool(
        result['reference_copy_identical_to_official'] and result['forward_bitwise_equal']
        and result['gradient_none_pattern_equal'] and result['gradients_bitwise_equal']
        and result['negative_control_dsn_cse_max_abs_diff'] > 1e-3
        and result['nai_pvr_classifier_fin_same_names_and_shapes'] and not result['model_parameter_names_added_vs_no_path_comparator']
        and not inventory['forbidden_name_matches'] and not inventory['dsn_inter_attention_present']
        and not inventory['inter_projection_present'] and not inventory['path_modules_present']
        and len(inventory['molecular_transformer_convs']) == 4 and inventory['fin_trainable']
        and result['forward_shapes']['logits'] == [n, 85] and inventory['pair_input'] == 256)
    EVIDENCE.parent.mkdir(parents=True, exist_ok=True)
    EVIDENCE.write_text(json.dumps(result, indent=1) + '\n')
    (EVIDENCE.parent / 'parameter_inventory_cpu.json').write_text(json.dumps(inventory, indent=1) + '\n')
    print(json.dumps(result, indent=1))
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
