"""Method-consistency check of the cRT sampler port against the OFFICIAL code.

Official ClassAwareSampler (references/cRT-official, f162adef...) needs labels
0..C-1; on the real train labels it is run on their rank (sorted class IDs ->
0..82), with the global `random` seeded. The port runs on the raw sparse IDs
with random.Random(seed). Both draw floor(N/512)*512 indices per epoch from a
fresh iterator over persistent cycle iterators for 10 epochs.
"""
import json
from itertools import islice
from pathlib import Path
import random
from types import SimpleNamespace
import numpy as np
import pandas as pd
from .sampler import ClassAwareSampler

ROOT = Path(__file__).resolve().parents[1]
OFFICIAL = ROOT / 'references/cRT-official/data/ClassAwareSampler.py'


def main():
    official = {}
    exec(compile(OFFICIAL.read_text(), str(OFFICIAL), 'exec'), official)
    report = {'official_source': str(OFFICIAL), 'official_revision': 'f162adef63fdf592d0b8a0f6e52bc481fc08785f'}
    for protocol in ['single_drugout', 'double_drugout']:
        frame = pd.read_parquet(ROOT / f'data/prepared/{protocol}/ddi.parquet')
        labels = frame[frame.split.eq('train')].event_id.to_numpy()
        class_ids = np.unique(labels)
        steps = len(labels) // 512
        random.seed(42)
        theirs = official['ClassAwareSampler'](SimpleNamespace(labels=np.searchsorted(class_ids, labels).tolist()), 4)
        ours = ClassAwareSampler(labels.tolist(), 4, 42)
        same, counts = True, np.zeros(85, np.int64)
        for epoch in range(10):
            a = list(islice(iter(theirs), steps * 512))
            b = list(islice(iter(ours), steps * 512))
            same &= a == b
            counts += np.bincount(labels[b], minlength=85)
        groups = labels[np.asarray(b)].reshape(-1, 4)
        report[protocol] = {
            'identical_index_stream_10_epochs': bool(same), 'train_rows': len(labels), 'steps_per_epoch': steps,
            'train_class_ids': class_ids.tolist(), 'absent_train_class_ids': sorted(set(range(85)) - set(class_ids.tolist())),
            'nominal_len_official': len(theirs), 'nominal_len_port': len(ours),
            'consecutive_groups_of_4_same_class': bool((groups == groups[:, :1]).all()),
            'sampled_rows_per_present_class_min_max': [int(counts[class_ids].min()), int(counts[class_ids].max())]}
    path = ROOT / 'crt_kpaths/evidence/sampler_official_check.json'
    path.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
