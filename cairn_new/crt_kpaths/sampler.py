"""Port of the official cRT class-aware sampler.

Source: references/cRT-official/data/ClassAwareSampler.py (facebookresearch/
classifier-balancing f162adef63fdf592d0b8a0f6e52bc481fc08785f). RandomCycleIter
and class_aware_sample_generator are unchanged except that the shared global
`random` module is one seeded random.Random stream. The class cycle runs over
the ACTUAL train class IDs (sparse dict, never renumbered); the head still
outputs all 85 logits. Nominal length = max class count * present classes, as
official; the trainer draws floor(Ntrain/batch) batches per epoch from the
persistent iterators (official epoch_steps), not this oversampled length.
"""
import random


class RandomCycleIter:

    def __init__(self, data, rng):
        self.data_list = list(data)
        self.length = len(self.data_list)
        self.i = self.length - 1
        self.rng = rng

    def __iter__(self):
        return self

    def __next__(self):
        self.i += 1
        if self.i == self.length:
            self.i = 0
            self.rng.shuffle(self.data_list)
        return self.data_list[self.i]


def class_aware_sample_generator(cls_iter, data_iter_list, n, num_samples_cls=1):
    i = 0
    j = 0
    while i < n:
        if j >= num_samples_cls:
            j = 0
        if j == 0:
            # num_samples_cls consecutive draws from ONE class iterator.
            temp_tuple = next(zip(*[data_iter_list[next(cls_iter)]] * num_samples_cls))
            yield temp_tuple[j]
        else:
            yield temp_tuple[j]
        i += 1
        j += 1


class ClassAwareSampler:

    def __init__(self, labels, num_samples_cls, seed):
        rng = random.Random(seed)
        cls_data_list = {}
        for i, label in enumerate(labels):
            cls_data_list.setdefault(int(label), []).append(i)
        self.class_ids = sorted(cls_data_list)
        self.class_iter = RandomCycleIter(self.class_ids, rng)
        self.data_iter_list = {c: RandomCycleIter(cls_data_list[c], rng) for c in self.class_ids}
        self.num_samples = max(len(x) for x in cls_data_list.values()) * len(cls_data_list)
        self.num_samples_cls = num_samples_cls

    def __iter__(self):
        return class_aware_sample_generator(self.class_iter, self.data_iter_list,
                                            self.num_samples, self.num_samples_cls)

    def __len__(self):
        return self.num_samples
