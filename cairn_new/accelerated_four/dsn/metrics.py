"""All classification averages specify the frozen 85-class universe."""
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, cohen_kappa_score,
                             confusion_matrix, f1_score, matthews_corrcoef,
                             precision_recall_fscore_support, roc_auc_score)

LABELS = np.arange(85)


def classification_metrics(y, probabilities):
    prediction = probabilities.argmax(1)
    precision, recall, f1, support = precision_recall_fscore_support(
        y, prediction, labels=LABELS, zero_division=0)
    records = []
    for label in LABELS:
        truth = y == label
        auc = float(roc_auc_score(truth, probabilities[:, label])) if truth.any() and not truth.all() else None
        ap = float(average_precision_score(truth, probabilities[:, label])) if truth.any() else None
        records.append({'event_id': int(label), 'precision': float(precision[label]),
                        'recall': float(recall[label]), 'f1': float(f1[label]), 'support': int(support[label]),
                        'predicted': int((prediction == label).sum()), 'roc_auc_ovr': auc, 'average_precision_ovr': ap})
    aucs = [r['roc_auc_ovr'] for r in records if r['roc_auc_ovr'] is not None]
    aps = [r['average_precision_ovr'] for r in records if r['average_precision_ovr'] is not None]
    metrics = {'rows': len(y), 'accuracy': float(accuracy_score(y, prediction)),
               'macro_f1_85': float(f1.mean()), 'macro_precision_85': float(precision.mean()),
               'macro_recall_85': float(recall.mean()),
               'weighted_f1': float(f1_score(y, prediction, labels=LABELS, average='weighted', zero_division=0)),
               'micro_f1': float(f1_score(y, prediction, labels=LABELS, average='micro', zero_division=0)),
               'mcc': float(matthews_corrcoef(y, prediction)), 'kappa': float(cohen_kappa_score(y, prediction, labels=LABELS)),
               'classes_present': int((support > 0).sum()),
               'macro_roc_auc_supported_classes': float(np.mean(aucs)) if aucs else None,
               'roc_auc_classes': len(aucs),
               'macro_ap_supported_classes': float(np.mean(aps)) if aps else None,
               'ap_classes': len(aps)}
    return metrics, pd.DataFrame(records), confusion_matrix(y, prediction, labels=LABELS)


def validation_selection_metric(y, probabilities):
    return float(f1_score(y, probabilities.argmax(1), labels=LABELS, average='macro', zero_division=0))
