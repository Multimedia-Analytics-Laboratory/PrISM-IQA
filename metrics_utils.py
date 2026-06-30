import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score, recall_score, roc_auc_score

from label_schema import EVAL_LABELS_MIN100_GLOBAL


ORDINAL_LEVELS = [0, 1, 2, 3]
THRESHOLD_LEVELS = [1, 2, 3]


def _nanmean(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0 or np.all(np.isnan(values)):
        return np.nan
    return float(np.nanmean(values))


def threshold_auc(y_true: np.ndarray, prob_levels: np.ndarray, thresholds=THRESHOLD_LEVELS):
    """
    Paper-defined threshold AUC.
    For 4 ordinal levels {0,1,2,3}, we convert the task into three binary tasks:
    - y >= 1 : "有没有问题"
    - y >= 2 : "是否至少到 severe"
    - y >= 3 : "是否到 critical"

    Each binary task uses score P(y >= t), and the final Threshold AUC is the mean
    over all thresholds that have both positive and negative examples.
    """
    y_true = np.asarray(y_true)
    prob_levels = np.asarray(prob_levels, dtype=np.float64)
    auc_by_threshold = {}

    for threshold in thresholds:
        binary_true = (y_true >= threshold).astype(np.int64)
        if np.unique(binary_true).size < 2:
            auc_by_threshold[threshold] = np.nan
            continue

        score = prob_levels[:, threshold:].sum(axis=1)
        try:
            auc_by_threshold[threshold] = float(roc_auc_score(binary_true, score))
        except Exception:
            auc_by_threshold[threshold] = np.nan

    return _nanmean(list(auc_by_threshold.values())), auc_by_threshold


def quadratic_weighted_kappa(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if np.unique(y_true).size < 2:
        return np.nan

    try:
        return float(cohen_kappa_score(y_true, y_pred, weights="quadratic", labels=ORDINAL_LEVELS))
    except Exception:
        return np.nan


def compute_per_attribute_metrics(
    gt_all: np.ndarray,
    pred_all: np.ndarray,
    prob_all: np.ndarray,
    attr_names,
    display_labels=None,
    observed_mask=None,
):
    """
    Return metric dictionaries keyed by attribute name plus an 'all' macro average.

    Inputs:
    - gt_all:   (N, A) integer ordinal ground truth in {0,1,2,3}
    - pred_all: (N, A) integer ordinal predictions in {0,1,2,3}
    - prob_all: (N, A, 4) ordinal class probabilities
    """
    gt_all = np.asarray(gt_all)
    pred_all = np.asarray(pred_all)
    prob_all = np.asarray(prob_all, dtype=np.float64)
    if observed_mask is None:
        observed_mask = np.ones(gt_all.shape, dtype=bool)
    else:
        observed_mask = np.asarray(observed_mask, dtype=bool)

    metric_names = ["acc", "auc", "qwk", "recall"]
    metrics = {name: {} for name in metric_names}
    count_dict = {}
    selected_attrs = list(display_labels) if display_labels is not None else list(EVAL_LABELS_MIN100_GLOBAL)
    selected_attr_set = set(selected_attrs)

    for attr_idx, attr_name in enumerate(attr_names):
        mask_i = observed_mask[:, attr_idx]
        gt_i = gt_all[mask_i, attr_idx]
        pred_i = pred_all[mask_i, attr_idx]
        prob_i = prob_all[mask_i, attr_idx, :]
        present_labels = np.unique(gt_i)

        count_dict[attr_name] = int(np.sum(gt_i > 0))
        if attr_name in selected_attr_set and gt_i.size > 0:
            metrics["acc"][attr_name] = float(accuracy_score(gt_i, pred_i))
            metrics["auc"][attr_name] = threshold_auc(gt_i, prob_i)[0]
            metrics["qwk"][attr_name] = quadratic_weighted_kappa(gt_i, pred_i)
            metrics["recall"][attr_name] = float(
                recall_score(gt_i, pred_i, labels=present_labels, average="macro", zero_division=0)
            )
        else:
            for metric_name in metric_names:
                metrics[metric_name][attr_name] = np.nan

    count_dict["all"] = int(gt_all.shape[0])
    for metric_name in metric_names:
        metrics[metric_name]["all"] = _nanmean(
            [metrics[metric_name][attr] for attr in selected_attrs if attr in metrics[metric_name]]
        )

    return metrics, count_dict, selected_attrs
