"""
slowfast/utils/metrics.py
"""

import torch


def topk_accuracies(preds, labels, ks=(1,)):
    """
    Compute top-k accuracy for each k in ks.
    preds:  [B, C] logits or [B] predicted class indices
    labels: [B]
    """
    results = []
    if preds.ndim == 2:
        maxk = max(ks)
        _, pred_topk = preds.topk(maxk, dim=1, largest=True, sorted=True)
    else:
        pred_topk = preds.unsqueeze(1)

    for k in ks:
        correct = pred_topk[:, :k].eq(labels.view(-1, 1).expand_as(pred_topk[:, :k]))
        acc = correct.any(dim=1).float().mean().item()
        results.append(acc)
    return results
