import torch
from sklearn.metrics import precision_recall_curve, auc

def compute_binary_iou(predictions,
                       targets,
                       threshold=0.5,
                       reduction="mean",
                       eps=1e-6):

    if predictions.dim() == 4:
        if predictions.shape[1] == 1:
            preds = (torch.sigmoid(predictions) > threshold)
            preds = preds.squeeze(1)
        else:
            raise ValueError(
                "Binary IoU expects 1 channel predictions"
            )
    else:
        preds = predictions.bool()

    if targets.dim() == 4:
        targets = targets.squeeze(1)

    targets = targets.bool()

    preds = preds.reshape(preds.shape[0], -1)
    targets = targets.reshape(targets.shape[0], -1)

    inter = (preds & targets).sum(dim=1).float()
    union = (preds | targets).sum(dim=1).float()

    iou = inter / (union + eps)

    empty_mask = union == 0
    iou[empty_mask] = 1.0

    if reduction == "none":
        return iou

    elif reduction == "mean":
        return iou.mean()

    else:
        raise ValueError(f"Unknown reduction: {reduction}")
    

def pr_auc_score(pred_logits: torch.Tensor,
                 targets: torch.Tensor,
                 reduction: str = "mean"):

    probs = torch.sigmoid(pred_logits).detach().cpu()

    if targets.ndim == 4:
        targets = targets.squeeze(1)

    targets = targets.detach().cpu()

    B = probs.shape[0]

    probs_flat = probs.reshape(B, -1).numpy()
    targets_flat = targets.reshape(B, -1).numpy()

    scores = []

    for b in range(B):
        precision, recall, _ = precision_recall_curve(
            targets_flat[b],
            probs_flat[b]
        )

        pr_auc = auc(recall, precision)
        scores.append(pr_auc)

    scores = torch.tensor(scores)

    if reduction == "none":
        return scores

    elif reduction == "mean":
        return scores.mean().item()

    else:
        raise ValueError(f"Unknown reduction: {reduction}")
    

def metrix(pred_logits: torch.Tensor,
           targets: torch.Tensor,
           threshold: float = 0.5,
           reduction: str = "mean",
           eps: float = 1e-7):

    probs = torch.sigmoid(pred_logits)

    preds = (probs > threshold).float()

    if targets.ndim == 3:
        targets = targets.unsqueeze(1)

    targets = targets.float()

    preds = preds.reshape(preds.shape[0], -1)
    targets = targets.reshape(targets.shape[0], -1)

    tp = (preds * targets).sum(dim=1)
    fp = (preds * (1 - targets)).sum(dim=1)
    fn = ((1 - preds) * targets).sum(dim=1)

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)

    f1 = 2 * precision * recall / (precision + recall + eps)

    if reduction == "none":
        return f1, precision, recall
    elif reduction == "mean":
        return f1.mean().item(), precision.mean().item(), recall.mean().item()

    else:
        raise ValueError(f"Unknown reduction: {reduction}")