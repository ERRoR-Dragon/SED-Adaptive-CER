"""The scene-level budget predictor and its supervision for frozen SED."""

import copy
import math

import torch
from torch import nn
from torch.nn import functional as F


BUDGETS = (4, 8, 16, 32, None)  # None dispatches to the dense decoder.
BUDGET_NAMES = ("k4", "k8", "k16", "k32", "all")
FEATURE_NAMES = (
    "mean_top1_score", "std_top1_score", "mean_top2_gap", "std_top2_gap",
    "near_tie_fraction", "mean_normalized_entropy", "log_winning_classes",
    "dominant_class_share", "log_effective_winning_classes", "log_vocabulary_size",
)


def scene_features(logits):
    """Summarize the first CER logits [1, classes, height, width]."""
    if logits.ndim != 4 or logits.shape[0] != 1:
        raise ValueError("CER features require a batch of one image.")
    classes = logits.shape[1]
    scores = logits[0].float().flatten(1)
    top, indices = scores.topk(min(2, classes), dim=0)
    confidence = top[0].sigmoid()
    gap = confidence - top[-1].sigmoid()
    log_prob = scores.log_softmax(dim=0)
    entropy = -(log_prob.exp() * log_prob).sum(dim=0).mean()
    counts = scores.new_zeros(classes)
    counts.scatter_add_(0, indices[0], torch.ones_like(confidence))
    shares = counts / confidence.numel()
    spatial_entropy = -torch.special.xlogy(shares, shares).sum()
    return torch.stack((
        confidence.mean(), confidence.std(unbiased=False),
        gap.mean(), gap.std(unbiased=False), (gap < 0.05).float().mean(),
        entropy / math.log(max(2, classes)), (counts > 0).sum().float().log1p(),
        shares.max(), spatial_entropy.exp().log1p(), scores.new_tensor(math.log1p(classes)),
    ))


def heuristic_from_features(features):
    unique = features[6].expm1().round()
    count = int((unique + (8 * features[5]).ceil()).clamp(4, 32).item())
    return next(k for k in BUDGETS[:-1] if k >= count)


def heuristic_budget(logits):
    return heuristic_from_features(scene_features(logits))


def oracle_index(scores, tolerance):
    """Smallest k whose ground-truth image IoU is within tolerance of dense."""
    dense = scores[-1]
    return next(i for i, score in enumerate(scores) if score + tolerance >= dense)


class BudgetHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("center", torch.zeros(len(FEATURE_NAMES)))
        self.register_buffer("scale", torch.ones(len(FEATURE_NAMES)))
        self.register_buffer("active", torch.ones(len(FEATURE_NAMES), dtype=torch.bool))
        self.layers = nn.Sequential(nn.Linear(len(FEATURE_NAMES), 32), nn.ReLU(), nn.Linear(32, len(BUDGETS)))

    def forward(self, features):
        return self.layers(self.active * (features - self.center) / self.scale)

    @torch.inference_mode()
    def budget_from_features(self, features):
        with torch.autocast(features.device.type, enabled=False):
            return BUDGETS[int(self(features).argmax().item())]

    def budget(self, logits):
        return self.budget_from_features(scene_features(logits))


def fit_head(train_features, train_labels, dev_features, dev_labels, seed):
    """Fit the CPU head and select its checkpoint on a disjoint development split."""
    torch.manual_seed(seed)
    shuffle = torch.Generator().manual_seed(seed)
    model = BudgetHead()
    model.center.copy_(train_features.mean(dim=0))
    std = train_features.std(dim=0, unbiased=False)
    model.active.copy_(std > 1e-6)
    model.scale.copy_(torch.where(model.active, std, 1.0))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.001)
    best_loss = float("inf")
    for epoch in range(1, 81):
        for batch in torch.randperm(len(train_labels), generator=shuffle).split(32):
            optimizer.zero_grad(set_to_none=True)
            F.cross_entropy(model(train_features[batch]), train_labels[batch]).backward()
            optimizer.step()
        with torch.no_grad():
            dev_logits = model(dev_features)
            loss = F.cross_entropy(dev_logits, dev_labels).item()
            if loss < best_loss:
                best_loss = loss
                best = copy.deepcopy(model.state_dict())
                best_epoch = epoch
                accuracy = (dev_logits.argmax(1) == dev_labels).float().mean().item()
    model.load_state_dict(best)
    return model.eval(), {"epoch": best_epoch, "dev_loss": best_loss, "dev_accuracy": accuracy}
