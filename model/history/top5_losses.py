"""Historical Top5 loss functions; no training entry point or dataset loading.
Source and limitations: ../docs/archive.md and ../docs/research_summary.md.
"""
import torch

def wpcc(preds, y):
    """按预测名次加权的 Pearson 相关（WPCC），保留参考脚本公式。

    权重依预测名次衰减；方差围绕普通均值而不是加权均值——不能偷偷换成另一种加权
    Pearson，只增加了常数截面的数值保护。每次输入必须是单个交易日的完整有效截面。
    """
    p = preds.reshape(-1)
    y = y.reshape(-1)
    n = p.numel()
    if n < 2:
        return p.sum() * 0
    order = torch.argsort(p, descending=True, stable=True)
    w = torch.empty_like(p)
    w[order] = torch.pow(.5, torch.arange(n, device=p.device, dtype=p.dtype) / (n - 1))
    w_sum = w.sum()
    cov = (p * y * w).sum() / w_sum - (p * w).sum() * (y * w).sum() / w_sum.square()
    vp = ((p - p.mean()).square() * w).sum() / w_sum
    vy = ((y - y.mean()).square() * w).sum() / w_sum
    return -cov / (vp * vy).clamp_min(1e-16).sqrt()

BASE_WPCC = wpcc

def rankmix(pred,y):
    p=pred.reshape(-1);y=y.reshape(-1)
    z=(p-p.mean())/p.std(unbiased=False).clamp_min(1e-5)
    target=torch.softmax(y.clamp(-3,3)/.5,dim=0)
    listwise=-(target*torch.log_softmax(z/.7,dim=0)).sum()
    return BASE_WPCC(pred,y)+.25*listwise

def top5_pair_loss(pred, y):
    """WPCC plus pairwise ranking of the actual top five against the low-return tail."""
    p = pred.reshape(-1)
    target = y.reshape(-1)
    n = p.numel()
    if n < 10:
        return BASE_WPCC(pred, y)
    order = torch.argsort(target, descending=True, stable=True)
    k = min(5, n // 2)
    tail = min(256, n - k)
    top_idx = order[:k]
    tail_idx = order[-tail:]
    margin = p[top_idx].unsqueeze(1) - p[tail_idx].unsqueeze(0)
    pairwise = torch.nn.functional.softplus(-margin / .2).mean()
    return BASE_WPCC(pred, y) + .10 * pairwise

def path_relative_win_loss(pred, y):
    """Rank above-mean path outcomes over below-mean outcomes, ignoring payoff size.

    The trainer centers each daily label cross-section before calling the loss, so
    y > 0 means the executable path return beat that day's universe mean.
    """
    p = pred.reshape(-1)
    relative_win = (y.reshape(-1) > 0).to(dtype=p.dtype)
    base = BASE_WPCC(pred, relative_win)
    winners = torch.nonzero(relative_win > 0, as_tuple=False).reshape(-1)
    losers = torch.nonzero(relative_win == 0, as_tuple=False).reshape(-1)
    if p.numel() < 10 or winners.numel() == 0 or losers.numel() == 0:
        return base
    top_idx = winners[:min(5, winners.numel())]
    tail_idx = losers[-min(256, losers.numel()):]
    margin = p[top_idx].unsqueeze(1) - p[tail_idx].unsqueeze(0)
    pairwise = torch.nn.functional.softplus(-margin / .2).mean()
    return base + .10 * pairwise

def top5_boundary_pair_loss(pred, y):
    """Pair the realized top five with the next twenty names at the seat boundary."""
    p = pred.reshape(-1)
    target = y.reshape(-1)
    n = p.numel()
    if n < 10:
        return BASE_WPCC(pred, y)
    order = torch.argsort(target, descending=True, stable=True)
    k = min(5, n // 2)
    boundary = min(20, n - k)
    top_idx = order[:k]
    boundary_idx = order[k:k + boundary]
    margin = p[top_idx].unsqueeze(1) - p[boundary_idx].unsqueeze(0)
    pairwise = torch.nn.functional.softplus(-margin / .2).mean()
    return BASE_WPCC(pred, y) + .10 * pairwise
