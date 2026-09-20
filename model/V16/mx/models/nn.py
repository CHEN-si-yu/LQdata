"""神经网络头（PyTorch）—— 目前唯一能**直接对策略目标求梯度**的模型族。

## 为什么值得单开一个模型族

线性和树的损失都得写成"每行一个预测、每行一个目标"的形状，所以它们只能通过
**样本权重**（外生的、固定的）间接表达"我关心头部"。神经网络没有这个限制：
整个当日截面在一个 batch 里，可以写任意**跨股票**的损失，autograd 负责求导。

于是 `L_top = −Σ_i p_i·z_i`（`p = softmax(s/τ)`，见 `mx/losses.py` 的推导）可以原样落地 ——
这是"IC 与 top_ret 耦合"最直接的表达：同一组参数，`z` 取秩高斯时它逼近排序质量，
`z` 取真实收益率时它逼近组合收益。

## 三个设计决定

1. **按"日"分批，不按行分批。** 逐日 softmax 和逐日 IC 都需要当天的**完整**截面；
   随机抽 rows 会把不同天的股票混进同一个 softmax 分母，目标就废了。
   `batch_days` = 每批几个交易日。
2. **混合损失**：`L = w_mse·加权MSE + w_ic·(1 − 平均IC) + w_top·L_top`。
   三个系数由单元声明、在榜单上比 —— 哪个有用是排出来的。
3. **`top_target` 可选 `z`（秩高斯）或 `ret`（真实收益）。**
   前者是"把头部排对"，后者是"把头部买对"，两者未必等价 —— 这正是要测的东西。

## ★ 设备

一律走 `resolve_device(cfg)`，**代码里不出现硬编码设备**（单元③ 硬约束 #5）。
CPU 上能跑通、换到有卡的机器同一份代码吃 GPU。
"""
from __future__ import annotations

import os

import numpy as np

from ..labels import day_bounds
from .base import ModelBase, register, resolve_device

# ★ 必须在任何 CUDA 算子之前设：让 cuBLAS 用确定性算法，
#   否则"同种子两次训练逐位一致"这条纪律在 GPU 上不成立（cuBLAS 默认启发式选核）。
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


class _MLP:
    """延迟构造 torch 模块（本文件在没有 torch 的机器上也能被 import）。"""

    @staticmethod
    def build(n_in: int, hidden: list[int], dropout: float, norm: str):
        import torch.nn as nn

        layers: list = []
        prev = n_in
        for h in hidden:
            layers.append(nn.Linear(prev, int(h)))
            if norm == "batch":
                layers.append(nn.BatchNorm1d(int(h)))
            elif norm == "layer":
                layers.append(nn.LayerNorm(int(h)))
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout(float(dropout)))
            prev = int(h)
        layers.append(nn.Linear(prev, 1))
        return nn.Sequential(*layers)


@register
class NNModel(ModelBase):
    type = "nn"
    fill_mode = "nn"

    # ------------------------------------------------------------------ 工具
    def _loss(self, torch, pred, y, raw, yv_ok, w):
        """混合损失 —— 三个分项按参数加权求和。"""
        p = self.params
        total = pred.new_zeros(())
        parts: dict[str, float] = {}

        w_mse = float(p.get("w_mse", 1.0))
        if w_mse > 0:
            sq = (pred - y) ** 2
            mse = ((sq * w).sum() / w.sum().clamp_min(1e-9)) if w is not None else sq.mean()
            total = total + w_mse * mse
            parts["mse"] = float(mse.detach())

        w_ic = float(p.get("w_ic", 0.0))
        if w_ic > 0 and yv_ok is not None:
            ic = _batch_ic(pred, y, yv_ok)
            if ic is not None:
                total = total + w_ic * (1.0 - ic)
                parts["ic"] = float(ic.detach())

        w_top = float(p.get("w_top", 0.0))
        if w_top > 0:
            tau = float(p.get("tau", 0.05))
            tgt = raw if str(p.get("top_target", "z")) == "ret" else y
            lt = _batch_softmax_top(pred, tgt, yv_ok, tau)
            if lt is not None:
                total = total + w_top * lt
                parts["top"] = float(lt.detach())

        # ★ 新增①：软秩 RankIC —— 显式地对"名次"给梯度（与 w_ic 的 Pearson 互补）
        w_rk = float(p.get("w_rankic", 0.0))
        if w_rk > 0 and yv_ok is not None:
            rk = _batch_soft_rankic(pred, y, yv_ok,
                                    tau_frac=float(p.get("soft_rank_tau", 0.05)),
                                    max_n=int(p.get("soft_rank_max", 2000)))
            if rk is not None:
                total = total + w_rk * (1.0 - rk)
                parts["rankic"] = float(rk.detach())

        # ★ 新增②：ListNet@top-k —— "把标签最高的 k 只排最前"，不关心头部内部幅度
        w_ln = float(p.get("w_listnet", 0.0))
        if w_ln > 0 and yv_ok is not None:
            ln = _batch_listnet_topk(pred, raw if raw is not None else y, yv_ok,
                                     k=int(p.get("listnet_k", 20)),
                                     tau_frac=float(p.get("listnet_tau", 0.10)))
            if ln is not None:
                total = total + w_ln * ln
                parts["listnet"] = float(ln.detach())

        self._last_parts = parts
        return total

    def _to_tensor_x(self, torch, Xb):
        t = torch.from_numpy(Xb).to(self.device_, non_blocking=True)
        return t.to(torch.float32)

    # ------------------------------------------------------------------ 训练
    def fit(self, X, y, Xv=None, yv=None, *, sample_weight=None,
            day=None, day_v=None, raw=None, raw_v=None, **kw):
        import torch

        if day is None:
            raise ValueError("nn 必须拿到 day —— 逐日 softmax 与逐日 IC 都要用当日完整截面")
        p = self.params
        self.device_ = resolve_device(self.cfg)
        torch.manual_seed(self.seed)
        np.random.seed(self.seed % (2 ** 31))
        if self.device_ == "cuda":
            torch.cuda.manual_seed_all(self.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        day = np.asarray(day)
        bnd = day_bounds(day)
        n_day = bnd.size - 1
        self.net_ = _MLP.build(int(X.shape[1]), list(p.get("hidden", [256, 64])),
                               float(p.get("dropout", 0.2)), str(p.get("norm", "batch")))
        self.net_.to(self.device_)
        opt = torch.optim.AdamW(self.net_.parameters(),
                                lr=float(p.get("lr", 1e-3)),
                                weight_decay=float(p.get("weight_decay", 1e-4)))
        epochs = int(p.get("epochs", 30))
        batch_days = int(p.get("batch_days", 16))
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=float(p.get("lr", 1e-3)), total_steps=max(1, epochs * max(1, n_day // batch_days)),
            pct_start=0.15)
        use_amp = bool(p.get("amp", True)) and self.device_ == "cuda"
        rng = np.random.default_rng(self.seed)

        yt = torch.from_numpy(np.asarray(y, dtype=np.float32))
        wt = (torch.from_numpy(np.asarray(sample_weight, dtype=np.float32))
              if sample_weight is not None else None)
        rawt = torch.from_numpy(np.asarray(raw, dtype=np.float32)) if raw is not None else None

        best, best_state, best_ep, bad = float("inf"), None, 0, 0
        patience = int(p.get("patience", 5))
        self.history_ = []
        for ep in range(1, epochs + 1):
            self.net_.train()
            order = rng.permutation(n_day)
            run, nb = 0.0, 0
            for s in range(0, n_day, batch_days):
                days = np.sort(order[s:s + batch_days])
                rows = np.concatenate([np.arange(bnd[d], bnd[d + 1]) for d in days])
                if rows.size == 0:
                    continue
                Xb = self._to_tensor_x(torch, X[rows])
                yb = yt[rows].to(self.device_)
                rb = rawt[rows].to(self.device_) if rawt is not None else None
                wb = wt[rows].to(self.device_) if wt is not None else None
                db = torch.from_numpy(day[rows].astype(np.int64)).to(self.device_)
                ok = _day_bounds_torch(torch, db)
                opt.zero_grad(set_to_none=True)
                ctx = (torch.autocast("cuda", dtype=torch.float16) if use_amp
                       else _null_ctx(torch))
                with ctx:
                    pred = self.net_(Xb).squeeze(-1)
                    loss = self._loss(torch, pred, yb, rb, ok, wb)
                    # ★ 新增③：R-Drop —— 同一样本前向两次（dropout 掩码不同），
                    #   逼两次输出一致。**只加正则、不改目标**：它不看标签，
                    #   所以不违反铁律二（不引入任何跨样本/跨日的统计量，
                    #   逐元素 MSE 与行序、日期无关）。
                    w_rd = float(p.get("rdrop", 0.0))
                    if w_rd > 0:
                        pred2 = self.net_(Xb).squeeze(-1)
                        rd = 0.5 * ((pred - pred2) ** 2).mean() \
                            + 0.5 * ((pred2 - pred) ** 2).mean()
                        loss = loss + w_rd * rd
                        self._last_parts["rdrop"] = float(rd.detach())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net_.parameters(),
                                               float(p.get("clip_grad", 5.0)))
                opt.step()
                if ep > 0:
                    try:
                        sched.step()
                    except Exception:            # noqa: BLE001  (步数估不准就退化成常数 lr)
                        pass
                run += float(loss.detach())
                nb += 1

            vloss = self._valid_loss(torch, Xv, yv, day_v, raw_v) if Xv is not None and len(Xv) else run / max(1, nb)
            self.history_.append({"epoch": ep, "train": run / max(1, nb), "valid": vloss,
                                  "parts": self._last_parts})
            if vloss < best - 1e-6:
                best, best_ep, bad = vloss, ep, 0
                best_state = {k: v.detach().clone() for k, v in self.net_.state_dict().items()}
            else:
                bad += 1
                if bad >= patience:
                    break

        if best_state is not None:
            self.net_.load_state_dict(best_state)
        self.best_epoch_ = best_ep
        self.valid_loss_ = best
        return self

    def _valid_loss(self, torch, Xv, yv, day_v, raw_v):
        """验证损失：只算 MSE 那一项（稳定的早停信号），不算 IC/top 那种带日结构的项。"""
        if Xv is None or len(Xv) == 0:
            return float("inf")
        self.net_.eval()
        day_v = np.asarray(day_v)
        bnd = day_bounds(day_v)
        yv_t = torch.from_numpy(np.asarray(yv, dtype=np.float32))
        tot, n = 0.0, 0
        bs = int(self.params.get("eval_batch_days", 64))
        with torch.no_grad():
            for s in range(0, bnd.size - 1, bs):
                rows = np.concatenate([np.arange(bnd[d], bnd[d + 1])
                                       for d in range(s, min(s + bs, bnd.size - 1))])
                Xb = self._to_tensor_x(torch, Xv[rows])
                pred = self.net_(Xb).squeeze(-1).float()
                tot += float(((pred - yv_t[rows].to(self.device_)) ** 2).mean())
                n += 1
        return tot / max(1, n)

    # ------------------------------------------------------------------ 推理
    def predict(self, X):
        import torch

        self.net_.eval()
        out = np.empty(X.shape[0], dtype=np.float32)
        bs = int(self.params.get("predict_batch", 262_144))
        with torch.no_grad():
            for s in range(0, X.shape[0], bs):
                Xb = self._to_tensor_x(torch, np.ascontiguousarray(X[s:s + bs]))
                out[s:s + bs] = self.net_(Xb).squeeze(-1).float().cpu().numpy()
        return out

    def loss_desc(self) -> str | None:
        """把本头的损失构成摊平报出来（见 `ModelBase.loss_desc` 的说明）。"""
        p = self.params
        bits = []
        for key, tag in (("w_mse", "mse"), ("w_ic", "ic"), ("w_rankic", "rankIC"),
                         ("w_top", "L_top"), ("w_listnet", "listnet"), ("rdrop", "rdrop")):
            v = float(p.get(key, 0.0) or 0.0)
            if v > 0:
                bits.append(f"{tag}×{v:g}")
        if float(p.get("w_top", 0.0) or 0.0) > 0:
            bits.append(f"τ={float(p.get('tau', 0.05)):g}")
            bits.append(f"top目标={p.get('top_target', 'z')}")
        if float(p.get("w_listnet", 0.0) or 0.0) > 0:
            bits.append(f"listnet_k={int(p.get('listnet_k', 20))}")
        bits.append(f"h={p.get('hidden', [256, 64])}·dropout={float(p.get('dropout', 0.2)):g}"
                    f"·lr={float(p.get('lr', 1e-3)):g}")
        return " ".join(bits)

    def importance(self):
        return None                     # MLP 没有"因子重要性"这个概念的干净对应物

    def info(self):
        d = super().info()
        d.update({"device": self.device_, "best_epoch": int(getattr(self, "best_epoch_", 0)),
                  "valid_loss": round(float(getattr(self, "valid_loss_", float("nan"))), 6),
                  "n_params": int(sum(q.numel() for q in self.net_.parameters()))})
        if getattr(self, "history_", None):
            d["history"] = self.history_[-int(self.params.get("history_tail", 8)):]
        return d


# ------------------------------------------------------------------ 损失的 torch 实现
def _day_bounds_torch(torch, day_t):
    """torch 版的分日边界：返回每行的"日序号"（0..n_day-1），供 scatter 归约用。"""
    n = day_t.numel()
    if n == 0:
        return day_t
    change = torch.ones(n, dtype=torch.bool, device=day_t.device)
    change[1:] = day_t[1:] != day_t[:-1]
    return (torch.cumsum(change.to(torch.int64), 0) - 1)


def _seg_sum(torch, values, seg, n_seg):
    out = torch.zeros(n_seg, dtype=values.dtype, device=values.device)
    return out.scatter_add_(0, seg, values)


def _seg_mean(torch, values, seg, n_seg):
    s = _seg_sum(torch, values, seg, n_seg)
    c = _seg_sum(torch, torch.ones_like(values), seg, n_seg).clamp_min(1.0)
    return s / c


def _batch_softmax_top(pred, target, seg, tau):
    """`−(1/天数)·Σ_days Σ_i softmax(pred/τ)_i · target_i`（torch 版，与 numpy 版公式一致）。"""
    if seg is None:
        return None
    import torch
    n_seg = int(seg.max().item()) + 1
    # 数值稳定：按日减最大值（scatter_reduce 的 amax）
    mx = torch.full((n_seg,), float("-inf"), dtype=pred.dtype, device=pred.device)
    mx = mx.scatter_reduce(0, seg, pred.detach(), reduce="amax", include_self=True)
    e = torch.exp((pred - mx[seg]) / float(tau))
    denom = _seg_sum(torch, e, seg, n_seg).clamp_min(1e-30)
    p = e / denom[seg]
    return -(p * target).sum() / float(n_seg)


def _pearson_t(torch, a, b):
    """两个同长向量的 Pearson 相关（torch 版，带数值保护）。"""
    a = a - a.mean()
    b = b - b.mean()
    d = (a.norm() * b.norm()).clamp_min(1e-12)
    return (a * b).sum() / d


def _batch_soft_rankic(pred, target, seg, tau_frac: float = 0.05, max_n: int = 2000):
    """可微的**软秩 RankIC**（逐日截面内）：把"名次"也变成可微的。

    ## 为什么要它（与已有的 `w_ic` 项有什么不同）

    `_batch_ic` 算的是预测与目标的 **Pearson**。它对预测是光滑可导的，
    所以"IC 项"其实已经在优化相关性了 —— 但它**不显式地看次序**：
    一个把两只票排反、幅度却差不多的预测，Pearson 惩罚得比"排对了但幅度偏"还轻。

    软秩版本用 `1 + Σ_j sigmoid((s_i − s_j)/τ)` 近似名次：τ→0 时它精确等于"名次"，
    τ 适中时处处可导。再与目标的**硬名次**求相关 ⇒ **直接对"排得对不对"给梯度**。
    这正是参考工程 `autodl-fs/tmp/model.py:rank_loss` 里 `−w·(IC + RankIC)` 的第二项。

    ## 成本与截断

    pairwise 是 **O(n_day²)**，必须在**当天**范围内做（跨日会串）。实测一天 ~3000 只
    时一次前向就是 9M 个数、反向还要留激活 —— 所以给一个 `max_n` 上限：
    超过就**无放回抽样**，用子集的软秩相关当当天值（有噪声但无偏，且它只是损失里的一项）。

    ★ 目标侧用**硬名次**（`argsort`）而不是软名次：它是常数，不需要梯度，
      而且硬名次更稳（参考实现也是这么做的）。
    """
    if seg is None:
        return None
    import torch
    n_seg = int(seg.max().item()) + 1
    vals = []
    for k in range(n_seg):
        m = seg == k
        n = int(m.sum())
        if n < 8:
            continue
        p = pred[m]
        t = target[m]
        if n > max_n:                        # 无放回抽样，控制 O(n²) 的开销
            idx = torch.randperm(n, device=p.device)[:max_n]
            p, t = p[idx], t[idx]
        tau = float(tau_frac) * p.detach().std().clamp_min(1e-6)
        # ★★ 广播方向是**递增**的：`A[i,j] = p_i − p_j`（`unsqueeze(1) − unsqueeze(0)`）。
        #   `Σ_j σ((p_i−p_j)/τ)` 对最大的 p_i 有 n 项接近 1 ⇒ 软秩最大 —— 与硬名次同向。
        #   ★ 参考工程 `autodl-fs/tmp/model.py:_soft_rank` 写的是反过来的
        #     `(x.unsqueeze(0) − x.unsqueeze(1))`，那给出的是**递减**的伪名次；
        #     与递增的硬名次求相关会得到反号的"RankIC"。这里实测踩到过
        #     （正相关输入返回 −0.82），已按递增方向修正 —— **不要照抄参考实现的这一行**。
        sp = 1.0 + torch.sigmoid((p.unsqueeze(1) - p.unsqueeze(0)) / tau).sum(dim=1)
        # 目标的名次：参数是常数，不需要梯度
        rt = torch.argsort(torch.argsort(t.detach())).to(p.dtype)
        vals.append(_pearson_t(torch, sp, rt))
    if not vals:
        return None
    return torch.stack(vals).mean()


def _batch_listnet_topk(pred, target, seg, k: int = 20, tau_frac: float = 0.10):
    """**ListNet@top-k**：只要求"把标签最高的 k 只排到最前面"。

    与 `_batch_softmax_top` 的区别（两者都叫"顶部"，但语义不同）：
      · `_batch_softmax_top` = `−Σ softmax(pred/τ)·z`，**对预测做 softmax**，
        再乘标签的**连续值** ⇒ 它关心头部的"收益幅度对不对"。
      · 本函数 = 交叉熵，**对标签做 top-k 硬截断**成一个均匀分布，
        与 `log_softmax(pred/τ)` 求交叉熵 ⇒ 它只说一句："**把这 k 只排前面**"，
        完全不关心这 k 只之间谁涨得多。

    「只买 5 只」的实盘口径下，第二种更贴题（我们不需要在头部内部排得极准，
    只需要"别把该买的漏在外面"）。参考工程用的是 `LISTNET_K=20`，本实现沿用。
    """
    if seg is None:
        return None
    import torch
    n_seg = int(seg.max().item()) + 1
    vals = []
    for s in range(n_seg):
        m = seg == s
        n = int(m.sum())
        if n < k + 2:
            continue
        p = pred[m]
        t = target[m].detach()
        tau = float(tau_frac) * p.detach().std().clamp_min(1e-6)
        logq = torch.log_softmax(p / tau, dim=0)
        _, top = torch.topk(t, int(k))
        tgt = torch.zeros_like(p)
        tgt[top] = 1.0 / float(k)
        vals.append(-(tgt * logq).sum())
    if not vals:
        return None
    return torch.stack(vals).mean()


def _batch_ic(pred, target, seg):
    """`1 − 平均逐日截面相关系数` 的分项（返回 IC 本身）。"""
    if seg is None:
        return None
    import torch
    n_seg = int(seg.max().item()) + 1
    mp = _seg_mean(torch, pred, seg, n_seg)
    mt = _seg_mean(torch, target, seg, n_seg)
    a = pred - mp[seg]
    b = target - mt[seg]
    cov = _seg_mean(torch, a * b, seg, n_seg)
    va = _seg_mean(torch, a * a, seg, n_seg)
    vb = _seg_mean(torch, b * b, seg, n_seg)
    corr = cov / (va.clamp_min(1e-12).sqrt() * vb.clamp_min(1e-12).sqrt())
    return corr.mean()


def _null_ctx(torch):
    from contextlib import nullcontext
    return nullcontext()
