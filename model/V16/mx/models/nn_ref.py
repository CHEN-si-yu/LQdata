"""参考工程两个模型的**引擎内复刻**（用户 2026-09-19 交办）。

来源 = `/autodl-fs/data/tmp/` 下的三个脚本：

| 文件 | 内容 | 本模块的复刻 |
|:--|:--|:--|
| `model.py` / `model3.py` | **逐字节相同**（md5 `a93d9101…`）——自称 "V8" | `nn_v8` |
| `model2.py` | 参考工程的通用训练模板（接季度 walk-forward、DDP、SWA） | `nn_wpcc` |

为什么要"复刻"而不是"直接跑原脚本"（三条，都是硬障碍）：

1. **原脚本要的数据本机没有。** 它们读 `/autodl-fs/data/lingqiData/trainingdata/{fac_all,
   label_ret_1d, label_ret_5d, buyable_mask}.fea`（宽表 feather）与 `Model/best/ridge_init.csv`，
   而 `/autodl-fs/data/lingqiData/` **整个目录不存在**；`model2.py` 更是指向另一台机器的
   `/home/user96/...`。⇒ 要跑就得先造一套 `.fea` 数据桥。
2. **原脚本的口径与我们的评价口径对不上。** `model.py` 是实战版（`LIVE_START='20260901'`、
   无 test 集），照原样跑，我们的数据到 2026-09-18 ⇒ 打分窗只剩 13 天，出不了可比结论。
3. **依赖缺失且部分源码不在文件里。** `pytorch_lightning` / `torchmetrics` 未装；
   `model2.py` 还 `from src.model import *`（`MultTime2dMixer`）与 `from kan import KAN` ——
   这两个模块**不在本机、也不在这三个文件里**，无法逐行还原。

⇒ 所以本模块把它们的**模型结构与损失**复刻进我们的引擎，跑我们自己的 4 折 + 固定 test 窗，
这样出来的数字能与 V2b~V12 直接比。**逐项保真度与已知偏离写在每个类的 docstring 里**
（用户要求"如实标注"，不许把偏离藏起来）。

★ 本文件**不改 `nn.py`**：新增类型、新增文件，现有单元的行为一字不变（零回归风险）。
"""
from __future__ import annotations

import numpy as np

from .base import ModelBase, register, resolve_device
from .nn import _day_bounds_torch, _pearson_t, _seg_sum

# ★★ 网络结构**必须建在模块级**：`ModelBase.save` 用 pickle 存整个模型对象，
#   而**函数内定义的类不可 pickle**（`Can't pickle local object ...<_build.<locals>._Net>`）。
#   冒烟实测踩到：训练跑完、落盘那一刻才炸。
# ★ 同时保持"没有 torch 的机器上本文件仍可 import"这条约定 —— 用 try/except 包起来，
#   拿不到 torch 时留 None（那两个头自然也用不了，但线性和树不受影响）。
try:
    import torch as _T

    class _V8Net(_T.nn.Module):
        """参考工程 V8 的双头网络：线性排序头 + 小 MLP 顶部分支。

        ★ 输出是**两个原始 logits**（lin / top），逐日 znorm 与加权在 `fit/predict` 里做 ——
          不放进 forward：znorm 需要"当天完整截面"这个结构信息，而 forward 只拿到一个 batch。
        """

        def __init__(self, n_in: int, top_hidden: list, top_dropout: float):
            super().__init__()
            import torch.nn as nn

            self.lin = nn.Linear(n_in, 1)
            layers: list = []
            prev = n_in
            n_h = len(top_hidden)
            for i, h in enumerate(top_hidden):
                layers += [nn.Linear(prev, int(h)), nn.GELU()]
                # ★★ dropout **只加在前 n−1 个隐层之后**，与参考实现逐层对齐：
                #   `tmp/model.py:312-319` 是
                #   `Linear(D,128) → GELU → Dropout(.3) → Linear(128,64) → GELU → Linear(64,1)`
                #   —— **最后一层隐层之后没有 dropout**。写成"每个隐层后都加"会多出一层：
                #   ① 改模型容量/正则；② 把 R-Drop 的随机源从 1 处变 2 处（lin 头没有 dropout，
                #   两次前向的差异**全部**来自 top 的 dropout）。
                #   ★ 这个错当初是"代码与自己的 docstring 矛盾"被抓出来的（保真度复核）。
                if top_dropout > 0 and i < n_h - 1:
                    layers.append(nn.Dropout(float(top_dropout)))
                prev = int(h)
            layers.append(nn.Linear(prev, 1))
            self.top = nn.Sequential(*layers)

        def forward(self, x):
            return self.lin(x).squeeze(-1), self.top(x).squeeze(-1)

    class _WPCCNet(_T.nn.Module):
        """`model2.py` 可见部分的 3 层 MLP（输出头换成 Linear —— 原本是不可得的 KAN）。"""

        def __init__(self, n_in: int, h1: int, h2: int, d1: float, d2: float):
            super().__init__()
            import torch.nn as nn

            self.net = nn.Sequential(
                nn.Linear(n_in, int(h1)), nn.LeakyReLU(inplace=True), nn.Dropout(float(d1)),
                nn.Linear(int(h1), int(h2)), nn.BatchNorm1d(int(h2)), nn.GELU(),
                nn.Dropout(float(d2)),
                nn.Linear(int(h2), 1),
            )

        def forward(self, x):
            return self.net(x).squeeze(-1)

except ImportError:                     # 没装 torch：这两个头用不了，但别连累线性/GBDT
    _V8Net = None                       # type: ignore[assignment]
    _WPCCNet = None                     # type: ignore[assignment]


def _znorm_day(torch, v, seg, n_seg):
    """**逐日** z-score。参考实现 `model.py:_znorm` 是在整个 batch 上做的 ——
    训练时 batch = 8 天 ⇒ 那是个跨日统计量；打分时 batch = 1 天 ⇒ 就是逐日。
    本实现统一成**逐日**（打分口径），这样"打分只由当天截面决定"，与铁律二一致。

    ★ 三个数值细节都对着参考实现抄：
      ① 标准差用**无偏**（`/ (n−1)`）—— `tmp/model.py:298` 是 `v.std()`，torch 默认 `unbiased=True`；
      ② `sd` **detach**（原实现写了 `.detach()`）⇒ 梯度不回流到"尺度"上，只回流到"相对次序"上；
         均值不 detach（原实现也没 detach）—— 这一条会改变标准化层的梯度，两边必须一致；
      ③ `clamp_min(1e-6)` 放在 sd 上（原实现的位置），不是放在方差上。
    """
    m = _seg_sum(torch, v, seg, n_seg)
    c = _seg_sum(torch, torch.ones_like(v), seg, n_seg).clamp_min(2.0)
    mu = m / c
    d = v - mu[seg]
    var = _seg_sum(torch, d * d, seg, n_seg) / (c - 1.0)
    sd = var.clamp_min(0.0).sqrt().clamp_min(1e-6).detach()
    return d / sd[seg]


def _v8_listnet(torch, pred, target, seg, k: int = 20, tau_frac: float = 0.15):
    """`model.py:_listnet_topk` 的原式复刻（含 **/ log(n)** 的归一化）。

    ★ 为什么不直接用引擎里的 `nn._batch_listnet_topk`：那一版**没有** `/log(n)`。
      参考实现除以 `log(n)`（n≈2000 时是 ÷7.6），所以同样写 `LISTNET_WEIGHT=0.30`，
      两项的实际量级差 7.6 倍 —— 复刻就不该动它。
    """
    if seg is None:
        return None
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
        vals.append(-(tgt * logq).sum() / float(np.log(n)))
    if not vals:
        return None
    return torch.stack(vals).mean()


def _soft_rankic_v8(torch, pred, target, max_n: int = 2000, tau_frac: float = 1.0):
    """`model.py:rank_loss` 的第二项：软秩 RankIC（**已修正方向**）。

    ★★ 参考实现的 `_soft_rank` 写的是 `(x.unsqueeze(0) − x.unsqueeze(1))`，
      给出的是**递减**的伪名次，与递增的硬名次求相关会得到反号的 "RankIC"。
      我们的 `nn._batch_soft_rankic` 已修正为递增方向（README 里记过实测：
      正相关输入返回 −0.82）。复刻沿用修正版 —— **这是纠错，不是改配方**。

    ★ `max_n`：pairwise 是 O(n²)，一天 2000 只就是 400 万个数、反向还要留激活。
      超过上限就**无放回抽样**（有噪声但无偏，而且它只是损失里的一项）——
      与引擎里 `nn._batch_soft_rankic` 的 `soft_rank_max` 是同一个保险。
      ⚠️ 参考实现是当天**全量** pairwise、不抽样 —— 这一条是本实现新增的近似
      （真实一天 ~3000 只时只取 2/3，该项因此带日间随机噪声）。

    ★★ `tau_frac` 默认 **1.0**：参考实现的 `_soft_rank` 是 `tau = x.std()`（**没有**系数），
      这里必须跟着用 1.0。引擎里 `nn._batch_soft_rankic` 默认 0.05 是**另一套**口径，
      两者 tau 差 20 倍（实测该项的值/梯度差约 5%）—— 复刻不许串用。
    """
    n = int(pred.numel())
    if n < 8:
        return None
    if n > int(max_n):
        idx = torch.randperm(n, device=pred.device)[:int(max_n)]
        pred, target = pred[idx], target[idx]
    tau = float(tau_frac) * pred.detach().std().clamp_min(1e-6)
    sp = 1.0 + torch.sigmoid((pred.unsqueeze(1) - pred.unsqueeze(0)) / tau).sum(dim=1)
    rt = torch.argsort(torch.argsort(target.detach())).to(pred.dtype)
    return _pearson_t(torch, sp, rt)


def _winsor_mad(torch, v, k: float = 5.0):
    """`model.py:_winsor_mad` —— 中位数 ± k×MAD 截尾（**逐日**做）。"""
    med = v.median()
    mad = (v - med).abs().median()
    if float(mad) <= 1e-12:
        return v
    return v.clamp(med - k * mad, med + k * mad)


# ==================================================================== V8 复刻
@register
class NNV8Model(ModelBase):
    """`tmp/model.py`（= `model3.py`）的复刻：**线性排序头（ridge 热启动）+ 小 MLP 顶部分支**。

    ## 结构
    ```
    lin = Linear(D, 1)                      ← ridge 热启动
    top = Linear(D,128) → GELU → Dropout(.3) → Linear(128,64) → GELU → Linear(64,1)
    mixed = znorm_day(lin) + znorm_day(top)     ← 最终打分
    ```

    ## 损失（逐日）
    ```
    L = −3.0·(pearson(lin, y_rg) + softRankIC(lin, y_rg))          # 1d 排序头
      + [−1.0·soft_top_ret(top, y_rg) + 0.30·ListNet@20(top, 截尾raw)]
      + 0.10·R-Drop(mixed)                                          # 同一样本两次前向
    ```
    再乘时间衰减权重（见下）。

    ## ★★ 与参考实现的偏离（逐条，2026-09-19 保真度复核后重写）

    复核方式：拿 `tmp/model.py` 的逐行源码与实测数值对拍（含把原实现的 `rank_loss`
    直接 exec 出来在合成数据上跑）。**已修正的偏离不再列在这里**（它们曾经是真错）：

    | # | 参考实现 | 本实现 | 为什么 |
    |:--|:--|:--|:--|
    | 1 | 一个模型同时训 `lin_1d`/`lin_5d` + top | **每个标签各训一个**（引擎是 (头×标签) 网格） | 引擎的产物/评价/策略都以 (头,标签) 为单位；改成多标签要重写整条链路。且参考工程**自己的**死胡同清单写着"联合共享躯干双头 = 信号互相破坏" |
    | 2 | 2 seeds × 4 折 bagging（8 个模型） | 本版 4 折（= 4 个模型） | 引擎的"折"已含种子维度；要 8 个就再开一个换种子的单元（`clone_seeds.py`） |
    | 3 | 时间衰减 `0.5^(日历天/600)`，**乘在当天整条 loss 上** | 引擎的 `sample_weight.half_life`（**交易日**）取 410，用**当天第一个样本**的权重当整天的系数 | 模型拿不到日历日期（只拿到日序号）。★ 取 `[0]` 仅在"权重逐日恒定"时成立 —— 本版 `spec` 只有 `half_life`（无 `tail_frac`），实测日内恒定；**若将来给这个头加尾部加权，这里必须改**（会静默取到"当天第一只票"的权重） |
    | 4 | `_soft_rank` 的 `(x.unsqueeze(0) − x.unsqueeze(1))` | 递增方向 | 参考实现这一行是 **bug**（给出递减伪名次，与递增硬名次求相关会**反号**）。<br>⚠️ **但它不是"无害纠错"**：复核实测原式 `ic+rankic` 在真实 IC 量级下**近乎相消**（+0.033 + (−0.037) ≈ −0.004，甚至反号），修正后 `−3.0·(ic−rankic)` 的**梯度约为原式的 7 倍**。⇒ 与参考工程的数字对比时，**线性头的有效学习信号强度不是同一个量** |
    | 5 | batch 内 `_znorm`（8 天一起） | **逐日** znorm | 打分时参考实现也是 1 天 1 batch ⇒ 逐日才是它的打分口径；且逐日满足"只用当天截面"的铁律 |
    | 6 | 可买掩码把 label 置 NaN | 沿用引擎的 `buyable`（切行） | 本库已有实测结论：训练侧剔行收益≈0、长期限上有害（§10）。**本版默认不开** |
    | 7 | 848 因子 | 快照特征数（306 去掉排除项 = 304） | 这是"复刻配方"不是"复刻数据" |
    | 8 | `_soft_rank` 当天**全量** pairwise | `soft_rank_max` 上限内抽样（默认 2000） | O(n²) 的成本保险；真实一天 ~3000 只时只取 2/3 ⇒ 该项带**日间随机噪声**（期望无偏）。参考实现不抽样 |
    | 9 | 线性头从外部 `ridge_init.csv` 热启动（"600 日截面秩线性拟合"） | 在**训练段全段**上现解闭式解（分块累加 `XᵀX`） | 那个 csv 本机没有。闭式解无随机性、PIT 安全；**但拟合段更长（全训练段 ≠ 600 日）**，且 `alpha=1.0` 对未经 z-score 的 [0,1] 秩特征近乎等于无正则 |
    | 10 | `_pearson` 在非有限时返回 0 | `_pearson_t` 可能返回 nan ⇒ 整个 batch 被 `isfinite` 跳过 | 参考实现会把那一天的 loss 当 0 计进去，本实现是"少训一个 batch"（`nn.py` 原有实现如此）。同一量级的事件极罕见 |

    ## 与 `nn` 的关系

    继承 `NNModel` 只借它的 `save/load/info` 等基础设施；`fit/predict/_valid_loss` 全部重写
    —— 因为参考实现是**三个输出头 + 各自不同的损失**，而基类是单输出。
    """

    type = "nn_v8"
    fill_mode = "nn"
    predict_needs_day = True            # mixed 里有逐日 znorm ⇒ 打分必须知道"哪些行是同一天"

    def _build(self, torch, n_in: int):
        if _V8Net is None:
            raise RuntimeError("nn_v8 需要 torch（当前环境没装）")
        p = self.params
        return _V8Net(n_in, list(p.get("top_hidden", [128, 64])),
                      float(p.get("top_dropout", 0.3)))

    def _ridge_warm_start(self, torch, X, y):
        """线性头用**岭回归闭式解**热启动（对应参考实现的 `ridge_init.csv`）。

        ★ 参考实现从一个外部 csv 读权重（`Model/best/ridge_init.csv`，本机没有）。
          这里改成**在训练集上现解** —— 同样是"用 600 日截面秩线性拟合"的语义，
          而且 PIT 安全（只用训练段）、可复现（闭式解，无随机性）。
        """
        if not bool(self.params.get("warm_start_ridge", True)):
            return
        # ★ 必须**分块**累加 XᵀX：整块转 float64 会多出一份 6 GB 的副本
        #   （250 万行 × 305 列 × 8B），两折并发时足以把容器顶穿。分块后峰值只有块大小。
        n_in = int(X.shape[1])
        yv = np.asarray(y, dtype=np.float64)
        chunk = int(self.params.get("warm_start_chunk", 200_000))
        XtX = np.zeros((n_in, n_in), dtype=np.float64)
        Xty = np.zeros(n_in, dtype=np.float64)
        s_x = np.zeros(n_in, dtype=np.float64)
        s_y = 0.0
        n_ok = 0
        for s in range(0, X.shape[0], chunk):
            Xc = np.asarray(X[s:s + chunk], dtype=np.float64)
            yc = yv[s:s + chunk]
            ok = np.isfinite(yc) & np.isfinite(Xc).all(axis=1)
            if not ok.any():
                continue
            Xc, yc = Xc[ok], yc[ok]
            XtX += Xc.T @ Xc
            Xty += Xc.T @ yc
            s_x += Xc.sum(axis=0)
            s_y += float(yc.sum())
            n_ok += int(ok.sum())
        if n_ok < 100:
            return
        mx_ = s_x / n_ok
        my_ = s_y / n_ok
        # 中心化后的正规方程： (X−μx)ᵀ(X−μx) w = (X−μx)ᵀ(y−μy)
        A = XtX - n_ok * np.outer(mx_, mx_) + float(self.params.get("warm_start_alpha", 1.0)) \
            * np.eye(n_in)
        b = Xty - n_ok * mx_ * my_
        try:
            w = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            return
        b0 = my_ - float(mx_ @ w)
        import torch
        with torch.no_grad():
            self.net_.lin.weight.copy_(
                torch.from_numpy(w).to(self.net_.lin.weight.dtype).view(1, -1))
            self.net_.lin.bias.fill_(float(b0))

    # ---------------------------------------------------------------- 训练
    def fit(self, X, y, Xv=None, yv=None, *, sample_weight=None,
            day=None, day_v=None, raw=None, raw_v=None, **kw):
        import torch

        if day is None:
            raise ValueError("nn_v8 必须拿到 day —— 逐日 znorm / 逐日 softmax 都要当天完整截面")
        p = self.params
        self.device_ = resolve_device(self.cfg)
        torch.manual_seed(self.seed)
        np.random.seed(self.seed % (2 ** 31))
        if self.device_ == "cuda":
            torch.cuda.manual_seed_all(self.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        day = np.asarray(day)
        from ..labels import day_bounds as _db
        bnd = _db(day)
        n_day = bnd.size - 1

        self.net_ = self._build(torch, int(X.shape[1])).to(self.device_)
        self._ridge_warm_start(torch, X, y)

        wd_lin = float(p.get("weight_decay_lin", 1e-3))
        wd_top = float(p.get("weight_decay_top", 3e-3))
        lin_params = list(self.net_.lin.parameters())
        top_params = list(self.net_.top.parameters())
        opt = torch.optim.AdamW([{"params": lin_params, "weight_decay": wd_lin},
                                 {"params": top_params, "weight_decay": wd_top}],
                                lr=float(p.get("lr", 5e-4)))
        epochs = int(p.get("epochs", 40))
        batch_days = int(p.get("batch_days", 8))
        steps = max(1, epochs * max(1, int(np.ceil(n_day / batch_days))))
        warmup = max(1, int(float(p.get("warmup_epochs", 2)) * steps / max(1, epochs)))
        lr_min_r = float(p.get("lr_min", 1e-5)) / max(1e-12, float(p.get("lr", 5e-4)))

        def lr_lambda(step):
            if step < warmup:
                return step / warmup
            prog = (step - warmup) / max(1, steps - warmup)
            return lr_min_r + 0.5 * (1.0 - lr_min_r) * (1.0 + np.cos(np.pi * prog))

        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
        use_amp = bool(p.get("amp", True)) and self.device_ == "cuda"
        rng = np.random.default_rng(self.seed)

        w_rank = float(p.get("w_rank", 3.0))
        w_top = float(p.get("w_top", 1.0))
        w_ln = float(p.get("w_listnet", 0.30))
        w_rd = float(p.get("rdrop", 0.10))
        tau_frac = float(p.get("tau_frac", 0.06))
        ln_k = int(p.get("listnet_k", 20))
        ln_tau = float(p.get("listnet_tau", 0.15))
        sr_max = int(p.get("soft_rank_max", 2000))

        yt = torch.from_numpy(np.asarray(y, dtype=np.float32))
        rawt = (torch.from_numpy(np.asarray(raw, dtype=np.float32))
                if raw is not None else None)
        wt = (torch.from_numpy(np.asarray(sample_weight, dtype=np.float32))
              if sample_weight is not None else None)

        best, best_state, best_ep, bad = float("inf"), None, 0, 0
        patience = int(p.get("patience", 10))
        min_delta = float(p.get("min_delta", 1e-4))
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
                Xb = torch.from_numpy(np.ascontiguousarray(X[rows])).to(self.device_)
                Xb = Xb.to(torch.float32)
                yb = yt[rows].to(self.device_)
                rb = rawt[rows].to(self.device_) if rawt is not None else yb
                wb = wt[rows].to(self.device_) if wt is not None else None
                db = torch.from_numpy(day[rows].astype(np.int64)).to(self.device_)
                seg = _day_bounds_torch(torch, db)
                n_seg = int(seg.max().item()) + 1
                ok = torch.isfinite(yb) & torch.isfinite(rb)
                opt.zero_grad(set_to_none=True)
                from contextlib import nullcontext
                ctx = (torch.autocast("cuda", dtype=torch.float16) if use_amp
                       else nullcontext())
                with ctx:
                    r1, t1 = self.net_(Xb)
                    r1f, t1f = r1.float(), t1.float()
                    mixed = _znorm_day(torch, r1f, seg, n_seg) + _znorm_day(torch, t1f, seg, n_seg)
                    # ★ R-Drop 的第二次前向：**同一份输入**再走一遍（dropout 掩码不同）。
                    #   本实现的 R-Drop 是**逐日**算的，理由见下面循环里的注释。
                    m2 = None
                    if w_rd > 0:
                        r2, t2 = self.net_(Xb)
                        m2 = (_znorm_day(torch, r2.float(), seg, n_seg)
                              + _znorm_day(torch, t2.float(), seg, n_seg))
                    parts: dict[str, float] = {}
                    total = None                    # 每天累加一次（见下面的 `total is None` 分支）
                    nd = 0
                    rd_sum = 0.0
                    for k in range(n_seg):
                        m = (seg == k) & ok
                        if int(m.sum()) < 50:
                            continue
                        yk = yb[m]
                        rk = r1f[m]
                        tk = t1f[m]
                        ic = _pearson_t(torch, rk, yk)
                        srk = _soft_rankic_v8(torch, rk, yk, max_n=sr_max)
                        if srk is not None:
                            ic = ic + srk
                        lrk = -w_rank * ic
                        ltop = rk.new_zeros(())
                        if w_top > 0:
                            ltop = ltop + w_top * _soft_top_ret_day(torch, tk, yk, tau_frac)
                        if w_ln > 0:
                            # ★ 只对本批的这一天做：seg 全 0（单日），n = 当天只数。
                            #   注意 `_v8_listnet` 收的是**扁平**张量 + 同长的 seg，
                            #   别把预测 unsqueeze 成 [1,n]（掩码长度会对不上）。
                            one = torch.zeros(int(m.sum()), dtype=torch.int64,
                                              device=tk.device)
                            lnv = _v8_listnet(torch, tk, _winsor_mad(torch, rb[m]),
                                              one, k=ln_k, tau_frac=ln_tau)
                            if lnv is not None:
                                ltop = ltop + w_ln * lnv
                        day_loss = lrk + ltop
                        # ★★ R-Drop 必须**在日循环内、且乘当天的衰减权重**：
                        #   参考实现是"按天处理"的（`DLDataset` 一天一条样本，
                        #   batch_size=8 只是一个 batch 装 8 天），
                        #   所以 `(lr1+lr5+ltop + RDROP_WEIGHT*rdrop) * _time_weight(day)`
                        #   里的 rdrop 同样被时间衰减缩放。
                        #   放在日循环外、不加权 ⇒ 衰减后 `mean(day_w) ≈ 0.31`，
                        #   R-Drop 的相对权重会被**放大约 3 倍**（实测 1/0.306 = 3.27）。
                        if m2 is not None:
                            rd = ((mixed[m] - m2[m]) ** 2).mean()
                            day_loss = day_loss + w_rd * rd
                            rd_sum += float(rd.detach())
                        day_w = wb[m][0] if wb is not None else rk.new_ones(())
                        total = day_loss * day_w if total is None else total + day_loss * day_w
                        nd += 1
                    if total is None or nd == 0:
                        continue
                    total = total / float(nd)
                    if m2 is not None:
                        parts["rdrop"] = rd_sum / max(1, nd)
                    parts["n_day"] = float(nd)
                # ★ `total` 为 None / 无梯度时不要 backward：`torch.zeros(())` 是叶子张量，
                #   backward 会抛 "element 0 of tensors does not require grad"（低概率但会整折崩）
                if total is None or not torch.isfinite(total) or not total.requires_grad:
                    continue
                total.backward()
                torch.nn.utils.clip_grad_norm_(self.net_.parameters(),
                                               float(p.get("clip_grad", 1.0)))
                opt.step()
                try:
                    sched.step()
                except Exception:            # noqa: BLE001
                    pass
                run += float(total.detach())
                nb += 1
                self._last_parts = parts

            vloss = (self._valid_loss(torch, Xv, yv, day_v, raw_v)
                     if Xv is not None and len(Xv) else run / max(1, nb))
            self.history_.append({"epoch": ep, "train": run / max(1, nb), "valid": vloss})
            # ★ `min_delta` 默认 1e-4：参考实现的 `EarlyStopping(min_delta=1e-4)`
            #   （`tmp/model.py:490-491`）。用 1e-6 会把噪声当改善 ⇒ 选出不同的 best epoch、
            #   而且更不容易早停 —— 这是**静默改变结果**的一类参数。
            if vloss < best - min_delta:
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
        """早停判据 = **负的逐日 Spearman 均值**（与参考实现 monitor `val_rankic` 同口径）。"""
        if Xv is None or len(Xv) == 0:
            return float("inf")
        sc = self.predict(Xv, day=np.asarray(day_v))
        return -_rank_ic_np(sc, np.asarray(yv), np.asarray(day_v))

    def predict(self, X, day=None):
        import torch

        self.net_.eval()
        X = np.asarray(X)
        if day is None:
            day = np.zeros(X.shape[0], dtype=np.int64)
        day = np.asarray(day).astype(np.int64)
        from ..labels import day_bounds as _db
        bnd = _db(day)
        out = np.empty(X.shape[0], dtype=np.float32)
        # 逐日 znorm 要求"一次前向里的行必须同属若干完整交易日" ⇒ 按**天**分块，
        # 不能按行分块（按行切会把一天切成两半，每半各自标准化，排名就变了）。
        chunk_days = max(1, int(self.params.get("predict_days", 32)))
        with torch.no_grad():
            for s in range(0, bnd.size - 1, chunk_days):
                e = min(s + chunk_days, bnd.size - 1)
                rows = np.concatenate([np.arange(bnd[d], bnd[d + 1]) for d in range(s, e)])
                Xb = torch.from_numpy(np.ascontiguousarray(X[rows])).to(self.device_).float()
                db = torch.from_numpy(day[rows]).to(self.device_)
                seg = _day_bounds_torch(torch, db)
                n_seg = int(seg.max().item()) + 1
                r1, t1 = self.net_(Xb)
                mixed = (_znorm_day(torch, r1.float(), seg, n_seg)
                         + _znorm_day(torch, t1.float(), seg, n_seg))
                out[rows] = mixed.float().cpu().numpy()
        return out

    def loss_desc(self) -> str | None:
        p = self.params
        return (f"V8复刻 lin+top · rank×{float(p.get('w_rank', 3.0)):g}"
                f"(IC+软秩RankIC) · top×{float(p.get('w_top', 1.0)):g}"
                f"(软TopRet+{float(p.get('w_listnet', 0.30)):g}×ListNet@{int(p.get('listnet_k', 20))})"
                f" · rdrop×{float(p.get('rdrop', 0.10)):g}"
                f" · top_h={p.get('top_hidden', [128, 64])}·dropout={float(p.get('top_dropout', 0.3)):g}"
                f" · lr={float(p.get('lr', 5e-4)):g} · wd(lin/top)="
                f"{float(p.get('weight_decay_lin', 1e-3)):g}/{float(p.get('weight_decay_top', 3e-3)):g}")

    def info(self):
        d = super().info()
        d.update({"device": self.device_, "best_epoch": int(getattr(self, "best_epoch_", 0)),
                  "valid_loss": round(float(getattr(self, "valid_loss_", float("nan"))), 6),
                  "n_params": int(sum(q.numel() for q in self.net_.parameters()))})
        if getattr(self, "history_", None):
            d["history"] = self.history_[-int(self.params.get("history_tail", 8)):]
        return d


def _soft_top_ret_day(torch, pred, target, tau_frac: float):
    """单日 `model.py:_soft_top_ret`：`Σ softmax(pred/τ)·z − mean(z)`。"""
    tau = float(tau_frac) * pred.detach().std().clamp_min(1e-6)
    w = torch.softmax((pred - pred.mean()) / tau, dim=0)
    return (w * target).sum() - target.mean()


def _rank_ic_np(score, y, day) -> float:
    """逐日 Spearman 的均值（给早停用）。

    ★ 必须向量化：它是**每个 epoch 都要算一次**的量（30~40 epoch × 78 万行），
      `groupby.apply(lambda ...)` 那种写法在这里要慢一两个数量级 —— 逐日 rank 用
      `transform` 一次算完，再做逐组 Pearson。

    ★ `MIN_DAY_STOCKS = 50`：参考实现的 `_evaluate_step` 是
      `if m.sum() >= MIN_DAY_STOCKS`（`tmp/model.py:52,435`）。门槛写错会让
      "最优 epoch"选在别处 —— **静默改变结果，不报任何错**。
    """
    import pandas as pd

    df = pd.DataFrame({"d": np.asarray(day), "s": np.asarray(score, dtype="float64"),
                       "y": np.asarray(y, dtype="float64")})
    df = df[np.isfinite(df["s"]) & np.isfinite(df["y"])]
    if df.empty:
        return 0.0
    g = df.groupby("d", observed=True)
    n = g["s"].transform("size")
    df = df[n >= 50]
    if df.empty:
        return 0.0
    g = df.groupby("d", observed=True)
    rs, ry = g["s"].rank(), g["y"].rank()
    ds = rs - rs.groupby(df["d"]).transform("mean")
    dy = ry - ry.groupby(df["d"]).transform("mean")
    num = (ds * dy).groupby(df["d"]).sum()
    den = ((ds ** 2).groupby(df["d"]).sum() * (dy ** 2).groupby(df["d"]).sum()).pow(0.5)
    ic = (num / den.replace(0.0, np.nan)).dropna()
    return float(ic.mean()) if len(ic) else 0.0


# ==================================================================== model2 复刻
@register
class NNWPCCModel(ModelBase):
    """`tmp/model2.py` 的复刻：**MLP + wpcc（按预测名次衰减加权的 Pearson）损失**。

    ## 结构（可见部分）

    ```
    Linear(D,256) → LeakyReLU → Dropout(0.2)
      → Linear(256,128) → BatchNorm1d(128) → GELU → Dropout(0.1)
      → Linear(128,1)
    ```

    ## 损失 `wpcc`（逐日，原式复刻）

    按**预测值的降序名次**给每只股票一个权重 `w_i = 0.5^((i−1)/(n−1))`
    （第一名 1.0、最后一名 0.5），再算**加权 Pearson**，取负。
    ⇒ 语义 = "**排在前面的那些票，相关要算得准**"，是一种**排序驱动的软头部加权**，
    与我们的"按原始收益排序取尾部加权"（`labels.tail_weight`）方向相同、依据不同
    （它按**预测**加权，我们按**真实收益**加权）—— 这正是值得试的地方。

    ## ★ 与参考实现的已知偏离（逐条）

    | # | 参考实现 | 本实现 | 为什么 |
    |:--|:--|:--|:--|
    | 1 | `src.model.MultTime2dMixer` + `kan.KAN` 输出头 | **换成 `Linear(128,1)`** | 这两个模块**不在本机、也不在这三个文件里**（`from src.model import *` / `from kan import KAN`），无法还原。⇒ 本版只复刻**可见的那部分结构**与**完整的 wpcc 损失** |
    | 2 | 标签在 train 阶段逐日 z-score、valid/test 不标准化 | 训练目标逐日 z-score（相同）；早停用逐日 Spearman（**尺度无关**，所以训/评不一致不影响它） | wpcc 是相关系数 ⇒ 对标签做正线性变换不变，z-score 与否只影响数值稳定；但**必须与"原值标签"配对**，所以单元的 `label_transform` 不设 |
    | 3 | 季度 walk-forward（`season`）+ DDP + SWA + 极端行情剔除 | 4 折固定 test 窗（本库口径） | 口径对齐才有可比性，这是"复刻配方"不是"复刻调度" |
    | 4 | `torch.autograd.set_detect_anomaly(True)` | 关（引擎默认） | 它只用于调试，开着会让训练慢一个量级 |
    | 5 | ReduceLROnPlateau(monitor=val_wei, factor=.5, patience=3) | 同（monitor 换成验证 RankIC） | `val_wei` 是它自带的现金级回测指标，本库的策略层在 `mx/strategy.py`，不塞进训练循环 |
    """

    type = "nn_wpcc"
    fill_mode = "nn"
    predict_needs_day = False

    def _build(self, torch, n_in: int):
        if _WPCCNet is None:
            raise RuntimeError("nn_wpcc 需要 torch（当前环境没装）")
        p = self.params
        return _WPCCNet(n_in, int(p.get("hidden1", 256)), int(p.get("hidden2", 128)),
                        float(p.get("dropout1", 0.2)), float(p.get("dropout2", 0.1)))

    def _wpcc(self, torch, pred, target):
        """单日 wpcc（原式复刻）：按**预测降序**给几何衰减权重，再算加权 Pearson。"""
        n = int(pred.numel())
        if n < 3:
            return None
        _, order = torch.sort(pred.detach(), descending=True, dim=0)
        w_new = torch.tensor([0.5 ** (i / (n - 1)) for i in range(n)],
                             device=pred.device, dtype=pred.dtype)
        w = torch.zeros(n, device=pred.device, dtype=pred.dtype)
        w[order] = w_new
        ws = w.sum().clamp_min(1e-9)
        mp = (pred * w).sum() / ws
        mt = (target * w).sum() / ws
        cov = (pred * target * w).sum() / ws - mp * mt
        sp = (((pred - pred.mean()) ** 2 * w).sum() / ws).clamp_min(1e-12).sqrt()
        st = (((target - target.mean()) ** 2 * w).sum() / ws).clamp_min(1e-12).sqrt()
        return -(cov / (sp * st))

    def fit(self, X, y, Xv=None, yv=None, *, sample_weight=None,
            day=None, day_v=None, raw=None, raw_v=None, **kw):
        import torch
        from contextlib import nullcontext

        if day is None:
            raise ValueError("nn_wpcc 必须拿到 day —— wpcc 是逐日截面内的量")
        p = self.params
        self.device_ = resolve_device(self.cfg)
        torch.manual_seed(self.seed)
        np.random.seed(self.seed % (2 ** 31))
        if self.device_ == "cuda":
            torch.cuda.manual_seed_all(self.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        day = np.asarray(day)
        from ..labels import day_bounds as _db
        bnd = _db(day)
        n_day = bnd.size - 1

        self.net_ = self._build(torch, int(X.shape[1])).to(self.device_)
        opt = torch.optim.AdamW(self.net_.parameters(), lr=float(p.get("lr", 1e-3)),
                                weight_decay=float(p.get("weight_decay", 2e-2)))
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="max", factor=float(p.get("lr_factor", 0.5)),
            patience=int(p.get("lr_patience", 3)),
            min_lr=float(p.get("lr_min", 5e-6)), cooldown=int(p.get("lr_cooldown", 2)))
        epochs = int(p.get("epochs", 30))
        batch_days = int(p.get("batch_days", 4))
        use_amp = bool(p.get("amp", True)) and self.device_ == "cuda"
        rng = np.random.default_rng(self.seed)

        yt = torch.from_numpy(np.asarray(y, dtype=np.float32))
        best, best_state, best_ep, bad = float("inf"), None, 0, 0
        patience = int(p.get("patience", 6))
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
                Xb = torch.from_numpy(np.ascontiguousarray(X[rows])).to(self.device_).float()
                yb = yt[rows].to(self.device_)
                db = torch.from_numpy(day[rows].astype(np.int64)).to(self.device_)
                seg = _day_bounds_torch(torch, db)
                n_seg = int(seg.max().item()) + 1
                ok = torch.isfinite(yb)
                opt.zero_grad(set_to_none=True)
                ctx = (torch.autocast("cuda", dtype=torch.float16) if use_amp
                       else nullcontext())
                with ctx:
                    pred = self.net_(Xb).squeeze(-1).float()
                    tot = pred.new_zeros(())
                    nd = 0
                    for k in range(n_seg):
                        m = (seg == k) & ok
                        if int(m.sum()) < 50:
                            continue
                        yk = yb[m]
                        # ★ 参考实现 train 阶段对标签逐日 z-score；wpcc 对正线性变换不变，
                        #   这里做同样的事只为数值稳定（不改变排序语义）。
                        sd = yk.std()
                        yk = (yk - yk.mean()) / (sd if float(sd) > 1e-12 else 1.0)
                        v = self._wpcc(torch, pred[m], yk)
                        if v is not None:
                            tot = tot + v
                            nd += 1
                    if nd:
                        tot = tot / float(nd)
                if not torch.isfinite(tot):
                    continue
                tot.backward()
                torch.nn.utils.clip_grad_norm_(self.net_.parameters(),
                                               float(p.get("clip_grad", 5.0)))
                opt.step()
                run += float(tot.detach())
                nb += 1

            sc_v = self.predict(Xv) if (Xv is not None and len(Xv)) else None
            val_ic = (_rank_ic_np(sc_v, np.asarray(yv), np.asarray(day_v))
                      if sc_v is not None else 0.0)
            try:
                sched.step(val_ic)
            except Exception:                # noqa: BLE001
                pass
            vloss = -val_ic
            self.history_.append({"epoch": ep, "train": run / max(1, nb),
                                  "valid": vloss, "val_rankic": val_ic})
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

    def predict(self, X, day=None):
        import torch

        self.net_.eval()
        X = np.asarray(X)
        out = np.empty(X.shape[0], dtype=np.float32)
        bs = int(self.params.get("predict_batch", 262_144))
        with torch.no_grad():
            for s in range(0, X.shape[0], bs):
                Xb = torch.from_numpy(np.ascontiguousarray(X[s:s + bs])).to(self.device_).float()
                out[s:s + bs] = self.net_(Xb).squeeze(-1).float().cpu().numpy()
        return out

    def loss_desc(self) -> str | None:
        p = self.params
        return (f"model2复刻 wpcc(预测名次几何衰减加权Pearson)"
                f" · h=[{int(p.get('hidden1', 256))},{int(p.get('hidden2', 128))}]"
                f"·dropout={float(p.get('dropout1', 0.2)):g}/{float(p.get('dropout2', 0.1)):g}"
                f" · lr={float(p.get('lr', 1e-3)):g}·wd={float(p.get('weight_decay', 2e-2)):g}"
                f" · ReduceLROnPlateau")

    def info(self):
        d = super().info()
        d.update({"device": self.device_, "best_epoch": int(getattr(self, "best_epoch_", 0)),
                  "valid_loss": round(float(getattr(self, "valid_loss_", float("nan"))), 6),
                  "n_params": int(sum(q.numel() for q in self.net_.parameters()))})
        if getattr(self, "history_", None):
            d["history"] = self.history_[-int(self.params.get("history_tail", 8)):]
        return d
