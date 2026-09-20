"""标签变换与样本权重 —— **损失函数设计**在引擎侧的唯一实现处。

## 为什么需要这个文件

现状的目标与手段是错配的：5 个标签存的是原始收益率 `open(T+1+h)/open(T+1) − 1`，
隐式损失是 MSE —— 它**均匀地**关心当日截面里每一只股票，而实盘只买 5 只。
截面的两端（尤其头部）决定 top_ret，中部只决定 IC 的分母。要同时抬 IC 和 top_ret，
就得把"优化预算"往头部挪，并且把标签换成与"排序"这个真实任务同构的形式。

这里提供四个可独立开关的组件，由单元的 `heads()` 通过 `label_transform` / `sample_weight`
两个参数声明；**哪个有用由榜单排出来，不靠拍脑袋**。

## ★★ 防泄漏铁律二：本文件每一个统计量都只在"当日截面"内计算

`day` 是行 → 交易日的下标（`panel.row_day`），本文件**所有**分组统计一律按 `day` 分组：
秩、分位、尾部阈值 —— 全部是"当日截面内的相对量"。
时间衰减权重虽然跨日，但它只是"日期"的函数（`0.5 ** (age/half_life)`），
不碰任何标签取值，也不碰任何特征取值 ⇒ 不构成前视。

★ 反面教材（不许出现）：全样本 `mean/std/quantile`、用 valid/test 的分布去标准化 train、
用当日**之后**的日期当衰减基准。`main.py audit-pit` 的截断复算会复验这条。
"""
from __future__ import annotations

import numpy as np

# 秩高斯化后的截断：n≈3000 时最极端分位是 0.5/3000 → Φ⁻¹ ≈ −3.6，
# 留一点余量到 ±4，避免出现 ±inf 把损失炸掉。
GAUSS_CLIP = 4.0


# ------------------------------------------------------------------ 工具
def day_bounds(day: np.ndarray) -> np.ndarray:
    """行 → 交易日下标的**连续段边界**（含首尾哨兵）。

    面板的行是"按日、日内按股票代码"排的（见 `mx/data.py` 的网格构造），
    所以同一天的行必然连续 ⇒ 用一次 `diff` 就能拿到所有分组的边界，O(n)。
    返回长度 = 天数 + 1 的下标数组，第 k 天 = `[b[k], b[k+1])`。
    """
    d = np.asarray(day)
    if d.ndim != 1:
        raise ValueError("day 必须是一维行→日下标")
    if d.size == 0:
        return np.zeros(1, dtype=np.int64)
    if d.size > 1 and not np.all(d[1:] >= d[:-1]):
        raise ValueError("day 必须先按交易日升序（面板网格天然满足；乱序说明上游出了事）")
    cut = np.flatnonzero(d[1:] != d[:-1]) + 1
    return np.concatenate([[0], cut, [d.size]]).astype(np.int64)


def _avg_rank(v: np.ndarray) -> np.ndarray:
    """组内平均秩（1-based，并列取平均）——不用 scipy，避免逐日调用的开销。"""
    n = v.size
    order = np.argsort(v, kind="stable")
    sv = v[order]
    new = np.empty(n, dtype=bool)
    new[0] = True
    np.not_equal(sv[1:], sv[:-1], out=new[1:])
    grp = np.cumsum(new) - 1                      # 排序后每个元素所属的并列组号
    counts = np.bincount(grp)
    right = np.cumsum(counts)                     # 组内最大秩
    left = right - counts                         # 组内最小秩 − 1
    avg = (left + right + 1) / 2.0                # 组内平均秩
    out = np.empty(n, dtype=np.float64)
    out[order] = avg[grp]
    return out


def _per_day(y: np.ndarray, day: np.ndarray, fn) -> np.ndarray:
    """按日分组逐段调用 `fn(段内有效值) -> 段内结果`，NaN 原样保留。"""
    y = np.asarray(y, dtype=np.float64)
    out = np.full(y.shape, np.nan, dtype=np.float64)
    b = day_bounds(day)
    for k in range(b.size - 1):
        lo, hi = int(b[k]), int(b[k + 1])
        v = y[lo:hi]
        ok = np.isfinite(v)
        if not ok.any():
            continue
        out[lo:hi][ok] = fn(v[ok])
    return out


# ------------------------------------------------------------------ ① 标签变换
def cs_rank_pct(y: np.ndarray, day: np.ndarray, *, center: bool = True) -> np.ndarray:
    """**当日截面**百分位。

    center=True  → `(r − 0.5)/n − 0.5` ∈ [−0.5, 0.5)　（0 是当日中位）
    center=False → `(r − 0.5)/n` ∈ (0, 1)

    PIT：只用当日截面内各股票的**相对次序**，与全样本分布、未来日期都无关。
    """
    def f(v):
        r = _avg_rank(v)
        p = (r - 0.5) / v.size
        return p - 0.5 if center else p
    return _per_day(y, day, f)


def cs_rank_gauss(y: np.ndarray, day: np.ndarray, *, clip: float = GAUSS_CLIP) -> np.ndarray:
    """**当日截面**秩高斯化：`Φ⁻¹((r − 0.5)/n)`。

    为什么用它当回归目标（而不是原始收益率）：
      · 与"排序"这个真实任务同构 —— 我们关心的就是次序，不是收益的绝对幅度；
      · 有界（±clip）：涨停/重组/退市整理那类把 MSE 拉爆的极端收益被自然压住；
      · 分布稳定：换一个波动率完全不同的年份，目标的量纲不变。

    PIT：同 `cs_rank_pct` —— 秩是当日截面内的相对量。
    """
    from scipy.special import ndtri
    p = cs_rank_pct(y, day, center=False)
    z = np.full(p.shape, np.nan, dtype=np.float64)
    ok = np.isfinite(p)
    z[ok] = ndtri(np.clip(p[ok], 1e-6, 1 - 1e-6))
    return np.clip(z, -clip, clip)


def cs_rank_z(y: np.ndarray, day: np.ndarray) -> np.ndarray:
    """**当日截面**标准化秩：`(rank − mean(rank)) / std(rank)` —— 均值 0、标准差 1，**线性于秩**。

    为什么单开一支（而不是复用 `cs_rank_gauss`）：
      · `cs_rank_gauss` 是 `Φ⁻¹(分位)`，与"秩"是**非线性**关系；
      · 本函数与秩是**严格线性**关系 —— 相关类目标（Pearson/Spearman）对它是尺度不变的，
        但**"加权求和型"的目标对取值形状敏感**：`−Σ softmax(s/τ)·y` 那一项只吃尾部，
        而两支在尾部差得最多（实测同一批数据：±1.71 vs ±2.44），
        换用会让该项的靶子量级差 **2~3 倍**。
      · 参考工程 `autodl-fs/tmp/model.py:_rank_gauss` **名字里带 gauss，做的却是这个**
        （`r = s.rank(); (r − r.mean())/r.std()`）。复刻它必须用这一支，
        否则名义上"同配方"、实际上目标函数的形状已经换了。

    PIT：只用当日截面内的相对次序，与 `cs_rank_gauss` 同源。
    """
    def f(v):
        r = _avg_rank(v)
        sd = r.std()
        if not np.isfinite(sd) or sd <= 1e-12:
            return np.zeros_like(r)
        return (r - r.mean()) / sd
    return _per_day(y, day, f)


_TRANSFORMS = {
    None: lambda y, day: np.asarray(y, dtype=np.float64),
    "none": lambda y, day: np.asarray(y, dtype=np.float64),
    "rank_gauss": cs_rank_gauss,
    "rank_pct": cs_rank_pct,
    "rank_z": cs_rank_z,
}


def cs_grade(y: np.ndarray, day: np.ndarray, *, levels: int = 10) -> np.ndarray:
    """把收益变成**当日截面内的等级** `0..levels-1`（LightGBM 排序目标的相关性标签）。

    排序目标（lambdarank）要求标签是非负整数等级，而不是连续收益。
    等分用当日截面内的百分位切 ⇒ PIT 安全，且**与市场整体涨跌无关**
    （同样的 +2% 在普涨日和普跌日会落在不同的等级，正是我们想要的）。

    ★ 低等级 = 收益低。做多策略只关心高等级那一端，靠 `lambdarank_truncation_level` 收窄。
    """
    p = cs_rank_pct(y, day, center=False)          # (0,1)
    g = np.floor(p * levels)
    return np.clip(np.nan_to_num(g, nan=0.0), 0, levels - 1).astype(np.float64)


def loss_spec(params: dict | None) -> dict:
    """把模型参数里的损失声明**规整成单一形状** —— 全流程只有这一个 spec 形状。

    模型参数里它长这样（两段，因为一个是"目标"、一个是"权重"）：

        {"label_transform": "rank_gauss", "sample_weight": {"half_life": 250, ...}}

    而 `make_target` / `make_weights` / `describe` 要的是**摊平**的一份：

        {"transform": "rank_gauss", "half_life": 250, ...}

    ★ 踩过的坑：`train_one` 把"模型参数"原样喂给 `describe()`，于是 `describe` 去找
      `spec["transform"]`（不存在，实际叫 `label_transform`）和 `spec["half_life"]`
      （实际嵌在 `sample_weight` 里）—— 两个都读不到，日志就谎报成"V2 口径（等权）"，
      而模型其实在用秩高斯 + 衰减权重。**日志说错话比不打印更危险**，
      因为它会让人在归因时把结论安到错误的配方上。摊平成一份就根除了这类错位。
    """
    p = params or {}
    w = p.get("sample_weight") or {}
    spec = {"transform": p.get("label_transform")}
    if isinstance(w, dict):
        spec.update(w)
    # ★ 可执行性掩码：`True` = 用默认阈值，dict = 传阈值给 `PanelData.buyable_mask`。
    spec["buyable"] = p.get("buyable")
    return spec


def make_buyable(panel, spec: dict | None) -> np.ndarray | None:
    """按单元的 `buyable` 声明造**训练样本的可执行性掩码**（不声明就返回 None）。

    声明形状（单元 `model.py:heads()` 里的 overrides）：

        "buyable": True                      # 默认阈值：limit_up_rank = 0.98
        "buyable": {"limit_up_rank": 0.98}   # 显式给阈值
        "buyable": {"limit_up_rank": 0}      # 只按覆盖率，不剔涨停

    ★ 为什么单独成一个函数（而不是塞进 `make_weights`）：掩码的语义是"**这些行不算样本**"，
      不是"这些行权重低"。LightGBM 对 `weight=0` 的行仍会计入 `min_data_in_leaf`、
      仍参与分箱 —— 用"零权重"表达"剔样本"会在树的生长上留下痕迹。
      所以 `mx/train.py` 直接**把行切掉**，而不是把权重压成 0。
    """
    b = (spec or {}).get("buyable")
    if not b:
        return None
    kw = b if isinstance(b, dict) else {}
    return panel.buyable_mask(limit_up_rank=kw.get("limit_up_rank", 0.98))


def make_target(y: np.ndarray, day: np.ndarray, spec: dict | None) -> np.ndarray:
    """按 `spec["transform"]` 把标签变成训练目标。未声明 = 原值（与 V2 完全一致）。"""
    name = (spec or {}).get("transform")
    fn = _TRANSFORMS.get(name)
    if fn is None:
        raise ValueError(f"未知的标签变换 {name!r}（可选：{sorted(k for k in _TRANSFORMS if k)}）")
    return fn(np.asarray(y, dtype=np.float64), day).astype(np.float32)


# ------------------------------------------------------------------ ② 时间衰减权重
def day_decay(dates_arr: np.ndarray, day: np.ndarray, *, half_life: float,
              anchor: str | None = None, floor: float = 1e-3) -> np.ndarray:
    """时间衰减样本权重 `0.5 ** (age / half_life)`，age 按**交易日序号**算。

    市场结构在漂移，2018 年的样本不该和 2025 年等权。`half_life` 是"每过多少个交易日
    权重减半"，250 ≈ 一年。

    ★ 为什么不是前视：权重只是**日期轴**的函数 —— 既不碰标签取值，也不碰特征取值，
      更不碰样本外的任何一天。`anchor` 默认取"本段样本里最晚的那一天"，
      而本段样本就是**训练段**（权重只在 `fit` 里用），所以它相对训练而言也是"当下"。
    """
    u, inv = np.unique(day, return_inverse=True)
    # 每天一个日期字符串 → 位置序号即交易日序号
    n_day = u.size
    if anchor is None:
        a = n_day - 1
    else:
        loc = np.searchsorted(np.asarray(dates_arr, dtype=object)[u], anchor)
        a = int(np.clip(loc, 0, n_day - 1))
    age = np.abs(a - np.arange(n_day, dtype=np.float64))
    w_day = np.maximum(0.5 ** (age / float(half_life)), float(floor))
    return w_day[inv]


# ------------------------------------------------------------------ ③ 尾部权重
def tail_weight(y: np.ndarray, day: np.ndarray, *, frac: float = 0.10,
                boost: float = 2.0, side: str = "top") -> np.ndarray:
    """尾部样本加权：当日截面里 `|秩|` 落在最极端 `frac` 的样本，权重 ×(1+boost)。

    `side`：`top` = 只加权最高分位（做多策略的真实关切）；
            `both` = 两端都加权（对排序的两头都提要求）。
    序号取 `|r − 中位|`，所以 `side="top"` 时用的是**当日截面的相对秩**。

    ★ PIT：阈值是**逐日**用当日截面算的分位，不是全样本分位。
    """
    def f(v):
        n = v.size
        k = max(1, int(round(n * float(frac))))
        r = _avg_rank(v)
        if side == "both":
            # 两端各 k 个：r ≤ k 或 r ≥ n−k+1 ⇔ |r − (n+1)/2| ≥ (n+1)/2 − k
            hit = np.abs(r - (n + 1) / 2.0) >= (n + 1) / 2.0 - k
        else:                                   # "top"：秩最大的 k 个
            hit = r > (n - k)
        return hit.astype(np.float64)
    hit = _per_day(y, day, f)
    return np.where(np.isfinite(hit), 1.0 + float(boost) * np.nan_to_num(hit), 1.0)


# ------------------------------------------------------------------ 组装
def make_weights(dates_arr: np.ndarray, day: np.ndarray, order_by: np.ndarray,
                 spec: dict | None) -> np.ndarray | None:
    """按 `spec` 组装样本权重（多个来源相乘）。全空 ⇒ 返回 None（= 等权，与 V2 一致）。

    参数
    ----
    order_by : 尾部权重的**排序依据**，传**原始收益率**。
      ★ "头部"的投资含义是"未来涨得最多的那几只"，这个定义只在原始收益率的次序下成立。
        秩高斯化是单调变换，用变换后的值排结果一样；显式传原值是为了让语义不依赖变换是否单调。
    spec : 例 `{"half_life": 250, "tail_frac": 0.10, "tail_boost": 2.0, "tail_side": "top"}`
    """
    spec = spec or {}
    hl, tf = spec.get("half_life"), spec.get("tail_frac")
    if not hl and not tf:
        return None
    w = np.ones(day.shape, dtype=np.float64)
    if hl:
        w *= day_decay(dates_arr, day, half_life=float(hl), anchor=spec.get("anchor"))
    if tf:
        w *= tail_weight(order_by, day, frac=float(tf), boost=float(spec.get("tail_boost", 2.0)),
                         side=str(spec.get("tail_side", "top")))
    return w.astype(np.float32)


def describe(spec: dict | None) -> str:
    """一行文字描述，落在日志/榜单里，方便日后归因。"""
    spec = spec or {}
    bits = []
    if spec.get("transform"):
        bits.append(f"标签={spec['transform']}")
    if spec.get("half_life"):
        bits.append(f"衰减hl={spec['half_life']}")
    if spec.get("tail_frac"):
        bits.append(f"尾部{spec.get('tail_side', 'top')}{spec['tail_frac']:.0%}×"
                    f"{1 + float(spec.get('tail_boost', 2.0)):.1f}")
    b = spec.get("buyable")
    if b:
        kw = b if isinstance(b, dict) else {}
        lu = kw.get("limit_up_rank", 0.98)
        bits.append("可执行掩码" + (f"(涨停秩≥{lu})" if lu else "(仅覆盖率)"))
    return " · ".join(bits) or "原值标签·等权（V2 口径）"
