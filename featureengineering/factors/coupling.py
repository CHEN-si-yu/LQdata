"""因子耦合（20 个）—— 把两个**已有因子**的信息交叉起来，或自算微观结构比率。

═══════════════════════════════════════════════════════════════════════════
一、为什么「耦合」值得单独做一族
═══════════════════════════════════════════════════════════════════════════

单因子回答的是「这个维度上谁高谁低」，而很多**有效信号只在条件里出现**：
低波动本身没有方向性，但「低波动 + 强动量」是经典的防守型动量；
筹码获利盘本身弱，但「获利盘高 + 近期超跌」是另一回事。
参考库里这一族有 147 个，但其中约 34 个是 `rank(A) × rank(B)` 的**同构堆砌**
（与母因子共线），23 个是单调变换（`log`/`sqrt`/`zscore` —— 因为下游只用 rank 列，
单调变换**逐位相同、零信息**），本文件按「信息论上真的有增量」筛出 20 个。

═══════════════════════════════════════════════════════════════════════════
二、实现约定（本族特有，别照抄到别的家族）
═══════════════════════════════════════════════════════════════════════════

1. **父因子用 `ctx.load_factor(name)` 读**，它读的是兄弟因子的**产物文件**（精确落格，
   缺失 NaN），所以：
   · 必须把父因子名写进 `deps` —— 引擎靠它做**水位失效传播**
     （父因子更新 → 本因子自动重算，见 `fea/engine.py::_factor_watermark`）；
   · 父因子必须**先算完**：`main.py run` 会分两趟执行（第一趟非耦合、第二趟耦合），
     跑在父因子前面会静默拿到全 NaN。
2. **标准化一律用 `ctx.cs_zscore(x, mask=ctx.universe)`**（逐交易日的截面标准化）。
   ★ 绝不能用全样本均值/标准差 —— 那是前视（用了未来的分布）。契约 §2.3 也禁止在因子里做 rank。
3. **乘积型耦合的方向**：`z(A) × z(B)` 的符号语义是「两个维度同时高/同时低」。
   为了让「值大 = 信号强」可解释，需要**同向**的两个维度相乘；若某个维度天然是「低优」
   （如波动率、非流动性），先取负号再相乘（本文件统一用 `_nz()` 明确表达这件事）。
4. **乘法与减法各管一件事**：乘法 = 共振（两维度同时高），减法 = 错配（一个高一个低）。
   参考库里 `rank(A) − rank(B)` 型有几十个，本文件只保留 3 个分歧型。

## 覆盖与真值约定

- 耦合因子的非空率 = **两个父因子同时有值**的交集。若某个父因子覆盖率低
  （如 `margin_*` 只有两融标的 ≈50%、`limit_up_count_20` 稀疏），交集会更低 ——
  每个因子在 note 里给出由父因子推出的**预期覆盖率**。
- 缺失一律 NaN（不用 0 填充）：`cs_zscore` 的 NaN 会传播，乘法自然得到 NaN。
"""

from __future__ import annotations

import numpy as np

from fea.spec import FactorSpec, register
from .coupling_private_basis import _private_basis_values, _PRIVATE_EQUITY, _PRIVATE_RETURN


GROUP = "coupling"

# 短窗耦合（父因子窗口 ≤ 20 交易日）：200 天够（含父因子 warmup + 冗余）
W_SHORT = 200
# 长窗耦合（父因子用到 60~120 日动量/波动）：给足，宁多勿少
W_LONG = 320


def _z(ctx, mat: np.ndarray) -> np.ndarray:
    """逐日截面 z-score（**只用当日截面**，绝无前视）。缺失保持 NaN。"""
    return np.asarray(ctx.cs_zscore(mat, mask=ctx.universe), dtype=np.float64)


def _f(ctx, name: str) -> np.ndarray:
    """读一个父因子的值（精确对齐到当前面板）。"""
    if name == "cp_internal_book_value":
        return np.asarray(_private_basis_values(ctx, _PRIVATE_EQUITY), dtype=np.float64)
    if name == "cp_internal_short_return":
        return np.asarray(_private_basis_values(ctx, _PRIVATE_RETURN), dtype=np.float64)
    return np.asarray(ctx.load_factor(name), dtype=np.float64)


def _nz(x: np.ndarray) -> np.ndarray:
    """取负号 —— 把「低优」维度翻成「高优」，让乘积的符号可解释。"""
    return -x


def _mul(ctx, a: str, b: str, flip_a: bool = False, flip_b: bool = False) -> np.ndarray:
    """两个父因子的标准化乘积（共振信号）。"""
    za = _z(ctx, _f(ctx, a))
    zb = _z(ctx, _f(ctx, b))
    if flip_a:
        za = _nz(za)
    if flip_b:
        zb = _nz(zb)
    return za * zb


def _diff(ctx, a: str, b: str) -> np.ndarray:
    """两个父因子的标准化之差（分歧/错配信号）。"""
    return _z(ctx, _f(ctx, a)) - _z(ctx, _f(ctx, b))


# ══════════════════════════════════════════════════════════════════════════
# A. 价格 × 风险（3 个）
# ══════════════════════════════════════════════════════════════════════════
@register(FactorSpec(
    name="cp_momentum_lowvol_20", version=2, group=GROUP,
    deps=('stock_daily', 'stock_adj_factor', 'idio_vol_60'),
    desc="防守型动量 = z(20日动量) × z(−特质波动)",
    formula="CP = z(momentum_20) × z(−idio_vol_60)",
    start=None, warmup_days=W_LONG, higher_is_better=True,
    note="★ 低波动本身没有方向（低波动异象的方向是「低波动跑赢」），乘以动量后表达的是"
         "「强势但波动小」的股票。父因子 `momentum_20` 与 `idio_vol_60` 实测在 2026 单年"
         "|RankIC| 分别 0.0424（负向）与 0.0752（负向），两者都强且相关性不高 → 乘积有增量。"
         "预期非空率 ≈ 两父因子交集（两者都是价格类，接近 100%）。",
))
def cp_momentum_lowvol_20(ctx):
    return _mul(ctx, 'cp_internal_short_return', "idio_vol_60", flip_b=True)






# ══════════════════════════════════════════════════════════════════════════
# B. 价值 × 质量 × 成长 × 资金结构（4 个）
# ══════════════════════════════════════════════════════════════════════════


@register(FactorSpec(
    name="cp_value_momentum_div", version=2, group=GROUP, deps=('stock_balancesheet', 'stock_daily', 'stock_finance', 'stock_adj_factor'),
    desc="价值−动量分歧 = z(BP) − z(20日动量)",
    formula="CP = z(bp) − z(momentum_20)",
    start=None, warmup_days=W_LONG, higher_is_better=True,
    note="★ 用**差**而不是积：表达的是「便宜但近期没涨」（错配），与「又便宜又强」（共振）"
         "是两种不同的信号。参考库里这种 `rank(A) − rank(B)` 型有几十个，"
         "本文件只保留 3 个分歧型（价值-动量、两融-趋势、筹码-换手各一）。",
))
def cp_value_momentum_div(ctx):
    return _diff(ctx, 'cp_internal_book_value', 'cp_internal_short_return')














# ══════════════════════════════════════════════════════════════════════════
# D. 筹码 × 价格/量（3 个）
# ══════════════════════════════════════════════════════════════════════════








# ══════════════════════════════════════════════════════════════════════════
# F. 自算微观结构（5 个）—— 不依赖父因子，直接由价格/成交额派生
# ══════════════════════════════════════════════════════════════════════════
@register(FactorSpec(
    name="rv_term_structure", version=2, group=GROUP, deps=("stock_daily", "stock_adj_factor"),
    desc="波动率期限结构 = 20 日 RV / 60 日 RV（短端相对长端的高低）",
    formula="RV_n = std(ret(1), n);  TermStructure = RV_20 / RV_60",
    start=None, warmup_days=W_LONG, higher_is_better=False,
    note="参考库的 `rv_term_structure_slope` 用的是 5min 已实现波动；本实现用**日频收益的"
         "滚动标准差**做同一件事（日频口径更稳、且不受 5min 数据起点限制）。"
         "值 >1 = 短期波动放大（不安定期），<1 = 波动回落。"
         "⚠️ 与已删除的 `vol_20`/`vol_60` 不同：那两个是**水平**，这个是**比率**（不含水平信息）。",
))
def rv_term_structure(ctx):
    r = ctx.ret(1)
    short = ctx.roll_std(r, 20)
    long = ctx.roll_std(r, 60)
    return ctx.safe_div(short, long, min_abs_den=1e-12)


@register(FactorSpec(
    name="vol_of_rv", version=2, group=GROUP, deps=("stock_daily", "stock_adj_factor"),
    desc="波动的波动 = 20 日 RV 在 60 日窗口内的标准差（不稳定度）",
    formula="VoV = std( roll_std(ret(1), 20), 60 )",
    start=None, warmup_days=W_LONG, higher_is_better=False,
    note="表达「波动率本身有多不稳定」。★ 与已删除的 `vol_of_vol_20` 不同源："
         "那个是「波动水平的窗口比较」（与 vol_20 ρ=0.968，已删）；本口径是**嵌套滚动**"
         "（先算 20 日 RV，再取它 60 日的标准差），衡量的是波动率的**变化节奏**。",
))
def vol_of_rv(ctx):
    rv20 = ctx.roll_std(ctx.ret(1), 20)
    return ctx.roll_std(rv20, 60)






