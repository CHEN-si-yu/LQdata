"""面板数学原语 —— 因子作者的数学工具箱。

**契约（与 `Panel.roll_sum` 完全一致，别破坏）**：
  1. 入参与返回都是 `(T, C)` 的 numpy 数组，不出现 DataFrame、不出现字符串列。
  2. 内部用 float64 计算、返回 float64（引擎最后统一转 float32）。
  3. **NaN 策略 = 毒化**：窗口内只要出现过非有限值，结果就是 NaN。
     `min_count=None`（默认）表示「满窗 **且** 无 NaN」；显式给 `min_count=k`
     表示「窗口内至少 k 个有效值」—— 这是**有意放松**，必须在调用点写清楚理由。
     参考库里的 `min_periods=n//2` 对应 `min_count=n//2`。
  4. **绝不就地修改入参**。需要写就 `np.array(..., copy=True)` 先复制。
     （实测踩过 `assignment destination is read-only`：`DataFrame.to_numpy()`
     在 pandas 3.0 下可能返回只读视图。）
  5. **不产生 warning**。全 NaN 行上的归约一律包 `np.errstate(all="ignore")` 收尾。
  6. 不依赖 scipy / numba（**本环境没装 scipy**，`df.corr(method='spearman')` 会直接抛错）。

复杂度：cumsum 类的都是 O(T·C)；`roll_max/min/rank/median` 走
`sliding_window_view` 跨步视图，是 O(T·C·n) 时间但 **O(T·C) 内存**
（视图不物化 (T,C,n) 张量）。
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "shift", "diff", "pct_change", "nan_fill_ffill", "nan_fill_ffill_2d",
    "roll_sum", "roll_mean", "roll_var", "roll_std", "roll_count",
    "roll_max", "roll_min", "roll_argmax", "roll_argmin", "roll_rank",
    "roll_cov", "roll_corr", "roll_skew", "roll_kurt", "roll_prod",
    "ewm_mean", "decay_linear", "signed_power",
    "cs_demean", "cs_zscore", "cs_winsor",
    "safe_div", "safe_log", "safe_sqrt",
]


# ══════════════════════════════════════════════════════════════════════
# 基础工具
# ══════════════════════════════════════════════════════════════════════

def _as_f64(mat) -> np.ndarray:
    """转 float64 的 (T,C) 数组。非有限值（inf）也当缺失。"""
    x = np.asarray(mat, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"mathx 只接受 (T, C) 二维数组，收到 {x.shape}")
    return x


def _finite(x: np.ndarray) -> np.ndarray:
    return np.isfinite(x)


def _poison(out: np.ndarray, x: np.ndarray, n: int, min_count) -> np.ndarray:
    """按 NaN 策略毒化。

    min_count is None -> 窗口必须满 n **且** 全为有限值
    min_count = k      -> 窗口内至少 k 个有限值
    """
    T = x.shape[0]
    if T < n:
        return out
    bad_of = ~_finite(x)
    valid = (~bad_of).astype(np.int64)
    cv = np.vstack([np.zeros((1, valid.shape[1]), dtype=np.int64),
                    np.cumsum(valid, axis=0)])            # (T+1, C)
    cnt = cv[n:] - cv[:-n]                                # (T-n+1, C)
    need = n if min_count is None else int(min_count)
    ok = cnt >= need
    tail = out[n - 1:]
    tail[~ok] = np.nan
    return out


# ══════════════════════════════════════════════════════════════════════
# 位移
# ══════════════════════════════════════════════════════════════════════

def shift(x, k: int, fill=np.nan) -> np.ndarray:
    """沿**交易日**轴位移。k>0 取过去（out[k:] = x[:-k]），k<0 取未来。

    ★ 面板本身就是交易日历，所以「移一行 = 移一个交易日」。
      k<0 是全框架**唯一**读未来的口子，只在标签（label）里用，调用点必须显眼。
    """
    x = _as_f64(x)
    T = x.shape[0]
    out = np.full(x.shape, fill, dtype=np.float64)
    k = int(k)
    if k == 0:
        return x.copy()
    if abs(k) >= T:
        return out
    if k > 0:
        out[k:] = x[:-k]
    else:
        out[:k] = x[-k:]
    return out


def diff(x, k: int = 1) -> np.ndarray:
    x = _as_f64(x)
    return x - shift(x, k)


def pct_change(x, k: int = 1, min_abs_den: float = 1e-12) -> np.ndarray:
    x = _as_f64(x)
    prev = shift(x, k)
    return safe_div(x - prev, np.abs(prev), min_abs_den=min_abs_den)


def nan_fill_ffill(x) -> np.ndarray:
    """前向填充（只在 NaN 处填上一个有限值），O(T·C) 无 Python 循环。"""
    x = _as_f64(x)
    T = x.shape[0]
    if T == 0:
        return x.copy()
    idx = np.where(_finite(x), np.arange(T)[:, None], 0)
    np.maximum.accumulate(idx, axis=0, out=idx)
    # idx 是**行**下标 (T,C)，列下标要另给；写成 x[arange(T), idx] 会张冠李戴
    return x[idx, np.arange(x.shape[1])[None, :]]


nan_fill_ffill_2d = nan_fill_ffill        # 别名，语义相同


# ══════════════════════════════════════════════════════════════════════
# 滚动 —— cumsum 类（O(T·C)）
# ══════════════════════════════════════════════════════════════════════

def _cumsum(x: np.ndarray) -> np.ndarray:
    z = np.zeros((1, x.shape[1]), dtype=np.float64)
    return np.vstack([z, np.cumsum(x, axis=0)])


def roll_sum(x, n: int, min_count=None) -> np.ndarray:
    """滚动求和（含当前，窗口 n）。窗口内出现 NaN 则结果为 NaN。"""
    x = _as_f64(x)
    T = x.shape[0]
    n = int(n)
    out = np.full(x.shape, np.nan, dtype=np.float64)
    if n <= 0 or T < n:
        return out
    xs = np.where(_finite(x), x, 0.0)
    cs = _cumsum(xs)
    out[n - 1:] = cs[n:] - cs[:-n]
    return _poison(out, x, n, min_count)


def roll_count(x, n: int, min_count: int = 0) -> np.ndarray:
    """滚动**有效值个数**。★ 这个函数**不毒化** —— 它本身就是用来度量缺口大小的。"""
    x = _as_f64(x)
    T = x.shape[0]
    n = int(n)
    out = np.full(x.shape, np.nan, dtype=np.float64)
    if n <= 0 or T < n:
        return out
    cs = _cumsum(_finite(x).astype(np.float64))
    out[n - 1:] = cs[n:] - cs[:-n]
    if min_count:
        tail = out[n - 1:]
        tail[~(tail >= min_count)] = np.nan
    return out


def roll_mean(x, n: int, min_count=None) -> np.ndarray:
    """滚动均值。同样先减全局均值（均值在常数平移下只差一个常数），
    避免大均值序列上 `cumsum` 的抵消误差。"""
    x = _as_f64(x)
    c0 = _col_anchor(x)
    s = roll_sum(x - c0, n, min_count)
    c = roll_count(x, n)
    with np.errstate(all="ignore"):
        out = s / c + c0
    out[~_finite(out)] = np.nan
    return out


def roll_var(x, n: int, min_count=None, ddof: int = 0) -> np.ndarray:
    """滚动方差（ddof=0，与 pandas 一致）。

    ★ **先减逐列全局均值再算**。这不是优化，是正确性要求：
      「平方和 − 和的平方/n」对均值很大的序列会灾难性抵消 —— 实测对
      `|x| ~ 1e8` 的列（成交量/成交额的真实量级）做 20 日窗，
      不减中心的相对误差达 **13.9**（结果完全是垃圾）。
      方差在常数平移下不变，所以先减全局均值在数学上等价，
      而把有效精度从 ~1e-16·x² 提到 ~1e-16·σ²。
      减完之后 `|x|~1e8` 的列与逐窗直接计算一致到 **2e-16**。

    ⚠️ **已知边界**：若某一列里混入了比该列典型波动大 5 个数量级的**孤立**
      离群点（例如 0.02 量级的收益序列里塞一个 1e6），列均值会被它拉走，
      此时窗口内所有值都远离中心，精度仍然会丢（实测相对误差 ~1e-2）。
      真实因子输入不会有这种构造；真遇到了请改用逐窗两遍算法。
    """
    x = _as_f64(x)
    T = x.shape[0]
    n = int(n)
    out = np.full(x.shape, np.nan, dtype=np.float64)
    if n <= ddof or T < n:
        return out
    z = _centered(x)
    ok = _finite(z)
    zs = np.where(ok, z, 0.0)
    s1 = _cumsum(zs)
    s2 = _cumsum(zs * zs)
    c = _cumsum(ok.astype(np.float64))
    m1 = s1[n:] - s1[:-n]
    m2 = s2[n:] - s2[:-n]
    cnt = c[n:] - c[:-n]
    # ★ 必须用**有效值个数**做分母，不能用固定 n：
    #   NaN 以 0 进了 cumsum，用 n 当分母会系统性低估方差
    #   （实测窗口 20 缺 3 个：算得 1.125 而正确值 1.219，偏小 8%）。
    need = n if min_count is None else max(int(min_count), ddof + 1)
    cdiv = np.maximum(cnt - ddof, 1.0)
    with np.errstate(all="ignore"):
        v = (m2 - m1 * m1 / np.maximum(cnt, 1.0)) / cdiv
    v[cnt < max(need, ddof + 1)] = np.nan
    v[~np.isfinite(v)] = np.nan
    v[v < 0] = 0.0                       # 浮点误差导致的极小负数
    out[n - 1:] = v
    return out


def roll_std(x, n: int, min_count=None, ddof: int = 0) -> np.ndarray:
    v = roll_var(x, n, min_count, ddof)
    with np.errstate(all="ignore"):
        return np.sqrt(v)


def roll_cov(x, y, n: int, min_count=None) -> np.ndarray:
    """滚动协方差。同样**先各自减因果锚点**（协方差也在常数平移下不变）。

    ★★ 分母 = **有效对数的实际计数**（2026-09-17 修的真实 bug）。
    原实现写的是 `mxy / n - (mx / n) * (my / n)`，而缺失对在上面被填成 0 参与求和 ——
    于是三项被**各自缩放成不同的倍数**（(k/n)·E[xy] − (k/n)²·E[x]E[y]，k = 有效对数），
    连符号都可能反。实测 `002203.SZ` 2015-06-01（60 日窗内 35 对有效）：
    手工真值 **+8.43e-5**，旧实现给 **+1.17e-4**（全局锚定）/ **−3.23e-4**（按年锚定）——
    比值型因子（`beta_60` = cov/var）因此整体失真，且**值还会随面板起点漂移**
    （减锚点后 E[x]、E[y] 不再是 0，缩放错误暴露得最明显）。
    同族的 `roll_var` / `roll_corr` 一直用的是实际计数，只有这里漏了。
    """
    x = _as_f64(x)
    y = _as_f64(y)
    T = x.shape[0]
    n = int(n)
    out = np.full(x.shape, np.nan, dtype=np.float64)
    if T < n:
        return out
    ok = _finite(x) & _finite(y)
    xz = np.where(ok, _centered(x), 0.0)
    yz = np.where(ok, _centered(y), 0.0)
    cxy = _cumsum(xz * yz)
    cx = _cumsum(xz)
    cy = _cumsum(yz)
    cc = _cumsum(ok.astype(np.float64))
    mxy = cxy[n:] - cxy[:-n]
    mx = cx[n:] - cx[:-n]
    my = cy[n:] - cy[:-n]
    cnt = np.maximum(cc[n:] - cc[:-n], 1.0)
    mxc = mx / cnt
    myc = my / cnt
    with np.errstate(all="ignore"):
        out[n - 1:] = mxy / cnt - mxc * myc
    xm = np.where(ok, x, np.nan)          # 任一侧缺失都算缺失
    return _poison(out, xm, n, min_count)


def roll_corr(x, y, n: int, min_count=None) -> np.ndarray:
    """滚动相关（WorldQuant `Correlation(x, y, n)`）。

    ★ 先各自减全局均值再算 —— 与 `roll_var` 同理，否则对 `|x| ~ 1e8` 的序列
      （比如成交量、成交额）结果完全是垃圾。减完之后可以直接喂原始价格。
    """
    x = _as_f64(x)
    y = _as_f64(y)
    T = x.shape[0]
    n = int(n)
    out = np.full(x.shape, np.nan, dtype=np.float64)
    if T < n:
        return out
    ok = _finite(x) & _finite(y)
    xz = np.where(ok, _centered(x), 0.0)
    yz = np.where(ok, _centered(y), 0.0)
    cxy = _cumsum(xz * yz)
    cx = _cumsum(xz)
    cy = _cumsum(yz)
    cx2 = _cumsum(xz * xz)
    cy2 = _cumsum(yz * yz)
    cc = _cumsum(ok.astype(np.float64))
    mxy = cxy[n:] - cxy[:-n]
    mx = cx[n:] - cx[:-n]
    my = cy[n:] - cy[:-n]
    mx2 = cx2[n:] - cx2[:-n]
    my2 = cy2[n:] - cy2[:-n]
    cnt = np.maximum(cc[n:] - cc[:-n], 1.0)
    # ★ 必须用**有效值对数**做中心化与缩放，不能沿用固定 n：
    #   沿用 n 等于把缺失对当成 (x̄, ȳ) 补进去，相关系数被系统性拉向 0
    #   （实测 −0.2063 vs 正确的 −0.2083）。
    mxc = mx / cnt
    myc = my / cnt
    sxx = mx2 - cnt * mxc * mxc
    syy = my2 - cnt * myc * myc
    sxy = mxy - cnt * mxc * myc
    with np.errstate(all="ignore"):
        den = np.sqrt(np.maximum(sxx, 0.0) * np.maximum(syy, 0.0))
        val = np.where(den > 0, sxy / np.where(den > 0, den, 1.0), np.nan)
    # ★ |ρ| ≤ 1 兜底：窗口内接近常数（长期停牌时前向填充出的平序列）时，
    #   分子分母都是浮点噪声，实测会吐出 |ρ| 高达 **14.4** 的垃圾值。
    #   这是「无定义判缺失」，不是 winsor —— 不能把它留给下游当极值信号。
    out[n - 1:] = np.where(np.abs(val) <= 1.0, val, np.nan)
    xm = np.where(ok, x, np.nan)
    return _poison(out, xm, n, min_count)


def _col_mean(x: np.ndarray) -> np.ndarray:
    """逐列均值，全 NaN 列为 0.0；**不产生 warning**。

    `np.nanmean` 在全 NaN 列上会抛 "Mean of empty slice" 的 RuntimeWarning，
    而本项目的面板里全 NaN 列很常见（某段时间还没上市的股票）。
    """
    fin = _finite(x)
    n = fin.sum(axis=0)
    s = np.where(fin, x, 0.0).sum(axis=0)
    with np.errstate(all="ignore"):
        mu = s / np.maximum(n, 1)
    return np.where(n > 0, mu, 0.0)


def _col_anchor(x: np.ndarray) -> np.ndarray:
    """逐列的**因果锚点** = 该列第一个有效值（全 NaN 列取 0.0）。

    ★ 为什么不用 `_col_mean`（整列均值）做数值中心：
      它在数学上会被减掉、不影响结果，但**浮点结果会依赖面板右端有多长** ——
      同一交易日 T 的值会随"这次算到哪天"发生 ~1e-12 量级的漂移
      （在近零的格子上相对误差能放大到 1e-4 以上）。实测：
      `main.py audit-pit` 的截断复算在 `dividend_yield_3y_avg` 上因此报出
      1.644e-04 的相对差 —— 不是前视，但**同样违反"同一个 T 任何时候算都一样"**。

      锚点取"第一个有效值"而不是"前 k 行均值"，有两个好处：
      ① 它只依赖面板**顶部**（= 任务窗口起点），与右端无关 → 截断重算逐位一致；
      ② 它自动贴合每个量级（成交量列 ~1e8、比率列 ~1e-3），
         与原来的"减列均值"有同样的条件数收益。
    """
    fin = _finite(x)
    T = x.shape[0]
    if T == 0:
        return np.zeros(x.shape[1], dtype=np.float64)
    has = fin.any(axis=0)
    first = np.argmax(fin, axis=0)                 # 全 NaN 列返回 0（下面用 has 掩掉）
    vals = x[first, np.arange(x.shape[1])]
    return np.where(has, vals, 0.0)


def _centered(x: np.ndarray) -> np.ndarray:
    """减去逐列**因果锚点**（该列第一个有效值）。

    三阶/四阶矩、方差、协方差都对「均值很大」的序列会灾难性抵消。
    它们在常数平移下不变，所以减一个中心值既去掉了抵消、又保持 O(T·C)。

    ★ 实测：不减的话，`rolling.std` 在 `|x| ~ 1e8` 的列上相对误差达 **13.9**；
      pandas 自己的 `rolling.corr` 在同一列上会算出 **−3.16**（相关系数越界！）。
      减完之后与直接计算逐窗结果一致到 1e-12。

    ★★ 中心值用 `_col_anchor`（列首第一个有效值）而**不是**整列均值：
      「同一个交易日 T 的值，任何时候算都必须一样」（用户的前视红线）。
      整列均值会把面板右端的信息带进浮点结果 —— 数学上等价，
      但审计实测能在近零格子上放大到 1.6e-4 的相对差。见 `_col_anchor` 的说明。
    """
    return x - _col_anchor(x)


def _cum_moments(z: np.ndarray, n: int, order: int):
    """围绕**逐列全局均值**的 1..order 阶滚动中心矩（已减去窗口均值）。

    减全局均值只是为了让三阶/四阶矩的「计算式」不灾难性抵消（矩在常数平移下不变）；
    减完之后仍然要按标准公式去掉窗口均值 m1 的贡献，才是真正的中心矩。
    """
    zf = np.where(_finite(z), z, 0.0)
    cs = [_cumsum(zf ** k) for k in range(1, order + 1)]
    m = [(c[n:] - c[:-n]) / n for c in cs]
    return m


def roll_skew(x, n: int, min_count=None) -> np.ndarray:
    """滚动**无偏**偏度，与 pandas `.rolling(n).skew()` 同口径。

    pandas 的约定：g1 = m3_c / var^1.5（有偏），再乘 `sqrt(n(n-1))/(n-2)` 去偏。
    """
    x = _as_f64(x)
    T = x.shape[0]
    n = int(n)
    out = np.full(x.shape, np.nan, dtype=np.float64)
    if n < 3 or T < n:
        return out
    m1, m2, m3 = _cum_moments(_centered(x), n, 3)
    var = m2 - m1 * m1
    m3c = m3 - 3 * m1 * m2 + 2 * m1 ** 3
    with np.errstate(all="ignore"):
        g1 = m3c / np.power(var, 1.5)
        out[n - 1:] = np.sqrt(n * (n - 1.0)) / (n - 2.0) * g1
    return _poison(out, x, n, min_count)


def roll_kurt(x, n: int, min_count=None) -> np.ndarray:
    """滚动**无偏超额**峰度，与 pandas `.rolling(n).kurt()` 同口径。

    ⚠️ 返回的是「减 3」之后的值（正态分布 ≈ 0），不是原始四阶矩比值。
    pandas 的约定：g2 = m4_c / var² − 3（有偏超额），
    再套 `g2' = ((n+1)g2 + 6) · (n−1)/((n−2)(n−3))` 去偏。
    """
    x = _as_f64(x)
    T = x.shape[0]
    n = int(n)
    out = np.full(x.shape, np.nan, dtype=np.float64)
    if n < 4 or T < n:
        return out
    m1, m2, m3, m4 = _cum_moments(_centered(x), n, 4)
    var = m2 - m1 * m1
    m4c = m4 - 4 * m1 * m3 + 6 * m1 * m1 * m2 - 3 * m1 ** 4
    with np.errstate(all="ignore"):
        g2 = m4c / (var * var) - 3.0
        out[n - 1:] = ((n + 1.0) * g2 + 6.0) * (n - 1.0) / ((n - 2.0) * (n - 3.0))
    return _poison(out, x, n, min_count)


# ══════════════════════════════════════════════════════════════════════
# 滚动 —— 跨步视图类（O(T·C·n) 时间 / O(T·C) 内存）
# ══════════════════════════════════════════════════════════════════════

def _windows(x: np.ndarray, n: int) -> np.ndarray:
    """(T-n+1, C, n) 的**只读跨步视图**，不物化。"""
    return np.lib.stride_tricks.sliding_window_view(x, n, axis=0)


def roll_max(x, n: int, min_count=None) -> np.ndarray:
    x = _as_f64(x)
    T = x.shape[0]
    n = int(n)
    out = np.full(x.shape, np.nan, dtype=np.float64)
    if n <= 0 or T < n:
        return out
    xf = np.where(_finite(x), x, -np.inf)
    with np.errstate(all="ignore"):
        out[n - 1:] = np.maximum.reduce(_windows(xf, n), axis=-1)
    return _poison(out, x, n, min_count)


def roll_min(x, n: int, min_count=None) -> np.ndarray:
    x = _as_f64(x)
    T = x.shape[0]
    n = int(n)
    out = np.full(x.shape, np.nan, dtype=np.float64)
    if n <= 0 or T < n:
        return out
    xf = np.where(_finite(x), x, np.inf)
    with np.errstate(all="ignore"):
        out[n - 1:] = np.minimum.reduce(_windows(xf, n), axis=-1)
    return _poison(out, x, n, min_count)


def _roll_arg(x: np.ndarray, n: int, find_max: bool) -> np.ndarray:
    """窗口内极值出现在「距今天多少个 bar 之前」（0 = 就是今天）。"""
    x = _as_f64(x)
    T = x.shape[0]
    n = int(n)
    out = np.full(x.shape, np.nan, dtype=np.float64)
    if n <= 0 or T < n:
        return out
    fill = -np.inf if find_max else np.inf
    xf = np.where(_finite(x), x, fill)
    Tn = T - n + 1
    step = 1.0
    for t in range(Tn):
        w = xf[t:t + n]                                  # (n, C) 视图
        if find_max:
            pos = np.argmax(w, axis=0)
        else:
            pos = np.argmin(w, axis=0)
        # 窗口最后一行的下标是 n-1；距今天 = (n-1) - pos
        out[t + n - 1] = (n - 1) - pos
    return _poison(out, x, n, None)


def roll_argmax(x, n: int) -> np.ndarray:
    """窗口内最大值距今天多少个 bar（0 = 今天创的窗口新高）。"""
    return _roll_arg(x, n, True)


def roll_argmin(x, n: int) -> np.ndarray:
    return _roll_arg(x, n, False)


def roll_rank(x, n: int, min_count=None, pct: bool = True) -> np.ndarray:
    """时间序列分位（WorldQuant `Ts_Rank`）：**窗口最后一个值在窗口内的百分位**。

    返回 [0,1]；`pct=False` 时返回名次（0..cnt-1）。
    """
    x = _as_f64(x)
    T = x.shape[0]
    n = int(n)
    out = np.full(x.shape, np.nan, dtype=np.float64)
    if n <= 1 or T < n:
        return out
    xf = np.where(_finite(x), x, np.nan)
    Tn = T - n + 1
    # 分块，避免一次物化过大的布尔张量
    step = max(1, 16)
    for s in range(0, Tn, step):
        e = min(Tn, s + step)
        w = xf[s:e + n - 1]                              # (e-s+n-1, C)
        win = np.lib.stride_tricks.sliding_window_view(w, n, axis=0)   # (e-s, C, n)
        last = win[:, :, -1][:, :, None]
        with np.errstate(all="ignore"):
            cnt = np.sum(np.isfinite(win), axis=-1)
            less = np.sum(win < last, axis=-1)
        if pct:
            with np.errstate(all="ignore"):
                val = np.where(cnt >= 2, less / np.maximum(cnt - 1, 1), np.nan)
        else:
            val = np.where(cnt >= 2, less.astype(np.float64), np.nan)
        out[s + n - 1: e + n - 1] = val
    return _poison(out, x, n, min_count)


def roll_quantile(x, n: int, q: float, min_count=None) -> np.ndarray:
    """滚动分位数（线性插值，与 pandas `.rolling(n).quantile(q)` 同口径）。

    用于历史模拟法的 VaR / CVaR。O(T·C·n·log n) —— n 别给太大。
    """
    x = _as_f64(x)
    T = x.shape[0]
    n = int(n)
    out = np.full(x.shape, np.nan, dtype=np.float64)
    if n <= 0 or T < n:
        return out
    Tn = T - n + 1
    step = max(1, 200 // max(n, 1) + 1)
    import warnings
    for s in range(0, Tn, step):
        e = min(Tn, s + step)
        w = np.lib.stride_tricks.sliding_window_view(x[s:e + n - 1], n, axis=0)
        with np.errstate(all="ignore"), warnings.catch_warnings():
            # 热路径上全 NaN 窗口很常见（停牌），numpy 会刷上千条
            # "All-NaN slice encountered" —— 纯噪音，吞掉。
            warnings.simplefilter("ignore", RuntimeWarning)
            out[s + n - 1: e + n - 1] = np.nanquantile(w, q, axis=-1)
    return _poison(out, x, n, min_count)


def roll_prod(x, n: int, min_count=None) -> np.ndarray:
    x = _as_f64(x)
    T = x.shape[0]
    n = int(n)
    out = np.full(x.shape, np.nan, dtype=np.float64)
    if n <= 0 or T < n:
        return out
    xf = np.where(_finite(x), x, 1.0)
    with np.errstate(all="ignore"):
        out[n - 1:] = np.prod(_windows(xf, n), axis=-1)
    return _poison(out, x, n, min_count)


# ══════════════════════════════════════════════════════════════════════
# 加权
# ══════════════════════════════════════════════════════════════════════

def ewm_mean(x, span: int | None = None, alpha: float | None = None,
             min_count: int = 1) -> np.ndarray:
    """指数加权均值（与 pandas `ewm(span=..., adjust=False).mean()` 同口径）。

    本环境没有 scipy，用 T 次 Python 循环实现 —— 每次是一整行的向量运算，
    723 行 × 3484 列约 3 ms，可接受。

    NaN 语义：遇到 NaN 时**保持上一个 EWMA 值不变**（pandas 的 ignore_na=False
    行为略有不同；这里选择「不把缺失当 0」），有效观测数 < min_count 前为 NaN。
    """
    x = _as_f64(x)
    T, C = x.shape
    if alpha is None:
        if span is None:
            raise ValueError("ewm_mean 需要 span 或 alpha")
        alpha = 2.0 / (span + 1.0)
    out = np.full((T, C), np.nan, dtype=np.float64)
    state = np.zeros(C, dtype=np.float64)
    seen = np.zeros(C, dtype=np.int64)
    started = np.zeros(C, dtype=bool)
    fin = _finite(x)
    for t in range(T):
        row = x[t]
        f = fin[t]
        # 已启动的列：state = a*x + (1-a)*state
        upd = started & f
        state[upd] = alpha * row[upd] + (1.0 - alpha) * state[upd]
        # 未启动但本行有效的列：直接用当行值起步
        new = (~started) & f
        state[new] = row[new]
        started |= new
        seen += f
        ok = started & (seen >= max(1, int(min_count)))
        out[t, ok] = state[ok]
    return out


def decay_linear(x, n: int, min_count=None) -> np.ndarray:
    """线性衰减加权和（WorldQuant `DecayLinear`）：权重 n, n-1, ..., 1（归一化）。"""
    x = _as_f64(x)
    T = x.shape[0]
    n = int(n)
    out = np.full(x.shape, np.nan, dtype=np.float64)
    if n <= 0 or T < n:
        return out
    w = np.arange(1, n + 1, dtype=np.float64)
    w /= w.sum()
    xf = np.where(_finite(x), x, 0.0)
    with np.errstate(all="ignore"):
        out[n - 1:] = np.tensordot(_windows(xf, n), w, axes=([-1], [0]))
    return _poison(out, x, n, min_count)


def signed_power(x, p: float) -> np.ndarray:
    """`sign(x) * |x| ** p`（WorldQuant `SignedPower`）。"""
    x = _as_f64(x)
    with np.errstate(all="ignore"):
        return np.sign(x) * np.power(np.abs(x), p)


# ══════════════════════════════════════════════════════════════════════
# 截面（逐行）
# ══════════════════════════════════════════════════════════════════════

def _apply_mask(x: np.ndarray, mask) -> np.ndarray:
    if mask is None:
        return x
    m = np.asarray(mask, dtype=bool)
    if m.shape != x.shape:
        raise ValueError(f"mask 形状 {m.shape} 与面板 {x.shape} 不符")
    return np.where(m, x, np.nan)


def _row_reduce(x: np.ndarray, fn, min_count: int):
    with np.errstate(all="ignore"):
        cnt = np.sum(_finite(x), axis=1)
        val = fn(x, axis=1)
    val = np.where(cnt >= max(1, int(min_count)), val, np.nan)
    return val, cnt


def cs_demean(x, mask=None, min_count: int = 1) -> np.ndarray:
    """逐行（逐交易日）减去截面均值。"""
    x = _apply_mask(_as_f64(x), mask)
    mu, _ = _row_reduce(x, np.nanmean, min_count)
    return x - mu[:, None]


def cs_zscore(x, mask=None, min_count: int = 1, clip: float | None = None) -> np.ndarray:
    """逐行标准化。`clip` 给定时对结果做 ±clip 截断（防极端值）。"""
    x = _apply_mask(_as_f64(x), mask)
    mu, cnt = _row_reduce(x, np.nanmean, min_count)
    sd, _ = _row_reduce(x, np.nanstd, min_count)
    with np.errstate(all="ignore"):
        z = (x - mu[:, None]) / sd[:, None]
    z[~_finite(z)] = np.nan
    if clip is not None:
        z = np.clip(z, -clip, clip)
    return z


def cs_winsor(x, lo: float = 0.01, hi: float = 0.99, mask=None) -> np.ndarray:
    """逐行缩尾。全 NaN 行保持全 NaN。"""
    x = _apply_mask(_as_f64(x), mask)
    T = x.shape[0]
    out = x.copy()
    with np.errstate(all="ignore"):
        for t in range(T):
            row = x[t]
            f = np.isfinite(row)
            if f.sum() < 2:
                continue
            q = np.quantile(row[f], [lo, hi])
            out[t] = np.clip(row, q[0], q[1])
    return out


# ══════════════════════════════════════════════════════════════════════
# 安全数学
# ══════════════════════════════════════════════════════════════════════

def safe_div(num, den, min_abs_den: float = 0.0) -> np.ndarray:
    """带分母保护的除法：分母非有限、或 |分母| 过小时返回 NaN 而不是 inf/巨值。

    与 `ctx.safe_div` 同语义（那个是静态方法，这里是供 mathx 内部复用）。
    """
    d = np.asarray(den, dtype=np.float64)
    num = np.asarray(num, dtype=np.float64)
    with np.errstate(all="ignore"):
        out = np.full(np.broadcast_shapes(num.shape, d.shape), np.nan, dtype=np.float64)
        ok = np.isfinite(d) & (np.abs(d) > min_abs_den)
        np.divide(num, d, out=out, where=ok)
    out[~np.isfinite(out)] = np.nan
    return out


def safe_log(x) -> np.ndarray:
    """log(x)，x <= 0 处返回 NaN（不产生 -inf / warning）。"""
    x = _as_f64(x)
    with np.errstate(all="ignore"):
        out = np.log(np.where(x > 0, x, np.nan))
    return out


def safe_sqrt(x) -> np.ndarray:
    x = _as_f64(x)
    with np.errstate(all="ignore"):
        return np.sqrt(np.where(x >= 0, x, np.nan))
