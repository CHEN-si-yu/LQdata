"""评价 —— 与模块② `fea/eval.py` **同一套口径**，但把 ICIR/t 算对。

口径（逐字对齐 `fea/eval.py`）：
  · **逐日截面**算 IC（Pearson）/ RankIC（Spearman = 先取秩再 Pearson），再对日**等权算术平均**；
  · 当日有效样本 < `min_n` 就**整天丢弃**；`min_n = min_cross_section // 3`（模块② 取 33）；
  · 分层用十分位 `D9 − D0`，单调性 = `spearman(arange(10), 各层均值)`。

★ 与模块② 的差异（有意为之）：**ICIR/t 从日频 IC 序列现算**。
  模块② 的 `summary.json` 里 `icir`/`t` 全是 NaN —— 它按"年份分区"算，而当前只有 2026 一个分区，
  于是守卫 `ics.size > 1` 永不成立（实测 0/225 有效）。模型侧直接给日频序列的 ICIR/t，别继承那个坑。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _rank(a: np.ndarray) -> np.ndarray:
    """平均名次秩（与 pandas rank 语义一致）——无 scipy 时的 Spearman 预处理。"""
    order = np.argsort(a, kind="stable")
    r = np.empty(len(a), dtype=np.float64)
    r[order] = np.arange(1, len(a) + 1, dtype=np.float64)
    # 并列取平均名次
    s = a[order]
    i = 0
    while i < len(s):
        j = i + 1
        while j < len(s) and s[j] == s[i]:
            j += 1
        if j - i > 1:
            r[order[i:j]] = r[order[i:j]].mean()
        i = j
    return r


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2:
        return np.nan
    a = a - a.mean()
    b = b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else np.nan


def daily_ic(dates: np.ndarray, score: np.ndarray, label: np.ndarray,
             min_n: int = 33) -> tuple[list[str], np.ndarray, np.ndarray]:
    """逐日截面 IC / RankIC（返回 `(天列表, ic 序列, rankic 序列)`）。"""
    s = pd.DataFrame({"d": np.asarray(dates, dtype=object), "p": score, "y": label})
    ok = np.isfinite(s["p"]) & np.isfinite(s["y"])
    s = s[ok]
    days, ics, rics = [], [], []
    for d, g in s.groupby("d", sort=True):
        if len(g) < min_n:
            continue
        p = g["p"].to_numpy(dtype=np.float64)
        y = g["y"].to_numpy(dtype=np.float64)
        ic = _pearson(p, y)
        ric = _pearson(_rank(p), _rank(y))
        if np.isfinite(ic):
            days.append(str(d))
            ics.append(ic)
            rics.append(ric)
    return days, np.asarray(ics, dtype=np.float64), np.asarray(rics, dtype=np.float64)


def summarize(ics: np.ndarray, rics: np.ndarray) -> dict:
    """IC 序列 → 汇总指标（含**从日频序列算的** ICIR / t）。"""
    out = {"n_days": int(len(ics))}
    for tag, v in (("ic", ics), ("rankic", rics)):
        if len(v) == 0:
            out.update({f"{tag}_mean": None, f"{tag}_std": None,
                        f"{tag}_icir": None, f"{tag}_t": None, f"{tag}_pos": None})
            continue
        m, sd = float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else 0.0
        out[f"{tag}_mean"] = m
        out[f"{tag}_std"] = sd
        out[f"{tag}_icir"] = float(m / sd) if sd > 0 else None
        out[f"{tag}_t"] = float(m / (sd / np.sqrt(len(v)))) if sd > 0 else None
        out[f"{tag}_pos"] = float(np.mean(v > 0))
    return out


def decile_table(dates: np.ndarray, score: np.ndarray, label: np.ndarray,
                 n: int = 10, min_n: int = 33) -> tuple[np.ndarray, float]:
    """十分位分层：返回 (各层日均收益, 单调性 spearman)。

    ⚠️ 与模块② 一致：用 `pd.qcut(duplicates='drop')` —— 打分离散时会少于 n 层。
    """
    s = pd.DataFrame({"d": np.asarray(dates, dtype=object), "p": score, "y": label})
    s = s[np.isfinite(s["p"]) & np.isfinite(s["y"])]
    rows = []
    for _d, g in s.groupby("d", sort=True):
        if len(g) < min_n:
            continue
        try:
            q = pd.qcut(g["p"], n, labels=False, duplicates="drop")
        except ValueError:
            continue
        if q.nunique() < 2:
            continue
        rows.append(g.groupby(q)["y"].mean())
    if not rows:
        return np.zeros(0), np.nan
    D = pd.concat(rows, axis=1).mean(axis=1).to_numpy(dtype=np.float64)
    mono = _pearson(_rank(np.arange(len(D), dtype=np.float64)), _rank(D))
    return D, mono


def topn_turnover(dates: np.ndarray, codes: np.ndarray, score: np.ndarray,
                  top_n: int = 50) -> float:
    """等权 top-N 组合的**日均单边换手率**（1.0 = 全部换掉）。"""
    s = pd.DataFrame({"d": np.asarray(dates, dtype=object),
                      "c": np.asarray(codes, dtype=object), "p": score})
    s = s[np.isfinite(s["p"])]
    prev: set | None = None
    turns = []
    for _d, g in s.groupby("d", sort=True):
        if len(g) < top_n:
            continue
        cur = set(g.nlargest(top_n, "p")["c"])
        if prev is not None and prev:
            turns.append(len(cur - prev) / float(top_n))
        prev = cur
    return float(np.mean(turns)) if turns else np.nan


def evaluate(panel_rows_dates, panel_rows_codes, score, label_vals, cfg,
             top_n: int | None = None) -> dict:
    """一次算全套指标。输入都是"同一批行"的等长数组。"""
    min_n = max(2, cfg.min_cross_section // 3)
    days, ics, rics = daily_ic(panel_rows_dates, score, label_vals, min_n=min_n)
    m = summarize(ics, rics)
    D, mono = decile_table(panel_rows_dates, score, label_vals, min_n=min_n)
    m["mono"] = None if not np.isfinite(mono) else float(mono)
    m["spread"] = float(D[-1] - D[0]) if len(D) >= 2 else None
    m["deciles"] = [float(x) for x in D] if len(D) else []
    m["coverage"] = float(np.isfinite(score).mean())
    m["turnover"] = topn_turnover(panel_rows_dates, panel_rows_codes, score,
                                  int(top_n or cfg.raw["backtest"]["top_n"]))
    return m


def _stability(tag: str, v: list[float], ks=(1, 2, 5)) -> dict:
    """**"种子中彩票"的照妖镜**：把最好的那几天剔掉，Σ 还剩多少。

    为什么必须有这一层：Σtop1 是 250 多个交易日**累加**出来的。若某天恰好押中一只涨停，
    那一天就能贡献整个 Σ 的一大截 —— 换个随机种子，那只票可能根本进不了候选。
    参考工程实测过同配方换种子 +73.7% → +2.7%（README §7）。所以本模块里
    **任何"新纪录"都必须先过这一关**：剔掉最好的 1/2/5 天之后还立得住，才算真信号。

    返回：
      · `_drop{k}`     = 剔掉贡献最大的 k 天之后的累加和（**最直接的稳健性判据**）
      · `_max_day`     = 单日最大贡献（绝对值上是否离谱，一眼看得出）
      · `_max_share`   = 单日最大贡献占 Σ 的比例（> 0.5 基本就是靠一天）
      · `_pos_frac`    = 正贡献天数占比
      · `_h1` / `_h2`  = 前半段 / 后半段的累加和（时间上稳不稳）
    """
    a = np.asarray(v, dtype=np.float64)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {}
    tot = float(a.sum())
    o = np.sort(a)[::-1]
    out = {f"{tag}_max_day": float(o[0]),
           f"{tag}_pos_frac": float((a > 0).mean()),
           f"{tag}_max_share": (float(o[0] / tot) if tot > 0 else float("nan"))}
    for k in ks:
        # 天数不够剔就不给数 —— 报一个"剔完还是正"的假数字比不报更危险
        out[f"{tag}_drop{k}"] = (float(tot - o[:k].sum()) if a.size > k else float("nan"))
    h = a.size // 2
    if h >= 5:
        out[f"{tag}_h1"] = float(a[:h].sum())
        out[f"{tag}_h2"] = float(a[h:].sum())
    return out


def costfree_cumsum(dates, score: np.ndarray, ret1d: np.ndarray,
                    enterable: np.ndarray | None = None, n_list=(1, 3, 5)) -> dict:
    """**可执行的无摩擦口径**：买 top-n、隔日换手的收益**累加求和**（主判据）。

    口径精确到可以照着下单：
      · 每天收盘后拿到打分 → 次日开盘买入当日分数最高的 n 只 → 再下一日开盘卖出并换仓；
      · 一天的收益 = 所持 n 只的 `label_ret_1d` 等权平均
        （`label_ret_1d(T) = open(T+2)/open(T+1) − 1`，正是"T+1 开盘买、T+2 开盘卖"）；
      · **累加**而不是连乘 —— 用户明确要求"收益用累计求和即可，没必要乘除法"。

    ★★ `enterable`（次日开盘**买得进**）不是可选项，是必须项。

      2026-09-18 实测踩到的坑：不套这个掩码时，`gbdt_w/label_ret_5d` 的 Σtop1 = **+3.638**；
      套上之后掉到 **+0.289** —— 因为模型有 **37.9% 的交易日**选中的是次日**一字涨停/停牌**的票，
      那些天平均 +3.49%，而买得进的天只有 +0.18%。**92% 的"收益"根本买不到。**

      模型为什么会这样：一字涨停通常意味着次日继续大涨，而"接近涨停"这件事本身
      在特征里是可见的（动量、量比、涨停计数…）。于是任何**不看可交易性**的目标函数
      都会自发地往这个方向收敛 —— 它优化的是一个**不可执行**的目标。
      这不是"交易摩擦"（费用/滑点），是**可执行性**，属于模块③ 的硬约束 #4。

      ⇒ 因此：`enterable` 为 None 时**照算但会在返回值里标记 `exec_filter=False`**，
        调用方必须显式选择；默认路径（`mx/analyze.py`）一律传真实的可买掩码。

    ★ 为什么是累加而不是连乘：小资金整手交易下，连乘会把"某天买不起一手而空仓"这种
      离散效应放大成一个说不清的数。带费用、整手、涨跌停顺延的**真实现金级净值**由
      `mx/strategy.py` 另出一份 —— 两者一起看：这份判断"信号有没有用"，
      那份判断"扣掉一切之后还赚不赚"。

    返回 `{top{n}_sum, uni_sum, excess_top{n}, days, exec_filter}`：
      · `*_sum` = 累加收益（单位与 label 一致，0.01 = 1%）
      · `uni_sum` = 当日可执行截面**等权全买**的累加收益（零信号对照）
        —— 跑不赢它说明只是吃到了 beta，不是 alpha
      · `exec_filter` = 是否套了可买掩码（False 的数字**不可用于结论**）
    """
    sc = np.asarray(score, dtype=np.float64)
    r1 = np.asarray(ret1d, dtype=np.float64)
    dd = np.asarray(dates, dtype=object)
    sel_ok = np.isfinite(sc) & np.isfinite(r1)
    if enterable is not None:
        sel_ok = sel_ok & np.asarray(enterable, dtype=bool)
    out: dict = {"days": 0, "uni_sum": 0.0, "exec_filter": enterable is not None}
    for n in n_list:
        out[f"top{n}_sum"] = 0.0
    tot_uni, n_day = 0.0, 0
    daily: dict[int, list[float]] = {n: [] for n in n_list}   # ★ 逐日贡献，供稳定性层用
    for d in np.unique(dd):
        m = (dd == d) & sel_ok
        if int(m.sum()) < max(n_list):
            continue
        r_sel, s_sel = r1[m], sc[m]
        tot_uni += float(np.mean(r_sel))          # 等权全买：零信号对照
        order = np.argsort(-s_sel, kind="stable")
        for n in n_list:
            v = float(np.mean(r_sel[order[:n]]))
            out[f"top{n}_sum"] += v
            daily[n].append(v)
        n_day += 1
    if n_day == 0:
        return {}
    out["days"] = n_day
    out["uni_sum"] = float(tot_uni)
    for n in n_list:
        out[f"excess_top{n}"] = out[f"top{n}_sum"] - tot_uni
        out.update(_stability(f"top{n}", daily[n]))      # ★ 稳健性：剔掉最好的 k 天还剩多少
    # ★ NaN → None：这些数会落进 leaderboard.json，而 JSON 里 `NaN` 是非法字面量
    #   （Python 自己能读回来，别的解析器会炸）。"算不出来"必须是 null，不能是 NaN。
    res = {}
    for k, v in out.items():
        if isinstance(v, bool) or isinstance(v, int):
            res[k] = v
        else:
            fv = float(v)
            res[k] = round(fv, 6) if np.isfinite(fv) else None
    return res


def disp_width(s: str) -> int:
    """终端显示宽度（CJK 字符占 2 列）—— 中文表头不对齐的根因。"""
    import unicodedata
    return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in str(s))


def pad(s, width: int, right: bool = False) -> str:
    """按**显示宽度**补齐（不是 len）。"""
    s = str(s)
    n = max(0, width - disp_width(s))
    return (" " * n + s) if right else (s + " " * n)


def fmt(v, p: int = 4) -> str:
    """数值格式化：None / NaN / inf 一律显示 `—`（不打印 nan 这种噪声）。"""
    if v is None:
        return "—"
    try:
        fv = float(v)
    except (TypeError, ValueError):
        return str(v)
    return "—" if not np.isfinite(fv) else f"{fv:+.{p}f}"


def render(metrics: dict, title: str = "") -> str:
    """一段式控制台输出（对齐模块② 的紧凑风格）。"""
    return "\n".join([
        f"  {title}" if title else "",
        "    RankIC {m} ± {s} · ICIR {icir} · t {t} · 正比例 {pos} · 天数 {nd}".format(
            m=fmt(metrics.get("rankic_mean")), s=fmt(metrics.get("rankic_std")),
            icir=fmt(metrics.get("rankic_icir"), 2), t=fmt(metrics.get("rankic_t"), 2),
            pos=fmt(metrics.get("rankic_pos"), 2), nd=metrics.get("n_days")),
        "    IC {ic} · 分层 D9−D0 {sp} · 单调 {mono} · 覆盖 {cov} · top-N 换手 {to}".format(
            ic=fmt(metrics.get("ic_mean")), sp=fmt(metrics.get("spread")),
            mono=fmt(metrics.get("mono"), 2), cov=fmt(metrics.get("coverage"), 3),
            to=fmt(metrics.get("turnover"), 3)),
    ]).strip("\n")
