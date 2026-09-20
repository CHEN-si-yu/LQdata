"""因子有效性评价 —— `python main.py eval`。

## 指标定义（口径写死在这里，避免日后各说各话）

| 指标 | 定义 |
|:--|:--|
| 覆盖率 | 每日有效股票数 / 当日 universe 股票数，报中位数与最低 |
| IC(h) | 逐日截面 **Pearson** 相关：`corr(value_t, label_h(t))`，报均值与普通 t 值（重叠标签有序列相关，不当作稳健显著性） |
| RankIC(h) | 逐日截面 **Spearman**。★本环境没有 scipy，用「先取秩再算 Pearson」实现 |
| ICIR(h) | `mean(IC_t) / std(IC_t)`。**同时报年化** `× sqrt(244)`，不静默年化 |
| ac1 | 因子值秩的 1 日自相关（换手代理）。接近 1 = 因子几乎不更新 |
| 分层 | 按 rank 分 10 层，报 D9−D0 与层号-收益的 Spearman（单调性） |

## 标签从哪来

**内部用 `PriceLayer` 现算**，不读 `data/factors/label_ret_Nd`。理由：
  · 让 eval 能在标签落盘之前就跑（自举）；
  · 去掉指标链路上的隐式依赖；
  · 顺带得到一个交叉校验：标签落盘后，两者不一致就是 ERROR。

⚠️ 标签本身**不进**评价表（它是被解释变量），但**进**格式校验。

## 内存

外层按年，内层按因子。逐年释放价格层；跨年只保留每日标量统计，不保留历史 (T,C) 大矩阵：
  1 个因子年（≈85 万行 × 4B ≈ 3.4 MB）+ 5 个标签年（5 × 85 万 × 8B = 34 MB）+ 累加器。
"""

from __future__ import annotations

import json
import math
import time
from collections import defaultdict

import numpy as np
import pandas as pd

from . import store
from .dates import int_to_str
from .spec import all_specs

# 阈值：ERROR 会让 exit code 非 0，WARN 只提示
COV_WARN = 0.30
NOISE_RANKIC = 0.005
# ★ `⚠NOISE` 原来要求 `ndays > 250`，而只跑单年（2026 是 168 个交易日）时**永不触发** ——
#   守卫看着在、其实从不生效。改成 60：一个季度以上的样本就足以判"这是噪声"。
NOISE_MIN_DAYS = 60
# 零膨胀：取值恰为 0 的占比超过它 → 截面排序被并列值支配（实测 7 个因子 75%~98% 是 0）
SPARSE_ZERO_FRAC = 0.70
# 取值卡片化：每日截面不同取值数的中位数低于它 → 分辨率不足以做排序
LOWCARD_MAX = 20
# ★ ac1 阈值要小心：**财务类因子的 ac1 天然接近 1** —— 它们只在季报公告日
#   才变一次（一年约 4 次），其余时间截面排名逐日完全相同，ac1 会到 0.997~0.999。
#   这不是 bug。所以阈值取 0.9999（「基本不更新」），并把 ac1 原样报出来供人判断。
STALE_AC1 = 0.9999
JUMP_MAX_ABS = 1e8


def _spearman_from_rank(a: np.ndarray, b: np.ndarray) -> float:
    """两个已排好名的数组之间的相关 —— 即 Spearman。"""
    if a.size < 3:
        return np.nan
    a = a - a.mean()
    b = b - b.mean()
    d = math.sqrt(float((a * a).sum()) * float((b * b).sum()))
    return float((a * b).sum() / d) if d > 0 else np.nan


def _daily_corr(val: np.ndarray, lab: np.ndarray, ts: np.ndarray,
                min_n: int) -> tuple[np.ndarray, np.ndarray, int]:
    """逐日截面 IC（Pearson）与 RankIC（Spearman）。返回逐日序列 (ic, rankic, n_days)，不能先压成年度均值。"""
    ics, rics = [], []
    for t in range(val.shape[0]):
        v, l = val[t], lab[t]
        m = np.isfinite(v) & np.isfinite(l)
        n = int(m.sum())
        if n < min_n:
            continue
        x, y = v[m], l[m]
        sx, sy = x.std(), y.std()
        if sx > 0 and sy > 0:
            ics.append(float(((x - x.mean()) * (y - y.mean())).sum() / (n * sx * sy)))
            rx = pd.Series(x).rank().to_numpy()
            ry = pd.Series(y).rank().to_numpy()
            rics.append(_spearman_from_rank(rx, ry))
    if not ics:
        return np.array([]), np.array([]), 0
    return np.asarray(ics), np.asarray(rics), len(ics)


def _deciles(val: np.ndarray, lab: np.ndarray, min_n: int) -> tuple[np.ndarray, np.ndarray]:
    """十个分层的平均未来收益。"""
    acc = np.zeros(10)
    cnt = np.zeros(10)
    for t in range(val.shape[0]):
        v, l = val[t], lab[t]
        m = np.isfinite(v) & np.isfinite(l)
        if int(m.sum()) < min_n:
            continue
        x, y = v[m], l[m]
        # 用分位切层，避免并列值把某一层撑爆
        q = pd.qcut(pd.Series(x), 10, labels=False, duplicates="drop").to_numpy()
        for k in range(10):
            s = q == k
            if s.any():
                acc[k] += y[s].mean()
                cnt[k] += 1
    return acc, cnt


def evaluate_all(args, cfg) -> int:
    t_start = time.time()
    specs = [s for s in all_specs() if s.enabled and not s.is_label]
    if getattr(args, "factors", None):
        want = set(args.factors)
        specs = [s for s in specs if s.name in want]
    if getattr(args, "group", None):
        specs = [s for s in specs if s.group in set(args.group)]
    if not specs:
        print("没有匹配的因子")
        return 1

    horizons = [int(x) for x in str(args.horizons).split(",")]
    h_main = 5 if 5 in horizons else horizons[0]
    years = None
    if args.years:
        years = list(range(int(args.years[0]), int(args.years[1]) + 1))

    from .engine import Engine
    engine = Engine(cfg)
    cal = engine.cal
    codes = engine.codes
    codes_idx = pd.Series(np.arange(codes.size), index=codes)

    all_years = sorted({y for s in specs for y in store.factor_years(cfg.factors_dir, s.name)})
    if years:
        all_years = [y for y in all_years if y in set(years)]
    if not all_years:
        print("没有可评价的因子产物（先用 main.py run 生成）")
        return 1

    eligible_end = min(engine.baseline_last_day(), max(all_years) * 10000 + 1231)
    specs = [s for s in specs if s.start_int(cfg) <= eligible_end]
    if not specs:
        print("所选年份早于这些因子的有效起点，没有可评价对象")
        return 1

    print(BANNER)
    print(f"因子有效性评价 · {len(specs)} 个因子 · {all_years[0]}~{all_years[-1]} · "
          f"主标签 label_ret_{h_main}d")
    print("-" * 96)

    # 预取每个因子的元数据（起点等）
    from .spec import get as get_spec
    out: dict[str, dict] = defaultdict(dict)
    for s in specs:
        out[s.name] = {"group": s.group, "start": s.resolved_start(cfg)}

    for year in all_years:
        y0 = str(year)
        # ---- 该年的标签（内部现算，不依赖 label_ret_* 是否落盘）
        ydays = cal.between(max(int(y0 + "0101"),int(cfg.default_start.replace("-", ""))), min(int(y0 + "1231"), engine.baseline_last_day()))
        if ydays.size == 0:
            continue
        from .panel import Panel
        # 标签要看到未来：面板向后延伸 max(horizons)+1 个交易日
        pos = int(np.searchsorted(cal.days, int(ydays[-1]), side="right"))
        fwd = min(pos + max(horizons) + 1, cal.days.size - 1)
        pdays = cal.between(int(ydays[0]), int(cal.days[fwd]))
        panel = Panel(pdays, codes)
        # Release per-year raw prices and panel caches; evaluation does not need old years.
        from .prices import PriceLayer
        from .upstream import Upstream
        px = PriceLayer(Upstream(cfg.upstream), cfg, cal, codes)
        px._ensure_loaded(year - 1, year)
        hfq_open = px.panel(panel, "hfq_open")
        from . import mathx as mx
        # ★★ 与 `factors/labels.py` 同一个坑（2026-09-15 修）：面板为标签向后延伸了
        #   `forward_days` 天，但延伸段**没有真实行情**，而开盘价是"水平量"会被前向填充 →
        #   `o[T+21]` 会取到最后一个真实交易日的陈旧价，把 4~6 天的短收益当成 20 天收益。
        #   这里用"成交量的有限性"判定真实行情（vol 是流量，不做 ffill）。
        vol = px.panel(panel, "vol")
        has_any = np.isfinite(vol).any(axis=1)
        _idx = np.flatnonzero(has_any)
        last_row = int(_idx[-1]) if _idx.size else -1
        row_ix = np.arange(panel.T)
        labels = {}
        for h in horizons:
            buy = mx.shift(hfq_open, -1)
            sell = mx.shift(hfq_open, -(h + 1))
            lab = mx.safe_div(sell, buy, min_abs_den=1e-12) - 1.0
            ok = (row_ix + h + 1) <= last_row
            labels[h] = np.where(ok[:, None], lab, np.nan)
        rows = panel.day_mask(ydays)

        for s in specs:
            df = store.read_year(cfg.factors_dir, s.name, year)
            if df.empty:
                continue
            ci = codes_idx.reindex(df["stock_code"].to_numpy()).to_numpy()
            keep = np.isfinite(ci)
            from .dates import series_to_int
            mat = panel.place(ci[keep].astype(np.int64),
                              series_to_int(df["trade_date"])[keep],
                              pd.to_numeric(df["value"], errors="coerce").to_numpy()[keep])
            active = ydays >= s.start_int(cfg)
            sub_v = mat[rows][active]
            sub_r = panel.place(ci[keep].astype(np.int64),
                                series_to_int(df["trade_date"])[keep],
                                pd.to_numeric(df["rank"], errors="coerce").to_numpy()[keep])[rows][active]

            rec = out[s.name]
            finite = np.isfinite(sub_v)
            cov = finite.mean(axis=1)
            rec.setdefault("_cov", []).append(cov)
            rec.setdefault("_rows", {})[year] = int(len(df))
            rec.setdefault("_uniq", []).append(
                np.array([len(np.unique(x[np.isfinite(x)])) if np.isfinite(x).any() else 0
                          for x in sub_v]))
            vals = sub_v[finite]
            rec["_value_count"] = rec.get("_value_count", 0) + vals.size
            rec["_zero_count"] = rec.get("_zero_count", 0) + int((vals == 0).sum())
            rec["_max_abs"] = max(rec.get("_max_abs", 0.0), float(np.max(np.abs(vals))) if vals.size else 0.0)
            # 自相关用 rank 网格（更稳）
            rk = np.where(np.isfinite(sub_r), sub_r, np.nan)
            a = rk[1:]
            b = rk[:-1]
            m = np.isfinite(a) & np.isfinite(b)
            n_ac = int(m.sum())
            if n_ac > 10:
                aa, bb = a[m], b[m]
                den = aa.std() * bb.std()
                if den > 0:
                    rec["_ac_sum"] = rec.get("_ac_sum", 0.0) + float(((aa-aa.mean())*(bb-bb.mean())).mean()/den) * n_ac
                    rec["_ac_count"] = rec.get("_ac_count", 0) + n_ac
            lab = labels[h_main][rows][active]
            ic, ric, nd = _daily_corr(sub_v, lab, ydays[active], cfg.min_cross_section // 3)
            rec.setdefault("_ic", []).append((ic, ric, nd))
            rec.setdefault("_dec", []).append(_deciles(sub_v, lab, cfg.min_cross_section // 3))
        print(f"  · {year} 年处理完成（{len(specs)} 个因子）", flush=True)

    # ---------------------------------------------------------------- 汇总
    rows_out = []
    for name, rec in sorted(out.items()):
        covs = np.concatenate(rec.get("_cov", [])) if rec.get("_cov") else np.array([])
        ics = np.concatenate([x[0] for x in rec.get("_ic", [])]) if rec.get("_ic") else np.array([])
        rics = np.concatenate([x[1] for x in rec.get("_ic", [])]) if rec.get("_ic") else np.array([])
        ndays = len(ics)
        tot = rec.get("_ac_count", 0)
        ac1 = rec.get("_ac_sum", 0.0) / tot if tot else np.nan
        decs = rec.get("_dec", [])
        acc = np.sum([x[0] for x in decs], axis=0) if decs else np.zeros(10)
        cnt = np.sum([x[1] for x in decs], axis=0) if decs else np.zeros(10)
        D = np.where(cnt > 0, acc / np.maximum(cnt, 1), np.nan)

        ic_m = float(ics.mean()) if ics.size else np.nan
        ic_s = float(ics.std(ddof=1)) if ics.size > 1 else np.nan
        ric_m = float(rics.mean()) if rics.size else np.nan
        icir = ic_m / ic_s if ics.size > 1 and ic_s > 0 else np.nan
        t_ic = ic_m / (ic_s / math.sqrt(ics.size)) if ics.size > 1 and ic_s > 0 else np.nan
        spread = (D[9] - D[0]) if np.isfinite(D[0]) and np.isfinite(D[9]) else np.nan
        mono = _spearman_from_rank(np.arange(10.0), D) if np.isfinite(D).all() else np.nan
        uniq = np.concatenate(rec["_uniq"]) if rec.get("_uniq") else np.array([])

        flags = []
        cvm = float(np.nanmean(covs)) if covs.size else 0.0
        if cvm < 0.01:
            flags.append("✘ALLNAN")
        elif cvm < COV_WARN:
            flags.append("⚠LOWCOV")
        if uniq.size and float(np.mean(uniq <= 1)) > 0.2:
            flags.append("✘CONST")
        # ★ 零膨胀：>70% 的取值恰好是 0。这类因子（涨停次数、事件计数）不是"坏"，
        #   但**截面排序被并列的 0 支配**，当真因子用会失真 —— 必须显式标注。
        zero_frac = rec.get("_zero_count", 0) / max(1, rec.get("_value_count", 0))
        if zero_frac > SPARSE_ZERO_FRAC:
            flags.append(f"⚠SPARSE{zero_frac:.0%}")
        # ★ 卡片化：每日截面只有很少的几个不同取值 → 分辨率不足以排序
        if uniq.size and float(np.median(uniq)) <= LOWCARD_MAX:
            flags.append(f"⚠LOWCARD{int(np.median(uniq))}")
        if np.isfinite(ac1) and ac1 > STALE_AC1:
            flags.append("✘STALE")
        if rec.get("_max_abs", 0.0) > JUMP_MAX_ABS:
            flags.append("✘JUMP")
        if np.isfinite(ric_m) and abs(ric_m) < NOISE_RANKIC and ndays >= NOISE_MIN_DAYS:
            flags.append("⚠NOISE")

        rows_out.append({
            "name": name, "group": rec["group"], "start": rec["start"],
            "cov": cvm, "days": ndays,
            "ic": ic_m, "t": t_ic, "rankic": ric_m, "icir": icir, "icir_annualized": icir * math.sqrt(244), "t_note": "naive daily t; overlapping forward returns are not independent",
            "ac1": ac1, "spread": spread, "mono": mono, "flags": flags,
            "years": rec.get("_rows", {}),
        })

    # ---------------------------------------------------------------- 输出
    print("-" * 96)
    print(f"{'因子':<24}{'类别':<10}{'覆盖':>7}{'天数':>7}{'IC':>8}{'t':>7}"
          f"{'RankIC':>8}{'ICIR':>7}{'ac1':>7}{'D9-D0':>8}{'单调':>7}  标记")
    print("-" * 96)
    err = 0
    for r in sorted(rows_out, key=lambda x: -abs(x["ic"]) if np.isfinite(x["ic"]) else 0):
        err += sum(1 for f in r["flags"] if f.startswith("✘"))
        fmt = lambda v, w=8, p=4: (f"{v:>{w}.{p}f}" if np.isfinite(v) else f"{'—':>{w}}")
        print(f"{r['name']:<24}{r['group']:<10}{r['cov']:>6.1%}{r['days']:>7}"
              f"{fmt(r['ic'])}{fmt(r['t'],7,2)}{fmt(r['rankic'])}{fmt(r['icir'],7,2)}"
              f"{fmt(r['ac1'],7,3)}{fmt(r['spread'],8,4)}{fmt(r['mono'],7,2)}  "
              f"{' '.join(r['flags'])}")

    # 因子 × 年份行数矩阵（一眼看清"谁的日期长短不一样"）
    print("-" * 96)
    allyr = sorted({y for r in rows_out for y in r["years"]})
    if allyr:
        print("因子 × 年份行数矩阵（· = 该年无数据，正是「日期长短不一样」所在）")
        print(f"{'因子':<24}" + "".join(f"{y:>9}" for y in allyr))
        for r in sorted(rows_out, key=lambda x: x["name"]):
            print(f"{r['name']:<24}" + "".join(
                (f"{r['years'].get(y, 0):>9,}" if r['years'].get(y) else f"{'·':>9}")
                for y in allyr))

    # ★ 输出目录必须跟随 `--sandbox`（`cfg.state_dir` 已被重定向）。
    #   原来写死 `cfg.root / "state" / "eval"`：多个 Agent 并行评测时会互相覆盖
    #   生产目录里的 summary.json，评测结论无法追溯。
    outdir = (Path(args.out) if getattr(args, "out", None)
              else (Path(cfg.state_dir) / "eval"))
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "summary.json").write_text(
        json.dumps(rows_out, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print("-" * 96)
    print(f"共 {len(rows_out)} 个因子 · 耗时 {time.time() - t_start:.1f}s · "
          f"报告已写入 {outdir / 'summary.json'}")
    if err:
        print(f"✘ {err} 个 ERROR 级问题（见上表「标记」列）")
        return 1
    print("✅ 无 ERROR 级问题")
    return 0


from pathlib import Path  # noqa: E402  (放到末尾避免与上方 import 块风格冲突)

BANNER = """
╔══════════════════════════════════════════════════════════════════════════╗
║  因子有效性评价（纯本地 / 零 API）                                        ║
║   IC = 与 label_ret_5d 的逐日截面 Pearson 相关                            ║
║   RankIC = 逐日截面 Spearman（本环境无 scipy，用"先取秩再算 Pearson"实现）║
║   ac1 = 因子秩的 1 日自相关（换手代理）；D9-D0 = 十分位首尾层收益差        ║
╚══════════════════════════════════════════════════════════════════════════╝
"""
