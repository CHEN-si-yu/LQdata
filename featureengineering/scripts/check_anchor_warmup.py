#!/usr/bin/env python3
"""锚点 A/B 抽检 —— 「全局锚定」vs「按年锚定」逐格比对。

## 为什么要有这个脚本

`run_year` 的面板下界锚点 2026-09-17 由「因子全局起点（= `default_start`）」
收窄为「**本任务所在年的 1 月 1 日**」（理由与收益见 `fea/engine.py::run_year`）。
两种锚定算出的值只在**浮点中心**的舍入上不同（末位 ULP 量级），但有一条必须证明：

    **每个因子的 `warmup_days` 足够长** —— 长到「本年第一格往前看 warmup 天」
    与「往前看整个历史」得到同一个数。

warmup 不足的因子在旧口径下被"从 2012 年一路算下来"掩盖了，换成按年锚定后
会在**每年年初**露出 NaN 或错值。这个脚本就是把两者都算一遍、逐格比对。

## 用法

    PY=/autodl-fs/data/miniconda3/bin/python
    $PY scripts/check_anchor_warmup.py                        # 默认抽样（各家族代表因子 × 3 年）
    $PY scripts/check_anchor_warmup.py --years 2015 2019      # 指定年份
    $PY scripts/check_anchor_warmup.py --factors bp etp5      # 指定因子
    $PY scripts/check_anchor_warmup.py --all                  # 全部因子 × 全部年份（很慢）

## 判据（三条，任何一条不为 0 都要逐条定性）

| 指标 | 期望 | 含义 |
|:--|:--|:--|
| 行集合差异 | **0** | 两边写的 (日期, 股票) 集合必须完全相同 |
| NaN 模式差异 | **0** | 一边有值另一边 NaN = warmup 不够（真问题） |
| 相对差 > 1e-6 | **0** | 超出浮点末位的差异（真问题） |
| 位级差异 | 允许 | 只差最后一个 ULP，属预期（浮点中心不同） |

只读上游、只写沙箱（`--keep` 才保留沙箱），不碰生产 `data/` 与 `state/`。
"""
from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fea import config as cfg_mod          # noqa: E402
from fea.engine import Engine              # noqa: E402
from fea.spec import REGISTRY, get         # noqa: E402

import factors                             # noqa: F401,E402  —— import 即注册

# 抽样：每个家族挑代表，且**特意包含**长 warmup / 滞后表 / 派生层 / 耦合 / 标签
DEFAULT_FACTORS = [
    "momentum_20", "momentum_250",          # 价格·短/长窗
    "vol_120", "beta_60",                   # 风险
    "bp", "etp5", "pegh5",                  # 估值·长 warmup（2300/2600 天）
    "dividend_yield_3y_avg",                # 1400 天 warmup
    "roe_ttm", "cash_conversion_cycle",     # 质量·财报
    "yoy_revenue", "asset_growth_qoq",      # 成长
    "chip_concentration", "winner_rate",    # 筹码（2018 起 + 派生层）
    "idt_rv_daily", "idt_overnight_gap",    # 日内（5min 派生层）
    "margin_balance_20d",                   # 滞后表（lagged_ok + lag_grid）
    "mf_net_inflow_5d", "fundflow_retail_inst_divergence",
    "limit_up_count_20", "consecutive_limit_up",
    "rsi_14", "turnover_f_20", "gap_fill_tendency_10d",
    "holder_number_chg",                    # 股东户数（2016 起）
    "cp_quality_momentum",                  # 耦合因子（读父因子产物）
    "label_ret_5d",                         # 标签（forward_days）
    # ★ 耦合因子必须**连同父因子**一起进样本，否则沙箱里父因子为空 →
    #   耦合因子全 NaN → 两边都 NaN → 得到"零差异"的**假通过**（实测踩过）。
    "mf_big_order_ratio", "mf_retail_dominance", "momentum_60",
]

DEFAULT_YEARS = [2015, 2019, 2023]


class _OldGlobalAnchor(Engine):
    """旧口径：锚在**因子全局起点**（= `default_start`），与本次缺口日无关。

    这就是 2026-09-15~09-17 期间的生产行为（那时 `--start` 参与锚点，
    但全量跑里 `override_start=None`，所以等价于这里的实现）。
    """

    def _anchor_for(self, spec, days):                      # noqa: ARG002
        return max(spec.start_int(self.cfg), self._start_floor)


def _run(mode: str, names: list[str], years: list[int], jobs: int, base: Path) -> Path:
    """把 `names` 的 `years` 年份算进一个沙箱，返回沙箱根目录。

    mode: `global` = 旧口径（锚在全局起点）；`year` = 新口径；
          `w3` = 新口径 + **warmup ×3**（用于「warmup 够不够」的全量筛查）。
    """
    cfg = cfg_mod.load()
    cfg.raw["default_start"] = "2012-01-01"
    sbx = base / mode
    for sub in ("factors", "state"):
        (sbx / sub).mkdir(parents=True, exist_ok=True)
    cfg.raw["paths"]["factors"] = str(sbx / "factors")
    cfg.raw["paths"]["state"] = str(sbx / "state")

    if mode.startswith("w") and mode[1:].isdigit() and mode != "w1":
        # ★ 测试专用：临时把样本因子的 warmup 放大 K 倍（mode = "w4" → ×4）。
        #   若值不变 → 该因子的实际回看窗口 ≤ 声明值（"按年锚定不切断时间轴"的充分条件）。
        k = int(mode[1:])
        for n in names:
            get(n).warmup_days = int(get(n).warmup_days * k)

    eng = _OldGlobalAnchor(cfg) if mode == "global" else Engine(cfg)
    for y in sorted(years):
        # 只产出本年：`--start` 的等价物。★ 它只截输出范围，不参与锚点
        # （旧实现里它会参与 —— 这正是本脚本要还原的差别，见 _OldGlobalAnchor）
        eng.override_start = y * 10000 + 101
        todo = eng.prebuild([get(n) for n in names], y * 10000 + 1231, rebuild=True)
        # ★ 两趟：耦合因子读的是**父因子的产物文件**，跑在父因子前面会静默全 NaN
        #   （与 main.py::cmd_run 同一套逻辑，别省）。
        def _coup(sp):
            return any(d in REGISTRY for d in sp.deps)
        waves = ([t for t in todo if not _coup(t[0])], [t for t in todo if _coup(t[0])])
        t0 = time.time()
        n = 0
        for wave in waves:
            tasks = [(sp.name, int(yy), plan[yy]) for sp, _, plan in wave for yy in sorted(plan)]
            if not tasks:
                continue
            res = eng.run_parallel(tasks, jobs)
            n += len(res)
            for e in [r for r in res if r.get("error")][:3]:
                print(f"     ✘ {e.get('factor')} {e.get('year')}: {e.get('error')}")
        print(f"  [{mode}] {y}: {n} 个任务 / {time.time() - t0:.0f}s", flush=True)
    return sbx


def _f32_bits(x: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(x.astype(np.float32)).view(np.uint32)


def _cmp_one(pa: Path, pb: Path, head_n: int = 20) -> dict | None:
    """逐格比对一年的产物。`head_n` = 另外单独统计**年初前 N 个交易日**。

    ★ 年初那几行是**连续性**的敏感点：按年锚定只把面板下界收窄到「年初 − warmup」，
      所以「warmup 够不够」只会在年初暴露。全局 vs 按年在全年都一致、但年初不一致，
      正是"时间轴被切断"的典型形态 —— 所以这一列要单独看。
    """
    a = pd.read_parquet(pa)
    b = pd.read_parquet(pb)
    if a.empty and b.empty:
        return None
    m = a.merge(b, on=["trade_date", "stock_code"], how="outer",
                suffixes=("_a", "_b"), indicator=True)
    days = np.sort(pd.unique(a["trade_date"].astype(str)))
    head = set(days[:head_n].tolist())
    is_head = m["trade_date"].astype(str).isin(head).to_numpy()

    out = {
        "rows_a": len(a), "rows_b": len(b),
        "only": int((m["_merge"] != "both").sum()),
        "nan_diff": 0, "bit_diff": 0, "over": 0, "max_rel": 0.0,
        "n_both": 0, "n_finite": 0,
        "head_n": int(is_head.sum()), "head_nan": 0, "head_over": 0,
        "head_bit": 0, "head_max_rel": 0.0,
        # ★★ 「实质差异」的真正判据：**rank 列的变化幅度**。
        #   模型消费的是 `rank`（截面百分位），而且 rank 是尺度无关的 ——
        #   相对差在「值接近 0」的格子上会爆表（实测 max_rel 2.0 全是这种），
        #   对判断"这一天这只股票的排序变了没有"毫无意义。
        "rank_gt_001": 0, "rank_gt_005": 0, "rank_max": 0.0,
    }
    both = m[m["_merge"] == "both"]
    out["n_both"] = len(both)
    if both.empty:
        return out
    hb = is_head[(m["_merge"] == "both").to_numpy()]
    for col in ("value", "rank"):
        va = pd.to_numeric(both[f"{col}_a"], errors="coerce").to_numpy(np.float64)
        vb = pd.to_numeric(both[f"{col}_b"], errors="coerce").to_numpy(np.float64)
        fa, fb = np.isfinite(va), np.isfinite(vb)
        nd = fa != fb
        out["nan_diff"] += int(nd.sum())
        out["head_nan"] += int(nd[hb].sum())
        f = fa & fb
        if not f.any():
            continue
        if col == "value":
            out["n_finite"] = int(f.sum())
        d = np.abs(va[f] - vb[f]) / np.maximum(
            np.maximum(np.abs(va[f]), np.abs(vb[f])), 1e-30)
        out["max_rel"] = max(out["max_rel"], float(d.max()))
        out["over"] += int((d > 1e-6).sum())
        bits = _f32_bits(va[f]) != _f32_bits(vb[f])
        out["bit_diff"] += int(bits.sum())
        if col == "rank":
            dr = np.abs(va[f] - vb[f])
            out["rank_gt_001"] = int((dr > 0.01).sum())
            out["rank_gt_005"] = int((dr > 0.05).sum())
            out["rank_max"] = float(dr.max())
        fh = hb[f]
        if fh.any():
            out["head_over"] += int((d[fh] > 1e-6).sum())
            out["head_bit"] += int(bits[fh].sum())
            out["head_max_rel"] = max(out["head_max_rel"], float(d[fh].max()))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="锚点 A/B 抽检（全局 vs 按年）")
    ap.add_argument("factors", nargs="*", help="只查这些因子")
    ap.add_argument("--years", nargs="*", type=int, default=None, help="年份，默认 2015 2019 2023")
    ap.add_argument("--all", action="store_true", help="全部因子 × 全部年份（慢）")
    ap.add_argument("--jobs", type=int, default=8, help="并行进程数（默认 8）")
    ap.add_argument("--keep", action="store_true", help="保留沙箱目录")
    ap.add_argument("--tmp", default=None, help="沙箱父目录（默认 /tmp）")
    ap.add_argument("--scaled-warmup", nargs="?", const=3, type=int, default=None,
                    metavar="K",
                    help="改跑「warmup ×1 vs ×K」（默认 K=3）：全量筛查每个因子的实际回看窗口"
                         "是否 ≤ 声明值。★ 配合 --all 就是全因子普查")
    ap.add_argument("--only", choices=("both", "a", "b"), default="both",
                    help="只跑一侧（大年份的 A 侧很贵/很吃内存，可以分两次跑再 --compare-only）")
    ap.add_argument("--compare-only", nargs=2, metavar=("DIR_A", "DIR_B"),
                    help="跳过计算，直接比对两个已有沙箱")
    args = ap.parse_args()

    names = args.factors or (sorted(REGISTRY) if args.all else DEFAULT_FACTORS)
    years = args.years or (list(range(2012, 2027)) if args.all else DEFAULT_YEARS)
    bad = [n for n in names if n not in REGISTRY]
    if bad:
        raise SystemExit(f"未知因子：{bad}")

    base = Path(tempfile.mkdtemp(prefix="anchor_ab_", dir=args.tmp)) \
        if not args.compare_only else None
    if base is not None:
        print(f"沙箱父目录 {base}")
    print(f"因子 {len(names)} 个 × 年份 {years}")
    if args.scaled_warmup:
        print("  A = 按年锚定 · warmup ×1")
        print(f"  B = 按年锚定 · warmup ×{args.scaled_warmup}"
              f"   → 若逐格相同 ⇒ 实际回看窗口 ≤ warmup（按年锚定不会切断时间轴）")
    else:
        print("  A = 全局锚定 [default_start − warmup, 年末]   ← 旧口径")
        print("  B = 按年锚定 [本年 1/1 − warmup, 年末]        ← 新口径")
    print("-" * 96)
    t0 = time.time()
    try:
        mode_a, mode_b = (("year", f"w{args.scaled_warmup}") if args.scaled_warmup
                          else ("global", "year"))
        if args.compare_only:
            dir_a, dir_b = Path(args.compare_only[0]), Path(args.compare_only[1])
        else:
            dir_a = _run(mode_a, names, years, args.jobs, base) \
                if args.only in ("both", "a") else base / mode_a
            dir_b = _run(mode_b, names, years, args.jobs, base) \
                if args.only in ("both", "b") else base / mode_b
        print("-" * 108)
        print(f"{'因子':<30}{'年':<6}{'行数':>10}{'NaN差':>10}{'rank差>0.01':>12}"
              f"{'rank差>0.05':>12}{'rank最大差':>11}{'位级差':>10}")
        print("-" * 108)
        n_row = n_nan = n_over = n_bit = 0
        h_nan = h_over = 0
        n_r001 = n_r005 = 0
        worst: list[tuple[float, str, int, int, int, float]] = []
        for n in names:
            for y in sorted(years):
                # ★ 沙箱里的产物在 <沙箱>/factors/<因子名>/year=YYYY/（别漏掉 factors 这一层 ——
                #   漏了会"两边都不存在"→ 空比对 → 打印出"0 差异"的**假通过**，实测踩过）
                pa = dir_a / "factors" / n / f"year={y}" / "data.parquet"
                pb = dir_b / "factors" / n / f"year={y}" / "data.parquet"
                if not pa.exists() and not pb.exists():
                    continue
                if not pa.exists() or not pb.exists():
                    print(f"{n:<34}{y:<6}  只有一边存在  ← ✘")
                    n_row += 1
                    continue
                r = _cmp_one(pa, pb)
                if r is None:
                    continue
                n_row += r["only"]
                n_nan += r["nan_diff"]
                n_over += r["over"]
                n_bit += r["bit_diff"]
                h_nan += r["head_nan"]
                h_over += r["head_over"]
                n_r001 += r["rank_gt_001"]
                n_r005 += r["rank_gt_005"]
                if r["rank_gt_001"] or r["nan_diff"] or r["only"]:
                    worst.append((r["rank_gt_001"], n, y, r["nan_diff"],
                                  r["only"], r["rank_max"]))
        # 只打「有实质差异」的行（rank 变了 / 缺失模式变了）—— 否则 230 行看不过来
        worst.sort(reverse=True)
        for _, n, y, nn, only, rmax in worst[:60]:
            print(f"{n:<30}{y:<6}{'':>10}{nn:>10,}{'':>12}{'':>12}{rmax:>11.4f}{'':>10}  ✘")
        if len(worst) > 60:
            print(f"  …另有 {len(worst) - 60} 个 (因子,年) 有差异（按 rank 差排序取前 60）")
        print("-" * 108)
        print(f"合计：行集合差 {n_row} · NaN 模式差 {n_nan}（其中年初 {h_nan}）· "
              f"**rank 差>0.01 的格子 {n_r001:,}**（>0.05 的 {n_r005:,}）· "
              f"相对差>1e-6 {n_over} · 位级差 {n_bit}（末位 ULP，允许）")
        ok = (n_row == 0 and n_nan == 0 and n_r001 == 0)
        print(f"{'✔ 按年锚定与全局锚定等价（rank 一致、无缺失模式差）' if ok else '✘ 有 rank/缺失层面的实质差异，必须逐条定性'}"
              f"  用时 {time.time() - t0:.0f}s")
        return 0 if ok else 1
    finally:
        if base is not None:
            if args.keep:
                print(f"沙箱保留在 {base}")
            else:
                shutil.rmtree(base, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
