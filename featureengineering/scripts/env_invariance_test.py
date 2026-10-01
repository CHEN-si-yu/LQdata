#!/usr/bin/env python
"""环境不变性测试（本次修复的核心验收）。

判据：把 numpy 的 **float32 超越函数核** 换成另一套实现（"双精度算完再舍入"），
因子输出必须**逐位不变**。若变了，说明这个因子还在把 float32 送进超越函数
—— 那条路径的末位随 numpy 版本/构建走，落盘值就不可复现。

为什么用"打补丁"而不是"换解释器"：容器里只有一套 numpy，换不了；而实测
numpy 的 float32 核与正确舍入在 24.76% 的输入上不一致（float64 下 0.000%），
所以"双精度算完再舍入"正是另一套环境会给出的值 —— 用它当替代实现是**可复现**的。

用法：python env_invariance_test.py <输出.json> [因子名 ...]
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

import factors  # noqa: F401
from fea import config as cfg_mod
from fea.context import FactorContext
from fea.engine import Engine, minus_days
from fea.panel import Panel
from fea.spec import all_specs, get

KERNELS = ("arcsinh", "log", "log1p", "log2", "log10", "exp", "expm1", "power",
           "tanh", "sinh", "cosh", "arctanh", "sin", "cos", "arcsin", "arccos", "arctan")

DEFAULT = ["afx_bs_cap_rese", "afx_bs_div_payable", "afx_is_basic_eps", "afx_fi_bps",
           "momentum_20", "momentum_120", "reversal_2d", "short_term_reversal_5",
           "rel_mom_ind_250d", "rel_turnover_ind_20d", "chip_peak_distance",
           "roe_ttm", "turnover_std_20", "liquidity_shock_20", "bollinger_width_20",
           "amount_ratio_20", "winner_rate_reversal_signal", "id2_am_ret",
           "open5_amt_log", "mkt_up_ratio", "mkt_am_ew_return", "label_ret_5d",
           "efx_limit_streak", "high_open_low_close_frac_20", "gap_event_decay_5"]


def patch_kernels():
    """把 float32 核替换成"双精度算完再舍入"。返回还原函数。"""
    saved = {}
    for name in KERNELS:
        f = getattr(np, name, None)
        if f is None:
            continue
        saved[name] = f

        def make(orig, nm):
            def wrapper(*a, **kw):
                # 注意：`np.log1p(series)` 传进来的是 pandas Series（不是 ndarray），
                # ufunc 内部仍按 float32 走 SIMD 核 —— 只查 isinstance 会漏掉它。
                if any(getattr(v, "dtype", None) == np.float32 for v in a):
                    a2 = [np.asarray(v, dtype=np.float64)
                          if getattr(v, "dtype", None) == np.float32 else v for v in a]
                    return np.asarray(orig(*a2, **kw)).astype(np.float32)
                return orig(*a, **kw)
            wrapper.__name__ = nm
            return wrapper
        setattr(np, name, make(f, name))

    def restore():
        for nm, f in saved.items():
            setattr(np, nm, f)
    return restore


def build(spec, eng, cfg, year_end):
    days = eng.cal.between(20260101, year_end)
    anchor = eng._anchor_for(spec, days)
    wlo = minus_days(anchor, spec.warmup_days)
    hi = int(days[-1])
    if spec.forward_days:
        pos = int(np.searchsorted(eng.cal.days, hi, side="right"))
        hi = int(eng.cal.days[min(pos + spec.forward_days, eng.cal.days.size - 1)])
    pdays = eng.cal.between(wlo, hi)
    panel = Panel(pdays, eng.codes)
    uni = eng.universe_for(panel)
    deriv = eng.deriv_for(wlo, hi, fields=eng._deriv_fields_run)
    ctx = FactorContext(panel, deriv, eng.up, cfg, uni,
                        prices=eng.prices_for(wlo, hi),
                        intraday=eng.intraday_layer(), chips=eng.chips_layer(),
                        cal=eng.cal, factor_io=eng.factor_io(),
                        open5=eng.open5_layer())
    return panel, ctx


def run_all(specs, cfg, patch: bool, end):
    """两套 Engine 各自建一次层，逐因子算完（避免每个因子重建价格层）。"""
    eng = Engine(cfg)
    eng.prebuild(specs, int(end))
    restore = patch_kernels() if patch else None
    outs = {}
    try:
        for spec in specs:
            panel, ctx = build(spec, eng, cfg, end)
            outs[spec.name] = np.asarray(spec.fn(ctx), dtype=np.float64)
    finally:
        if restore:
            restore()
    return outs


def main():
    out_path = Path(sys.argv[1])
    names = sys.argv[2:] or DEFAULT
    cfg = cfg_mod.load()
    # ★ 审计工具绝不能写生产 state/：`Engine.prebuild` 在检测到 recipe 变化时会把
    #   "已清空的 manifest" 落盘（引擎的有意行为：中途崩了也不能让旧 coverage 冒充完成）。
    #   实测踩过（2026-09-25）：审计脚本没隔离 state，直接把 544 个生产台账清空了。
    #   这里把 state 重定向到临时目录 —— 审计只读生产产物，不碰任何状态。
    import tempfile
    cfg.raw["paths"]["state"] = str(Path(tempfile.mkdtemp(prefix="fea_audit_state_")) / "state")
    specs = [get(n) for n in names]
    end = Engine(cfg).baseline_last_day()
    print(f"环境不变性测试：{len(specs)} 个因子 · 双引擎各建一次层 · 截止 {end}", flush=True)
    t0 = time.time()
    normal = run_all(specs, cfg, patch=False, end=end)
    print(f"  未打补丁一轮完成 {time.time()-t0:.0f}s", flush=True)
    t1 = time.time()
    patched = run_all(specs, cfg, patch=True, end=end)
    print(f"  打补丁一轮完成 {time.time()-t1:.0f}s", flush=True)

    recs, bad = [], 0
    for i, spec in enumerate(specs):
        name = spec.name
        a, b = normal[name], patched[name]
        fa = a.astype(np.float32)
        fb = b.astype(np.float32)
        same_bits = (fa.view(np.uint32) == fb.view(np.uint32)) | (np.isnan(fa) & np.isnan(fb))
        diff = int((~same_bits).sum())
        finite = int((np.isfinite(fa) & np.isfinite(fb)).sum())
        rec = {"factor": name, "shape": list(a.shape), "finite_cells": finite,
               "bit_diff_cells": diff}
        recs.append(rec)
        bad += bool(diff)
        mark = "✔" if not diff else "✘"
        print(f"[{i+1}/{len(names)}] {mark} {name:<28} 不同位 {diff:>6} / 有限格 {finite:>7} "
              "", flush=True)
    summary = {"objects": len(recs), "bad": bad, "patch": "float32 kernel -> float64 then round"}
    print(json.dumps(summary, ensure_ascii=False))
    out_path.write_text(json.dumps({"summary": summary, "records": recs}, ensure_ascii=False, indent=1))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
