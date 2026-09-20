"""推理与前视审计 —— 打分是**逐行**的，这条约束由"截断复算"来守。

★ 防泄漏：打分只用第 i 行自己的特征，不依赖任何跨行/跨日统计量。
  判据（与模块② 同一条）：**T 日能算出的打分，只能用到 ≤T 的数据。**
  做法（`audit_pit_unit`）：把面板截到 `cut` 重新打分，与该折已落盘的全量打分逐格比对；
  任何不一致都说明链路里藏了跨行/全局统计量（未来函数）。
"""
from __future__ import annotations

import numpy as np

from . import preprocess, state, store
from .config import Cfg
from .data import frame_for, load_for_unit
from .models.base import ModelBase

# 截断复算的浮点容差：见 audit_pit_unit 的说明（1e-5 相对误差以内算"一致"）
AUDIT_RTOL = 1e-5
AUDIT_ATOL = 1e-7


def _X_for(cfg: Cfg, panel, model_name: str) -> np.ndarray:
    """按该头的填充策略准备输入矩阵（与训练时**同一条**路径）。"""
    mode = preprocess.fill_mode_for(cfg, model_name)
    X = preprocess.clip_features(panel.X, cfg.raw["preprocess"].get("clip"))
    X, _notes = preprocess.fill_for(X, mode)
    return X


def _compare(j, cut: str, log) -> int:
    """逐格比对（生产 vs 截断复算）。返回 0 = 一致。"""
    a = j["value_prod"].to_numpy(dtype=np.float64)
    b = j["value_cut"].to_numpy(dtype=np.float64)
    both_nan = ~np.isfinite(a) & ~np.isfinite(b)
    # ★ 容差：float32 的矩阵乘在不同数组形状下 BLAS 分块不同 → 求和顺序不同 →
    #   相对误差 ~1e-6 是**预期噪声**（V1 首审实测）。
    #   真正的未来函数（用了未来日期的统计量）会在**信号量级**上炸开（相对误差 1e-1 以上），
    #   两者相差 5 个数量级，用 1e-5 的容差区分得干干净净。
    scale = np.maximum(np.abs(a), np.abs(b))
    rel = np.where(np.isfinite(a) & np.isfinite(b),
                   np.abs(a - b) / np.maximum(scale, 1e-12), 0.0)
    diff = ~both_nan & ~np.isclose(a, b, rtol=AUDIT_RTOL, atol=AUDIT_ATOL, equal_nan=True)
    mx = float(np.nanmax(rel)) if len(rel) and np.isfinite(rel).any() else 0.0
    log(f"  比对 {len(j):,} 格（{cut} 及以前）· 不一致 {int(diff.sum())}"
        f"（最大相对偏差 {mx:.2e}，容差 rtol={AUDIT_RTOL:g}）")
    if diff.sum():
        log("  ✘ 有不一致 —— 说明打分依赖了截断点之后的数据（未来函数）：")
        for _, r in j[diff].head(5).iterrows():
            log(f"     {r['trade_date']} {r['stock_code']}: "
                f"{r['value_prod']} vs {r['value_cut']}")
        return 1
    log("  ✅ 截断复算逐格一致（打分只用 ≤T 的数据）")
    return 0


# ---------------------------------------------------------------- 单元（V{N}）前视审计
def audit_pit_unit(cfg: Cfg, unit, fold: int, head: str, label: str,
                   cut: str | None = None, smoke: bool = False, sample: bool = False,
                   log=print) -> int:
    """对**单元某一折的头**做截断复算审计（做法见模块 docstring）。

    ★ 截断复算要求"按日期重切"，`load(cfg, end=cut)` 就是干这个的 ——
      这正是初加工产物必须支持按日期切片的原因。
    """
    panel_all = load_for_unit(cfg, unit, sample=sample, log=lambda *_: None)
    cut = cut or panel_all.dates[-2]
    mp = state.unit_model_path(cfg, unit.name, fold, head, label, smoke)
    if not mp.exists():
        raise SystemExit(f"没有训好的模型 {mp}（先训练折 {fold}）")
    prod_p = state.unit_fold_score_path(cfg, unit.name, fold, head, label, smoke)
    if not prod_p.exists():
        raise SystemExit(f"没有该折的打分产物 {prod_p}")
    log(f"  截断复算：面板截到 {cut} 重新打分，与 {prod_p.name} 比对 …")
    m = ModelBase.load(mp, cfg)
    panel_cut = load_for_unit(cfg, unit, end=cut, sample=sample, log=lambda *_: None)
    if len(panel_cut.dates) == 0:
        log(f"  ⚠️ 截断到 {cut} 后面板为空 —— 换一个更晚的 --cut")
        return 1
    _Xc = _X_for(cfg, panel_cut, head)
    # ★ 与 `train.py` 同一条分支：逐日标准化的头（`nn_v8`）必须拿到日边界，
    #   否则整个截断面板会被当成一天去标准化，截断复算会报出假不一致。
    sc = (m.predict(_Xc, day=panel_cut.row_day) if getattr(m, "predict_needs_day", False)
          else m.predict(_Xc)).astype(np.float32)
    # ★ 必须套与训练/落盘**同一条**"有特征"判据：一个特征都没有的行在产物里是 NaN，
    #   不套掩码就会拿"填充后的预测值"去比 NaN —— 全是假不一致（V1 首审实测踩到）。
    avail = np.isfinite(panel_cut.X).sum(axis=1) >= 1
    sc = np.where(avail, sc, np.nan)
    got = frame_for(panel_cut, np.arange(panel_cut.n_rows), sc, name="value")
    prod = store.read_frame(prod_p)
    j = prod.merge(got[["trade_date", "stock_code", "value"]],
                   on=["trade_date", "stock_code"], how="inner",
                   suffixes=("_prod", "_cut"))
    if j.empty:
        log("  ⚠️ 没有可比对的格子（产物与截断面板不重叠？）")
        return 1
    return _compare(j, str(cut), log)
