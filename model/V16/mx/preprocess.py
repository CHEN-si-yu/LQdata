"""预处理 —— **一切统计量只用当日截面**（模块③ 的第二条铁律）。

三条规矩：
  ① 不许出现全样本 mean/std/quantile（那是未来信息，模块② `FACTORS.md:735` 同一条红线）；
  ② 缺失怎么处理**按模型分**：GBDT 保留 NaN（原生处理缺失，填了反而丢信息）；
     线性模型填一个**固定常数**（不是数据统计量）+ 附一列"缺失个数"；
  ③ 方向统一已经在 `data.load` 里做了（乘 `higher_is_better` 的 ±1）。
"""
from __future__ import annotations

import numpy as np

from .config import Cfg

RANK_FILL = 0.5          # rank 特征的中性填充值（rank ∈ [0,1]，0.5 ≈ 中位）


def clip_features(X: np.ndarray, clip: list | None, *, already=None) -> np.ndarray:
    """按配置夹一层（对 rank 列的端点做微裁；NaN 原样保留）。

    ★ `already` = "这份矩阵已经在这个区间里夹过了"。面板在 `data.load` 时**夹一次**，
      训练时把生效区间传进来，这里就**直接返回、不再拷贝**。
      为什么值得为这点较真：全历史面板一份 X ≈ 6.6 GB，而每个头都要走一次预处理 ——
      以前是"每头一拷"，三个头就是三份 6.6 GB。夹取是**逐元素**的（无跨样本统计量），
      所以提前到载入时做与在训练时做**结果逐位相同**，只是省掉了 N−1 次全量拷贝。
    """
    if not clip:
        return X
    lo, hi = float(clip[0]), float(clip[1])
    if already is not None and float(already[0]) == lo and float(already[1]) == hi:
        return X
    return np.clip(X, lo, hi)


def fill_for(X: np.ndarray, mode: str) -> tuple[np.ndarray, list[str]]:
    """按模型类型准备输入矩阵。返回 (矩阵, 附注)。"""
    if mode == "lgbm":
        return X, ["保留 NaN（LightGBM 原生处理缺失）"]
    if mode == "linear":
        nan_cnt = np.isnan(X).sum(axis=1, keepdims=True).astype(np.float32)
        nan_frac = nan_cnt / max(1, X.shape[1])
        out = np.nan_to_num(X, nan=RANK_FILL, posinf=np.nan, neginf=np.nan)
        out = np.nan_to_num(out, nan=RANK_FILL, posinf=1.0, neginf=0.0)
        out = np.concatenate([out, nan_frac], axis=1)
        return out, [f"NaN→{RANK_FILL}（固定常数，非数据统计量）+ 1 列缺失占比"]
    if mode == "nn":
        nan_cnt = np.isnan(X).sum(axis=1, keepdims=True).astype(np.float32)
        nan_frac = nan_cnt / max(1, X.shape[1])
        out = np.nan_to_num(X, nan=RANK_FILL)
        out = np.concatenate([out, nan_frac], axis=1)
        return out, [f"NaN→{RANK_FILL} + 1 列缺失占比"]
    raise ValueError(f"未知 fillna 模式 {mode}")


def fill_mode_for(cfg: Cfg, model_name: str) -> str:
    """配置 `preprocess.fillna`：`per_model`=按模型类型自动选；否则用显式值。

    ★ 填充模式取自**模型类自己的 `fill_mode` 属性**，而不是在这里维护一张
      `{"gbdt": ..., "linear": ...}` 的映射表。踩过的坑：新增 `linear_w` / `gbdt_rank`
      这些类型后，映射表查不到就 `.get(..., "lgbm")` **静默退回保留 NaN** ——
      于是加权岭拿到带 NaN 的矩阵，sklearn 直接抛"Input X contains NaN"。
      查不到类型现在**报错**，不再有静默默认值。
    """
    v = str(cfg.raw["preprocess"].get("fillna", "per_model"))
    if v != "per_model":
        return v
    from .models import REGISTRY                # 延迟 import：避免 preprocess ← models 的环
    mtype = (cfg.model_cfgs.get(model_name) or {}).get("type")
    cls = REGISTRY.get(str(mtype))
    if cls is None:
        raise SystemExit(f"✘ 模型 {model_name!r} 的类型 {mtype!r} 未注册，"
                         f"无法确定填充模式（已注册：{sorted(REGISTRY)}）")
    return cls.fill_mode


def extra_width(mode: str) -> int:
    """`fill_for` 会给矩阵加几列（特征重要性对齐时要用）。"""
    return 1 if mode in ("linear", "nn") else 0


def screen_report(cfg: Cfg, feat_names: list[str], log=print) -> dict:
    """特征初筛报告（**只报告，不自动删**）——用模块② 的先验做参考。

    数据来源（都是模块② 的产物，零成本）：
      · `state/eval/summary.json`：单因子 IC/RankIC/ac1/分层/flags
      · `state/dedup/report.json`：225×225 秩相关矩阵（|ρ|≥0.95 的簇）
    """
    # ★ 自包含单元里自动降级为空报告：本函数要模块② 的产物，而单元不该依赖上游。
    #   它是**纯报告**功能（只提示、不自动删特征），降级不影响任何数值 ——
    #   特征初筛的结论在初加工阶段就定了，已经写进快照。
    try:
        from . import upstream
        ev = {r["name"]: r for r in upstream.eval_summary(cfg)}
    except (ImportError, TypeError, AttributeError, SystemExit, KeyError):
        log("  特征初筛：上游评价表不可用（自包含运行）→ 跳过报告，不影响训练与评价")
        return {"rows": [], "flagged": []}
    rows = []
    for f in feat_names:
        r = ev.get(f)
        if not r:
            rows.append({"factor": f, "note": "无评价记录"})
            continue
        rows.append({"factor": f, "group": r.get("group"), "ic": r.get("ic"),
                     "rankic": r.get("rankic"), "ac1": r.get("ac1"),
                     "cov": r.get("cov"), "days": r.get("days"),
                     "flags": r.get("flags") or []})
    flagged = [r for r in rows if r.get("flags")]
    log(f"  特征初筛：{len(feat_names)} 个特征，模块② 标了 flag 的 {len(flagged)} 个：")
    for r in flagged[:15]:
        log(f"     {r['factor']:32} {' '.join(r['flags'])}")
    if len(flagged) > 15:
        log(f"     …另有 {len(flagged)-15} 个")
    return {"rows": rows, "flagged": flagged}
