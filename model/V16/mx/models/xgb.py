"""XGBoost 头 —— **树模型里唯一能吃到 GPU 的那条路**。

## 为什么要单独开这一族

LightGBM 官方 wheel 是**纯 CPU** 的（`mx/README` 里记过这条），所以树模型一直是训练时间的瓶颈：
实测 V3 一折 19 分钟里，`gbdt_w` 占 ~15 分钟，线性头只占 1 分钟。
而 `xgboost` 的官方 wheel **带 CUDA**（本机 3.4.1 / CUDA 13.3），`device="cuda"` 实测快 3.4 倍。

于是这里的定位有两层：
  ① **纯性能替代**：同样的加权回归配方，换成 XGBoost + GPU，把迭代速度提上来；
  ② **不同的实现即不同的先验**：LightGBM 是 leaf-wise + 直方图，XGBoost 是 depth-wise + 预排序/直方图，
      正则化写法也不同（`lambda`/`alpha` vs `reg_lambda`/`min_child_samples`）。
      两者在榜单上并列比较，本身就是一次"模型族"级别的对照 —— 不必假设它们等价。

## ★ 设备

`device` 走 `resolve_device(cfg)`（引擎硬约束 #5：**代码里不出现硬编码设备**）。
CUDA 构建缺失、驱动不匹配这类情况**在 fit 时兜底退回 CPU 并记一行日志** ——
不能让"这台机器没卡"变成"这个单元跑不起来"。
"""
from __future__ import annotations

import numpy as np

from ..labels import cs_grade
from .base import ModelBase, register, resolve_device


def _xgb_device(cfg, params: dict) -> str:
    """XGBoost 的设备串：`auto` → 有 CUDA 就 `cuda`，否则 `cpu`。"""
    want = str(params.get("device", "auto")).lower()
    if want in ("cpu", "cuda"):
        return want
    return resolve_device(cfg)


class _XGBBase(ModelBase):
    """XGBoost 的公共部分（构造 / 早停 / 设备兜底）。"""

    default_objective = "reg:squarederror"

    def _params(self) -> dict:
        p = {k: v for k, v in self.params.items()
             if k not in ("labels", "label_transform", "sample_weight", "n_estimators",
                          "num_boost_round", "grade_levels", "device", "buyable")}
        p.setdefault("tree_method", "hist")
        p["random_state"] = self.seed
        p["verbosity"] = 0
        n_jobs = p.pop("n_jobs", None)
        if n_jobs:
            p["nthread"] = int(n_jobs)
        return p

    #: 用哪个 sklearn 估计器 —— 子类覆盖（排序必须用 `XGBRanker`，
    #: 因为只有它的 `fit` 收 `group`/`qid`；`XGBRegressor.fit` 根本没这两个参数）
    estimator = "regressor"

    def _fit_xgb(self, X, y, Xv, yv, *, sample_weight, day, day_v,
                 label_fn=None, extra_params=None):
        import xgboost as xgb

        n = int(self.params.get("n_estimators", 500))
        kw = dict(self._params())
        kw.update(extra_params or {})
        lab = label_fn(y, day) if label_fn else y
        ranked = self.estimator == "ranker"
        yv_lab = (label_fn(yv, day_v) if label_fn else yv) if yv is not None else None

        for dev in self._device_plan():
            kw["device"] = dev
            try:
                self.model_ = _make(xgb, kw, self.default_objective, n, ranked)
                fit_kw: dict = {"verbose": False}
                if sample_weight is not None:
                    fit_kw["sample_weight"] = self._group_weights(sample_weight, day) if ranked \
                        else np.asarray(sample_weight, dtype=np.float64)
                if ranked:
                    fit_kw["group"] = self._group(day)
                if Xv is not None and yv is not None and len(Xv):
                    fit_kw["eval_set"] = [(Xv, yv_lab)]
                    if ranked:
                        fit_kw["eval_group"] = [self._group(day_v)]
                if not _supports_early_stopping(self.model_):
                    fit_kw.pop("eval_set", None)      # 没有早停就不要白跑验证集
                    fit_kw.pop("eval_group", None)
                self.model_.fit(X, lab, **fit_kw)
                self.device_ = dev
                return self
            except Exception as exc:                      # noqa: BLE001
                if dev == "cpu":
                    raise
                self._cuda_err = f"{type(exc).__name__}: {str(exc)[:160]}"
        raise SystemExit("✘ XGBoost 的 CPU 兜底也没成功")

    def _device_plan(self) -> list[str]:
        d = _xgb_device(self.cfg, self.params)
        return ["cpu"] if d == "cpu" else [d, "cpu"]      # GPU 失败自动退回 CPU

    @staticmethod
    def _group(day) -> np.ndarray:
        """分组 = **每个交易日的样本数**（数据必须按组连续——面板天然满足）。

        XGBoost 要的是"组大小"的序列（不是每行的组号），与 LightGBM 的 `group` 同义。
        """
        from ..labels import day_bounds
        return np.diff(day_bounds(np.asarray(day))).astype(np.uint32)

    @staticmethod
    def _group_weights(sample_weight, day) -> np.ndarray:
        """**把按行的样本权重压成按组（每个交易日一个）** —— 排序目标必须这么喂。

        ★ 踩过的坑（2026-09-18，V4 首跑整折崩）：排序目标下把**按行**的权重直接传给
          `XGBRanker.fit`，XGBoost 会当成"每组一个权重"，于是报
          `Check failed: group_weights.size() == group_ptr.size() - 1 (3108185 vs. 1106)`
          —— 3108185 是行数、1106 是交易日数。错误信息不指向根因，只有对着数字才看懂。

        ★ 为什么取**组内均值**就够、且不是近似：本模块喂给排序头的权重只有**时间衰减**
          （`labels.day_decay`，`half_life` 是"每过 N 个交易日减半"），它在**同一天内是常数**
          ⇒ 组内均值 = 每行原值，语义完全一致。
          （若将来给排序头加上**日内变化**的权重如尾部加权，这里需要改成别的归约方式 ——
            那时得重新定义"一个交易日算多重"，不能沿用均值。）
        """
        from ..labels import day_bounds
        w = np.asarray(sample_weight, dtype=np.float64)
        b = day_bounds(np.asarray(day))          # [0, 第1日末尾, 第2日末尾, …, n]
        starts, g = b[:-1].astype(np.int64), np.diff(b).astype(np.int64)
        if w.size != int(g.sum()):
            raise ValueError(f"权重长度 {w.size} 与分组总长 {int(g.sum())} 对不上"
                             f"—— 说明权重与 day 不是同一批行")
        return np.add.reduceat(w, starts) / g

    def predict(self, X):
        return np.asarray(self.model_.predict(X), dtype=np.float32)

    def importance(self):
        try:
            return np.asarray(self.model_.feature_importances_, dtype=np.float32)
        except Exception:            # noqa: BLE001
            return None

    def info(self):
        d = super().info()
        d["device"] = getattr(self, "device_", "?")
        if getattr(self, "_cuda_err", None):
            d["cuda_fallback"] = self._cuda_err
        try:
            d["best_iteration"] = int(getattr(self.model_, "best_iteration", 0) or 0)
        except Exception:            # noqa: BLE001
            d["best_iteration"] = 0
        return d


def _make(xgb, kw: dict, objective: str, n: int, ranked: bool):
    """构造估计器。★ 排序必须用 `XGBRanker` —— `XGBRegressor.fit` 不收 `group`/`qid`。"""
    rounds = int(kw.pop("early_stopping_rounds", 50))
    base = dict(objective=objective, n_estimators=n)
    base.update(kw)
    cls = xgb.XGBRanker if ranked else xgb.XGBRegressor
    try:
        return cls(**base, early_stopping_rounds=rounds)
    except TypeError:                         # 该版本不支持构造期早停 → 不带它构造
        return cls(**base)


def _supports_early_stopping(model) -> bool:
    """这个估计器实例到底有没有吃上 `early_stopping_rounds`（老版本会静默忽略）。"""
    return int(getattr(model, "early_stopping_rounds", 0) or 0) > 0


@register
class XGBRegModel(_XGBBase):
    """加权 XGBoost 回归（`xgb`）—— 与 `gbdt_w` **同配方、不同实现、走 GPU**。"""

    type = "xgb"

    def fit(self, X, y, Xv=None, yv=None, *, sample_weight=None, day=None, day_v=None, **kw):
        return self._fit_xgb(X, y, Xv, yv, sample_weight=sample_weight,
                             day=day, day_v=day_v)


@register
class XGBRankModel(_XGBBase):
    """XGBoost 排序（`xgb_rank`）—— 与 `gbdt_rank` 同构，原生 `rank:ndcg`，走 GPU。

    相关性标签 = **当日截面内的收益等级**（`labels.cs_grade`），按 `qid` 分组。
    `lambdarank_pair_method="topk"` + `lambdarank_num_pair_per_sample` =
    只从**头部**取样构对 —— 与"只买 5 只"的实盘口径对齐。
    """

    type = "xgb_rank"
    default_objective = "rank:ndcg"
    estimator = "ranker"

    def fit(self, X, y, Xv=None, yv=None, *, sample_weight=None, day=None, day_v=None, **kw):
        if day is None:
            raise ValueError("xgb_rank 必须拿到 day（排序只在当日截面内进行）")
        levels = int(self.params.get("grade_levels", 10))
        return self._fit_xgb(
            X, y, Xv, yv, sample_weight=sample_weight, day=day, day_v=day_v,
            label_fn=lambda a, d: cs_grade(a, d, levels=levels),
            extra_params={"lambdarank_pair_method": "topk",
                          "lambdarank_num_pair_per_sample":
                              int(self.params.get("truncation_level", 20))})
