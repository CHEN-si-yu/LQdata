"""LightGBM 基线 —— 主力模型。

为什么它是量化横截面回归的默认选择：
  · 天然处理 NaN（因子覆盖率参差，本库 87.5%），不需要填补；
  · 非线性 + 特征交互自动，且对异常值不敏感（树只关心排序）；
  · CPU 上就很快（32 核），GPU 机器上不改代码也能跑 —— 完美贴合"两手准备"。

★ 早停用的是验证集 L2，不是 IC：L2 稳定、快；IC 型早停留作后续优化（见 docs）。
"""
from __future__ import annotations

import numpy as np

from .. import losses as S
from ..labels import cs_grade
from .base import ModelBase, register


@register
class GBDTModel(ModelBase):
    type = "gbdt"

    def fit(self, X, y, Xv=None, yv=None, **kw):
        import lightgbm as lgb

        params = dict(self.params)
        rounds = int(self.cfg.raw["train"].get("early_stopping_rounds", 50))
        self.model_ = lgb.LGBMRegressor(random_state=self.seed, **params)
        callbacks = []
        fit_kw = {}
        if Xv is not None and yv is not None and len(Xv):
            fit_kw["eval_set"] = [(Xv, yv)]
            fit_kw["eval_metric"] = "l2"
            callbacks.append(lgb.early_stopping(rounds, verbose=False))
        self.model_.fit(X, y, callbacks=callbacks or None, **fit_kw)
        return self

    def predict(self, X):
        return self.model_.predict(X).astype(np.float32)

    def importance(self):
        try:
            return np.asarray(self.model_.feature_importances_, dtype=np.float32)
        except Exception:            # noqa: BLE001
            return None

    def info(self):
        d = super().info()
        d["best_iteration"] = int(getattr(self.model_, "best_iteration_", 0) or 0)
        return d


# ==================================================================== 加权 / 排序 / 顶部焦点
# 三种"把拟合预算挪到截面头部"的树模型。为什么走 LightGBM **原生 API**：
#   `LGBMRegressor.fit` 不接受 `group=`（实测 4.7.0 直接 TypeError），而排序目标必须按日分组。
#   原生 API 还能顺带控制 `lambdarank_truncation_level`、自定义目标、以及 `free_raw_data`。
# ★ 这些键是**引擎给模型的口径声明**（目标变换、样本权重、分组），不是 LightGBM 的超参 ——
#   原样传给 `lgb.train` 会报 `Unknown type of parameter:sample_weight, got:dict`。
#   **新增这类键时必须同步加到这里**；漏了会在下面被 "值是 dict" 的兜底检查当场抓住。
_NON_LGB_KEYS = {"labels", "label_transform", "sample_weight", "tau",
                 "n_estimators", "num_boost_round", "grade_levels", "buyable"}


class _NativeBooster(ModelBase):
    """LightGBM 原生 API 的公共部分（构造 Dataset / 早停 / 取重要性）。"""

    #: 子类若要换掉 objective，在这里给出（默认取 params["objective"]）。
    default_objective = "regression"

    def _lgb_params(self) -> dict:
        import lightgbm as lgb      # noqa: F401  （确认装了，早失败）
        p = {k: v for k, v in self.params.items() if k not in _NON_LGB_KEYS}
        p.setdefault("objective", self.default_objective)
        p.setdefault("verbose", -1)
        p["num_threads"] = int(p.get("num_threads", self.params.get("n_jobs", 8)))
        p.pop("n_jobs", None)
        p["seed"] = self.seed
        p["feature_fraction_seed"] = self.seed + 1
        p["bagging_seed"] = self.seed + 2
        # ★ 兜底：口径声明必然是 dict/list，LightGBM 超参必然是标量或字符串。
        #   漏进一个 dict 就会得到一个**语义不明**的 LightGBM 报错；这里直接点名是哪个键，
        #   并提示该往 `_NON_LGB_KEYS` 里加。
        bad = [k for k, v in p.items() if isinstance(v, (dict, list, tuple))
               and k not in ("interaction_constraints", "monotone_constraints")]
        if bad:
            raise SystemExit(
                f"✘ {self.name}：这些键看起来是**口径声明**而不是 LightGBM 超参：{bad} —— "
                f"把它们加进 `mx/models/gbdt.py:_NON_LGB_KEYS`")
        return p

    def _rounds(self) -> int:
        n = self.params.get("n_estimators", self.params.get("num_boost_round"))
        if n is None:
            n = self.cfg.raw["models"]["lgbm"]["params"].get("n_estimators", 500)
        return int(n)

    @staticmethod
    def _group_sizes(day) -> np.ndarray:
        from ..labels import day_bounds
        return np.diff(day_bounds(day)).astype(np.int32)

    def _fit_native(self, X, y, Xv, yv, *, sample_weight, day, day_v,
                    label_fn=None, obj=None, metric=None):
        import lightgbm as lgb

        params = self._lgb_params()
        if metric is not None:
            params["metric"] = metric
        # ★ LightGBM 4.x 把 `fobj` 从 `train()` 的参数表里拿掉了（实测 4.7.0 直接 TypeError）——
        #   自定义目标现在走 `params["objective"] = 可调用对象`。
        if obj is not None:
            params["objective"] = obj
        lab = label_fn(y, day) if label_fn else y
        ds_kw = {"free_raw_data": True}
        if day is not None:
            ds_kw["group"] = self._group_sizes(day)
        if sample_weight is not None:
            ds_kw["weight"] = np.asarray(sample_weight, dtype=np.float64)
        dtrain = lgb.Dataset(np.ascontiguousarray(X), label=lab, **ds_kw)

        valid_sets, valid_names, cbs = [], [], [lgb.log_evaluation(0)]
        if Xv is not None and yv is not None and len(Xv):
            v_kw = {"reference": dtrain, "free_raw_data": True}
            if day_v is not None:
                v_kw["group"] = self._group_sizes(day_v)
            dvalid = lgb.Dataset(np.ascontiguousarray(Xv),
                                 label=(label_fn(yv, day_v) if label_fn else yv), **v_kw)
            valid_sets, valid_names = [dvalid], ["valid"]
            cbs.append(lgb.early_stopping(
                int(self.cfg.raw["train"].get("early_stopping_rounds", 50)), verbose=False))

        self.booster_ = lgb.train(params, dtrain, num_boost_round=self._rounds(),
                                  valid_sets=valid_sets or None,
                                  valid_names=valid_names or None, callbacks=cbs)
        self.best_iteration_ = int(getattr(self.booster_, "best_iteration", 0) or 0)
        return self

    def predict(self, X):
        n = self.best_iteration_ or 0
        return self.booster_.predict(np.ascontiguousarray(X),
                                     num_iteration=n or None).astype(np.float32)

    def importance(self):
        try:
            return np.asarray(self.booster_.feature_importance("gain"), dtype=np.float32)
        except Exception:            # noqa: BLE001
            return None

    def info(self):
        d = super().info()
        d["best_iteration"] = self.best_iteration_
        return d


@register
class GBDTWModel(_NativeBooster):
    """加权 LightGBM（`gbdt_w`）—— 与 `GBDTModel` 同架构，只多两样：样本权重、秩高斯目标。

    "只换损失、不换架构"那条假设的最干净落点：`gbdt` vs `gbdt_w` 的差异
    可以 100% 归因到权重与标签变换上。
    """

    type = "gbdt_w"
    default_objective = "regression"

    def fit(self, X, y, Xv=None, yv=None, *, sample_weight=None, day=None, day_v=None, **kw):
        return self._fit_native(X, y, Xv, yv, sample_weight=sample_weight,
                                day=day, day_v=day_v, metric="l2")


@register
class GBDTRankModel(_NativeBooster):
    """排序 LightGBM（`gbdt_rank`）—— **换了一个模型族**，不是换损失。

    回归模型问的是"这只股票涨多少"，排序模型问的是"这只股票该排在谁前面"。
    目标换成 LightGBM 原生的 `lambdarank`：
      · 相关性标签 = **当日截面内的收益等级**（`labels.cs_grade`，等分位切）；
      · `group` = 每个交易日的股票数（排序只在日内进行 —— 跨日的分数不可比）；
      · `lambdarank_truncation_level` 只对**头部**算梯度 ⇒ 天然是"顶部焦点"；
      · 早停指标用 **NDCG@5**，与我们"只买 5 只"的实盘口径直接对齐。
    """

    type = "gbdt_rank"
    default_objective = "lambdarank"

    def fit(self, X, y, Xv=None, yv=None, *, sample_weight=None, day=None, day_v=None, **kw):
        if day is None:
            raise ValueError("gbdt_rank 必须拿到 day（排序只在当日截面内进行）")
        levels = int(self.params.get("grade_levels", 10))
        return self._fit_native(
            X, y, Xv, yv, sample_weight=sample_weight, day=day, day_v=day_v,
            label_fn=lambda a, d: cs_grade(a, d, levels=levels),
            metric="ndcg", )


@register
class GBDTFocusModel(_NativeBooster):
    """顶部焦点 LightGBM（`gbdt_focus`）—— 目标函数本身带"往头部看"的权重。

    用 `mx/losses.py:top_focus_grad_hess`：每一轮用**当前预测的逐日 softmax** 当权重，
    把拟合预算压到模型自己认为的头部（`τ` 越小越极端）。

    与 `gbdt_w` 的区别：`gbdt_w` 的权重是**外生的**（按历史收益的秩给的固定权重），
    这条的权重是**内生的**（随预测实时变化）。
    """

    type = "gbdt_focus"
    default_objective = "none"

    def fit(self, X, y, Xv=None, yv=None, *, sample_weight=None, day=None, day_v=None, **kw):
        if day is None:
            raise ValueError("gbdt_focus 必须拿到 day（softmax 分母在当日截面内）")
        tau = float(self.params.get("tau", 0.05))
        day_arr = np.asarray(day)

        def fobj(preds, dset):
            w = dset.get_weight()
            w = np.asarray(w, dtype=np.float64) if w is not None and len(w) else None
            return S.top_focus_grad_hess(dset.get_label(), preds, day_arr, tau=tau, weight=w)

        # 自定义目标没有自带 metric ⇒ 显式给 l2 供早停用（早停看的是"分值准不准"）
        return self._fit_native(X, y, Xv, yv, sample_weight=sample_weight,
                                day=day, day_v=day_v, obj=fobj, metric="l2")
