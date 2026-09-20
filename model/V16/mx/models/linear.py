"""线性基线（Ridge）—— 最透明的那个：系数就是因子权重，出问题一眼能看出来。

用途有两个：
  ① **管线验证**：秒级训完，先证明"数据 → 样本 → 训练 → 打分 → 评估"这条链是通的；
  ② **可解释对照**：系数排序 = 模型眼里的因子强弱，可以直接和模块② 的单因子 RankIC 排序对照
     （两者差太远就说明预处理或样本构造有问题）。
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge

from .base import ModelBase, register


@register
class RidgeModel(ModelBase):
    type = "linear"
    fill_mode = "linear"

    def fit(self, X, y, Xv=None, yv=None, **kw):
        self.model_ = Ridge(alpha=float(self.params.get("alpha", 1.0)),
                            fit_intercept=bool(self.params.get("fit_intercept", True)),
                            random_state=self.seed)
        self.model_.fit(X, y)
        return self

    def predict(self, X):
        return self.model_.predict(X).astype(np.float32)

    def importance(self):
        return np.abs(self.model_.coef_).astype(np.float32)

    def info(self):
        d = super().info()
        d["n_features_in"] = int(getattr(self.model_, "n_features_in_", 0))
        return d


@register
class RidgeWModel(RidgeModel):
    """加权岭回归（`linear_w`）—— **把"排序"当回归目标来做**的那条基线。

    与 `RidgeModel` 的差别只有两处，但两处都是要害：

    ① **吃 `sample_weight`**：权重 = 时间衰减 × 尾部加权（`mx/labels.py`）。
       截面中部的样本被降权、头部被升权 ⇒ 同样的线性模型，拟合预算挪到了我们真正会买的那一端。
    ② **目标是秩高斯化的标签**（由 `train_one` 传进来）：`y` 已是 `Φ⁻¹(当日分位)`，
       不再是原始的收益率。MSE 在有界、等方差的 z 上求解，比在重尾的收益率上稳得多。

    ★ 求解器显式写 `cholesky`：样本量 = 数百万、特征 = 224，`XᵀWX` 只有 224×224，
      一次 BLAS 就解完；`solver="auto"` 在带权重时会退到 sag/lsqr 那种迭代解法，慢得多。
    """

    type = "linear_w"

    def fit(self, X, y, Xv=None, yv=None, *, sample_weight=None, **kw):
        alpha = float(self.params.get("alpha", 1.0))
        self.model_ = Ridge(alpha=alpha,
                            fit_intercept=bool(self.params.get("fit_intercept", True)),
                            solver=str(self.params.get("solver", "cholesky")),
                            random_state=self.seed)
        self.model_.fit(X, y, sample_weight=sample_weight)
        return self

    def info(self):
        d = super().info()
        d["weighted"] = True
        return d
