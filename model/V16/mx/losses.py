"""损失函数 —— 把"截面头部拟合得好"翻译成可优化的目标。

## 问题陈述

实盘只买 5 只，但 MSE 均匀地关心当日截面的 3000 只。IC 与 top_ret 的耦合点就在这：
**头部决定 top_ret，中部只决定 IC 的分母**。本文件给出三种"把预算挪到头部"的机制。

## 三种机制（按风险从低到高）

| 机制 | 实现 | 落点 | 风险 |
|:--|:--|:--|:--|
| **样本权重** | `labels.make_weights`（衰减 × 尾部） | 原生 `sample_weight` | 无 —— sklearn/LightGBM 都原生支持 |
| **顶部焦点** | `softmax_weights` + `top_focus_grad_hess` | LightGBM 自定义目标 | 中 —— 数学是启发式的，但 Hessian 恒正、不会发散 |
| **顶部可微损失** | `softmax_top_loss_np`（torch 版在 `mx/models/nn.py`） | 神经网络 | 低 —— autograd 直接算真梯度 |

## 顶部可微损失：为什么是它

`L_top = −Σ_i p_i · z_i`，其中 `p_i = softmax(s_i / τ)` 是**当日截面内**归一化的选股权重。

- `τ → 0` 时 `p` 退化成 one-hot ⇒ 就是"买 top-1"的目标；
- `τ` 适中时 `p` 是个平滑的 top-k ⇒ 对应"买 top-5、等权"；
- 它对 `s` 处处可微，梯度 `∂L/∂s_j = −(1/τ)·p_j·(z_j − Σ p_i z_i)`
  —— 直观含义：**预测分越高的股票，模型越应该为它的排序误差负责**。

这条式子就是"IC 与 top_ret 耦合"的落点：同一组参数，`z` 取秩高斯时它逼近排序质量（IC 侧），
`z` 取真实收益率时它逼近组合收益（top_ret 侧）。

## ★ PIT

所有按日归约（softmax 的分母、平均 IC、分位）一律在**当日截面内**完成，
绝不跨日、绝不碰全样本统计量。`day` 的构造见 `mx/labels.py:day_bounds`。
"""
from __future__ import annotations

import numpy as np

from .labels import day_bounds


def _check(day: np.ndarray, n: int) -> np.ndarray:
    """自定义目标里最难防的一类错：LightGBM 传进来的行数与训练段对不上。

    一旦对不上，按日分组就会把不同天的股票混在一起算 softmax 分母 —— 那是**静默的错误结果**。
    所以这里直接抛错而不是打日志降级：宁可训练挂掉，也不要训出一个说不清的东西。
    """
    d = np.asarray(day)
    if d.size != n:
        raise ValueError(
            f"day 的长度({d.size}) 与传入的样本数({n}) 不一致 —— "
            f"自定义目标必须拿到完整的训练段（不能是 bagging 子集），否则按日归约会算错")
    return d


# ------------------------------------------------------------------ 顶部焦点
def softmax_weights(pred: np.ndarray, day: np.ndarray, *, tau: float) -> np.ndarray:
    """`p_i = softmax(pred_i / tau)`，**分母在当日截面内**求和。

    ★ 数值稳定：先减去当日最大值再取指数（Softmax 的标准写法），
      否则 `pred/tau` 在 tau 很小时会直接溢出成 inf。
    """
    p = np.asarray(pred, dtype=np.float64).copy()
    out = np.zeros_like(p)
    b = day_bounds(_check(day, p.size))
    for k in range(b.size - 1):
        lo, hi = int(b[k]), int(b[k + 1])
        v = p[lo:hi] / float(tau)
        v -= np.nanmax(v)
        e = np.exp(v)
        s = e.sum()
        out[lo:hi] = e / s if s > 0 else 1.0 / (hi - lo)
    return out


def top_focus_grad_hess(y_true: np.ndarray, y_pred: np.ndarray, day: np.ndarray,
                        *, tau: float = 0.05, weight: np.ndarray | None = None
                        ) -> tuple[np.ndarray, np.ndarray]:
    """LightGBM 自定义目标：**顶部焦点加权 MSE**。

    `L = Σ_i p_i · (ŷ_i − y_i)²`，其中 `p_i = softmax(ŷ_i / τ)` **按当次迭代的预测算、
    但对求导视作常数**（IRLS / Gauss-Newton 的标准做法）。

    - 梯度 `∂L/∂ŷ_j = 2 p_j (ŷ_j − y_j)`
    - Hessian `∂²L/∂ŷ_j² = 2 p_j`（★ 恒正，LightGBM 不会发散）

    ★ 为什么不直接用 `L_top` 的真梯度：那个 Hessian 是
      `−(1/τ²)·p_j·(z_j − S)·(1 − 2p_j)`，符号不定 —— LightGBM 要求 hess > 0，
      只能靠取绝对值硬掰，实战里会抖。顶部焦点 MSE 是同一意图下**数值上安全**的那个写法：
      每轮用"当前预测的 softmax"当权重，把拟合预算压到模型自己认为的头部。

    语义上它和 `L_top` 不同：`L_top` 关心"头部的**收益**对不对"，
    这条关心"头部的**分值**准不准"。两者都试，让榜单说话。
    """
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)
    p = softmax_weights(yp, day, tau=tau)
    if weight is not None:
        p = p * np.asarray(weight, dtype=np.float64)
    grad = 2.0 * p * (yp - yt)
    hess = 2.0 * np.maximum(p, 1e-6)
    return grad, hess


# ------------------------------------------------------------------ 顶部可微损失（numpy 参考实现）
def softmax_top_loss_np(pred: np.ndarray, target: np.ndarray, day: np.ndarray,
                        *, tau: float) -> float:
    """`−Σ_i p_i · z_i` 的 numpy 参考实现（torch 版在 `mx/models/nn.py`）。

    ★ 存在的意义是**给 torch 版做数值对拍**：两边的公式一旦漂移，测试会立刻发现。
      神经网络那些"训练不收敛"的事故，九成死在损失写错而没人对拍。
    """
    p = softmax_weights(pred, day, tau=tau)
    z = np.asarray(target, dtype=np.float64)
    return float(-np.sum(p * z))


def softmax_top_grad_np(pred: np.ndarray, target: np.ndarray, day: np.ndarray,
                        *, tau: float) -> np.ndarray:
    """`∂L_top/∂s_j = −(1/τ)·p_j·(z_j − S_d)`，`S_d = Σ_{i∈当日} p_i z_i`。用于和 torch 对拍。

    ★ `S_d` 必须是**当日截面内**的和。踩过的坑：写成 `Σ_all p_i z_i`（全数组求和）
      时，单日用例逐位正确、多日用例静默错掉 —— 因为 softmax 是逐日归一的，
      跨日求和等于把"每天 z 的均值"混在一起。`tests/test_labels_losses.py` 盯的就是这条。
    """
    p = softmax_weights(pred, day, tau=tau)
    z = np.asarray(target, dtype=np.float64)
    s = np.zeros(p.shape, dtype=np.float64)          # 每行所属日的 S_d
    b = day_bounds(_check(day, p.size))
    for k in range(b.size - 1):
        lo, hi = int(b[k]), int(b[k + 1])
        s[lo:hi] = float(np.sum(p[lo:hi] * z[lo:hi]))
    return -(1.0 / float(tau)) * p * (z - s)


# ------------------------------------------------------------------ IC 损失
def daily_pearson(pred: np.ndarray, target: np.ndarray, day: np.ndarray) -> np.ndarray:
    """逐日截面 Pearson 相关（数组按天）。`std=0` 或样本数 < 3 的那天返回 NaN。

    ★ 只用当日截面 —— 这是"IC 损失"的参考实现，同时也是 `mx/evaluate.py`
      之外第二处算 IC 的地方，两者对不上就说明有一处写错了。
    """
    p = np.asarray(pred, dtype=np.float64)
    z = np.asarray(target, dtype=np.float64)
    b = day_bounds(_check(day, p.size))
    out = np.full(b.size - 1, np.nan)
    for k in range(b.size - 1):
        lo, hi = int(b[k]), int(b[k + 1])
        a_, b_ = p[lo:hi], z[lo:hi]
        ok = np.isfinite(a_) & np.isfinite(b_)
        if ok.sum() < 3:
            continue
        a_, b_ = a_[ok], b_[ok]
        sa, sb = a_.std(), b_.std()
        if sa <= 0 or sb <= 0:
            continue
        out[k] = float(((a_ - a_.mean()) * (b_ - b_.mean())).mean() / (sa * sb))
    return out


def describe(tau: float | None = None, **kw) -> str:
    """一行文字描述，进日志/榜单。"""
    bits = []
    if tau:
        bits.append(f"顶部softmax τ={tau}")
    for k, v in kw.items():
        if v:
            bits.append(f"{k}={v}")
    return " · ".join(bits) or "无额外损失项"
