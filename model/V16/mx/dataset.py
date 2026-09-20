"""样本构造与**时序切分** —— 模块③ 的防泄漏核心。

## 为什么必须 purge + embargo（而不是普通的时间切分）

标签是**前视**的：`label_ret_hd(T) = open(T+1+h)/open(T+1) − 1`。
于是相邻交易日的样本**共享未来 h 天的行情** —— 第 T 天的标签用到了 (T+1, T+1+h) 的价格，
第 T+1 天的标签用到了 (T+2, T+2+h) 的价格，两者重叠 h−1 天。

后果：如果只在某个日期上"切一刀"，**训练集最后 h 天**的标签与**测试期最前 h 天**的行情重叠
→ 模型见过测试期的未来 → 指标虚高。所以边界两侧都要按该标签的 h 让开：
    · **purge**：切点**前** h 天的样本不参与训练（它们的标签伸进了测试期）；
    · **embargo**：切点**后** h 天不参与训练（特征与训练期末尾强相关）。
本实现两侧各丢 `embargo_days`（默认 = 该标签的 h）。

## 两套口径（用户 2026-09-17 拍板）

| 口径 | 单元 | 切法 |
|:--|:--|:--|
| `date` **研究** | `V{N}` | **固定 test 窗**（如 2025-09-01 ~ 2026-09-01）；train+valid 只能用到 `train_end`（如 2025-08-10）。<br>★ 每个标签还会按自己的 h 再往前退：**effective train_end = min(train_end, test 起点 − (h+1) 个交易日)**<br>（标签 T 用到 open(T+1+h)，要等 h+1 天才定型）。所以 20d 标签实际退到 2025-08-01 左右、<br>1d 标签停在 08-10 —— 用户说的"隔断 20 天"就是这么严格成立的，不必人工填第二个日期 |
| `date` **实战** | `best` | **无 test**：`train_end` 放开到数据末日，train+valid 用全部数据（见 `unit.split_cfg` 的生产覆盖） |

多折时 test 窗**固定不动**（全折共用 → 指标可比、不泄漏），valid 窗在可训练段内**等距轮换**
（让线性模型也是真 bagging：同数据换种子得到的是同一份模型，那是幻觉）。

`ratio` 模式是遗留口径（按比例 60/20/20 三段），只给 V1 这类历史单元复现用。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import Cfg
from .data import PanelData

MIN_DAYS_PER_BLOCK = 20          # 任何一块少于这么多交易日就直接报错（样本太短切不出）


@dataclass
class Split:
    label: str
    embargo: int
    train: np.ndarray                 # 行号（指向 panel.X）
    valid: np.ndarray
    test: np.ndarray | None           # ★ None = 本口径**没有 test 集**（实战单元 best）
    days: dict = field(default_factory=dict)   # {"train": [...], "valid": [...], "test": [...]}

    @property
    def has_test(self) -> bool:
        return self.test is not None and len(self.test) > 0

    def as_dict(self) -> dict:
        return {"label": self.label, "embargo": self.embargo,
                "n_train": int(len(self.train)), "n_valid": int(len(self.valid)),
                "n_test": int(len(self.test)) if self.test is not None else 0,
                "has_test": bool(self.has_test),
                "days": {k: [v[0], v[-1], len(v)] for k, v in self.days.items()}}


def _rows_by_day(panel: PanelData, mask: np.ndarray) -> tuple[list[str], list[np.ndarray]]:
    """把样本按交易日分组（返回有样本的交易日列表 + 每天的行号数组）。"""
    T = len(panel.dates)
    has = np.zeros(T, dtype=bool)
    np.logical_or.at(has, panel.row_day[mask], True)
    idx = np.flatnonzero(has)
    days = [panel.dates[i] for i in idx]
    order = np.argsort(panel.row_day[mask], kind="stable")
    r = np.flatnonzero(mask)[order]
    d = panel.row_day[r]
    bounds = np.searchsorted(d, idx, side="left")
    ends = np.searchsorted(d, idx, side="right")
    return days, [r[a:b] for a, b in zip(bounds, ends)]


def embargo_for(cfg: Cfg, label: str, sp: dict | None = None) -> int:
    e = (sp or cfg.split_cfg).get("embargo_days")
    return int(e) if e is not None else cfg.label_horizon(label)


def _last_le(days: list[str], d: str) -> int:
    """最后一个 <= d 的日索引（没有则 −1）。"""
    lo, hi = 0, len(days) - 1
    out = -1
    while lo <= hi:
        mid = (lo + hi) // 2
        if days[mid] <= str(d):
            out, lo = mid, mid + 1
        else:
            hi = mid - 1
    return out


def _first_ge(days: list[str], d: str) -> int:
    """第一个 >= d 的日索引（没有则 len(days)）。"""
    for i, x in enumerate(days):
        if x >= str(d):
            return i
    return len(days)


def make_splits(panel: PanelData, label: str, cfg: Cfg, log=print,
                min_days: int = MIN_DAYS_PER_BLOCK,
                fold: int | None = None, n_folds: int = 1,
                split: dict | None = None) -> list[Split]:
    """按配置生成切分。`date` 返回 1 折；`ratio`/`walk_forward` 见下。

    `split` = 本单元**生效**的切分配置（单元 `SPLIT` 覆盖全局后的结果，见 `unit.split_cfg`）；
    不传则用全局 `cfg.split_cfg`。
    """
    sp = dict(split if split is not None else cfg.split_cfg)
    mask = panel.sample_mask(label)
    if not mask.any():
        raise SystemExit(f"标签 {label} 没有任何可用样本")
    days, rows_by_day = _rows_by_day(panel, mask)
    e = embargo_for(cfg, label, sp)
    mode = str(sp.get("mode", "date"))

    if mode == "date":
        return [_date_split(panel, label, sp, e, days, rows_by_day,
                            fold=fold, n_folds=n_folds, min_days=min_days, log=log)]
    if mode == "ratio":
        return _ratio_split(panel, label, sp, e, days, rows_by_day,
                            fold=fold, n_folds=n_folds, min_days=min_days, log=log)
    if mode == "walk_forward":
        return _walk_forward(panel, label, sp, e, days, rows_by_day,
                             min_days=min_days, log=log)
    raise SystemExit(f"未知 split.mode = {mode}")


# ---------------------------------------------------------------- 现行口径：固定日期
def _date_split(panel: PanelData, label: str, sp: dict, e: int,
                days: list[str], rows_by_day: list[np.ndarray], *,
                fold: int | None, n_folds: int, min_days: int, log=print) -> Split:
    n = len(days)
    t_start, t_end = sp.get("test_start"), sp.get("test_end")
    train_end = sp.get("train_end")

    # ---- test 窗（固定日期；无 start/end 即"没有 test 集"）
    if t_start and t_end:
        te = list(range(_first_ge(days, t_start), _last_le(days, t_end) + 1))
        if not te:
            raise SystemExit(
                f"✘ 标签 {label}：test 窗 [{t_start}, {t_end}] 内没有任何可用样本"
                f"（面板覆盖 {days[0]} ~ {days[-1]}）—— 检查切分配置或先补建这些年份的数据")
    else:
        te = []

    # ---- 可训练上界 = min(train_end 给的, 按标签 h purge 给的)
    # ★ 标签 T 用到 open(T+1+h)，要等 **h+1** 个交易日才定型，所以 test 起点前 (h+1) 天
    #   都不能进训练/验证 —— 这才是"隔断 20 天"严格成立的地方。
    b_train = _last_le(days, train_end) if train_end else n - 1
    b_purge = (te[0] - 1 - e) if te else n - 1
    upper = min(b_train, b_purge, n - 1)
    usable = upper + 1
    need = 2 * min_days + e
    if usable < need:
        raise SystemExit(
            f"✘ 标签 {label}：可训练段只有 {usable} 天（test 从 {days[te[0]] if te else '—'} 起，"
            f"purge {e} 天）< 最少 {need} 天 —— 训练块 + 验证块切不出来。"
            f"面板覆盖 {days[0]} ~ {days[-1]}")

    # ---- valid 窗：在 [0, upper] 内等距轮换（test 固定，valid 与 train 随折变化）
    ratio = float(sp.get("valid_ratio", 0.20))
    v = max(min_days, int(round(usable * ratio)))
    hi = max(min_days + e, upper - v + 1)          # valid 末尾必须落在 upper 之内
    lo = max(min_days + e, int(round(usable * 0.5)))   # 训练块至少占一半
    step = (hi - lo) / (n_folds - 1) if (n_folds > 1 and hi > lo) else 0.0
    if n_folds > 1 and step < min_days:
        log(f"  ⚠️ {label}: 验证窗轮换的步长只有 {step:.0f} 天（< {min_days}）—— 各折划分过于接近，"
            f"折间离散度会偏小，别把这种「稳定」当成真信号")
    # fold=None（评价用）取**最后一折**的验证窗：离当下最近的那段
    v0 = hi if fold is None else min(hi, int(round(lo + (fold - 1) * step)))
    va = list(range(v0, min(usable, v0 + v)))
    tr = list(range(0, max(0, v0 - e)))
    for nm, blk in (("train", tr), ("valid", va)):
        if len(blk) < min_days:
            raise SystemExit(f"✘ 标签 {label} 的 {nm} 只有 {len(blk)} 个交易日（< {min_days}）")

    sp_out = Split(label, e,
                   np.concatenate([rows_by_day[i] for i in tr]),
                   np.concatenate([rows_by_day[i] for i in va]),
                   np.concatenate([rows_by_day[i] for i in te]) if te else None,
                   {"train": [days[tr[0]], days[tr[-1]]],
                    "valid": [days[va[0]], days[va[-1]]],
                    **({"test": [days[te[0]], days[te[-1]]]} if te else {})})
    tag = f" · 折{fold}/{n_folds}(验证窗轮换)" if (fold and n_folds > 1) else ""
    if te and b_purge < b_train:
        purged = f"（★ 按 h={e} purge：训练上界退到 {days[upper]}）"
    elif te and train_end:
        purged = f"（train_end={train_end} → 末个交易日 {days[upper]}）"
    else:
        purged = ""
    log(f"  切分 {label}{tag}: train {days[tr[0]]}~{days[tr[-1]]}({len(tr)}d)"
        f" | valid {days[va[0]]}~{days[va[-1]]}({len(va)}d)"
        + (f" | test {days[te[0]]}~{days[te[-1]]}({len(te)}d)" if te else " | test 无（实战口径）")
        + f" | purge/embargo {e} 天{purged}")
    return sp_out


# ---------------------------------------------------------------- 遗留口径：按比例
def _ratio_split(panel: PanelData, label: str, sp: dict, e: int,
                 days: list[str], rows_by_day: list[np.ndarray], *,
                 fold: int | None, n_folds: int, min_days: int, log=print) -> Split:
    tr_r, va_r = float(sp.get("train_ratio", 0.60)), float(sp.get("valid_ratio", 0.20))
    n = len(days)
    i1, i2 = int(n * tr_r), int(n * (tr_r + va_r))
    if fold is not None and n_folds > 1:
        v = max(min_days, int(round(n * va_r)))
        v_end = max(v + min_days + e, i2)
        lo = max(min_days + e, int(round(n * tr_r * 0.5)))
        hi = max(lo, v_end - v)
        step = (hi - lo) / (n_folds - 1) if n_folds > 1 else 0.0
        v0 = int(round(min(hi, lo + (fold - 1) * step)))
        va = list(range(v0, min(n, v0 + v)))
        tr = list(range(0, max(0, v0 - e)))
    else:
        tr = list(range(0, max(0, i1 - e)))
        va = list(range(min(n, i1 + e), max(0, i2 - e)))
    te = list(range(min(n, i2 + e), n))
    for nm, blk in (("train", tr), ("valid", va), ("test", te)):
        if len(blk) < min_days:
            raise SystemExit(
                f"✘ {label} 的 {nm} 只有 {len(blk)} 个交易日（< {min_days}）—— 样本太短。"
                f"总可用天数 {n}，embargo {e}。（ratio 是遗留口径，请优先用 mode: date）")
    split = Split(label, e,
                  np.concatenate([rows_by_day[i] for i in tr]),
                  np.concatenate([rows_by_day[i] for i in va]),
                  np.concatenate([rows_by_day[i] for i in te]),
                  {"train": [days[tr[0]], days[tr[-1]]],
                   "valid": [days[va[0]], days[va[-1]]],
                   "test": [days[te[0]], days[te[-1]]]})
    log(f"  切分 {label}（ratio 遗留口径）: train {split.days['train'][0]}~{split.days['train'][1]}"
        f"({len(tr)}d) | valid {split.days['valid'][0]}~{split.days['valid'][1]}({len(va)}d)"
        f" | test {split.days['test'][0]}~{split.days['test'][1]}({len(te)}d) | embargo {e} 天")
    return split


# ---------------------------------------------------------------- 遗留口径：滚动扩展窗
def _walk_forward(panel: PanelData, label: str, sp: dict, e: int,
                  days: list[str], rows_by_day: list[np.ndarray], *,
                  min_days: int, log=print) -> list[Split]:
    wf = sp.get("walk_forward") or {}
    folds = int(wf.get("folds", 4))
    min_train = int(wf.get("min_train_days", 250))
    n = len(days)
    if n < min_train + MIN_DAYS_PER_BLOCK + 2 * e:
        raise SystemExit(f"✘ {label}: 总天数 {n} 不足以做 {folds} 折 walk-forward")
    out: list[Split] = []
    seg = max(MIN_DAYS_PER_BLOCK, (n - min_train) // max(1, folds))
    for k in range(folds):
        i1 = min_train + k * seg
        i2 = min(n - e, i1 + seg)
        tr = list(range(0, max(0, i1 - e)))
        va = list(range(min(n, i1 + e), i2))
        if len(tr) < min_days or len(va) < min_days:
            continue
        out.append(Split(label, e,
                         np.concatenate([rows_by_day[i] for i in tr]),
                         np.concatenate([rows_by_day[i] for i in va]),
                         None,                      # ★ 折内没有独立 test（要报指标就跑 date 口径）
                         {"train": [days[tr[0]], days[tr[-1]]],
                          "valid": [days[va[0]], days[va[-1]]]}))
    if not out:
        raise SystemExit(f"✘ {label}: walk-forward 一折都没切出来")
    log(f"  切分 {label}: walk-forward {len(out)} 折，embargo {e} 天")
    return out
