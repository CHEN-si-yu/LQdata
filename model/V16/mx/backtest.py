"""组合回测（**最小可用版**）—— 可交易性 + 等权 top-N + 费用 + 净值/回撤/换手。

## 口径（与标签定义严格对齐）

标签是 `T 日信号 → T+1 开盘买 → T+1+h 开盘卖`（见模块② `factors/labels.py`）。
所以**每一笔交易的收益就是该 (股票, T) 的标签值** —— 不需要再自己拼价格，
避免"回测口径与训练目标不一致"这一类最常见的错。

## 组合构造：重叠"批次"（sleeve）

每个信号日 i 选 top-N 组成一个 sleeve，持有 h 个交易日（每天有 h 个 sleeve 在持仓，
每天刷新 1/h）。好处是：① 直接给出**日频净值曲线**（可算回撤/夏普）；
② 换手天然由 h 决定（`1/h × sleeve 换手`），能直接看"持有期越长越省成本"。
★ 近似：一个 sleeve 的 h 日收益按几何平均摊到每一天（日频 mark-to-market 的近似，
真实净值要用逐日行情做 mark —— 那是下一轮的活）。

## 可交易性（用户 2026-09-15 裁定属于模型侧）

- **停牌**：`PriceLayer('traded')` 为 False → T+1 买不进 → 该股从 sleeve 剔除；
- **一字板**：T+1 的 `open == high == low`（且涨跌幅触及限价）→ 也买不进 → 剔除；
- 卖不掉（T+1+h 一字跌停/停牌）**本轮不建模**（标签假定可执行），已在文档里标注为已知简化。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import prices as PR
from .config import Cfg
from .data import load

TRADING_DAYS = 244.0            # A 股年化用的交易日数（与模块② 的文档口径一致）


def _price_arrays(cfg: Cfg, dates: list[str], codes: np.ndarray):
    """取价格层里回测要用的几块面板：开盘/最高/最低/涨跌幅/是否成交。

    ★ 2026-09-19 起价格来自 `trainingdata/P` 快照（`mx/prices.py`），**不再读上游**。
      原先这里要 `upstream._engine()` + `from fea.panel import Panel` + `assert_price_axis`，
      三条外部依赖把回测锁死在"有模块①/② 的机器上"。改成读快照后，回测与训练一样
      只需 `trainingdata/` 一个外部文件夹。
    ★ 轴错位的隐患一并消除：旧实现按**列号**落格（甲乙股票的价会静默互换），
      新实现按 `(trade_date, stock_code)` **显式对齐**，对不上就是 NaN。
    """
    panel, lay = PR.load(cfg, dates, codes)
    return {
        "open": lay.panel(panel, "open"),
        "high": lay.panel(panel, "high"),
        "low": lay.panel(panel, "low"),
        "ret1": lay.panel(panel, "ret1"),
        "traded": lay.panel(panel, "traded").astype(bool),
    }


def tradable_entry(cfg: Cfg, dates: list[str], codes: np.ndarray) -> np.ndarray:
    """(T, C) bool —— 在**每个日期**能否以开盘价买入（用于 T+1 那一行）。

    判据：当日有成交（非停牌） 且 **不是一字板**（open==high==low 且涨跌幅触及限价）。
    """
    px = _price_arrays(cfg, dates, codes)
    o, h, l = px["open"], px["high"], px["low"]
    r1 = px["ret1"]
    lim = float(cfg.raw["backtest"].get("limit_pct", 0.098))
    eq = lambda a, b: np.abs(a - b) <= 1e-6 * np.maximum(1.0, np.abs(a))   # noqa: E731
    one_word = eq(o, h) & eq(h, l)
    at_limit = np.abs(r1) >= lim - 1e-3
    return px["traded"] & ~(one_word & at_limit)


def _daily_from_tranches(sig_days: list[str], sleeve_ret: np.ndarray, h: int,
                         all_days: list[str]) -> pd.Series:
    """把"每个信号日一个 sleeve、持有 h 日"的收益摊成日频序列（几何近似）。"""
    idx = {d: i for i, d in enumerate(all_days)}
    out = np.zeros(len(all_days), dtype=np.float64)
    cnt = np.zeros(len(all_days), dtype=np.float64)
    for i, d in enumerate(sig_days):
        r = sleeve_ret[i]
        if not np.isfinite(r):
            continue
        j0 = idx.get(d)
        if j0 is None:
            continue
        daily = (1.0 + r) ** (1.0 / h) - 1.0
        for k in range(1, h + 1):
            j = j0 + k
            if j < len(all_days):
                out[j] += daily
                cnt[j] += 1
    with np.errstate(invalid="ignore", divide="ignore"):
        series = np.where(cnt > 0, out / np.maximum(cnt, 1), 0.0)
    return pd.Series(series, index=all_days)


def run(cfg: Cfg, model: str, label: str, log=print, top_n: int | None = None,
        hold: int | None = None, frame=None, panel=None,
        dates_lo: str | None = None, dates_hi: str | None = None) -> dict:
    """跑一次组合回测，返回指标 + 净值曲线。

    `frame`：直接给 4 列打分（单元 analysis 用；不给则从 `data/predictions` 读）。
    `panel`：复用已加载的面板（省一次列栈）。
    `dates_lo/dates_hi`：**只在这个日期窗内跑**（研究单元 = 固定 test 窗；实战单元 = 最后一个
      验证窗）。★ 不给窗就是全样本 —— 那含训练期，只能当"历史对照"，不能当业绩。
      （窗口末尾的 sleeve 会持有到窗外，这里按窗口截断，属于保守处理）
    """
    bt = cfg.raw["backtest"]
    top_n = int(top_n or bt["top_n"])
    h = int(hold or bt.get("hold_days") or cfg.label_horizon(label))
    cost = (float(bt["cost_bps"]) + float(bt["slippage_bps"])) / 1e4      # 单边总成本

    panel = panel if panel is not None else load(cfg, log=lambda *_: None)
    if frame is None:
        raise SystemExit("回测必须给 frame（单元打分）—— 旧的 data/predictions 落点已随重构移除")
    pred = frame

    grid = panel.sub(np.arange(panel.n_rows))
    m = grid.merge(pred[["trade_date", "stock_code", "value"]],
                   on=["trade_date", "stock_code"], how="left")
    score = m["value"].to_numpy(dtype=np.float64)
    score_2d = score.reshape(len(panel.dates), len(panel.codes))
    y_2d = panel.y[label].astype(np.float64).reshape(len(panel.dates), len(panel.codes))

    entry_ok = tradable_entry(cfg, panel.dates, panel.codes)     # (T, C) 能否在 T 日开盘买
    log(f"  可交易性：{entry_ok.mean():.1%} 的格子可在开盘买入"
        f"（其余为停牌或一字板）")

    # ---- 每个信号日选 top-N（T 日信号 → 用 **T+1 的 entry_ok** 过滤 → 用 T 的标签算收益）
    sig_days: list[str] = []
    sleeve_ret: list[float] = []
    turnovers: list[float] = []
    prev: set | None = None
    for i, d in enumerate(panel.dates):
        if dates_lo and d < str(dates_lo):
            continue
        if dates_hi and d > str(dates_hi):
            break
        s = score_2d[i]
        yv = y_2d[i]
        j = i + 1                                    # T+1 才成交
        if j >= len(panel.dates):
            break
        ok = np.isfinite(s) & np.isfinite(yv) & entry_ok[j]
        if ok.sum() < top_n:
            continue
        cand = np.flatnonzero(ok)
        pick = cand[np.argsort(-s[cand], kind="stable")[:top_n]]
        sel = set(panel.codes[pick])
        if prev is not None:
            turnovers.append(len(sel - prev) / float(top_n))
        prev = sel
        sig_days.append(d)
        sleeve_ret.append(float(np.mean(yv[pick])))

    if not sig_days:
        raise SystemExit("没有一个信号日能选出组合（检查 top_n / 可交易性）")

    r = np.asarray(sleeve_ret, dtype=np.float64)
    # ---- 费用：每天刷新 1/h 的 sleeve；单边成本 × 双边
    turn = float(np.mean(turnovers)) if turnovers else 1.0
    cost_per_day = (turn * 2.0 * cost) / h
    daily = _daily_from_tranches(sig_days, r, h, panel.dates)
    # ★ 限窗时把净值轴也截到窗内：否则窗口外的空日会照样被扣费用（把夏普压低）
    eq_days = list(panel.dates)
    if dates_lo or dates_hi:
        eq_days = [d for d in panel.dates
                   if (not dates_lo or d >= str(dates_lo)) and (not dates_hi or d <= str(dates_hi))]
        daily = daily.reindex(eq_days).fillna(0.0)
    net = daily - cost_per_day
    equity = (1.0 + net).cumprod()

    # ---- 指标
    ann_ret = float(equity.iloc[-1] ** (TRADING_DAYS / max(1, len(net))) - 1.0)
    vol = float(net.std(ddof=1) * np.sqrt(TRADING_DAYS)) if len(net) > 1 else float("nan")
    sharpe = float(ann_ret / vol) if vol and np.isfinite(vol) and vol > 0 else float("nan")
    dd = float((equity / equity.cummax() - 1.0).min())
    gross_ann = float((1.0 + daily).prod() ** (TRADING_DAYS / max(1, len(net))) - 1.0)

    out = {
        "model": model, "label": label, "hold_days": h, "top_n": top_n,
        "n_signals": len(sig_days), "first_day": sig_days[0], "last_day": sig_days[-1],
        "window": [dates_lo, dates_hi],
        "ann_return_net": ann_ret, "ann_return_gross": gross_ann,
        "ann_vol": vol, "sharpe": sharpe, "max_drawdown": dd,
        "sleeve_turnover": turn, "daily_turnover": turn / h,
        "cost_per_day_bps": cost_per_day * 1e4,
        "mean_sleeve_ret": float(r.mean()), "win_rate": float((r > 0).mean()),
        "equity": [round(float(x), 6) for x in equity.to_numpy()],
        "equity_days": eq_days,
    }
    win_txt = f" · 窗口 {dates_lo}~{dates_hi}" if (dates_lo or dates_hi) else " · ⚠️ 全样本（含训练期）"
    log(f"  回测 {model}/{label}（top {top_n} · 持有 {h} 日{win_txt}）："
        f"净年化 {ann_ret:+.2%} · 毛年化 {gross_ann:+.2%} · 波动 {vol:.2%} · "
        f"夏普 {sharpe:.2f} · 最大回撤 {dd:.2%} · 日换手 {turn/h:.1%}")
    return out


def render(res: dict) -> str:
    """ASCII 净值曲线（终端里直接看走势，不依赖 matplotlib）。"""
    eq = res.get("equity") or []
    days = res.get("equity_days") or []
    if not eq:
        return "（无净值数据）"
    lo, hi = min(eq), max(eq)
    W, H = 60, 8
    rng = (hi - lo) or 1.0
    grid = [[" "] * W for _ in range(H)]
    for k, v in enumerate(eq):
        x = int(k * (W - 1) / max(1, len(eq) - 1))
        y = H - 1 - int((v - lo) / rng * (H - 1))
        grid[y][x] = "█"
    lines = [f"  净值 {res['model']}/{res['label']} · {days[0]} ~ {days[-1]}",
             f"  {hi:.3f} ┌" + "".join(grid[0])]
    for row in grid[1:-1]:
        lines.append("        │" + "".join(row))
    lines.append(f"  {lo:.3f} └" + "".join(grid[-1]))
    return "\n".join(lines)
