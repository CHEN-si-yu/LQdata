"""出图 —— 回测收益曲线（用户 2026-09-18 交办）。

> "回测结果的时候能生成一些收益的曲线图，类似原来那个 git 项目中 analysis 里
>   会生成一年的收益累计曲线。"

## 落点

每个单元的 `model_pred/` 桶（图是**回测的输出**，所以和榜单/回测 JSON 摆在一起；
冒烟跑落在 `model_pred/smoke/`）：

    model_pred/nav_<打分源>_<策略>.png      单条策略的**累计收益曲线**（跨折集成打分）
    model_pred/nav_folds_<打分源>.png       同一条策略的**逐折曲线 + 折间带**（min~max）
    model_pred/leaderboard_top1.png         主判据 Σtop1 的横向条形图

## ★ 中文标签：本机没有 CJK 字体，所以默认走英文

`matplotlib` 全机只有 DejaVu（无 CJK）。中文标签会渲染成**豆腐块**（□□□），
比英文更糟。所以这里做**自动降级**：检测到可用的 CJK 字体就用中文，否则用英文。
将来装了字体（`fc-list` 能看到 CJK），标题自动变回中文，不用改代码。

★ 铁律：**图只画"评价窗口"内的那条曲线**（研究单元 = test 窗）。把训练期也画进去是自欺 ——
  净值在样本内好看没有任何意义（README §7）。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

_CJK_CACHE: bool | None = None


def has_cjk() -> bool:
    """本机有没有可用的中文字体（结果缓存，避免每次画图都扫字体表）。"""
    global _CJK_CACHE
    if _CJK_CACHE is None:
        try:
            from matplotlib import font_manager as fm
            names = {f.name for f in fm.fontManager.ttflist}
            _CJK_CACHE = any(k in n for n in names for k in
                             ("Noto Sans CJK", "Noto Serif CJK", "WenQuanYi",
                              "Source Han", "SimHei", "Microsoft YaHei",
                              "PingFang", "Droid Sans Fallback", "AR PL"))
        except Exception:               # noqa: BLE001
            _CJK_CACHE = False
    return bool(_CJK_CACHE)


def L(zh: str, en: str) -> str:
    """有中文字体就用中文，否则退回英文（**不要**在图上硬写中文）。"""
    return zh if has_cjk() else en


def _plt():
    import matplotlib
    matplotlib.use("Agg")           # 无显示环境；必须在 pyplot 之前设
    import matplotlib.pyplot as plt
    if has_cjk():
        plt.rcParams["font.sans-serif"] = [
            "Noto Sans CJK SC", "WenQuanYi Zen Hei", "Source Han Sans SC",
            "SimHei", "Microsoft YaHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 120
    plt.rcParams["savefig.bbox"] = "tight"
    return plt


# ---------------------------------------------------------------- 工具
def _equity_to_cumret(res: dict) -> tuple[np.ndarray, np.ndarray] | None:
    """把一次回测的净值序列转成 (日期, 累计收益率)。缺数据就返回 None。"""
    nav = res.get("equity")
    dates = res.get("nav_dates")
    cap = float(res.get("capital") or 0.0)
    if nav is None or dates is None or not cap:
        return None
    nav = np.asarray(nav, dtype=np.float64)
    ok = np.isfinite(nav)
    if ok.sum() < 2:
        return None
    # ★ 防御：净值与日期两条序列长度必须一致（2026-09-18 修 `_metrics` 前它们差一天）。
    #   真要再不一致，就按"末尾对齐"截齐 —— 两条序列都以同一个交易日收尾，
    #   截尾巴保住的正是评价窗最后那段，画出来的曲线不会悄悄错位。
    k = min(len(nav), len(dates))
    nav, dates = nav[-k:], list(dates)[-k:]
    ok = np.isfinite(nav)
    return np.asarray(dates, dtype=object)[ok], nav[ok] / cap - 1.0


def _per_fold_cells(matrix: dict) -> dict[str, list[dict]]:
    """`{打分源: [各折的 cell]}`（只取逐折口径）。"""
    out: dict[str, list[dict]] = {}
    for c in (matrix or {}).get("cells", []):
        if c.get("scope") != "fold":
            continue
        out.setdefault(c["source"].split(":", 1)[1], []).append(c)
    return out


def _ens_cell(matrix: dict, source: str) -> dict | None:
    for c in (matrix or {}).get("cells", []):
        if c.get("scope") == "ens" and c.get("source") == f"ens:{source}":
            return c
    return None


# ---------------------------------------------------------------- 图 1：单条曲线
def nav_curve(res: dict, *, title: str, out: Path, benchmark: dict | None = None) -> Path | None:
    """画一条策略的**累计收益率曲线**（评价窗口内）。`benchmark` 可选（同结构的回测结果）。"""
    got = _equity_to_cumret(res)
    if got is None:
        return None
    d, y = got
    plt = _plt()
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.plot(range(len(y)), y * 100, lw=1.6, color="#1f77b4",
            label=L("策略", "strategy"))
    if benchmark is not None:
        bg = _equity_to_cumret(benchmark)
        if bg is not None:
            bd, by = bg
            n = min(len(y), len(by))
            ax.plot(range(n), by[:n] * 100, lw=1.3, ls="--", color="#7f7f7f",
                    label=L("等权全买（基准）", "equal-weight benchmark"))
    ax.axhline(0, color="#999", lw=0.8)
    ax.set_xlabel(L("交易日（评价窗口内）", "trading days (eval window)"))
    ax.set_ylabel(L("累计收益率 %", "cumulative return %"))
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=9)
    step = max(1, len(y) // 8)
    ax.set_xticks(list(range(0, len(y), step)))
    ax.set_xticklabels([str(d[i]) for i in range(0, len(y), step)], rotation=30, fontsize=8)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return out


# ---------------------------------------------------------------- 图 2：折间带
def nav_with_folds(matrix: dict, source: str, strategy: str, params: dict, *,
                   out: Path, window: str = "") -> Path | None:
    """**跨折集成曲线 + 逐折曲线 + 折间 min~max 带** —— 一图看清"信号还是运气"。

    ★ 为什么必须带折间带：本轮的种子冗余实测（README §8.1）显示
      **模型稳、组合抖**（RankIC 极差 0.0037，而 ≤5 只组合收益极差 20.4 个百分点）。
      只画一条集成曲线会把这个抖动完全藏起来。
    """
    ens = _ens_cell(matrix, source)
    if ens is None:
        return None
    # 同一条（策略, 参数）的所有 cell
    def match(c):
        return (c.get("strategy") == strategy and (c.get("params") or {}) == (params or {}))
    ens_c = next((c for c in (matrix or {}).get("cells", [])
                  if c.get("scope") == "ens" and c.get("source") == f"ens:{source}" and match(c)), None)
    if ens_c is None:
        return None
    got = _equity_to_cumret(ens_c)
    if got is None:
        return None
    d, y = got
    folds = [c for c in (matrix or {}).get("cells", [])
             if c.get("scope") == "fold" and c["source"].split(":", 1)[1] == source and match(c)]
    plt = _plt()
    fig, ax = plt.subplots(figsize=(9, 4.4))
    curves = []
    for c in sorted(folds, key=lambda x: x.get("fold") or 0):
        g = _equity_to_cumret(c)
        if g is None:
            continue
        fd, fy = g
        n = min(len(y), len(fy))
        curves.append(fy[:n])
        ax.plot(range(n), fy[:n] * 100, lw=0.9, alpha=0.55, color="#bbbbbb",
                label=L(f"第 {c.get('fold')} 折", f"fold {c.get('fold')}")
                if c is folds[0] else None)
    if curves:
        M = np.vstack([np.pad(c, (0, len(y) - len(c)), constant_values=np.nan)
                       for c in curves if len(c) <= len(y)] or [np.full(len(y), np.nan)])
        ax.fill_between(range(len(y)), np.nanmin(M, 0) * 100, np.nanmax(M, 0) * 100,
                        color="#1f77b4", alpha=0.12,
                        label=L("折间 min~max", "fold min~max"))
    ax.plot(range(len(y)), y * 100, lw=1.9, color="#d62728",
            label=L("跨折集成", "ensemble of folds"))
    ax.axhline(0, color="#999", lw=0.8)
    ax.set_xlabel(L("交易日（评价窗口内）", "trading days (eval window)"))
    ax.set_ylabel(L("累计收益率 %", "cumulative return %"))
    pp = ", ".join(f"{k}={v}" for k, v in (params or {}).items())
    ax.set_title(f"{source} / {strategy}{(' ' + pp) if pp else ''}"
                 + (f"  [{window}]" if window else ""))
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=9)
    step = max(1, len(y) // 8)
    ax.set_xticks(list(range(0, len(y), step)))
    ax.set_xticklabels([str(d[i]) for i in range(0, len(y), step)], rotation=30, fontsize=8)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return out


# ---------------------------------------------------------------- 图 3：榜单条形图
def leaderboard_bars(rows: list[dict], *, out: Path, top: int = 14) -> Path | None:
    """主判据 Σtop1 的横向条形图（带"剔1天"的浅色叠加，一眼看出靠不靠一两天）。"""
    r = [x for x in rows if (x.get("costfree") or {}).get("top1_sum") is not None]
    if not r:
        return None
    r.sort(key=lambda x: -(x["costfree"]["top1_sum"]))
    r = r[:top]
    names = [f"{x['head']}/{x['label'].replace('label_ret_', '')}" for x in r]
    v1 = [x["costfree"]["top1_sum"] for x in r]
    v2 = [x["costfree"].get("top1_drop1") or 0.0 for x in r]
    plt = _plt()
    fig, ax = plt.subplots(figsize=(8.5, max(3.0, 0.34 * len(r) + 1.2)))
    yy = np.arange(len(r))[::-1]
    ax.barh(yy, v1, color="#1f77b4", height=0.62, label=L("Σtop1", "sum top1"))
    ax.barh(yy, v2, color="#d62728", height=0.30,
            label=L("剔掉最好的 1 天后", "after dropping best 1 day"))
    ax.axvline(0, color="#999", lw=0.8)
    ax.set_yticks(yy)
    ax.set_yticklabels(names, fontsize=8.5)
    ax.set_xlabel(L("买 top-1 隔日换手的收益累加（可执行口径）",
                    "cumulative top-1 next-day return (executable)"))
    ax.set_title(L("主判据 Σtop1（评价窗口）", "Main criterion: sum top1 (eval window)"))
    ax.grid(alpha=0.25, axis="x")
    ax.legend(loc="lower right", fontsize=9)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return out


# ---------------------------------------------------------------- 入口
def make_all(summary: dict, pics_dir: Path, *, top_strategies: int = 3, log=print) -> list[str]:
    """给一个单元的 analysis summary 出全套图，返回写出的文件名列表。"""
    pics_dir = Path(pics_dir)
    written: list[str] = []
    bt = summary.get("backtest") or {}
    # ① 回测曲线（单元组合 + 评价窗最强单头）
    for key, tag in (("unit", summary.get("unit", "unit")), ("best_head", None)):
        res = bt.get(key)
        if not res:
            continue
        head = res.get("head") or tag or key
        label = res.get("label") or ""
        p = nav_curve(res, title=f"{head} / {label}  [{res.get('window', '')}]",
                      out=pics_dir / f"nav_bt_{key}_{head}.png",
                      benchmark=bt.get("unit") if key != "unit" else None)
        if p:
            written.append(p.name)
    # ② 策略矩阵：挑 Σtop1 最优的前几条策略，出"折间带"图
    mx = summary.get("strategy_matrix") or {}
    ens = [c for c in mx.get("cells", []) if c.get("scope") == "ens"]
    if ens:
        ens.sort(key=lambda c: -(c.get("total_ret") or -9e9))
        seen = set()
        for c in ens:
            src = c["source"].split(":", 1)[1]
            key = (src, c.get("strategy"), str(c.get("params")))
            if key in seen:
                continue
            seen.add(key)
            p = nav_with_folds(mx, src, c.get("strategy"), c.get("params") or {},
                               out=pics_dir / f"nav_folds_{src}_{c.get('strategy')}"
                                              f"_{len(seen)}.png",
                               window=mx.get("window_kind") or "")
            if p:
                written.append(p.name)
            if len(seen) >= top_strategies:
                break
    # ③ 榜单条形图
    p = leaderboard_bars(summary.get("leaderboard") or [],
                         out=pics_dir / "leaderboard_top1.png")
    if p:
        written.append(p.name)
    if written:
        log(f"  出图：{len(written)} 张 → {pics_dir.name}/（{'、'.join(written[:4])}"
            + ("…" if len(written) > 4 else "") + "）")
    return written
