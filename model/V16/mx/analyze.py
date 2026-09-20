"""单元分析 —— **折 → 头 → 集成 → 评价 → 榜单 → 回测**。

这是 `Model/V{N}/analysis.py` 背后干的全部事情。三件事按顺序：

1. **装配**：读该单元**每一折**的打分产物 → 每个头做「逐日截面 z 后跨折平均」（跨折集成），
   再按单元 `model.py` 的 `combine()` 把头合成"这一版的打分"。
2. **评价 + 榜单**：每个 (标签 × 头) 在 valid/test 上出指标（RankIC/ICIR/分层/换手），
   并按 test RankIC 排名；同时给出**跨折（换种子）离散度** —— 这是"单种子纪录 = 种子中奖"
   纪律的量化体现，没有它就不许宣称新纪录。
3. **回测**：对主标签的组合打分跑一次含费用/可交易性的最小可用回测。

⚠️ 口径诚实声明：榜单是**在 test 窗口上排的**，用来选版本就存在选择偏差；
   参考工程的做法是"分半 + 平台区 + 多种子"三件套，我们目前只做到**多种子离散度**，
   分半与 bootstrap 留到下一轮（写进 REPORT 的待办）。
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from . import backtest, combine, dataset, evaluate, state, store
from .config import Cfg
from .data import load_for_unit
from .unit import Unit


def fold_ids(cfg: Cfg, unit: Unit, smoke: bool = False, sample: bool = False) -> list[int]:
    """已经训完的折号（**有打分产物**的折）。

    ★★ 判据必须扫**全部 (头 × 标签)**，不能只看"第一个头 × 第一个标签"。
      单元可以让某个头只训一部分标签（`heads()` 里的 `labels`），
      于是"第一个标签"可能根本没有头去训 —— 那样四折明明都训完了，
      却会报"还没有任何折的打分产物"。

      **实测踩到（2026-09-18，V10b）**：`LABELS = None`（单元层面全 5 个标签）
      + `nn_cons` 只训 `[5d, 20d]` ⇒ 判据去看 `nn_cons/label_ret_1d`，永远不存在 ⇒ 四折全被当成没训。
      ★ V6/V9 之所以没暴露，是因为它们的单元级 `LABELS` 恰好把第一个标签放进了每个头的列表里 ——
      **靠的是巧合**，不是设计。这类"靠巧合通过"的判据迟早会在新单元上翻车。
    """
    heads = [m.name for m in unit.heads(cfg, unit.seeds[0])]
    labels = unit.my_labels(cfg)
    out = []
    for f in range(1, len(unit.seeds) + 1):
        found = any(
            state.unit_fold_score_path(cfg, unit.name, f, h, lb, smoke, sample).exists()
            for h in heads for lb in labels)
        if found:
            out.append(f)
    return out


def head_names(cfg: Cfg, unit: Unit) -> list[str]:
    """本版的头名（用第一个种子构造一次即可，只为读名字）。"""
    return [m.name for m in unit.heads(cfg, unit.seeds[0])]


def eval_split(cfg: Cfg, unit: Unit, panel, label: str, smoke: bool = False):
    """**评价窗口**：研究单元用固定的 test 窗（样本外）；实战单元（无 test）诚实退到 valid。

    返回 `(split, window)`，`window` ∈ {"test", "valid"} —— 榜单里必须把这件事写明白：
    把 valid 的数字标成 test 是自欺，把无 test 当成"没有样本外可比"才是实话。
    """
    sp_cfg = unit.split_cfg(cfg)
    min_days = 5 if smoke else dataset.MIN_DAYS_PER_BLOCK
    sp = dataset.make_splits(panel, label, cfg, log=lambda *_: None,
                             min_days=min_days, split=sp_cfg)[0]
    return sp, ("test" if sp.has_test else "valid")


def _window_rows(split: dataset.Split, window: str) -> np.ndarray:
    rows = split.test if window == "test" else split.valid
    return np.asarray([] if rows is None else rows)


def assemble(cfg: Cfg, unit: Unit, smoke: bool = False, sample: bool = False, log=print) -> dict[str, dict[str, pd.DataFrame]]:
    """{标签: {头: 跨折集成的打分}}。"""
    folds = fold_ids(cfg, unit, smoke, sample)
    if not folds:
        raise SystemExit(f"✘ 单元 {unit.name} 还没有任何折的打分产物 —— 先跑 train.sh")
    out: dict[str, dict[str, pd.DataFrame]] = {}
    for label in unit.my_labels(cfg):
        per_head: dict[str, pd.DataFrame] = {}
        for head in head_names(cfg, unit):
            frames = {}
            for f in folds:
                p = state.unit_fold_score_path(cfg, unit.name, f, head, label, smoke, sample)
                if p.exists():
                    frames[f] = store.read_frame(p)
            if not frames:
                continue
            per_head[head] = (next(iter(frames.values())) if len(frames) == 1
                              else combine.zscore_across_folds(frames))
        if per_head:
            try:      # ★ 本版的汇总口径（跨头）也算"一份打分"：评价、落盘、回测都用它
                per_head[unit.name] = unit.combine(per_head)
            except Exception as exc:           # noqa: BLE001
                log(f"  ⚠️ {unit.name} 的 combine() 失败：{exc}")
        out[label] = per_head
    log(f"  装配完成：{len(folds)} 折 × {len(head_names(cfg, unit))} 头 × "
        f"{len(out)} 标签（折号 {folds}）")
    return out


def _score_on_grid(panel, df: pd.DataFrame) -> np.ndarray:
    """把 4 列打分对齐到面板网格（返回 (T*C,) 的数组）。"""
    grid = panel.sub(np.arange(panel.n_rows))
    m = grid.merge(df[["trade_date", "stock_code", "value"]],
                   on=["trade_date", "stock_code"], how="left")
    return m["value"].to_numpy(dtype=np.float64)


def _seed_spread(cfg: Cfg, unit: Unit, head: str, label: str, smoke: bool,
                 window: str = "test", sample: bool = False) -> dict:
    """跨折（换种子）的 RankIC 离散度 —— 单种子纪录不作数的量化依据。"""
    vals = []
    for f in range(1, len(unit.seeds) + 1):
        s = state.load_json(state.unit_eval_path(cfg, unit.name, f, head, label, smoke, sample), {}) or {}
        v = ((s.get(window) or {}).get("rankic_mean"))
        if v is None:                     # 无 test 的单元：折里只有 valid
            v = ((s.get("test") or s.get("valid") or {}).get("rankic_mean"))
        if v is not None:
            vals.append(float(v))
    if not vals:
        return {}
    a = np.asarray(vals)
    return {"n_seeds": len(vals), "min": float(a.min()), "max": float(a.max()),
            "mean": float(a.mean()), "std": float(a.std(ddof=0)), "values": [round(x, 4) for x in vals]}


def _own_window_metrics(cfg: Cfg, unit: Unit, head: str, label: str,
                        folds: list[int], smoke: bool, sample: bool = False) -> dict:
    """各折在**自己那个验证窗**上的指标均值（诚实口径）。

    为什么不用一个统一的 valid 窗：本单元的折是**验证窗轮换**的，"统一的 valid 窗"
    对某些折其实落在它的训练集里（读出来的数字会虚高）。所以 valid 列 = 各折自窗均值，
    test 列 = 固定窗（全折共用、从未参与训练/选择），判定看 test。
    """
    ms = []
    for f in folds:
        d = state.load_json(state.unit_eval_path(cfg, unit.name, f, head, label, smoke, sample), {}) or {}
        v = d.get("valid")
        if v:
            ms.append(v)
    if not ms:
        return {}
    out = {}
    for k in set().union(*[m.keys() for m in ms]):
        vals = [m[k] for m in ms if isinstance(m.get(k), (int, float))]
        if vals:
            out[k] = float(np.mean(vals))
    out["n_folds"] = len(ms)
    return out


def entry_mask_next_day(enterable, panel, rr) -> np.ndarray:
    """把「**T 日**能否开盘买入」的掩码**顺延一天** ⇒ 「**T+1 日**能否开盘买入」。

    ★★ 2026-09-18 抓到的 off-by-one（主判据口径，见 README §8.5）：主判据的成交发生在
      **T+1 开盘**（买 top-n、T+2 开盘卖并换仓），所以"买不买得进"必须按 **T+1** 判。
      原先把 `entry_ok[T]` 传给了 `costfree_cumsum` —— 那是"昨天能不能买"，与本次成交无关。
      实测代价（V3 折1 `gbdt_w/label_ret_5d`，253 天）：

        | 掩码 | Σtop1 | 复利 |
        |:--|--:|--:|
        | 不套 | +3.777 | +2285% |
        | **T 日（错位）** | **+0.637** | +42.3% |
        | **T+1 日（正确）** | **−1.576** | **−87.0%** |

      ⇒ 错位让主判据**系统性偏乐观**（模型专挑"T 日收盘涨停 ⇒ T+1 一字买不进"的票，
        按 T 日的掩码恰好过滤不掉它们）。修完之后主判据与现金级策略（−93%）
        终于对上了 —— 这是它对齐正确的旁证。

    ★ 末一天没有 T+1，一律判 False（宁可少算一天，也不能拿别的日子顶上）。
    """
    d = panel.row_day[rr]
    c = panel.row_code[rr]
    nd = d + 1
    out = np.zeros(rr.size, dtype=bool)
    ok = nd < enterable.shape[0]
    out[ok] = enterable[nd[ok], c[ok]]
    return out


def entry_mask(cfg: Cfg, panel, smoke: bool = False, log=print):
    """能买掩码 `(T, C) bool` —— **次日开盘买得进**（非停牌、非一字涨停）。

    ★ 为什么榜单必须要它：无摩擦口径如果不管"买不买得进"，就会奖励"挑明天一字板"的模型。
      2026-09-18 实测：`gbdt_w/5d` 有 37.9% 的交易日选中不可买的票，Σtop1 从 +3.638 掉到 +0.289。
      可交易性属于模块③（硬约束 #4），所以这个掩码必须由模型侧自己算。

    价格层从模块② 现读（`mx/strategy.py:price_ctx`）—— 这意味着**回测结论依赖因子侧
    当时的价格快照**，因子侧改动期间引用它的结论要标注这一点。
    """
    try:
        from . import strategy as S
        return S.price_ctx(cfg, panel.dates, panel.codes)["entry_ok"]
    except Exception as exc:                     # noqa: BLE001
        log(f"  ⚠️ 拿不到价格层（{type(exc).__name__}: {str(exc)[:120]}）——"
            f" 无摩擦口径将**不套可买掩码**，那些数字不可用于结论")
        return None


def evaluate_all(cfg: Cfg, panel, scores: dict[str, dict[str, pd.DataFrame]],
                 unit: Unit, smoke: bool = False, sample: bool = False,
                 enterable=None, log=print) -> list[dict]:
    """逐 (标签 × 头) 出指标。

    窗口 = `eval_split`：研究单元是**固定的 test 窗**（全折共用、样本外）；
           实战单元（无 test）退到 valid 窗，并在 `eval_window` 里**标明**。
    valid 列 = 各折**自窗**指标均值（见 `_own_window_metrics`，折是验证窗轮换的）。
    """
    folds = fold_ids(cfg, unit, smoke, sample)
    rows: list[dict] = []
    for label, per_head in scores.items():
        try:
            split, window = eval_split(cfg, unit, panel, label, smoke)
        except SystemExit as exc:
            log(f"  ⊘ {label} 无法切分，跳过评价：{exc}")
            continue
        rr = _window_rows(split, window)
        if rr.size == 0:
            log(f"  ⊘ {label} 的评价窗口为空，跳过")
            continue
        for head, df in per_head.items():
            sc = _score_on_grid(panel, df)
            rec = {"label": label, "head": head, "n_scored": int(np.isfinite(sc).sum()),
                   "eval_window": window,
                   "eval_window_days": split.days.get(window)}
            rec["eval"] = evaluate.evaluate(
                panel.dates_arr[panel.row_day[rr]], panel.codes[panel.row_code[rr]],
                sc[rr], panel.y[label][rr], cfg)
            # ★ 无摩擦口径（用户 2026-09-18 拍板的主判据）：买 top-n 隔日换手的收益累加。
            #   用**1 日标签**算，因为它就是"T+1 开盘买、T+2 开盘卖"这一个动作的收益，
            #   与当前评价的标签期限无关 —— 5 日标签的头也按 1 日持有来比，才可横比。
            if "label_ret_1d" in panel.y:
                # ★ 掩码必须对齐到**成交日 T+1**，不是信号日 T（见 `entry_mask_next_day`）
                ent = (None if enterable is None
                       else entry_mask_next_day(enterable, panel, rr))
                rec["costfree"] = evaluate.costfree_cumsum(
                    panel.dates_arr[panel.row_day[rr]], sc[rr], panel.y["label_ret_1d"][rr],
                    enterable=ent)
            if window == "test":
                rec["test"] = rec["eval"]
            rec["valid"] = (_own_window_metrics(cfg, unit, head, label, folds, smoke, sample)
                            if head != unit.name else {})
            rec["seed_spread"] = (_seed_spread(cfg, unit, head, label, smoke, window, sample)
                                  if head != unit.name else {})
            rows.append(rec)
    return rows


def leaderboard(rows: list[dict]) -> list[dict]:
    """按评价窗口的 RankIC 排序（跨标签同表，便于看"哪条腿最强"）。"""
    return sorted(rows, key=lambda r: -(r.get("eval", {}).get("rankic_mean") or -9e9))


def _eval_window_dates(cfg: Cfg, unit: Unit, panel, label: str, smoke: bool):
    """评价窗口的日期边界（回测/策略矩阵都用它，保证与榜单同一个窗口）。"""
    try:
        split, window = eval_split(cfg, unit, panel, label, smoke)
    except SystemExit:
        return None, None, "none"
    rr = _window_rows(split, window)
    if rr.size == 0:
        return None, None, "none"
    d = panel.dates_arr[panel.row_day[rr]]
    return str(np.min(d)), str(np.max(d)), window


def run(cfg: Cfg, unit: Unit, smoke: bool = False, sample: bool = False,
        top_n: int | None = None, do_backtest: bool = True, label: str | None = None,
        log=print) -> dict:
    """分析主入口：装配 → 评价 → 榜单 → 落盘单元打分 → 回测。

    `label`：**主腿**（策略矩阵 + 对照回测 + 落盘打分都用它）。默认取单元声明的
    `PRIMARY_LABEL`。★ 需要能改的理由：本轮实测**只有 20d 期限的可执行 Σtop1 为正**，
    而多数单元声明的是 5d —— 不指定主腿就永远只能看到那条不赚钱的腿的净值。
    """
    if getattr(unit, "archived", False):
        raise SystemExit(f"✘ {unit.name} 是**已冻结的历史单元**（ARCHIVED）—— 不要重跑分析："
                         f"它的产物是旧口径的证据，覆盖掉就再也回不来了。")
    t0 = time.time()
    state.unit_ensure_dirs(cfg, unit.name, smoke, sample)
    log(f"══ 分析单元 {unit.name}（{unit.title}）{'· smoke' if smoke else ''}"
        f"{' · 小样本' if sample else ''}")
    if unit.drift and unit.drift.get("engine"):
        log(f"  ⚠️ 引擎哈希与配方冻结值不同 —— 下面的数字与 README 记录不可直接比较")

    scores = assemble(cfg, unit, smoke, sample=sample, log=log)
    # ★ 必须用**单元的数据窗**加载 —— 与训练同一把尺子，否则行下标会对不上
    panel = load_for_unit(cfg, unit, sample=sample, log=log)
    # ★ 可买掩码只算一次，榜单与策略矩阵共用（同一份价格层、同一套涨跌停口径）
    enterable = None if sample else entry_mask(cfg, panel, smoke, log=log)
    rows = evaluate_all(cfg, panel, scores, unit, smoke, sample=sample,
                        enterable=enterable, log=log)

    # ---- 单元打分落盘（每个标签一份，4 列契约；best/ 晋级时直接取用）
    pred_dir = state.unit_pred_dir(cfg, unit, smoke, sample)
    written = {}
    for label, per_head in scores.items():
        df = per_head.get(unit.name)
        if df is None:
            continue
        p = store.write_frame(pred_dir / f"score__{unit.name}__{label}", df)
        written[label] = str(p)
    log(f"  单元打分落盘：{len(written)} 个标签 → {pred_dir.relative_to(cfg.unit_dir(unit.name))}/")

    # ---- 回测（主标签 + 评价窗最强的一个头，做对照）
    # ★ 回测**只跑评价窗口**（研究单元 = test 窗；实战单元 = 最后一个验证窗）——
    #   以前它跑的是全样本，等于把训练期也拿去算净值，那是样本内自欺。
    bt: dict = {}
    if do_backtest:
        primary = label or unit.primary_label or (rows[0]["label"] if rows else "")
        lo, hi, win = _eval_window_dates(cfg, unit, panel, primary, smoke) if primary else (None, None, "none")
        log(f"  回测窗口：{lo} ~ {hi}（{win}）")
        if primary in scores and unit.name in scores[primary]:
            log(f"  回测 {unit.name}/{primary}（含费用+可交易性）…")
            bt["unit"] = backtest.run(cfg, unit.name, primary, log=log, top_n=top_n,
                                      frame=scores[primary][unit.name], panel=panel,
                                      dates_lo=lo, dates_hi=hi)
            bt["unit"]["head"] = unit.name
            bt["unit"]["window"] = win
        if rows:
            best = max((r for r in rows if r["label"] == primary and r["head"] != unit.name),
                       key=lambda r: r.get("eval", {}).get("rankic_mean") or -9e9, default=None)
            if best:
                log(f"  对照回测 {best['head']}/{primary}（评价窗最强的单头）…")
                bt["best_head"] = backtest.run(cfg, best["head"], primary, log=log, top_n=top_n,
                                               frame=scores[primary][best["head"]], panel=panel,
                                               dates_lo=lo, dates_hi=hi)
                bt["best_head"]["head"] = best["head"]
                bt["best_head"]["window"] = win

    # ---- 策略矩阵（策略 × 打分源 × 折）—— 小资金口径，策略与模型同步迭代的落点
    matrix: dict = {}
    if do_backtest:
        matrix = strategy_matrix(cfg, panel, unit, scores, label=label,
                                 smoke=smoke, sample=sample, log=log)

    summary = {"unit": unit.name, "title": unit.title, "smoke": bool(smoke),
               "sample": bool(sample),
               "finished_at": state.now(), "seconds": round(time.time() - t0, 1),
               "folds": fold_ids(cfg, unit, smoke),
               "engine_hash": unit.drift.get("engine_hash") if unit.drift else None,
               "recipe": unit.recipe, "n_rows": len(rows),
               "split": unit.split_cfg(cfg),
               "eval_window": (rows[0].get("eval_window") if rows else None),
               "eval_window_days": (rows[0].get("eval_window_days") if rows else None),
               "leaderboard": rows, "pred": written, "strategy_matrix": matrix,
               "backtest": bt,
               "panel": {"n_days": len(panel.dates), "n_codes": int(len(panel.codes)),
                         "panel_digest": panel.meta.get("panel_digest"),
                         "digest": panel.meta.get("digest")}}
    # ---- 出图（用户 2026-09-18 交办）：收益累计曲线 / 折间带 / Σtop1 条形图
    # ★ 只画**评价窗口内**的曲线 —— 把训练期也画进去是自欺（README §7）。
    # ★ 整段包 try：出图失败不能连累榜单落盘（榜单是结论的载体，图只是附属）。
    if do_backtest:
        try:
            from . import plots
            summary["pics"] = plots.make_all(
                summary, state.unit_pics_dir(cfg, unit.name, smoke, sample), log=log)
        except Exception as exc:                    # noqa: BLE001
            log(f"  ⚠️ 出图失败（{type(exc).__name__}: {str(exc)[:120]}）—— 榜单不受影响")
    state.save_json(state.unit_leaderboard_path(cfg, unit, smoke, sample), summary)
    log(f"  ✔ 分析完成：用时 {summary['seconds']}s → "
        f"{state.unit_leaderboard_path(cfg, unit, smoke, sample).relative_to(cfg.unit_dir(unit.name))}")
    return summary


def render(summary: dict, top: int = 20) -> str:
    """控制台榜单：标签 × 头 × (valid/评价窗) RankIC / ICIR / 分层 / 换手 / 种子离散度。"""
    rows = leaderboard(summary.get("leaderboard") or [])
    pad, fmt = evaluate.pad, evaluate.fmt
    win = (rows[0].get("eval_window") if rows else None) or "test"
    win_txt = "test(样本外)" if win == "test" else "valid(★ 实战口径：无 test，退到验证窗)"
    head = (pad("标签", 15) + pad("头", 9) + pad("valid IC*", 10, True)
            + pad("评价 IC", 10, True) + pad("ICIR", 8, True) + pad("t", 7, True)
            + pad("D9-D0", 10, True) + pad("换手", 8, True)
            + pad("Σtop1", 9, True) + pad("剔1天", 9, True) + pad("剔2天", 9, True)
            + pad("Σtop5", 9, True) + pad("超额1", 9, True)
            + pad("种子离散", 18, True))
    W = 139
    out = ["=" * W,
           f"  单元 {summary['unit']} · {summary.get('title','')} · "
           f"折 {summary.get('folds')} · {summary.get('finished_at')}",
           f"  评价窗口：{win_txt} · {((rows[0].get('eval_window_days') if rows else None) or ['?','?','?'])[0]}"
           f" ~ {((rows[0].get('eval_window_days') if rows else None) or ['?','?','?'])[1]}",
           "-" * W, "  " + head, "-" * W]
    for r in rows[:top]:
        v, t = r.get("valid", {}), r.get("eval", {})
        cf = r.get("costfree") or {}
        sp = r.get("seed_spread") or {}
        sp_s = (f"{sp['min']:.4f}~{sp['max']:.4f}" if sp else "—")
        out.append("  " + pad(r["label"], 15) + pad(r["head"], 9)
                   + pad(fmt(v.get("rankic_mean")), 10, True)
                   + pad(fmt(t.get("rankic_mean")), 10, True)
                   + pad(fmt(t.get("rankic_icir"), 2), 8, True)
                   + pad(fmt(t.get("rankic_t"), 2), 7, True)
                   + pad(fmt(t.get("spread"), 4), 10, True)
                   + pad(fmt(t.get("turnover"), 2), 8, True)
                   + pad(fmt(cf.get("top1_sum"), 3), 9, True)
                   + pad(fmt(cf.get("top1_drop1"), 3), 9, True)
                   + pad(fmt(cf.get("top1_drop2"), 3), 9, True)
                   + pad(fmt(cf.get("top5_sum"), 3), 9, True)
                   + pad(fmt(cf.get("excess_top1"), 3), 9, True)
                   + pad(sp_s, 18, True))
    cf0 = (rows[0].get("costfree") or {}) if rows else {}
    out.append("-" * W)
    out.append("  （valid IC* = **各折自窗**均值（诚实口径，折是验证窗轮换的）；"
               "评价 IC 列 = 上表那个窗口上的集成打分；种子离散 = 各折同窗口 RankIC 的 min~max）")
    out.append(f"  ★ 主判据（可执行的无摩擦口径）：Σtop1/Σtop5 = 买 top-1/top-5 隔日换手的收益**累加**；"
               f"超额1 = Σtop1 − 当日等权全买（零信号对照）")
    out.append("  ★ 剔1天/剔2天 = 从 Σtop1 里**去掉贡献最大的 1/2 个交易日**之后还剩多少 —— "
               "**防「随机种子中彩票」的照妖镜**：要是一天就撑起整个 Σ，那换个种子大概率就没了")
    if cf0:
        f = "已套" if cf0.get("exec_filter") else "★未套（数字不可用于结论）"
        out.append(f"     可买掩码：{f}（次日一字涨停/停牌的不进候选）· 共 {cf0.get('days')} 个交易日")
        out.append(f"     基准：等权全买 Σ = {fmt(cf0.get('uni_sum'), 3)}"
                   f" —— 超不过它就是在吃 beta，不是 alpha")
        if cf0.get("top1_max_share") is not None:
            out.append(f"     Σtop1 的单日最大贡献占比 {fmt(cf0.get('top1_max_share'), 2)}"
                       f" · 正贡献天数占比 {fmt(cf0.get('top1_pos_frac'), 2)}"
                       f" · 前半段 {fmt(cf0.get('top1_h1'), 3)} / 后半段 {fmt(cf0.get('top1_h2'), 3)}")
    # ---- ★ 按**主判据**再排一次：榜单默认按 RankIC 排，但实盘看的是 Σtop1。
    #   两榜常常不一致（README §10：「IC 升但 Top1 崩」是对面踩过的坑），所以两个都要摆在眼前。
    ranked = [r for r in rows if (r.get("costfree") or {}).get("top1_sum") is not None]
    if ranked:
        ranked.sort(key=lambda r: -(r["costfree"]["top1_sum"]))
        out.append("")
        out.append("  ★ 按主判据 Σtop1 重排（同一批行，换个排序看「哪条腿真赚钱」）")
        out.append("  " + pad("标签", 15) + pad("头", 9) + pad("Σtop1", 9, True)
                   + pad("剔1天", 9, True) + pad("剔2天", 9, True)
                   + pad("正比例", 8, True) + pad("单日占比", 10, True)
                   + pad("评价 IC", 10, True) + pad("换手", 8, True))
        for r in ranked[:10]:
            cf, t = r["costfree"], r.get("eval", {})
            out.append("  " + pad(r["label"], 15) + pad(r["head"], 9)
                       + pad(fmt(cf.get("top1_sum"), 3), 9, True)
                       + pad(fmt(cf.get("top1_drop1"), 3), 9, True)
                       + pad(fmt(cf.get("top1_drop2"), 3), 9, True)
                       + pad(fmt(cf.get("top1_pos_frac"), 2), 8, True)
                       + pad(fmt(cf.get("top1_max_share"), 2), 10, True)
                       + pad(fmt(t.get("rankic_mean")), 10, True)
                       + pad(fmt(t.get("turnover"), 2), 8, True))
    out.append("=" * W)
    return "\n".join(out)


# ================================================================ 策略矩阵（策略 × 打分源 × 折）
def strategy_specs(unit: Unit) -> list[dict]:
    """单元声明的策略清单（`Model/V{N}/model.py: STRATEGIES`）。"""
    specs = getattr(unit.mod, "STRATEGIES", None) or [{"name": "top1_daily", "params": {}}]
    return [dict(s) for s in specs]


def _fold_head_scores(cfg: Cfg, unit: Unit, fold: int, label: str, smoke: bool,
                      sample: bool = False) -> dict:
    """某一折的各头打分（外加"该折内跨头汇总"），用于**逐折**跑策略。"""
    frames: dict[str, pd.DataFrame] = {}
    for h in head_names(cfg, unit):
        p = state.unit_fold_score_path(cfg, unit.name, fold, h, label, smoke, sample)
        if p.exists():
            frames[h] = store.read_frame(p)
    if frames:
        try:
            frames[unit.name] = unit.combine(dict(frames))
        except Exception:                       # noqa: BLE001
            pass
    return frames


def _score_grid(panel, df: pd.DataFrame) -> np.ndarray:
    """4 列打分 → (T, C) 网格（NaN = 不可选）。"""
    sc = _score_on_grid(panel, df)
    return sc.reshape(len(panel.dates), len(panel.codes))


# ---- 矩阵并行用的**进程内共享状态**：fork 出来的子进程直接继承，避免把大数组来回 pickle。
#      （打分网格 (255×3484) float64 ≈ 7 MB/份，几十份来回序列化就是 GB 级的 IPC 开销）
_JOBS: dict = {}


def _matrix_worker(task: tuple) -> dict:
    """策略矩阵的**一个格子**（策略 × 打分源）。

    ★ 必须是**模块级函数** —— 闭包/局部函数不可 pickle，进程池起不来。
    ★ 大数组（价格层 ctx_px / 打分网格 / 股票池）一律从 `_JOBS` 取，不随任务传。
    """
    from . import strategy as S
    cfg, spec, src, scope, fold = task
    r = S.simulate(cfg, _JOBS["ctx_px"], _JOBS["scores"][src],
                   strategy=spec["name"], params=spec.get("params") or {},
                   universe_2d=_JOBS["uni"],
                   dates_lo=_JOBS["lo"], dates_hi=_JOBS["hi"])
    r.update({"source": src, "scope": scope, "fold": fold})
    return r


def _matrix_jobs(n_task: int) -> int:
    """矩阵并行度。可用 `MX_JOBS` 覆盖；默认取容器可用核数的 1/3（与训练错开）。"""
    import os
    from . import unit as U
    want = os.environ.get("MX_JOBS")
    if want:
        return max(1, min(n_task, int(want)))
    return max(1, min(n_task, U.avail_cores() // 3))


def strategy_matrix(cfg: Cfg, panel, unit: Unit, scores: dict, label: str | None = None,
                    smoke: bool = False, sample: bool = False, log=print) -> dict:
    """策略 × 打分源 矩阵：**每个折（种子）各跑一遍**，报集成口径与折间离散。

    小资金策略的随机干扰很重（用户 2026-09-17 提醒），所以矩阵里必须同时给出
    "全折集成打分下的结果"和"单折（换种子）结果的区间" —— 只看单个数会自欺。

    ★ 每个格子互相独立（纯函数），所以**并行跑**：串行时这一块是分析阶段的全部瓶颈
      （11 策略 × 25 打分源 × 255 天 ≈ 7 万次日循环，单核要十几分钟），
      并行后与核数同阶。见 `_matrix_worker` / `_matrix_jobs`。
    """
    from . import strategy as S
    label = label or unit.primary_label or next(iter(scores))
    if label not in scores:
        return {}
    folds = fold_ids(cfg, unit, smoke, sample)
    ctx_px = S.price_ctx(cfg, panel.dates, panel.codes)
    uni = panel.universe.reshape(len(panel.dates), len(panel.codes))
    specs = strategy_specs(unit)
    ens_head_scores = {h: df for h, df in scores[label].items()}
    # ★ 只评估**评价窗口**（研究单元 = test 窗，模型从未见过；实战单元 = 最后一个验证窗）
    #   —— 样本内跑策略的净值是自欺。无窗口就明确报"不跑"，不再悄悄退成全样本。
    d_lo, d_hi, win = _eval_window_dates(cfg, unit, panel, label, smoke)
    log(f"  策略矩阵窗口：{d_lo} ~ {d_hi}（{win}）"
        + ("　★ 实战口径无 test，退到最后一段验证窗" if win == "valid" else ""))
    if not d_lo:
        log("  ⊘ 没有可用的评价窗口 —— 策略矩阵跳过（不拿全样本冒充样本外）")
        return {}

    # ---- 打分源清单（集成口径 + 逐折口径）→ 全部铺成 (T,C) 网格
    meta: dict[str, tuple[str, int | None]] = {}
    raw: dict[str, pd.DataFrame] = {}
    for h, df in ens_head_scores.items():
        meta[f"ens:{h}"] = ("ens", None)
        raw[f"ens:{h}"] = df
    for k in folds:
        for h, df in _fold_head_scores(cfg, unit, k, label, smoke, sample).items():
            meta[f"f{k}:{h}"] = ("fold", k)
            raw[f"f{k}:{h}"] = df
    scores_grid = {k: _score_grid(panel, df) for k, df in raw.items()}
    del raw

    tasks = [(cfg, spec, src, meta[src][0], meta[src][1])
             for spec in specs for src in scores_grid]
    jobs = _matrix_jobs(len(tasks))
    log(f"  策略矩阵：{len(specs)} 策略 × {len(ens_head_scores)} 打分源(集成) "
        f"+ {len(folds)} 折 × {len(ens_head_scores)} 打分源(逐折) "
        f"= {len(tasks)} 格 · **并行 {jobs} 进程** …")

    cells: list[dict] = []
    _JOBS.update({"ctx_px": ctx_px, "scores": scores_grid, "uni": uni,
                  "lo": d_lo, "hi": d_hi})
    try:
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor
        ctx = mp.get_context("fork")        # fork：子进程继承 _JOBS，零拷贝
        done = 0
        with ProcessPoolExecutor(max_workers=jobs, mp_context=ctx) as ex:
            for r in ex.map(_matrix_worker, tasks, chunksize=max(1, len(tasks) // (jobs * 8))):
                cells.append(r)
                done += 1
                if done % max(1, len(tasks) // 4) == 0:
                    log(f"    …已完成 {done}/{len(tasks)} 格")
    except Exception as exc:                # noqa: BLE001 —— 并行起不来就退回串行，不能让分析挂掉
        log(f"  ⚠️ 并行矩阵失败（{type(exc).__name__}: {str(exc)[:120]}）→ 退回串行")
        cells = [_matrix_worker(t) for t in tasks]
    finally:
        _JOBS.clear()

    out = {"label": label, "folds": folds, "window": [d_lo, d_hi], "window_kind": win,
           "headline": f"ens:{unit.name}", "cells": cells,
           "n_cells": len(cells)}
    log(f"  策略矩阵完成：{len(cells)} 个 (策略 × 打分源 × 折) 单元")
    return out


def strategy_table(matrix: dict, top: int = 14) -> str:
    """控制台：行 = 策略，列 = 打分源（集成口径），格 = 累计收益 / 折间区间。"""
    if not matrix or not matrix.get("cells"):
        return "  （没有策略结果）"
    pad = evaluate.pad
    cells = matrix["cells"]
    sources = sorted({c["source"] for c in cells if c["scope"] == "ens"},
                     key=lambda s: (s != matrix.get("headline"), s))
    rows: dict[str, dict[str, dict]] = {}
    spread: dict[tuple, list] = {}
    for c in cells:
        key = (c["strategy"], tuple(sorted((c.get("params") or {}).items())))
        rows.setdefault(key, {})[c["source"]] = c
        if c["scope"] == "fold" and c.get("total_ret") is not None:
            spread.setdefault((key, c["source"].split(":", 1)[1]), []).append(c["total_ret"])
    out = ["=" * 118,
           f"  策略矩阵（标签 {matrix['label']} · 窗口 {matrix.get('window',[None,None])[0]}"
           f"~{matrix.get('window',[None,None])[1]} · {matrix.get('window_kind','test')}）"
           f"· 现金级真实净值 · 最多 5 只 · T+1 开盘执行 · 含费用",
           "-" * 118,
           "  " + pad("策略", 30) + "".join(pad(s, 16, True) for s in sources)
           + pad("折间区间(主源)", 22, True)]
    for key, bysrc in rows.items():
        name, ps = key[0], dict(key[1])
        label_txt = name + (f" {ps}" if ps else "")
        line = "  " + pad(label_txt[:29], 30)
        for s in sources:
            v = (bysrc.get(s) or {}).get("total_ret")
            line += pad("—" if v is None else f"{v*100:+.2f}%", 16, True)
        sp = spread.get((key, matrix["headline"].split(":", 1)[1]))
        line += pad("—" if not sp else f"{min(sp)*100:+.1f}%~{max(sp)*100:+.1f}%", 22, True)
        out.append(line)
    out += ["-" * 118,
            "  （列 = 打分源：`ens:X` = 全折集成打分；折间区间 = 逐折(换种子)跑同一策略的累计收益 min~max）",
            "=" * 118]
    return "\n".join(out)
