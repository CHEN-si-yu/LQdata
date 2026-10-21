# -*- coding: utf-8 -*-
"""三个实战单元（V31 / V36 / V38）的**集成**：打分等权平均，接口与单元内的 `--live` 一致。

为什么要有这一层：
  三个单元的打分**逐日秩相关 0.92~0.97** —— 它们几乎是同一个模型，差别集中在每天
  top5 里那 2~3 只不同的票上。单看某一个单元的 62 天账户收益（−18.9% / −0.3% / −8.0%），
  差 18pp，但那全是"抓到哪几只"的运气。**等权平均是几乎免费的降方差**，也避免了
  "按最近一段收益挑模型"这个陷阱。

口径（与单元内 `analysis.py --live` 逐字一致）：
  · 每个单元的打分 = 该单元伪季度 `2026Q3` 的**四折集成**（四个子模型打分平均）；
  · 本脚本的集成 = 三个单元的打分再平均 ⇒ 等价于 **12 个子模型等权**；
  · 评估窗 = `splits(quarter='2026Q3')['test']` = 2026-07-01 → 数据末日（62 天），
    对 12 个子模型的训练集都不含这段 ⇒ 干净样本外；
  · 账户口径 = `analysis.py:cash_backtest`（10 万元、含佣金/印花/过户/3bp 滑点、整手、T+1、拒单）。

用法：python -B ensemble.py            # 打印并写 releases/ensemble.json
"""
import importlib
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

RELEASES = Path(__file__).resolve().parent
UNITS = ('V31', 'V36', 'V38')
QUARTER = '2026Q3'
MONEY = 100000.
REF_UNIT = 'V38'          # 借它的 model.py / analysis.py 当实现（三个单元同源）


def load_impl():
    root = RELEASES / REF_UNIT
    sys.path.insert(0, str(root))
    try:
        m = importlib.import_module('model')
        a = importlib.import_module('analysis')
    finally:
        sys.path = [p for p in sys.path if p != str(root)]
    return m, a


def unit_scores(unit):
    """单元自己的四折集成打分（与 `analysis.py --live` 逐字同一算法）。"""
    acc = None
    for k in (1, 2, 3, 4):
        fp = RELEASES / unit / 'model_train' / QUARTER / f'fold{k}' / 'score_predictions.npy'
        sc = np.load(fp).astype(np.float64)
        acc = sc if acc is None else acc + sc
    return acc / 4


def single_fold_scores():
    """12 个子模型的原始打分，用来算"每个子模型对集成的重要性"之外的诊断。"""
    out = {}
    for u in UNITS:
        for k in (1, 2, 3, 4):
            fp = RELEASES / u / 'model_train' / QUARTER / f'fold{k}' / 'score_predictions.npy'
            out[f'{u}/fold{k}'] = np.load(fp).astype(np.float64)
    return out


def ric(pred, panel, ev, label):
    o = []
    for i, d in enumerate(ev):
        p, y = pred[i], panel.Y[label][d]
        ok = np.isfinite(p) & np.isfinite(y)
        if ok.sum() >= 3:
            o.append(float(np.corrcoef(rankdata(p[ok]), rankdata(y[ok]))[0, 1]))
    return np.asarray(o)


def topk_daily(pred, panel, ev, k, label='label_ret_1d'):
    o = []
    for i, d in enumerate(ev):
        p, y = pred[i], panel.Y[label][d]
        ok = np.isfinite(p) & np.isfinite(y)
        if ok.sum() < k:
            o.append(np.nan)
            continue
        ix = np.flatnonzero(ok)[np.argsort(-p[ok], kind='stable')][:k]
        o.append(float(np.mean(y[ix])) * 100)
    return np.asarray(o)


# =============================================================================
# REPORT.md 的顶栏：**最新一日推演**（由本脚本自动刷新，勿手改）
#
#   每天数据更新之后只要跑一次 `python -B ensemble.py`，REPORT.md 最上面那段就跟着更新；
#   两个标记之间的内容会被整体替换，标记之外的部分（分析正文）不动。
# =============================================================================

DAILY_BEGIN = '<!-- DAILY:BEGIN 由 ensemble.py 生成，勿手改；更新方式见下方"每日更新" -->'
DAILY_END = '<!-- DAILY:END -->'

#: 历史逐日 Top-10 的起始日（用户 2026-09-27 指定）
HISTORY_START = '2026-09-01'
HIST_BEGIN = '<!-- HISTORY:BEGIN 由 ensemble.py 生成，勿手改 -->'
HIST_END = '<!-- HISTORY:END -->'


def render_daily(out):
    """把最新一日推演渲染成 REPORT.md 顶栏（展示 top10：前 5 名建仓，后 5 名顺位替补）。"""
    lt = out['latest']
    picks = lt['picks'][:10]
    rows = '\n'.join(
        f"| {p['rank']} | {p['code']} | {p['name']} | {p['score']:.4f}"
        f"{' | **建仓** |' if p['rank'] <= out['topn'] else ' | 替补 |'}"
        for p in picks)
    return f"""## ★ 最新一日推演

**信号日 `{lt['signal_day']}`（收盘）→ `{lt['buy_day']}` 开盘买入 → `{lt['sell_day']}` 开盘卖出**

| 排名 | 代码 | 名称 | 集成打分 | 用途 |
| ---: | --- | --- | ---: | --- |
{rows}

**执行口径**：`{out['strategy']}` —— 持仓 **{out['topn']} 只**，按上表**前 {out['topn']} 名建仓**；
第 {out['topn'] + 1}~10 名是**顺位替补**：某只票买不进（一字板 / 涨停 / 停牌 / 无量）时就往下顺延一名。
每个交易日检查调仓，`band={out['band']}` 缓冲区让只有掉出前 {out['band']} 名的票才卖。

★ 顺位回补的代价项目实测过：**−0.0124 %/天**（很小），比"追涨停买不进的票"划算得多。
★ 打分 = **12 个子模型等权平均**（`{' + '.join(out['units'])}` 各四折）。
★ 这只是**模型输出**，不含可买性判断 —— 表格不区分能不能买进，能不能买进要靠下单时自己判断。
★ 账户口径 10 万元、含佣金/印花/过户/3bp 滑点、整手、T+1、拒单。

---

"""


def render_history(out, panel, ens, ev, days):
    """历史逐日 Top-10（`HISTORY_START` 起）：直接给原始打分。"""
    names_all = None
    parts, n_days = [], 0
    for i, d in enumerate(ev):
        day = str(days[d])
        if day < HISTORY_START:
            continue
        n_days += 1
        p = ens[i]
        ok = np.isfinite(p)
        order = np.flatnonzero(ok)[np.argsort(-p[ok], kind='stable')][:10]
        if names_all is None:
            names_all = None
        from analysis import stock_names as _sn
        nm = _sn([str(panel.codes[c]) for c in order])
        rows = '\n'.join(f"| {r} | {str(panel.codes[c])} | {nm.get(str(panel.codes[c]), '')} | {p[c]:.4f} |"
                          for r, c in enumerate(order, 1))
        parts.append(f"### {day}\n\n| 排名 | 代码 | 名称 | 集成打分 |\n| ---: | --- | --- | ---: |\n{rows}\n")
    head = (f"## ★ 历史每日 Top-10（{HISTORY_START} 起，共 {n_days} 个交易日）\n\n"
            "★ 打分 = 12 个子模型等权平均；每天取前 10 名，**前 5 名建仓、6~10 名顺位替补**。\n"
            "★ 这张表由 `ensemble.py` 自动生成（见「每日更新」），与顶栏同一份数据。\n\n")
    return head + '\n'.join(parts)


def sync_history(text, body):
    block = HIST_BEGIN + '\n\n' + body + '\n' + HIST_END
    if HIST_BEGIN in text and HIST_END in text:
        head, rest = text.split(HIST_BEGIN, 1)
        _, tail = rest.split(HIST_END, 1)
        return head + block + tail
    return text.rstrip() + '\n\n---\n\n' + block + '\n'


def sync_report(out):
    """刷新 `REPORT.md` 里两个标记之间的内容；标记缺失则在主标题后插入。"""
    path = RELEASES / 'REPORT.md'
    if not path.is_file():
        return
    text = path.read_text(encoding='utf-8')
    block = DAILY_BEGIN + '\n\n' + render_daily(out) + DAILY_END
    if DAILY_BEGIN in text and DAILY_END in text:
        head, rest = text.split(DAILY_BEGIN, 1)
        _, tail = rest.split(DAILY_END, 1)
        new = head + block + tail
    else:                                  # 首次：插到第一个 '---' 之后（主标题下面）
        marker = text.find('\n---\n')
        new = (text[:marker] + '\n\n' + block + text[marker:]) if marker > 0 else block + '\n' + text
    if new != text:
        path.write_text(new, encoding='utf-8')
        print(f'REPORT.md 顶栏已刷新（信号日 {out["latest"]["signal_day"]}）')


def sync_history_block(out, panel, ens, ev, days):
    """把历史逐日 Top-10 写进 REPORT.md 尾部（`HISTORY` 标记之间）。"""
    path = RELEASES / 'REPORT.md'
    if not path.is_file():
        return
    body = render_history(out, panel, ens, ev, days)
    text = path.read_text(encoding='utf-8')
    new = sync_history(text, body)
    if new != text:
        path.write_text(new, encoding='utf-8')
        print(f'REPORT.md 历史逐日 Top-10 已刷新（{HISTORY_START} 起）')


def main():
    M, A = load_impl()
    panel = M.Panel(M.RECIPE['features'], load_x=False)
    px = M.Prices(panel)
    days = panel.days
    ev = M.splits(days, quarter=QUARTER)[0]['test']
    ev_bt = ev[ev <= len(days) - 3]
    spec = A.STRATEGIES[0]

    singles = single_fold_scores()
    per_unit = {u: unit_scores(u) for u in UNITS}
    ens = sum(per_unit.values()) / len(UNITS)          # 等价于 12 个子模型等权

    def account(pred):
        st, curve, _ = A.cash_backtest(pred[:len(ev_bt)].astype(np.float32), panel, px, ev_bt,
                                       n=spec['topn'], period=1, band=spec['band'], money=MONEY)
        eq = np.asarray([c['equity'] for c in curve], float)
        prev = np.r_[MONEY, eq[:-1]]
        r = (eq - prev) / prev
        return dict(compounded_pct=float(eq[-1] / MONEY - 1) * 100,
                    net_annual_pct=float(r.mean()) * 242 * 100 - st['total_fees'] / MONEY * 100 * (242 / len(r)),
                    max_drawdown_pct=float(st['max_drawdown']) * 100, trades=int(st['trades']),
                    total_fees=float(st['total_fees']), blocked_entries=int(st.get('blocked_entries', 0)))

    def quality(pred):
        r5 = ric(pred, panel, ev, 'label_ret_5d')
        r5 = r5[np.isfinite(r5)]
        d1 = topk_daily(pred, panel, ev, 5)
        d1 = d1[np.isfinite(d1)]
        return dict(rank_ic_5d=float(r5.mean()), icir=float(r5.mean() / r5.std(ddof=1)),
                    t_rank_ic=float(r5.mean() / (r5.std(ddof=1) / np.sqrt(len(r5)))),
                    top5_daily_pct=float(d1.mean()),
                    t_top5=float(d1.mean() / (d1.std(ddof=1) / np.sqrt(len(d1)))),
                    win_rate_pct=float((d1 > 0).mean() * 100))

    bench = np.asarray([float(np.mean(panel.Y['label_ret_1d'][d][np.isfinite(panel.Y['label_ret_1d'][d])])) * 100
                        for d in ev])
    excess = {}
    for k in (5, 10, 20, 50, 100):
        e = topk_daily(ens, panel, ev, k)
        e = e[np.isfinite(e)] - bench[np.isfinite(e)]
        excess[f'top{k}'] = dict(mean_pct=float(e.mean()),
                                 t=float(e.mean() / (e.std(ddof=1) / np.sqrt(len(e)))))

    # 逐日打分秩相关：集成 vs 单模型，证明"几乎是同一个模型"
    from scipy.stats import spearmanr
    agree = {}
    for u in UNITS:
        rs = [float(spearmanr(ens[i], per_unit[u][i], nan_policy='omit').statistic) for i in range(len(ev))]
        agree[u] = float(np.mean(rs))

    # 最新一日排序
    axis = A.trade_axis(panel)
    last_day = str(days[ev[-1]])
    last = ens[-1]
    ok = np.isfinite(last)
    order = np.flatnonzero(ok)[np.argsort(-last[ok], kind='stable')][:20]
    names = A.stock_names([str(panel.codes[c]) for c in order])
    picks = [dict(rank=r, code=str(panel.codes[c]), name=names.get(str(panel.codes[c]), ''),
                  score=float(last[c])) for r, c in enumerate(order, 1)]
    try:
        p0 = axis.index(last_day)
        buy_day, sell_day = axis[p0 + 1], axis[p0 + 2]
    except (ValueError, IndexError):
        buy_day, sell_day = '次一交易日开盘', '再下一交易日开盘'

    out = dict(
        units=list(UNITS), quarter=QUARTER,
        definition='三个单元的四折集成打分等权平均 ⇒ 等价于 12 个子模型等权',
        eval_window=dict(start=str(days[ev[0]]), end=str(days[ev[-1]]), days=int(len(ev)),
                         backtest_days=int(len(ev_bt))),
        strategy=spec['name'], topn=spec['topn'], band=spec['band'], account_money=MONEY,
        ensemble=dict(quality=quality(ens), account=account(ens)),
        per_unit={u: dict(quality=quality(per_unit[u]), account=account(per_unit[u])) for u in UNITS},
        topk_excess_vs_cross_section=excess,
        agreement_with_ensemble=agree,
        latest=dict(signal_day=last_day, buy_day=buy_day, sell_day=sell_day, picks=picks),
    )
    (RELEASES / 'ensemble.json').write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
    sync_report(out)
    sync_history_block(out, panel, ens, ev, days)

    q, ac = out['ensemble']['quality'], out['ensemble']['account']
    print(f"集成（{'+'.join(UNITS)}）：评估窗 {out['eval_window']['start']} → {out['eval_window']['end']}"
          f"（{out['eval_window']['days']} 天）")
    print(f"  RankIC_5d {q['rank_ic_5d']:+.4f}（ICIR {q['icir']:.2f}，t {q['t_rank_ic']:+.2f}）"
          f" | 理想 top5 {q['top5_daily_pct']:+.4f}%/日（t {q['t_top5']:+.2f}）")
    print(f"  账户：复利 {ac['compounded_pct']:+.2f}% | 年化 {ac['net_annual_pct']:+.2f}%"
          f" | 回撤 {ac['max_drawdown_pct']:+.2f}% | {ac['trades']} 笔 / ¥{ac['total_fees']:,.0f}"
          f" | 被拒买 {ac['blocked_entries']}")
    print(f"  与各单模型的逐日秩相关：{' '.join(f'{u} {v:+.4f}' for u, v in agree.items())}")
    print(f"  最新一日 {last_day} → 买入 {buy_day} → 卖出 {sell_day}")
    for p in picks[:10]:
        print(f"    {p['rank']:>2}. {p['code']}  {p['name']:<8}{p['score']:>9.4f}")
    return out


if __name__ == '__main__':
    main()
