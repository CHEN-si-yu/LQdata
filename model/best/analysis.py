"""best：实战单元的**集成 / 门槛 / 出票**半边。

不训练（训练在 `model.py`），**不做任何 test 回测、不出任何评价指标** —— 用户 2026-09-21 明确：
实战组不需要；有效性由 valid 块（四折轮换）承担。这里只做三件事：

1. 把九组 × 四折的打分合成**一份实战打分**；
2. 过**固定 60 日市场趋势门槛**（T 收盘定、T+1 开盘执行）；
3. 出**明天的最终排名**与**明天的具体操作策略**（`orders.md`）。

## 口径（照"得分最高的来源"复现，与 R38 / V75/combine.py 逐字一致，可对账）

1. **组内**：该组四折 `score_predictions.npy` **直接相加**（SPEC 铁律二；与官方 analysis.py 同口径）。
   NaN 用求和传播：某票某折没有分数时结果仍是"没有分数"，不当成 0 分。
2. **组间**：以第一组（`s3253`）为参考，其余各组**逐日按截面标准差**对齐后**等权平均**
   （线性缩放不改当天排名 ⇒ k=1 时逐位等于该组自己的分）。
3. **门槛**：固定 2115 池、等权、复权收盘市场指数 ≥ 过去 60 交易日均值（`market_gate_research.py` 同口径，
   含截断因果性自检）。
4. **选股不加覆盖率掩码** —— 与官方 `costfree` / `cash_backtest` 的选股口径一致
   （那里只要求"分数有限"且"次日开盘买得进"）。

## 用法

    python analysis.py                # 集成 + 门槛 + 出票
    python analysis.py --jobs 6       # 只影响读盘并发，不影响口径
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import evaluation_core as E

ROOT = Path(__file__).resolve().parent

# 九组，顺序即集成顺序；第一组是跨组对齐的参考（与 model.PROD_GROUPS 一致）。
GROUPS = ['s3253', 's3254', 's3255', 's15361', 's28403', 's49117', 's3301', 's3302', 's3303']
FOLDS = (1, 2, 3, 4)
SLIPPAGE = .0003            # 与项目其余部分一致的单边滑点
GATE_WINDOW = 60
TOP_SHOW = 50              # 排名表输出前 N 名
INITIAL_CASH = 100000.0    # 展示用的作战资金（用户口径：小资金 10 万）


def group_stream(group):
    """一组的集成分 = 四折打分直接相加；逐折核验 run_id 与折间日期轴。"""
    per_fold, dates = [], None
    for fold in FOLDS:
        folder = ROOT / 'model_train' / group / f'fold{fold}'
        if not (folder / 'complete.json').exists():
            raise RuntimeError(f'{group} fold{fold} 尚未训练完成（缺 complete.json）')
        done = json.loads((folder / 'complete.json').read_text())
        lock = json.loads((folder / 'recipe.lock.json').read_text())
        if lock['run_id'] != done['run_id']:
            raise RuntimeError(f'{group} fold{fold}: run_id 不一致（配方/数据与产物不符）')
        dts = list(done['score_dates'])
        if dates is None:
            dates = dts
        elif dates != dts:
            raise RuntimeError(f'{group} fold{fold}: 折间推演窗不一致')
        per_fold.append(np.load(folder / 'score_predictions.npy'))
    return dates, np.sum(per_fold, axis=0).astype(np.float32)


def scale_to(x, ref):
    """把 x 逐日截面标准差缩放到与 ref 相同；线性缩放不改当天排名（照抄 V75/combine.py）。"""
    out = np.full(x.shape, np.nan, np.float32)
    for i in range(x.shape[0]):
        a, b = ref[i], x[i]
        ma, mb = np.isfinite(a), np.isfinite(b)
        if ma.sum() < 3 or mb.sum() < 3:
            continue
        sa, sb = np.std(a[ma]), np.std(b[mb])
        if sb > 0:
            out[i, mb] = b[mb] * (sa / sb)
    return out


def market_gate(days, px):
    """固定 60 日市场趋势门槛：等权、复权收盘市场指数 ≥ 过去 60 交易日均值。

    与 `V62/market_gate_research.py` 逐字同口径，并保留它的**因果性自检**：
    拿截断到 2026-01-01 的数据重算，前缀必须逐位相同 —— 这条是"没用未来数据"的机器证明。
    """
    adjusted = px.close * px.adj

    def gate(prices):
        ret = prices[1:] / prices[:-1] - 1
        good = np.isfinite(ret)
        count = good.sum(axis=1)
        assert count.min() > 100, '市场指数当日有效样本太少，门槛不可信'
        mr = np.r_[0., np.where(good, ret, 0.).sum(axis=1) / count]
        idx = 100 * np.cumprod(1 + mr)
        ma = pd.Series(idx).rolling(GATE_WINDOW, min_periods=GATE_WINDOW).mean().to_numpy()
        return idx, idx >= ma

    index, risk_on = gate(adjusted)
    cut = int(np.searchsorted(days, '2026-01-01'))
    past_index, past_gate = gate(adjusted[:cut])
    assert np.array_equal(past_index, index[:cut]) and np.array_equal(past_gate, risk_on[:cut]), \
        '门槛的因果性自检失败：截断重算的前缀与全量不一致'
    return index, risk_on


def stock_names():
    """代码 → 当前名称，只用于展示。

    ⚠️ 厂商把**整段历史**都改写成**当前**名称，所以它只能用于展示"当前"名称，
    绝不能拿去判 ST / 风险 / 分类（见 featureengineering/README.md T3）。
    """
    root = Path(os.environ.get('MX_RAW_DATA', ROOT.parent.parent / 'datadownload' / 'data'))
    files = sorted((root / 'stock_daily').glob('year=*/data.parquet'))
    if not files:
        print(f'⚠️ 找不到 {root}/stock_daily/year=*/data.parquet：排名表只出代码、不出名称', flush=True)
        return {}
    df = pq.read_table(files[-1], columns=['trade_date', 'stock_code', 'stock_name']).to_pandas()
    df['trade_date'] = df['trade_date'].astype(str).str[:10]
    df['stock_code'] = df['stock_code'].astype(str)
    df = df.sort_values('trade_date').drop_duplicates('stock_code', keep='last')
    return df.set_index('stock_code')['stock_name'].to_dict()


def size_order(budget, price):
    """整手股数：与 evaluation_core 买入块逐字一致（预留下单费用，买不起就减一手）。"""
    quantity = int(max(0, budget - 5) / (price * (1 + .00025 + .00001)) / 100) * 100
    while quantity > 0 and quantity * price + E.fees(quantity * price) > budget:
        quantity -= 100
    return quantity


def write_ensemble(scores, dates, panel):
    """打分落盘，契约与项目其余部分一致：4 列 + 年分区 + 原子写。"""
    out = ROOT / 'model_pred' / 'ensemble'
    frame = pd.DataFrame(dict(
        trade_date=np.repeat(np.asarray(dates), len(panel.codes)),
        stock_code=np.tile(np.asarray(panel.codes), len(dates)),
        value=scores.reshape(-1).astype(np.float32)))
    # rank = 当日截面百分位（只对有限分数排名），与 write_scores 同口径。
    ranks = []
    for row in scores:
        r = np.full(len(row), np.nan, np.float32)
        ok = np.isfinite(row)
        if ok.sum():
            order = np.argsort(np.argsort(row[ok]))
            r[ok] = (order + 1) / ok.sum()
        ranks.append(r)
    frame['rank'] = np.concatenate(ranks).astype(np.float32)
    for year, grp in frame.groupby(frame['trade_date'].str[:4]):
        d = out / f'year={year}'
        d.mkdir(parents=True, exist_ok=True)
        tmp = d / 'data.parquet.tmp'
        grp.to_parquet(tmp, index=False)
        os.replace(tmp, d / 'data.parquet')
    return out


def main():
    ap = argparse.ArgumentParser(description='best：九组集成的实战打分 + 60日门槛 + 出票')
    ap.add_argument('--out', default='model_pred/latest')
    args = ap.parse_args()

    panel = E.Panel(load_x=False)       # 只要 days/codes/amount/Y，省掉最大的一块内存
    px = E.Prices(panel)

    # --- ① 九组集成 ---
    streams, dates, prov = {}, None, []
    for g in GROUPS:
        d, s = group_stream(g)      # 注意顺序：group_stream 返回 (dates, scores)
        if dates is None:
            dates = d
        elif dates != d:
            raise RuntimeError(f'{g} 的推演窗与参考组不一致')
        streams[g] = s
        prov.append(dict(group=g, sha256=hashlib.sha256(
            np.ascontiguousarray(s).tobytes()).hexdigest()[:16]))
        print(f'已载入 {g}：{s.shape[0]} 天 × {s.shape[1]} 只', flush=True)

    ref = streams[GROUPS[0]]
    ens = np.zeros_like(ref)
    for g in GROUPS:
        ens += (ref if g == GROUPS[0] else scale_to(streams[g], ref))
    ens = (ens / len(GROUPS)).astype(np.float32)

    days_ix = np.searchsorted(panel.days, dates)
    if not np.array_equal(panel.days[days_ix], np.asarray(dates)):
        raise RuntimeError('推演窗与面板轴对不上')

    # --- ② 固定 60 日市场趋势门槛 ---
    index, risk_on = market_gate(panel.days, px)
    last = len(dates) - 1
    d_last = int(days_ix[last])
    gate_now = bool(risk_on[d_last])

    # 门槛后的打分：risk_off 当日把分数置 NaN ⇒ 不持有（与 market_gate_research.py 同法）。
    gated = np.where(risk_on[days_ix][:, None], ens, np.nan).astype(np.float32)

    # --- ③ 出票 ---
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    name_of = stock_names()
    nm = lambda code: name_of.get(str(code), '')   # 入参是**股票代码**，不是列号

    def top_of(row, k):
        ok = np.flatnonzero(np.isfinite(row))
        return ok[np.argsort(-row[ok], kind='stable')][:k]

    # 排名表：只给"门槛开着"的那份（门槛关了就空仓，没有可买清单）。
    pick_row = ens[last] if gate_now else np.full(ens.shape[1], np.nan)
    top = top_of(pick_row, TOP_SHOW)
    close = px.close[d_last]
    amount = panel.amount[d_last]
    # 信号日到底有没有成交：停牌股拿到的是**前向填充的陈旧收盘价**，必须显式标出来。
    traded_today = np.isfinite(px.raw['open'][d_last]) & (px.raw['vol'][d_last] > 0)
    ranking = pd.DataFrame([dict(
        排名=i, 代码=str(panel.codes[c]), 名称=nm(panel.codes[c]),
        打分=round(float(ens[last][c]), 4),
        收盘=round(float(close[c]), 2) if np.isfinite(close[c]) else None,
        信号日成交额亿元=round(float(amount[c]) / 1e8, 2) if np.isfinite(amount[c]) else None,
        信号日有成交=bool(traded_today[c]),
    ) for i, c in enumerate(top, 1)])
    ranking.to_csv(out / 'ranking.csv', index=False)

    # 全池排名（可查任意一只）
    write_ensemble(ens, dates, panel)

    # 近 10 个信号日选了哪只 —— 只服务"换仓连续性"（昨天选的是不是同一只 → hold 还是 switch）。
    recent = []
    for i in range(len(dates)):
        d = int(days_ix[i])
        if not risk_on[d]:                      # 门槛关着的日子理论上是空仓
            recent.append(dict(信号日=str(dates[i]), 门槛='空仓', 排名1代码='', 名称='', 打分=''))
            continue
        sel = top_of(ens[i], 1)
        c = int(sel[0]) if len(sel) else None
        recent.append(dict(信号日=str(dates[i]), 门槛='持有',
                           排名1代码='' if c is None else str(panel.codes[c]),
                           名称='' if c is None else nm(panel.codes[c]),
                           打分='' if c is None else round(float(ens[i][c]), 4)))
    pd.DataFrame(recent).to_csv(out / 'recent_picks.csv', index=False)

    # 下单指令
    top1 = int(top[0]) if len(top) else None
    top5 = [int(c) for c in top_of(pick_row, 5)]
    prev_row = recent[-2] if len(recent) >= 2 else None
    prev_code = prev_row['排名1代码'] if prev_row and prev_row['门槛'] == '持有' else ''
    same = bool(top1 is not None and prev_code and prev_code == str(panel.codes[top1]))
    if not gate_now:
        action = '空仓（不建仓；若有持仓则开盘清掉）'
    elif same:
        # 上一信号日已经选过同一只：已建仓的人不用动，空仓的人仍要建仓 —— 两种状态都写清。
        action = (f'**已按上一信号日建仓的人：持有不动**（同一只，明天不用交易）；'
                  f'**现在空仓的人：明天开盘建仓买入**')
    elif prev_code:
        action = '换仓（与上一信号日不是同一只：**开盘先卖后买**）'
    else:
        action = '建仓（上一信号日门槛关闭 / 无持仓：**明天开盘买入**）'

    def ticket(c, n_hold, cash):
        price = float(close[c])
        amt = float(amount[c]) if np.isfinite(amount[c]) else np.nan
        budget = min(cash, cash / n_hold, amt * .01 if np.isfinite(amt) else 0.)
        qty = size_order(budget, price) if np.isfinite(price) and price > 0 else 0
        return dict(code=str(panel.codes[c]), name=nm(panel.codes[c]), price=price, budget=budget,
                    qty=qty, notional=qty * price, score=float(ens[last][c]))

    t1 = ticket(top1, 1, INITIAL_CASH) if top1 is not None else None
    t5 = [ticket(c, 5, INITIAL_CASH) for c in top5]

    ma_now = float(pd.Series(index[:d_last + 1]).rolling(
        GATE_WINDOW, min_periods=GATE_WINDOW).mean().to_numpy()[-1])
    mkt = dict(信号日=str(dates[last]), 市场指数=round(float(index[d_last]), 4),
               ma60=round(ma_now, 4),
               门槛='risk_on（可持仓）' if gate_now else 'risk_off（空仓）',
               risk_on=gate_now)

    lines = [f'# {str(dates[last])} 收盘 · 实战下单指令（best 单元）', '',
             '> **本组为实战口径：没有 test 窗、不出任何样本外证据。** 有效性由四折轮换的 valid 块承担。',
             '> 下面的历史数字（如有引用）来自 R38 在**评价窗**上的实测，是"这套配方历史上值多少"，',
             '> **不是这只模型明天的预期。**', '',
             '## ① 先看门槛（这一条决定明天做不做）', '',
             f'- 信号日：**{mkt["信号日"]} 收盘**',
             f'- 市场指数（固定 2115 池、等权、复权收盘）：**{mkt["市场指数"]}**',
             f'- 过去 60 日均值：**{mkt["ma60"]}**',
             f'- **判定：{mkt["门槛"]}**', '']
    if gate_now:
        lines += ['## ② 买什么（门槛开着 → 明天开盘建仓）', '',
                  f'**Top1（全部资金一只）**：`{t1["code"]}` {t1["name"]}'
                  f'（打分 {t1["score"]:.4f}）', '',
                  f'- 参考价（{mkt["信号日"]} 收盘）：**{t1["price"]:.2f} 元**',
                  f'- 预算：**{t1["budget"]:,.0f} 元**（= min(现金, 资金/1, 信号日成交额×1%)）',
                  f'- **买入 {t1["qty"]} 股**（整手，按参考价估算，约 {t1["notional"]:,.0f} 元）',
                  f'- ★ 实际股数按**明天开盘价**用同一条公式重算：'
                  f'`int((预算−5)/(开盘价×1.00026)/100)×100`', '']
        lines += ['**Top5（分散备选，各 1/5 资金）**：', '',
                  '| 排名 | 代码 | 名称 | 参考价 | 预算(元) | 股数 | 打分 |',
                  '|--:|:--|:--|--:|--:|--:|--:|']
        for i, t in enumerate(t5, 1):
            lines.append(f'| {i} | {t["code"]} | {t["name"]} | {t["price"]:.2f} | '
                         f'{t["budget"]:,.0f} | {t["qty"]} | {t["score"]:.4f} |')
        lines += ['']
    else:
        lines += ['## ② 明天不建仓', '',
                  '门槛为 **risk_off** ⇒ **明天空仓**。若手上已有持仓，**开盘清掉**'
                  '（跌停/停牌卖不掉则顺延到能卖的那天）。', '']

    lines += [
        '## ③ 换仓连续性', '',
        f'- 上一个信号日（{prev_row["信号日"] if prev_row else "无"}）的第 1 名：'
        f'`{prev_code or "（空仓 / 无）"}`',
        f'- 本次第 1 名：`{t1["code"] if t1 else "（空仓）"}`',
        f'- **判定：{action}**', '',
        '## ④ 之后的规则（每个交易日重复）', '',
        '1. **T 日收盘**出分 → **T+1 开盘**按新排名调仓（本项目固定 T+2 开盘换仓口径：'
        'T 日收盘定、T+1 开盘执行该次调仓）。',
        '2. **拒单顺延**：T+1 开盘一字涨停（触及按 pre_close 算的 10% 上限）或停牌 ⇒ **买不进，顺延**；',
        '   跌停或停牌 ⇒ **卖不掉，顺延**到能成交的那天。',
        '3. **门槛转 risk_off 的当天收盘**即触发清仓信号，**次日开盘清仓**。', '',
        '## ⑤ 费用（回测口径，实盘以券商为准）', '',
        '- 佣金 万 2.5（最低 5 元）＋ 过户 万 0.1 ＋ 卖出印花 万 5 ＋ 单边滑点 3bp。',
        '- 单笔买入不超过**信号日成交额的 1%**。', '',
        '---', '',
        f'集成口径：九组（{"、".join(GROUPS)}）各四折打分**直接相加** → '
        '以 `s3253` 为参考**逐日截面标准差对齐** → **等权平均**。',
        f'门槛口径：{GATE_WINDOW} 日等权复权收盘市场指数 ≥ 其 {GATE_WINDOW} 日均值。',
        '2115 只固定池含**幸存者偏差**，绝对值不可当真钱预期读。',
    ]
    (out / 'orders.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')

    pd.DataFrame(dict(date=panel.days, market_index=index, risk_on=risk_on)).to_csv(
        out / 'market_signal.csv', index=False)

    plan = dict(unit='best', mode='production_full_data', groups=list(GROUPS), folds=list(FOLDS),
                ensemble='每组的四折 score_predictions 直接相加；组间逐日截面标准差对齐到首组后等权平均',
                gate=dict(window=GATE_WINDOW,
                          rule='equal-weight adjusted-close market index >= trailing 60-trading-day mean',
                          execution='Signal at T close; sell/buy at T+1 open under original fees, lots and blocking rules.'),
                score_window=[dates[0], dates[last]], signal_day=str(dates[last]),
                slippage=SLIPPAGE, initial_cash=INITIAL_CASH, gate_now=gate_now,
                provenance=prov,
                note='实战口径：无 test 窗、不出评价指标；有效性由四折轮换的 valid 块承担。')
    (out / 'plan.json').write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding='utf-8')

    print('\n=== 前 10 名（' + str(dates[last]) + ' 收盘）===', flush=True)
    print(ranking.head(10).to_string(index=False), flush=True)
    print(f'\n门槛：{mkt["门槛"]}（指数 {mkt["市场指数"]} vs 60日均值 {mkt["ma60"]}）', flush=True)
    print(f'产物：{out}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
