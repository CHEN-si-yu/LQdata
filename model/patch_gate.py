"""给 V76/V77/V78/V79 加「60 日市场趋势门槛」格（2026-09-22，RD11 第二轮）。

门槛口径**逐字沿用** `V62/market_gate_research.py`（该口径在 V32 上实测把 top1 从
+123.35% 抬到 +160.57%、回撤从 −27.31% 压到 −12.40%）：

    固定 2115 池 · 等权 · 复权收盘指数 ≥ 过去 60 交易日均值 ⇒ risk_on
    risk_off 当日把分数置 NaN ⇒ 不持有（次日开盘清仓）

并保留它的**因果性自检**：截断到 2026-01-01 重算，指数与信号必须**前缀逐位相同**。

★ 这是**加法**：既有非门槛格一个都不动。新增格一律以 `g_` 前缀命名，便于对读。
"""
import sys
from pathlib import Path

ROOT = Path('/autodl-fs/data/model')
UNITS = ['V76', 'V77', 'V78', 'V79']

GATE_FN = '''
def market_gate(panel, px):
    """单一 60 日市场趋势门槛（`V62/market_gate_research.py` 逐字同口径 + 因果性自检）。

    固定池、等权、复权收盘市场指数 ≥ 过去 60 交易日均值 ⇒ risk_on。
    返回 (index, risk_on)，两者都按 `panel.days` 的**全轴**对齐（不是评价窗）。

    ★ 自检不是装饰：把指数截断到 2026-01-01 重算，前缀必须与全量逐位相同 ——
      对不上说明门槛用了未来信息（`LESSONS §2` 铁律二），直接拒跑。
    """
    adjusted = px.close * px.adj

    def build(prices):
        ret = prices[1:] / prices[:-1] - 1
        good = np.isfinite(ret)
        count = good.sum(axis=1)
        if count.min() <= 100:
            raise ValueError('市场指数：有效股票数不足 100，数据不可用')
        mr = np.r_[0., np.where(good, ret, 0.).sum(axis=1) / count]
        idx = 100 * np.cumprod(1 + mr)
        ma = pd.Series(idx).rolling(60, min_periods=60).mean().to_numpy()
        return idx, idx >= ma

    index, on = build(adjusted)
    cut = int(np.searchsorted(panel.days, '2026-01-01'))
    past_index, past_on = build(adjusted[:cut])
    assert np.array_equal(past_index, index[:cut]), '市场门槛因果性自检失败：指数前缀不一致'
    assert np.array_equal(past_on, on[:cut]), '市场门槛因果性自检失败：信号前缀不一致'
    return index, on


'''

# --- 门槛格（预登记）：台地四点全加门槛 + 主实验/分解三条
GATE_STRATS = """# --- ★ 2026-09-22 RD11 第二轮：把已验证的 60 日市场门槛接到五日换手台地上 ---
# 门槛口径见 `market_gate(panel, px)`；risk_off 日分数置 NaN ⇒ 不持有。
# 预登记：台地四点全部加门槛；主实验格 P 加门槛；再留只止盈/只止损两条做分解。
STRATEGIES += [
    dict(name='g_re400_t5_h5', topn=5, exit_rank=400, min_hold=5, gate=True),
    dict(name='g_re500_t5_h5', topn=5, exit_rank=500, min_hold=5, gate=True),
    dict(name='g_re600_t5_h5', topn=5, exit_rank=600, min_hold=5, gate=True),
    dict(name='g_re700_t5_h5', topn=5, exit_rank=700, min_hold=5, gate=True),
    dict(name='g_re500_t5_h5_sl8_tp20', topn=5, exit_rank=500, min_hold=5, stop_loss=.08, take_profit=.20, gate=True),
    dict(name='g_re500_t5_h5_tp20', topn=5, exit_rank=500, min_hold=5, take_profit=.20, gate=True),
    dict(name='g_re500_t5_h5_sl8', topn=5, exit_rank=500, min_hold=5, stop_loss=.08, gate=True),
]

STRATEGIES += [dict(name=f'buffer_r{r}_t{n}', topn=n, exit_rank=r, min_hold=1)"""

OLD_MARK = "STRATEGIES += [dict(name=f'buffer_r{r}_t{n}', topn=n, exit_rank=r, min_hold=1)"

# 调度层：把门槛作用到分数上，并把 gate 纳入防呆
OLD_DISPATCH = """            for spec in STRATEGIES:
                stat, curve, trades = cash_backtest(
                    ev, panel, px, eval_days, n=spec['topn'], period=spec.get('period', 1),"""
NEW_DISPATCH = """            for spec in STRATEGIES:
                # ★ 门槛格：risk_off 当日把分数置 NaN ⇒ 次日开盘清仓（与 V62 门槛研究同法）。
                pred_run = ev
                if spec.get('gate'):
                    if gate_on is None:
                        raise RuntimeError(f"{spec['name']} 声明了 gate 但没有价格层")
                    pred_run = np.where(gate_on[eval_days][:, None], ev, np.nan).astype(np.float32)
                stat, curve, trades = cash_backtest(
                    pred_run, panel, px, eval_days, n=spec['topn'], period=spec.get('period', 1),"""

OLD_GUARD = """                for _k in ('take_profit', 'stop_loss', 'trailing'):"""
NEW_GUARD = """                stat['gate'] = bool(spec.get('gate'))
                for _k in ('take_profit', 'stop_loss', 'trailing'):"""

OLD_PRIMARY = "    primary = 'ensemble' if 'ensemble' in variants else 'fold1'"
NEW_PRIMARY = """    # 门槛指数只依赖 panel/prices，算一次给全部门槛格共用（带因果性自检）。
    gate_index, gate_on = market_gate(panel, px) if px else (None, None)
    primary = 'ensemble' if 'ensemble' in variants else 'fold1'"""

PATCHES = [
    ('门槛函数', "def fees(notional, sell=False):", GATE_FN.lstrip('\n') + "def fees(notional, sell=False):"),
    ('门槛格', OLD_MARK, GATE_STRATS),
    ('门槛指数预计算', OLD_PRIMARY, NEW_PRIMARY),
    ('调度层应用门槛', OLD_DISPATCH, NEW_DISPATCH),
    ('防呆加 gate', OLD_GUARD, NEW_GUARD),
]

for unit in UNITS:
    p = ROOT / unit / 'analysis.py'
    src = p.read_text(encoding='utf-8')
    for label, old, new in PATCHES:
        n = src.count(old)
        if n != 1:
            print(f'✘ {unit}: [{label}] 命中 {n} 次，期望 1 —— 中止', file=sys.stderr)
            raise SystemExit(1)
        src = src.replace(old, new, 1)
    p.write_text(src, encoding='utf-8')
    print(f'✔ {unit}/analysis.py 打完 {len(PATCHES)} 处门槛补丁')
