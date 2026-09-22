"""给 V76/V77 的 analysis.py 打 RD11 补丁（2026-09-22）。

三处改动，全部是**加法**，不改任何既有口径：
  ① `costfree(..., horizon=1)` —— 新增可选参数，默认 1 与旧行为逐位相同；
  ② `cash_backtest(..., take_profit=None)` —— 新增**固定止盈**（原引擎只有 stop_loss 与 trailing）；
  ③ 新增 `main_metrics_5d.csv`（用户 2026-09-22 口径变更：主指标 1d → 5d）；
  ④ STRATEGIES 增加 RD11 预登记格（台地 + 止盈止损）。

★ 判定时序沿用引擎既有口径：**信号日收盘判定 → 执行日开盘执行**（与全引擎同一条时序，PIT 安全）。
  RD11 文档里原先写的"盘中触发、按 min(open, 止损价) 成交"需要日线分时数据，本引擎不做这个假设。
"""
import sys
from pathlib import Path

ROOT = Path('/autodl-fs/data/model')
UNITS = ['V76', 'V77']

# ---------------------------------------------------------------- ① costfree 加 horizon
OLD_SIG = "def costfree(pred, panel, prices, days, n):\n    \"\"\"信号日选前 n、次日开盘买得进才算，用逐日 1d 标签求和。\n\n    这是可执行无摩擦口径，不是现金账户净值——两者不能混用。\n    返回 (汇总指标, 逐日收益, 逐笔选股)。\n    \"\"\""
NEW_SIG = "def costfree(pred, panel, prices, days, n, horizon=1):\n    \"\"\"信号日选前 n、次日开盘买得进才算，用逐日 {horizon}d 标签求和。\n\n    ★ 2026-09-22：新增 `horizon` 可选参数（用户口径变更 1d→5d）。**默认 1 与旧行为逐位相同**，\n      horizon=5 时取 `label_ret_5d`；掩码与选股逻辑一字不动 —— 只换求和用的那条标签。\n\n    这是可执行无摩擦口径，不是现金账户净值——两者不能混用。\n    返回 (汇总指标, 逐日收益, 逐笔选股)。\n    \"\"\""

# ★ 锚点必须带上下两行：`write_daily_picks` 里也有同一句（那是近 10 日明细的展示表，
#   口径是 T+1 买 / T+2 卖的 1d 收益，本次**不动**它 —— 它的买卖日也是写死的）。
OLD_Y = ("        y = panel.Y['label_ret_1d'][d]\n"
         "        if len(picks) and not np.isfinite(y[picks]).all():")
NEW_Y = ("        y = panel.Y[f'label_ret_{horizon}d'][d]\n"
         "        if len(picks) and not np.isfinite(y[picks]).all():")

# ---------------------------------------------------------------- ② cash_backtest 加 take_profit
OLD_BS_SIG = "def cash_backtest(pred, panel, px, days, n=5, period=1, exit_rank=None, min_hold=1,\n                  stop_loss=None, trailing=None, skip=0, sell_at='open'):"
NEW_BS_SIG = "def cash_backtest(pred, panel, px, days, n=5, period=1, exit_rank=None, min_hold=1,\n                  stop_loss=None, take_profit=None, trailing=None, skip=0, sell_at='open'):"

OLD_GUARD = "                    if stop_loss is None and trailing is None:\n                        continue"
NEW_GUARD = "                    if stop_loss is None and take_profit is None and trailing is None:\n                        continue"

OLD_SL = """                    if stop_loss is not None and price <= entry[c] * (1 - stop_loss):
                        exits.add(c)
                    elif trailing is not None and price <= peak[c] * (1 - trailing):
                        exits.add(c)"""
NEW_SL = """                    if stop_loss is not None and price <= entry[c] * (1 - stop_loss):
                        exits.add(c)
                    elif take_profit is not None and price >= entry[c] * (1 + take_profit):
                        exits.add(c)
                    elif trailing is not None and price <= peak[c] * (1 - trailing):
                        exits.add(c)"""

# ★ 原 return 的下一行已经有 `stop_loss=stop_loss, trailing=trailing, ...`，
#   所以这里只**追加** take_profit，不能重复写 stop_loss（第一版就踩了这个，py_compile 当场炸）。
OLD_RET = "    return dict(topn=n, rebalance_days=period, exit_rank=exit_rank, min_hold=min_hold,"
NEW_RET = "    return dict(topn=n, rebalance_days=period, exit_rank=exit_rank, min_hold=min_hold, take_profit=take_profit,"

# ---------------------------------------------------------------- ④ RD11 策略格
OLD_STRAT_TAIL = "STRATEGIES += [dict(name=f'buffer_r{r}_t{n}', topn=n, exit_rank=r, min_hold=1)"
NEW_STRAT_TAIL = """# --- ★ 2026-09-22 RD11（用户指令）：台地 + 止盈止损 ---
# 基座固定用台地**中心** re500_t5_h5（R26 的台地是 topn=5 / min_hold=5 / exit_rank 400~700，
# 中心取 500）。预登记**固定格，不扫描**：
#   B  = 台地本身（上面已有的 re400/500/600/700_t5_h5，无风控）
#   P  = 主比较格：止损 -8% / 止盈 +20%
#   S  = 敏感性 2×2：{止损 -6%,-10%} × {止盈 +15%,+25%}，只报不选
#   D  = 分解：只止损 / 只止盈（判断效果来自哪一条腿）
# ★ 既有反证：LESSONS §10 死胡同列「止损<6%或>10%与移动止盈」、RD10 判定「止损/移动止盈不成立」，
#   但那是在**排名退出 2 只**族上测的。本次在**五日换手台地**上重开，是配对重测不是推翻。
STRATEGIES += [
    dict(name='re500_t5_h5_sl8_tp20', topn=5, exit_rank=500, min_hold=5, stop_loss=.08, take_profit=.20),
    dict(name='re500_t5_h5_sl6_tp15', topn=5, exit_rank=500, min_hold=5, stop_loss=.06, take_profit=.15),
    dict(name='re500_t5_h5_sl6_tp25', topn=5, exit_rank=500, min_hold=5, stop_loss=.06, take_profit=.25),
    dict(name='re500_t5_h5_sl10_tp15', topn=5, exit_rank=500, min_hold=5, stop_loss=.10, take_profit=.15),
    dict(name='re500_t5_h5_sl10_tp25', topn=5, exit_rank=500, min_hold=5, stop_loss=.10, take_profit=.25),
    dict(name='re500_t5_h5_sl8', topn=5, exit_rank=500, min_hold=5, stop_loss=.08),
    dict(name='re500_t5_h5_tp20', topn=5, exit_rank=500, min_hold=5, take_profit=.20),
]

STRATEGIES += [dict(name=f'buffer_r{r}_t{n}', topn=n, exit_rank=r, min_hold=1)"""

# ---------------------------------------------------------------- ②b 调度层转发 + 防呆
# ★★ 第一版漏了这里：`cash_backtest` 加了 take_profit 参数，但**调度层没转发 spec 里的值**，
#    于是所有策略都在 take_profit=None 下跑 —— re500_t5_h5_tp20 与 re500_t5_h5 **逐位相同**、
#    sl8_tp20 与 sl8 逐位相同。不报错、不崩，只有"两行数字一模一样"这一个线索。
#    这正是 LESSONS §11-26 那条：「入口要问"这里读的是配方那个值，还是另写了一遍"」。
OLD_DISPATCH = """                    stop_loss=spec.get('stop_loss'), trailing=spec.get('trailing'),
                    skip=spec.get('skip', 0), sell_at=spec.get('sell_at', 'open'))
                result['strategies'].append(stat)"""
NEW_DISPATCH = """                    stop_loss=spec.get('stop_loss'), take_profit=spec.get('take_profit'),
                    trailing=spec.get('trailing'),
                    skip=spec.get('skip', 0), sell_at=spec.get('sell_at', 'open'))
                # ★ 防呆：声明了 take_profit/stop_loss 就必须真的落到结果里，否则它是个死参数。
                #   把静默失败变成当场崩溃（LESSONS §11-24：降级成警告的 try/except 比崩更危险）。
                for _k in ('take_profit', 'stop_loss', 'trailing'):
                    if spec.get(_k) != stat.get(_k):
                        raise AssertionError(
                            f"{spec['name']}：{_k} 声明 {spec.get(_k)!r} 但回测里是 {stat.get(_k)!r}"
                            ' —— 调度层没转发，结果不可用')
                result['strategies'].append(stat)"""

# ---------------------------------------------------------------- ③ main_metrics_5d
OLD_METRICS = """        def metrics_1d(pred, days_):
            ic = next(x for x in ic_table(pred, panel, days_) if x['label'] == 'label_ret_1d')
            stat, daily, _ = costfree(pred, panel, px, days_, 1)
            rets = np.asarray([x['return_value'] for x in daily])
            return dict(IC=ic['RankIC'], ICIR=ic['ICIR'], top_return=stat['sum_return'],
                        top_stability=float(rets.mean() / rets.std()) if rets.std() > 0 else 0.)

        main_metrics = []
        for q in quarters(panel.days):
            loc = np.array([str(pd.Period(panel.days[d], freq='Q')) == q for d in eval_days])
            if loc.any():
                main_metrics.append(dict(quarter=q, days=int(loc.sum()),
                                         **metrics_1d(ev_main[loc], eval_days[loc])))
        main_metrics.append(dict(quarter='总计', days=len(eval_days), **metrics_1d(ev_main, eval_days)))
        pd.DataFrame(main_metrics).to_csv(tb / 'main_metrics_1d.csv', index=False)"""

NEW_METRICS = """        def metrics_at(pred, days_, horizon):
            ic = next(x for x in ic_table(pred, panel, days_) if x['label'] == f'label_ret_{horizon}d')
            stat, daily, _ = costfree(pred, panel, px, days_, 1, horizon=horizon)
            rets = np.asarray([x['return_value'] for x in daily])
            return dict(IC=ic['RankIC'], ICIR=ic['ICIR'], top_return=stat['sum_return'],
                        top_stability=float(rets.mean() / rets.std()) if rets.std() > 0 else 0.)

        # 两份表并存（2026-09-22 用户口径变更）：1d 是历史口径、**不删**；5d 是新主口径。
        for horizon in (1, 5):
            rows = []
            for q in quarters(panel.days):
                loc = np.array([str(pd.Period(panel.days[d], freq='Q')) == q for d in eval_days])
                if loc.any():
                    rows.append(dict(quarter=q, days=int(loc.sum()),
                                     **metrics_at(ev_main[loc], eval_days[loc], horizon)))
            rows.append(dict(quarter='总计', days=len(eval_days), **metrics_at(ev_main, eval_days, horizon)))
            pd.DataFrame(rows).to_csv(tb / f'main_metrics_{horizon}d.csv', index=False)"""

PATCHES = [
    ('costfree 签名', OLD_SIG, NEW_SIG),
    ('costfree 标签', OLD_Y, NEW_Y),
    ('cash_backtest 签名', OLD_BS_SIG, NEW_BS_SIG),
    ('cash_backtest 守卫', OLD_GUARD, NEW_GUARD),
    ('cash_backtest 止损/止盈', OLD_SL, NEW_SL),
    ('cash_backtest 返回值', OLD_RET, NEW_RET),
    ('RD11 策略格', OLD_STRAT_TAIL, NEW_STRAT_TAIL),
    ('调度层转发 take_profit', OLD_DISPATCH, NEW_DISPATCH),
    ('main_metrics 双口径', OLD_METRICS, NEW_METRICS),
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
    print(f'✔ {unit}/analysis.py 打完 8 处补丁')
