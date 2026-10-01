#!/usr/bin/env python3
"""V13: same monthly 60-session Top2, rebalance only on membership changes."""
from __future__ import annotations
import os
for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[k] = "1"
import hashlib, json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
V1 = Path("/root/autodl-fs/model/experiments2/V1")
sys.path.insert(0, str(ROOT))
import account_engine as E
sys.path.insert(0, str(V1))
import model as M

TEST_START, TEST_END, TEST_EXIT = "2025-07-01", "2026-06-30", "2026-07-01"
EXT_START, EXT_END = "2023-01-03", "2026-09-24"
LOOKBACK, K = 60, 2
INITIAL = float(E.COST["account_money"])

def month_first(days):
    months = np.array([str(x)[:7] for x in days])
    return np.flatnonzero(np.r_[True, months[1:] != months[:-1]])

def create_schedule(days, codes, arrays, signal_start, signal_end, changed_only):
    adj = pd.DataFrame(arrays["adj_factor"]).ffill().to_numpy(dtype=float)
    ac = arrays["close"] * adj
    schedule, records = {}, []
    previous = None
    for i in month_first(days):
        day = str(days[i])
        if day < signal_start or day > signal_end or i < LOOKBACK or i + 1 >= len(days):
            continue
        old, now = ac[i-LOOKBACK], ac[i]
        valid = np.isfinite(old) & (old > 0) & np.isfinite(now) & (now > 0)
        ids = np.flatnonzero(valid)
        if len(ids) == 0:
            continue
        ret = np.full(len(codes), np.nan)
        ret[valid] = now[valid] / old[valid] - 1.0
        ranked = sorted(ids.tolist(), key=lambda c: (-ret[c], str(codes[c])))
        chosen = ranked[:K]
        members = frozenset(chosen)
        changed = previous is None or members != previous
        if not changed_only or changed:
            schedule[int(i+1)] = np.asarray(chosen, dtype=int)
        records.append({
            "signal_date": day, "execution_date": str(days[i+1]),
            "selected": [str(codes[c]) for c in chosen],
            "membership_changed": bool(changed),
            "scheduled_rebalance": bool((not changed_only) or changed),
            "return_60d": {str(codes[c]): float(ret[c]) for c in ranked}
        })
        previous = members
    return schedule, records

def mark_residual(curve, arrays, n):
    errors, last = [], np.full(n, np.nan)
    for i, row in enumerate(curve):
        op, close = arrays["open"][i], arrays["close"][i]
        mark = np.where(np.isfinite(op)&(op>0), op,
                        np.where(np.isfinite(close)&(close>0), close, last))
        mark = np.where(np.isfinite(mark)&(mark>0), mark, last)
        mv = float(np.sum(np.asarray(row["shares"], float)*np.nan_to_num(mark, nan=0.0)))
        errors.append(abs(float(row["equity"])-float(row["cash"])-mv))
        last = mark.copy()
    return max(errors) if errors else 0.0

def select_window(all_days, all_arrays, start, last_mark):
    ix = np.flatnonzero((all_days >= start) & (all_days <= last_mark))
    if not len(ix) or str(all_days[ix[0]]) != start or str(all_days[ix[-1]]) != last_mark:
        raise RuntimeError(f"window boundary absent: {start}..{last_mark}")
    return ix, all_days[ix], {k:v[ix] for k,v in all_arrays.items()}

def run_variant(all_days, codes, all_arrays, start, signal_end, exit_day, changed_only):
    ix, days, arrays = select_window(all_days, all_arrays, start, exit_day)
    all_schedule, signals = create_schedule(all_days, codes, all_arrays, start, signal_end, changed_only)
    schedule = {int(i-ix[0]): names for i,names in all_schedule.items()
                if ix[0] <= i <= ix[-1]-1}
    if not schedule:
        raise RuntimeError("empty execution schedule")
    result = E.account_sim(days, codes, arrays, schedule)
    stats = E.summarize(result["curve"], INITIAL)
    curve = result["curve"]
    cash_min = min(float(x["cash"]) for x in curve)
    audit = {
        "mark_days_including_exit_execution": int(len(days)),
        "signal_window_session_count": int(np.count_nonzero((days >= start) & (days <= signal_end))),
        "monthly_signal_count": len(signals),
        "membership_change_count": int(sum(x["membership_changed"] for x in signals)),
        "scheduled_rebalance_events": len(schedule),
        "cash_min": cash_min, "ending_cash": float(result["ending_cash"]),
        "ending_positions": int(np.count_nonzero(np.asarray(result["ending_shares"]) > 1e-7)),
        "maximum_simultaneous_holdings": max(int(x["holdings"]) for x in curve),
        "blocked_entries": int(result["blocked_entries"]),
        "blocked_exits": int(result["blocked_exits"]),
        "trade_count": int(result["ntr"]), "fees": float(result["fees"]),
        "buy_notional": float(result["buy_notional"]),
        "sell_notional": float(result["sell_notional"]),
        "gross_traded_notional": float(result["buy_notional"]+result["sell_notional"]),
        "minimum_cash_nonnegative": bool(cash_min >= -1e-7),
        "final_liquidation_complete": bool(np.count_nonzero(np.asarray(result["ending_shares"]) > 1e-7) == 0),
        "max_abs_equity_cash_mark_residual": mark_residual(curve, arrays, len(codes)),
        "forced_exit_date": exit_day,
        "matched_method": "previous execution-day shares x previous-open to current-open adjusted return, compounded without fees"
    }
    return {"stats":stats,"audit":audit,"curve":curve,"trades":result["trades"],"signals":signals}

def baseline(all_days, codes, all_arrays, start, signal_end, exit_day):
    ix, days, arrays = select_window(all_days, all_arrays, start, exit_day)
    all_schedule, signals = create_schedule(all_days, codes, all_arrays, start, signal_end, False)
    local = {int(i-ix[0]):names for i,names in all_schedule.items()
             if ix[0] <= i <= ix[-1]-1}
    monthly_result = E.account_sim(days, codes, arrays, local)
    first = min(local)
    initial = E.account_sim(days, codes, arrays, {first: local[first]})
    eq = E.account_sim(days, codes, arrays, {first: np.arange(len(codes), dtype=int)})
    def pack(res):
        return {"stats":E.summarize(res["curve"], INITIAL),
                "audit":{"trade_count":int(res["ntr"]),"fees":float(res["fees"]),
                         "buy_notional":float(res["buy_notional"]),
                         "sell_notional":float(res["sell_notional"]),
                         "gross_traded_notional":float(res["buy_notional"]+res["sell_notional"]),
                         "blocked_entries":int(res["blocked_entries"]),
                         "blocked_exits":int(res["blocked_exits"]),
                         "ending_positions":int(np.count_nonzero(np.asarray(res["ending_shares"])>1e-7))},
                "curve":res["curve"],"trades":res["trades"]}
    return pack(monthly_result), pack(initial), pack(eq), signals

def segmented(curve, frequency, end_date=None):
    d = pd.DataFrame(curve)
    if end_date is not None:
        d = d.loc[d.date <= end_date].copy()
    dt = pd.to_datetime(d.date)
    keys = dt.dt.to_period(frequency).astype(str)
    rows = []
    for key in sorted(keys.unique()):
        g = d.loc[keys == key]
        net = float(np.prod(1.0 + g.daily_return.to_numpy(float))-1.0)
        matched = float(np.prod(1.0 + g.matched_return.to_numpy(float))-1.0)
        rows.append({"period":str(key),"days":int(len(g)),"net_return":net,
                     "matched_return":matched,"matched_friction":net-matched})
    return rows

def total_to(curve, date):
    d = pd.DataFrame(curve)
    row = d.loc[d.date == date]
    return float(row.equity.iloc[-1]/INITIAL-1.0) if len(row) else None

def main():
    t0=time.time()
    panel=M.load_panel()
    days, codes=panel.days, panel.codes
    arrays={k:panel.prices.raw[k] for k in ("open","high","low","pre_close","close","vol","adj_factor")}
    arrays["amount"]=panel.amount
    codes=[str(x) for x in codes]
    if tuple(codes) != tuple(M.CODES):
        raise RuntimeError(f"unexpected universe {codes}")
    testdays=np.flatnonzero((days>=TEST_START)&(days<=TEST_END))
    if len(testdays)!=242:
        raise RuntimeError(f"expected 242 official sessions through {TEST_END}, got {len(testdays)}")
    if str(days[np.flatnonzero(days==TEST_EXIT)[0]])!=TEST_EXIT:
        raise RuntimeError("forced exit day missing")

    # 242 in-window marks plus next-session forced liquidation.
    v13=run_variant(days,codes,arrays,TEST_START,TEST_END,TEST_EXIT,True)
    # Re-run all comparators on the identical ledger horizon, ending with the same forced exit.
    monthly, initial, equal, v4signals=baseline(days,codes,arrays,TEST_START,TEST_END,TEST_EXIT)
    v13["quarterly"]=segmented(v13["curve"],"Q",TEST_END)
    monthly["quarterly"]=segmented(monthly["curve"],"Q",TEST_END)
    initial["quarterly"]=segmented(initial["curve"],"Q",TEST_END)
    equal["quarterly"]=segmented(equal["curve"],"Q",TEST_END)
    v13["ending_return_on_exit_day"]=v13["stats"]["cumulative_net_return"]
    v13["ending_return_at_last_signal_window_mark"]=total_to(v13["curve"],TEST_END)
    v13["performance_days_including_exit_open"]=len(v13["curve"])
    v13["comparison"]={
        "net_gain_vs_same_ledger_monthly_reweighted_v4_pp":100*(v13["stats"]["cumulative_net_return"]-monthly["stats"]["cumulative_net_return"]),
        "net_gain_vs_first_top2_hold_pp":100*(v13["stats"]["cumulative_net_return"]-initial["stats"]["cumulative_net_return"]),
        "net_gain_vs_four_bank_equal_hold_pp":100*(v13["stats"]["cumulative_net_return"]-equal["stats"]["cumulative_net_return"]),
        "trades_avoided_vs_monthly_v4":monthly["audit"]["trade_count"]-v13["audit"]["trade_count"],
        "gross_notional_avoided_vs_monthly_v4":monthly["audit"]["gross_traded_notional"]-v13["audit"]["gross_traded_notional"],
        "fees_avoided_vs_monthly_v4":monthly["audit"]["fees"]-v13["audit"]["fees"],
    }

    ext=run_variant(days,codes,arrays,EXT_START,EXT_END,EXT_END,True)
    ext_monthly, ext_initial, ext_equal, ext_v4signals=baseline(days,codes,arrays,EXT_START,EXT_END,EXT_END)
    ext["yearly"]=segmented(ext["curve"],"Y")
    ext_monthly["yearly"]=segmented(ext_monthly["curve"],"Y")
    ext_initial["yearly"]=segmented(ext_initial["curve"],"Y")
    ext_equal["yearly"]=segmented(ext_equal["curve"],"Y")
    ext["comparison"]={
        "net_gain_vs_same_ledger_monthly_reweighted_v4_pp":100*(ext["stats"]["cumulative_net_return"]-ext_monthly["stats"]["cumulative_net_return"]),
        "net_gain_vs_first_top2_hold_pp":100*(ext["stats"]["cumulative_net_return"]-ext_initial["stats"]["cumulative_net_return"]),
        "net_gain_vs_four_bank_equal_hold_pp":100*(ext["stats"]["cumulative_net_return"]-ext_equal["stats"]["cumulative_net_return"]),
        "trades_avoided_vs_monthly_v4":ext_monthly["audit"]["trade_count"]-ext["audit"]["trade_count"],
        "gross_notional_avoided_vs_monthly_v4":ext_monthly["audit"]["gross_traded_notional"]-ext["audit"]["gross_traded_notional"],
        "fees_avoided_vs_monthly_v4":ext_monthly["audit"]["fees"]-ext["audit"]["fees"],
    }
    script_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    out={
        "unit":"experiments2/V13 membership-change-only equal-weight monthly momentum",
        "built_at":time.strftime("%Y-%m-%d %H:%M:%S"),
        "snapshot_built_at":panel.data_built_at,"snapshot_last_day":str(days[-1]),
        "universe":codes,
        "protocol":{
            "signal":"first observed trading session of each calendar month at close",
            "factor":"adjusted close(T)/adjusted close(T-60 trading sessions)-1; fixed Top2",
            "execution":"T+1 open; equal-value targets only when Top2 membership changes; otherwise preserve share counts; shared cash ledger copied from V4; 100-share lots, capacity, price locks, fees and slippage",
            "official_signal_window":[TEST_START,TEST_END],
            "official_session_count":242,"official_forced_exit":TEST_EXIT,
            "extended_signal_window":[EXT_START,EXT_END],"extended_forced_exit":EXT_END,
            "no_training":True,"no_parameter_sweep":True,
            "matched_return_interpretation":"same realized holdings path without fees; net-minus-matched is execution friction, not selection alpha",
            "engine_copy":"V13/account_engine.py copied from V4/account_engine.py"
        },
        "official":{
            "v13":{k:v for k,v in v13.items() if k not in ("curve","trades","signals")},
            "same_ledger_monthly_reweighted_v4":{"stats":monthly["stats"],"audit":monthly["audit"],"quarterly":monthly["quarterly"]},
            "first_selected_top2_hold":{"stats":initial["stats"],"audit":initial["audit"],"quarterly":initial["quarterly"]},
            "four_bank_equal_weight_hold":{"stats":equal["stats"],"audit":equal["audit"],"quarterly":equal["quarterly"]}
        },
        "extended":{
            "v13":{k:v for k,v in ext.items() if k not in ("curve","trades","signals")},
            "same_ledger_monthly_reweighted_v4":{"stats":ext_monthly["stats"],"audit":ext_monthly["audit"],"yearly":ext_monthly["yearly"]},
            "first_selected_top2_hold":{"stats":ext_initial["stats"],"audit":ext_initial["audit"],"yearly":ext_initial["yearly"]},
            "four_bank_equal_weight_hold":{"stats":ext_equal["stats"],"audit":ext_equal["audit"],"yearly":ext_equal["yearly"]}
        },
        "runner_sha256":script_hash,"runtime_seconds":time.time()-t0
    }
    (ROOT/"result.json").write_text(json.dumps(out,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
    for name,payload in [("official",v13),("extended",ext)]:
        pd.DataFrame(payload["curve"]).drop(columns=["shares"]).to_csv(ROOT/f"{name}_daily.csv",index=False)
        pd.DataFrame(payload["trades"]).to_csv(ROOT/f"{name}_trades.csv",index=False)
        pd.DataFrame(payload["signals"]).to_json(ROOT/f"{name}_signals.json",orient="records",force_ascii=False,indent=2)
    for name,payload in [("official_v4_reweighted",monthly),("extended_v4_reweighted",ext_monthly)]:
        pd.DataFrame(payload["trades"]).to_csv(ROOT/f"{name}_trades.csv",index=False)
    protocol={
        "objective":"Compare drift-preserving Top2 momentum with V4 monthly reweighting",
        "rule":out["protocol"],"runner_sha256":script_hash,
        "engine_sha256":hashlib.sha256((ROOT/"account_engine.py").read_bytes()).hexdigest(),
        "panel_loader":str(V1/"model.py"),"created_at":out["built_at"]
    }
    (ROOT/"protocol.json").write_text(json.dumps(protocol,ensure_ascii=False,indent=2),encoding="utf-8")

    def fmt(x): return f"{x:.2%}"
    off=v13["stats"]; va=v13["audit"]; mm=monthly["stats"]
    report=[
        "# V13：Top2 成分变化时才等权再平衡",
        "",
        f"- 数据快照 {panel.data_built_at}，截至 {days[-1]}；未训练模型、未扫描参数。",
        "- 固定信号与V4相同：每月首个交易日收盘按60交易间隔复权收益选择Top2，次日开盘执行。",
        "- 持仓Top2成员未变化时不动股数；成员变化时才按当时账户权益重新等值配置。共享现金、整手、容量、涨跌停限制、费用与滑点均由独立复制的V4账本执行。",
        f"- 官方窗包含 {len(testdays)} 个信号窗口交易日（{TEST_START} 至 {TEST_END}），在 {TEST_EXIT} 开盘强制清仓；终值包含退出日开盘变动及清仓费用。",
        "- 同持仓匹配收益按实际上一执行日股数计算无费开盘到开盘收益；净收益与匹配收益之差只表示费用和执行摩擦，不是选股alpha。",
        "",
        "## 官方窗（含 2026-07-01 清仓）",
        "",
        "| 方案 | 清仓后净收益 | 最大回撤 | 平均敞口 | 手续费 | 成交数 | 总买卖额 | 净收益差 vs V13 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    rows=[
        ("V13 成分变化才再平衡",v13["stats"],v13["audit"],"—"),
        ("同账本月频重等权 V4",monthly["stats"],monthly["audit"],f"{(mm['cumulative_net_return']-off['cumulative_net_return'])*100:+.2f}pp"),
        ("首次Top2持有",initial["stats"],initial["audit"],f"{(initial['stats']['cumulative_net_return']-off['cumulative_net_return'])*100:+.2f}pp"),
        ("四股等权持有",equal["stats"],equal["audit"],f"{(equal['stats']['cumulative_net_return']-off['cumulative_net_return'])*100:+.2f}pp"),
    ]
    for name,st,au,diff in rows:
        report.append(f"| {name} | {fmt(st['cumulative_net_return'])} | {fmt(st['max_drawdown'])} | {st['avg_exposure']:.1%} | {au['fees']:.2f} | {au['trade_count']} | {au['gross_traded_notional']:.2f} | {diff} |")
    report += [
        "",
        f"- V13 在 {TEST_END} 开盘价标记的清仓前账本收益为 {fmt(v13['ending_return_at_last_signal_window_mark'])}；加上 {TEST_EXIT} 开盘至清仓后的官方终值收益为 {fmt(off['cumulative_net_return'])}。",
        f"- 对比同账本月频V4，V13少成交 {v13['comparison']['trades_avoided_vs_monthly_v4']} 笔、少买卖额 {v13['comparison']['gross_notional_avoided_vs_monthly_v4']:.2f}、少费用 {v13['comparison']['fees_avoided_vs_monthly_v4']:.2f}；相对V4净收益差 {v13['comparison']['net_gain_vs_same_ledger_monthly_reweighted_v4_pp']:+.2f}pp。",
        "", "### 242日信号窗季度收益（不含7月1日清算日）", "",
        "| 季度 | 交易日 | 净收益 | 同持仓无费收益 | 成本/执行摩擦差 |",
        "|---|---:|---:|---:|---:|"
    ]
    for r in v13["quarterly"]:
        report.append(f"| {r['period']} | {r['days']} | {fmt(r['net_return'])} | {fmt(r['matched_return'])} | {fmt(r['matched_friction'])} |")
    report += ["", "### 扩展窗年度收益（最终于2026-09-24开盘清仓）","",
               "| 年份 | 交易日 | 净收益 | 同持仓无费收益 | 成本/执行摩擦差 |",
               "|---:|---:|---:|---:|---:|"]
    for r in ext["yearly"]:
        report.append(f"| {r['period']} | {r['days']} | {fmt(r['net_return'])} | {fmt(r['matched_return'])} | {fmt(r['matched_friction'])} |")
    report += [
        "",
        f"- 扩展窗累计净收益 {fmt(ext['stats']['cumulative_net_return'])}，最大回撤 {fmt(ext['stats']['max_drawdown'])}；对同账本月频V4净收益差 {ext['comparison']['net_gain_vs_same_ledger_monthly_reweighted_v4_pp']:+.2f}pp。",
        f"- 扩展窗少于V4的成交数 {ext['comparison']['trades_avoided_vs_monthly_v4']} 笔、总买卖额 {ext['comparison']['gross_notional_avoided_vs_monthly_v4']:.2f}、费用 {ext['comparison']['fees_avoided_vs_monthly_v4']:.2f}。",
        "",
        "## 账本核验",
        "",
        f"- 官方：最低现金 {va['cash_min']:.4f}，最多同时持仓 {va['maximum_simultaneous_holdings']} 只，阻塞买入/卖出 {va['blocked_entries']}/{va['blocked_exits']}，清仓后持仓数 {va['ending_positions']}，权益-现金-持仓市值最大残差 {va['max_abs_equity_cash_mark_residual']:.3g}。",
        f"- 扩展：最低现金 {ext['audit']['cash_min']:.4f}，最多同时持仓 {ext['audit']['maximum_simultaneous_holdings']} 只，阻塞买入/卖出 {ext['audit']['blocked_entries']}/{ext['audit']['blocked_exits']}，最终持仓数 {ext['audit']['ending_positions']}，残差 {ext['audit']['max_abs_equity_cash_mark_residual']:.3g}。",
        "- 成分不变月份没有生成调仓事件；策略结果仅是固定历史样本回测，不能据此确认稳定超额。"
    ]
    (ROOT/"REPORT.md").write_text("\n".join(report)+"\n",encoding="utf-8")
    print("V13_RESULT",ROOT/"result.json",flush=True)
    print("official",json.dumps({"v13":off,"v4":monthly["stats"],"comparison":v13["comparison"],"audit":va},ensure_ascii=False),flush=True)
    print("extended",json.dumps({"v13":ext["stats"],"v4":ext_monthly["stats"],"comparison":ext["comparison"],"audit":ext["audit"]},ensure_ascii=False),flush=True)
    print("runtime_seconds",time.time()-t0,flush=True)

if __name__=="__main__":
    main()
