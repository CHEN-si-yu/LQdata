#!/usr/bin/env python3
"""Fixed inverse-vol allocation for the four-bank monthly 60d-momentum Top2."""
from __future__ import annotations
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_k] = "1"
import hashlib, json, math, time
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

OUT = Path(__file__).resolve().parent
DATA = Path("/root/autodl-fs/model/trainingdata")
CODES = ("601288.SH", "601398.SH", "601939.SH", "601988.SH")
START_OFFICIAL, SIGNAL_END_OFFICIAL, EXIT_OFFICIAL = "2025-07-01", "2026-06-30", "2026-07-01"
START_EXTENDED = "2023-01-03"
TOP_K, MOMENTUM_LOOKBACK, VOL_LOOKBACK = 2, 60, 20
COST = {"account_money":100000.0, "lot_size":100, "max_participation":0.01,
        "commission_rate":0.00025, "min_commission":5.0, "transfer_rate":0.00001,
        "stamp_sell_rate":0.0005, "slippage_rate":0.0003}


def fee(notional, side):
    return max(COST["min_commission"], notional*COST["commission_rate"]) + notional*COST["transfer_rate"] + (notional*COST["stamp_sell_rate"] if side == "sell" else 0.0)


def limit_rate(code):
    s = str(code)
    if s.startswith(("300", "301", "688", "689")): return 0.195
    if s.startswith(("4", "8", "92")): return 0.295
    return 0.095


def load_price_panel():
    meta = json.loads((DATA / "meta.json").read_text(encoding="utf-8"))
    years = sorted(int(y) for y in meta["built_years"] if int(y) >= 2022)
    price_cols = ["open", "high", "low", "pre_close", "close", "vol", "adj_factor"]
    frames = []
    amount_col = None
    for year in years:
        pp = DATA / "prices" / f"year={year}" / "data.parquet"
        ap = DATA / "amount" / f"year={year}" / "data.parquet"
        if not pp.is_file() or not ap.is_file(): continue
        pschema = pq.ParquetFile(pp).schema_arrow.names
        aschema = pq.ParquetFile(ap).schema_arrow.names
        missing = [c for c in price_cols if c not in pschema]
        if missing: raise RuntimeError(f"price columns missing in {year}: {missing}")
        if amount_col is None:
            amount_col = next((c for c in aschema if c not in ("trade_date", "stock_code")), None)
            if amount_col is None: raise RuntimeError("amount parquet has no numeric column")
        p = pq.read_table(pp, columns=["trade_date", "stock_code", *price_cols],
                          filters=[("stock_code", "in", list(CODES))]).to_pandas()
        a = pq.read_table(ap, columns=["trade_date", "stock_code", amount_col],
                          filters=[("stock_code", "in", list(CODES))]).to_pandas()
        p["trade_date"] = pd.to_datetime(p["trade_date"]).dt.strftime("%Y-%m-%d")
        a["trade_date"] = pd.to_datetime(a["trade_date"]).dt.strftime("%Y-%m-%d")
        p["stock_code"] = p["stock_code"].astype(str)
        a["stock_code"] = a["stock_code"].astype(str)
        frames.append(p.merge(a, on=["trade_date", "stock_code"], validate="one_to_one"))
    if not frames: raise RuntimeError("no price/amount rows loaded")
    frame = pd.concat(frames, ignore_index=True)
    frame = frame.sort_values(["trade_date", "stock_code"]).drop_duplicates(["trade_date", "stock_code"], keep="last")
    days = np.asarray(sorted(frame["trade_date"].unique()), dtype=str)
    index = pd.MultiIndex.from_product([days, CODES], names=["trade_date", "stock_code"])
    frame = frame.set_index(["trade_date", "stock_code"]).reindex(index)
    shape = (len(days), len(CODES))
    arrays = {k: frame[k].to_numpy(dtype=float).reshape(shape) for k in [*price_cols, amount_col]}
    arrays["amount"] = np.nan_to_num(arrays.pop(amount_col), nan=0.0, posinf=0.0, neginf=0.0)
    adj = arrays["adj_factor"].copy()
    for c in range(adj.shape[1]):
        good = np.flatnonzero(np.isfinite(adj[:, c]) & (adj[:, c] > 0))
        if len(good):
            adj[:good[0], c] = adj[good[0], c]
            for i in range(good[0]+1, len(days)):
                if not np.isfinite(adj[i, c]) or adj[i, c] <= 0: adj[i, c] = adj[i-1, c]
    arrays["adj_factor"] = adj
    return days, np.asarray(CODES, dtype=str), arrays, meta, amount_col


def month_first_indices(days):
    months = np.asarray([str(d)[:7] for d in days])
    return np.flatnonzero(np.r_[True, months[1:] != months[:-1]])


def build_schedules(days, codes, arrays, start, signal_end):
    adjusted = arrays["close"] * arrays["adj_factor"]
    ret = adjusted[1:] / adjusted[:-1] - 1.0
    schedule_iv, schedule_equal, signals = {}, {}, []
    for si in month_first_indices(days):
        d = str(days[si])
        if d < start or d > signal_end or si < MOMENTUM_LOOKBACK or si + 1 >= len(days): continue
        exec_i = int(si + 1)
        if str(days[exec_i]) > signal_end: continue
        old, now = adjusted[si-MOMENTUM_LOOKBACK], adjusted[si]
        valid = np.isfinite(old) & (old > 0) & np.isfinite(now) & (now > 0)
        candidates = np.flatnonzero(valid)
        momentum = np.full(len(codes), np.nan)
        momentum[valid] = now[valid] / old[valid] - 1.0
        ranked = sorted(candidates.tolist(), key=lambda c: (-momentum[c], str(codes[c])))
        if len(ranked) < TOP_K: raise RuntimeError(f"fewer than Top2 valid banks at {d}")
        chosen = ranked[:TOP_K]
        # Fixed rank tilt: momentum rank 1 receives 60%; rank 2 receives 40%.
        weights = np.asarray([0.60, 0.40], dtype=float)
        selected = np.asarray(chosen, dtype=int)
        schedule_iv[exec_i] = {"selected":selected, "weights":weights.copy(), "ranked":ranked[:TOP_K]}
        schedule_equal[exec_i] = {"selected":selected.copy(), "weights":np.full(TOP_K, 1.0/TOP_K), "ranked":ranked[:TOP_K]}
        signals.append({"signal_date":d, "execution_date":str(days[exec_i]),
            "selected":[str(codes[c]) for c in chosen],
            "momentum_60d":{str(codes[c]):float(momentum[c]) for c in chosen},
            "momentum_rank":{str(codes[c]):int(rank+1) for rank,c in enumerate(chosen)},
            "momentum_rank_weights":{str(codes[c]):float(w) for c,w in zip(chosen,weights)},
            "weights_sum":float(weights.sum()), "weight_asof_date":d})
    return schedule_iv, schedule_equal, signals


def simulate(days, codes, arrays, schedule, final_close_exit=True):
    op, hi, lo, prev, close = (arrays[k] for k in ("open", "high", "low", "pre_close", "close"))
    vol, amount, adj = arrays["vol"], arrays["amount"], arrays["adj_factor"]
    nday, nc = op.shape
    shares = np.zeros(nc, dtype=float); cash = float(COST["account_money"]); prev_eq = cash
    lastmark = np.full(nc, np.nan); shares_prev = shares.copy(); lastmark_prev = lastmark.copy()
    ntr = 0; fee_total = 0.0; buys = sells = 0.0; blocked_entries = blocked_exits = 0
    curve, trades = [], []
    for i in range(nday):
        if i > 0:
            ratio = np.divide(adj[i], adj[i-1], out=np.ones(nc), where=np.isfinite(adj[i]) & (adj[i]>0) & np.isfinite(adj[i-1]) & (adj[i-1]>0))
            shares *= ratio
        is_final = i == nday-1
        use_close = is_final and final_close_exit
        raw_mark = close[i] if use_close else op[i]
        fallback = op[i] if use_close else close[i]
        mark = np.where(np.isfinite(raw_mark)&(raw_mark>0), raw_mark, np.where(np.isfinite(fallback)&(fallback>0), fallback, lastmark))
        mark = np.where(np.isfinite(mark)&(mark>0), mark, lastmark)
        if i > 0:
            prior_mv = np.nan_to_num(shares_prev * lastmark_prev, nan=0.0, posinf=0.0)
            prior_w = prior_mv / max(prev_eq, 1e-12)
            r = np.divide(mark*adj[i], lastmark_prev*adj[i-1], out=np.ones(nc),
                where=np.isfinite(mark)&(mark>0)&np.isfinite(adj[i])&(adj[i]>0)&np.isfinite(lastmark_prev)&(lastmark_prev>0)&np.isfinite(adj[i-1])&(adj[i-1]>0)) - 1.0
            bench = float(np.sum(prior_w*np.nan_to_num(r, nan=0.0)))
        else: bench = 0.0
        equity_pre = cash + float(np.nansum(shares*np.nan_to_num(mark, nan=0.0)))
        desired = schedule.get(i)
        if is_final: desired = {"selected":np.asarray([], dtype=int), "weights":np.asarray([], dtype=float)}
        exec_px = close[i] if use_close else op[i]
        if desired is not None:
            sel = desired["selected"]; wts = desired["weights"]
            if len(sel) != len(wts): raise RuntimeError("selected/weight length mismatch")
            if len(wts) and abs(float(np.sum(wts))-1.0) > 1e-12: raise RuntimeError("target weights do not sum to one")
            target_qty = {}
            for c,w in zip(sel,wts):
                px = exec_px[c]
                target_qty[int(c)] = math.floor(max(equity_pre,0.0)*float(w)/px/COST["lot_size"])*COST["lot_size"] if np.isfinite(px) and px>0 else 0
            sig_i = i-1
            for c in np.flatnonzero(shares>1e-7):
                want = float(target_qty.get(int(c),0)); delta = shares[c]-want
                if delta <= 1e-7: continue
                px = exec_px[c]
                if not (np.isfinite(px) and px>0 and np.isfinite(vol[i,c]) and vol[i,c]>0):
                    blocked_exits += 1; continue
                lim = limit_rate(codes[c]); one = np.isfinite(hi[i,c]) and np.isfinite(lo[i,c]) and abs(hi[i,c]-lo[i,c])<=1e-8
                down = one and np.isfinite(prev[i,c]) and px<=prev[i,c]*(1.0-lim+0.005)
                if down: blocked_exits += 1; continue
                capamt = float(amount[sig_i,c]) if sig_i>=0 else 0.0
                cap = math.floor(max(capamt,0.0)*COST["max_participation"]/px/COST["lot_size"])*COST["lot_size"]
                req = shares[c] if want<=1e-8 else math.floor(delta/COST["lot_size"])*COST["lot_size"]
                qty = min(req,cap)
                if want<=1e-8 and qty>=shares[c]-1e-7: qty=shares[c]
                else: qty=math.floor(qty/COST["lot_size"])*COST["lot_size"]
                if qty<=0: continue
                fill = px*(1.0-COST["slippage_rate"]); notional=qty*fill; f=fee(notional,"sell")
                cash += notional-f; shares[c] -= qty; fee_total += f; sells += notional; ntr += 1
                trades.append({"date":str(days[i]),"signal_date":str(days[sig_i]) if sig_i>=0 else "","code":str(codes[c]),"side":"sell","qty":float(qty),"reference_price":float(px),"fill_price":float(fill),"notional":float(notional),"fee":float(f),"exit_at_close":bool(use_close)})
            order = list(sel)
            for c0 in order:
                c=int(c0); want=float(target_qty.get(c,0)); delta=want-shares[c]
                if delta<COST["lot_size"]-1e-7: continue
                px=exec_px[c]
                if not (np.isfinite(px) and px>0 and np.isfinite(vol[i,c]) and vol[i,c]>0):
                    blocked_entries += 1; continue
                lim=limit_rate(codes[c]); one=np.isfinite(hi[i,c]) and np.isfinite(lo[i,c]) and abs(hi[i,c]-lo[i,c])<=1e-8
                up=one and np.isfinite(prev[i,c]) and px>=prev[i,c]*(1.0+lim-0.005)
                if up: blocked_entries += 1; continue
                capamt=float(amount[sig_i,c]) if sig_i>=0 else 0.0
                cap=math.floor(max(capamt,0.0)*COST["max_participation"]/px/COST["lot_size"])*COST["lot_size"]
                qty=min(math.floor(delta/COST["lot_size"])*COST["lot_size"],cap)
                fill=px*(1.0+COST["slippage_rate"])
                while qty>0:
                    notional=qty*fill; f=fee(notional,"buy")
                    if notional+f<=cash+1e-8: break
                    qty-=COST["lot_size"]
                if qty<=0: continue
                cash-=notional+f; shares[c]+=qty; fee_total+=f; buys+=notional; ntr+=1
                trades.append({"date":str(days[i]),"signal_date":str(days[sig_i]) if sig_i>=0 else "","code":str(codes[c]),"side":"buy","qty":float(qty),"reference_price":float(px),"fill_price":float(fill),"notional":float(notional),"fee":float(f),"exit_at_close":bool(use_close)})
        equity=cash+float(np.nansum(shares*np.nan_to_num(mark,nan=0.0)))
        invested=float(np.nansum(shares*np.nan_to_num(mark,nan=0.0)))
        daily=equity/prev_eq-1.0 if prev_eq>0 else 0.0
        curve.append({"date":str(days[i]),"equity":equity,"daily_return":daily,"matched_return":bench,"cash":cash,
                      "exposure":invested/max(equity,1e-12),"invested_value":invested,"holdings":int(np.count_nonzero(shares>1e-7)),"shares":shares.tolist(),"mark_type":"close_exit" if use_close else "open"})
        shares_prev=shares.copy(); lastmark_prev=mark.copy(); prev_eq=equity; lastmark=mark.copy()
    return {"curve":curve,"trades":trades,"fees":fee_total,"ntr":ntr,"buy_notional":buys,"sell_notional":sells,
            "blocked_entries":blocked_entries,"blocked_exits":blocked_exits,"ending_shares":shares.tolist(),"ending_cash":cash}


def summarize(curve, fees, result):
    d=pd.DataFrame(curve); daily=d.daily_return.to_numpy(float); matched=d.matched_return.to_numpy(float)
    eq=np.r_[COST["account_money"],d.equity.to_numpy(float)]
    matched_total=float(np.prod(1.0+matched)-1.0)
    return {"days":int(len(d)),"cumulative_net_return":float(d.equity.iloc[-1]/COST["account_money"]-1.0),
      "annualized_return":float(max(d.equity.iloc[-1]/COST["account_money"],1e-12)**(242.0/len(d))-1.0),
      "max_drawdown":float(np.min(eq/np.maximum.accumulate(eq)-1.0)),"strict_matched_gross_return":matched_total,
      "strict_matched_friction_net_minus_matched":float(d.equity.iloc[-1]/COST["account_money"]-1.0-matched_total),
      "avg_exposure":float(d.exposure.mean()),"max_exposure":float(d.exposure.max()),
      "average_holdings":float(d.holdings.mean()),"ending_equity":float(d.equity.iloc[-1]),"fees":float(fees),
      "trades":int(result["ntr"]),"blocked_entries":int(result["blocked_entries"]),"blocked_exits":int(result["blocked_exits"]),
      "minimum_cash":float(d.cash.min()),"maximum_cash":float(d.cash.max())}


def residual_audit(curve, arrays):
    errors=[]
    for i,row in enumerate(curve):
        px=arrays["close"][i] if row["mark_type"]=="close_exit" else arrays["open"][i]
        alt=arrays["open"][i] if row["mark_type"]=="close_exit" else arrays["close"][i]
        marks=np.where(np.isfinite(px)&(px>0),px,np.where(np.isfinite(alt)&(alt>0),alt,np.nan))
        errors.append(abs(float(row["equity"])-float(row["cash"])-float(np.nansum(np.asarray(row["shares"])*np.nan_to_num(marks,nan=0.0)))))
    return max(errors) if errors else 0.0


def subarrays(days, arrays, start, end):
    ix=np.flatnonzero((days>=start)&(days<=end))
    if not len(ix) or str(days[ix[0]])!=start or str(days[ix[-1]])!=end: raise RuntimeError(f"missing requested endpoint {start}..{end}")
    return ix, days[ix], {k:v[ix] for k,v in arrays.items()}


def run_window(all_days,codes,all_arrays,start,signal_end,exit_day,schedules,signals):
    ix,days,arrays=subarrays(all_days,all_arrays,start,exit_day)
    lo,hi=int(ix[0]),int(ix[-1])
    iv={i-lo:v for i,v in schedules[0].items() if lo<=i<=hi}
    equal={i-lo:v for i,v in schedules[1].items() if lo<=i<=hi}
    if not iv or not equal: raise RuntimeError("empty strategy schedule")
    base_i=1
    four_hold={base_i:{"selected":np.arange(len(codes),dtype=int),"weights":np.full(len(codes),1.0/len(codes))}}
    strat=simulate(days,codes,arrays,iv); strat_eq=simulate(days,codes,arrays,equal); four=simulate(days,codes,arrays,four_hold)
    s0=summarize(strat["curve"],strat["fees"],strat); s1=summarize(strat_eq["curve"],strat_eq["fees"],strat_eq); s2=summarize(four["curve"],four["fees"],four)
    curve=strat["curve"]
    audit={"session_days_including_exit":len(days),"signal_window_start":start,"signal_window_end":signal_end,"exit_only_day":exit_day,
      "signal_window_session_count":int(np.sum((days>=start)&(days<=signal_end))),"monthly_rebalances":len(iv),
      "signals_selected_same_as_equal_weight_60d_top2":all(list(map(int,schedules[0][i]["selected"]))==list(map(int,schedules[1][i]["selected"])) for i in schedules[0] if lo<=i<=hi),
      "all_target_weights_sum_to_one":all(abs(float(v["weights"].sum())-1.0)<1e-12 for i,v in schedules[0].items() if lo<=i<=hi),
      "causal_weight_asof_equals_signal_day":all(s["weight_asof_date"]==s["signal_date"] and s["execution_date"]>s["weight_asof_date"] for s in signals if start<=s["signal_date"]<=signal_end),
      "cash_nonnegative":bool(min(r["cash"] for r in curve)>=-1e-7),"cash_min":min(float(r["cash"]) for r in curve),
      "max_exposure":max(float(r["exposure"]) for r in curve),"blocked_entries":int(strat["blocked_entries"]),"blocked_exits":int(strat["blocked_exits"]),
      "final_exit_at_close":curve[-1]["mark_type"]=="close_exit","ending_positions":int(np.count_nonzero(np.asarray(strat["ending_shares"])>1e-7)),
      "final_liquidation_complete":bool(np.count_nonzero(np.asarray(strat["ending_shares"])>1e-7)==0),
      "max_abs_equity_cash_mark_residual":residual_audit(curve,arrays),
      "strict_matched_definition":"previous session realized shares valued at prior mark; current mark return on adjusted prices; excludes fees/slippage and is only a same-holdings execution-friction reference, not selection alpha"}
    if not audit["signals_selected_same_as_equal_weight_60d_top2"] or not audit["all_target_weights_sum_to_one"] or not audit["causal_weight_asof_equals_signal_day"] or not audit["cash_nonnegative"] or not audit["final_liquidation_complete"]:
        raise RuntimeError(f"critical audit failed: {audit}")
    # Make quarterly and annual compounded-return summaries from the same inclusive lifecycle.
    df=pd.DataFrame(curve); dates=pd.to_datetime(df.date)
    def grouped(keys, labels, mask=None):
        ans=[]
        for key in sorted(set(keys)):
            take=np.asarray(keys)==key
            if mask is not None: take &= mask
            if not take.any(): continue
            g=df.loc[take]
            net=float(np.prod(1+g.daily_return.to_numpy(float))-1); mat=float(np.prod(1+g.matched_return.to_numpy(float))-1)
            ans.append({labels:key,"days":int(len(g)),"net_return":net,"matched_return":mat,"matched_friction":net-mat})
        return ans
    quarters=np.asarray([str(x.to_period("Q")) for x in dates]); years=np.asarray([str(x.year) for x in dates])
    common_end=curve[-1]["date"]
    return {"strategy":s0,"equal_weight_60d_top2":s1,"four_bank_equal_weight_hold":s2,"audit":audit,
      "quarterly":grouped(quarters,"quarter"),"annual":grouped(years,"year"),
      "net_gain_pp_vs_equal_weight_top2":100*(s0["cumulative_net_return"]-s1["cumulative_net_return"]),
      "net_gain_pp_vs_four_bank_hold":100*(s0["cumulative_net_return"]-s2["cumulative_net_return"]),
      "signals":[s for s in signals if start<=s["signal_date"]<=signal_end],
      "curve":curve,"trades":strat["trades"],"equal_weight_curve":strat_eq["curve"],"equal_weight_trades":strat_eq["trades"],
      "four_bank_curve":four["curve"],"four_bank_trades":four["trades"]}


def main():
    t0=time.time(); print("Loading four-bank price and amount columns only",flush=True)
    days,codes,arrays,meta,amount_col=load_price_panel()
    latest=str(days[-1]); print(f"Loaded {len(days)} sessions {days[0]}..{latest}; universe={list(codes)}",flush=True)
    official_sched=build_schedules(days,codes,arrays,START_OFFICIAL,SIGNAL_END_OFFICIAL)
    official=run_window(days,codes,arrays,START_OFFICIAL,SIGNAL_END_OFFICIAL,EXIT_OFFICIAL,official_sched[:2],official_sched[2])
    for key in ("curve","equal_weight_curve","four_bank_curve"):
        for row in official[key]: row.pop("shares",None)
    # Persist the official lifecycle immediately before starting the extended window.
    official_record={"window":"242 signal sessions plus 2026-07-01 close exit",
        "strategy":official["strategy"],"equal_weight_60d_top2":official["equal_weight_60d_top2"],
        "four_bank_equal_weight_hold":official["four_bank_equal_weight_hold"],"audit":official["audit"],
        "quarterly":official["quarterly"],"annual":official["annual"],
        "net_gain_pp_vs_equal_weight_top2":official["net_gain_pp_vs_equal_weight_top2"],
        "net_gain_pp_vs_four_bank_hold":official["net_gain_pp_vs_four_bank_hold"],"signals":official["signals"]}
    (OUT/"official_result.json").write_text(json.dumps(official_record,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
    pd.DataFrame(official["curve"]).to_csv(OUT/"official_lifecycle_243d_daily.csv",index=False)
    pd.DataFrame(official["trades"]).to_csv(OUT/"official_lifecycle_243d_trades.csv",index=False)
    pd.DataFrame(official["equal_weight_curve"]).to_csv(OUT/"official_lifecycle_243d_equal_top2_daily.csv",index=False)
    pd.DataFrame(official["four_bank_curve"]).to_csv(OUT/"official_lifecycle_243d_four_bank_hold_daily.csv",index=False)
    (OUT/"official_lifecycle_243d_signals.json").write_text(json.dumps(official["signals"],ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
    pd.DataFrame(official["quarterly"]).to_csv(OUT/"official_quarterly.csv",index=False)
    (OUT/"official_audit.json").write_text(json.dumps(official["audit"],ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
    print(json.dumps({"official_net":official["strategy"]["cumulative_net_return"],
        "official_equal_top2":official["equal_weight_60d_top2"]["cumulative_net_return"],
        "official_four_bank_hold":official["four_bank_equal_weight_hold"]["cumulative_net_return"],
        "official_mdd":official["strategy"]["max_drawdown"],"official_fees":official["strategy"]["fees"],
        "official_audit":official["audit"]},indent=2),flush=True)
    print("Official result persisted; starting extended window.",flush=True)
    ext_sched=build_schedules(days,codes,arrays,START_EXTENDED,latest)
    extended=run_window(days,codes,arrays,START_EXTENDED,latest,latest,ext_sched[:2],ext_sched[2])
    for key in ("curve","equal_weight_curve","four_bank_curve"):
        for row in extended[key]: row.pop("shares",None)
    script_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result={"unit":"experiments2/V14 momentum-rank 60/40 allocation","built_at":time.strftime("%Y-%m-%d %H:%M:%S"),
      "data_built_at":meta.get("built_at"),"snapshot_last_day":latest,"price_only_amount_column":amount_col,"universe":list(codes),
      "rule":{"signal":"first observed session of each calendar month at T close",
        "selection":"fixed 60-session adjusted-close return Top2 ranking as V4",
        "allocation":"momentum rank 1 receives 60% and rank 2 receives 40%; fixed weights, no volatility scaling",
        "execution":"T+1 open; final lifecycle exit at closing price; audited cash-only ledger; 100-share lots, 1% prior signal-day amount capacity, one-price limit locks, 3bp slippage, commission/transfer/sell stamp fees",
        "no_training":True,"no_parameter_sweep":True,"cash_constraints":COST,
        "interpretation":"strict matched friction is same-holdings gross reference minus trading costs/slippage; not selection alpha"},
      "official_lifecycle_242_signal_days_plus_20260701_close_exit":official_record,
      "extended_20230103_to_latest":{k:v for k,v in extended.items() if k not in ("curve","trades","equal_weight_curve","equal_weight_trades","four_bank_curve","four_bank_trades")},
      "runtime_seconds":time.time()-t0,"script_sha256":script_hash}
    (OUT/"result.json").write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
    for name,payload in (("extended",extended),):
        for suffix,key in (("daily","curve"),("trades","trades"),("equal_top2_daily","equal_weight_curve"),("equal_top2_trades","equal_weight_trades"),("four_bank_hold_daily","four_bank_curve"),("four_bank_hold_trades","four_bank_trades")):
            pd.DataFrame(payload[key]).to_csv(OUT/f"{name}_{suffix}.csv",index=False)
        (OUT/f"{name}_signals.json").write_text(json.dumps(payload["signals"],ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
    pd.DataFrame(extended["quarterly"]).to_csv(OUT/"extended_quarterly.csv",index=False)
    pd.DataFrame(extended["annual"]).to_csv(OUT/"extended_annual.csv",index=False)
    pd.DataFrame([{"window":"official",**x} for x in official["quarterly"]]+
                 [{"window":"extended",**x} for x in extended["quarterly"]]).to_csv(OUT/"quarterly.csv",index=False)
    (OUT/"audits.json").write_text(json.dumps({"official":official["audit"],"extended":extended["audit"]},ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
    protocol={"objective":"fixed four-bank monthly 60-session momentum Top2 with rank-based 60/40 allocation",
      "official_signal_window":[START_OFFICIAL,SIGNAL_END_OFFICIAL],"official_exit_only_day":EXIT_OFFICIAL,
      "official_signal_sessions":242,"extended_window":[START_EXTENDED,latest],"signal_rule":result["rule"],
      "source_snapshot":str(DATA/"meta.json"),"amount_column":amount_col,"script_sha256":script_hash,
      "price_loader":"parquet filters only four bank codes and reads price plus amount; no model/features loaded"}
    (OUT/"protocol.json").write_text(json.dumps(protocol,ensure_ascii=False,indent=2),encoding="utf-8")
    report=["# V14 momentum-rank 60/40 allocation candidate","",
      f"Snapshot: {latest}; this run loaded only four-bank prices and traded amount. No model retraining or parameter sweep.","",
      "## Fixed rule","","At each first trading session of the month, rank the four banks by 60-session adjusted-close return. Hold the same Top2 as V4; allocate 60% to the stronger momentum name and 40% to the second name. Execute at the next open and rebalance monthly. Liquidate at the designated final close. The cash ledger enforces lots, liquidity participation, price-limit locks and transaction costs.","",
      "## Official lifecycle","","Includes 242 signal-window sessions from 2025-07-01 through 2026-06-30 and the 2026-07-01 closing liquidation. The result compares against monthly equal-weight 60d Top2 and equal four-bank hold. See official_result.json, official_quarterly.csv and official_audit.json.","",
      "## Extended lifecycle","","Runs from 2023-01-03 through the latest available session. See extended_quarterly.csv, extended_annual.csv, result.json and audits.json.","",
      "Strict matched friction is net return minus a no-fee/no-slippage return on the same realized holdings path; it describes execution friction, not stock-selection alpha. This fixed candidate is exploratory and is not evidence of stable improvement.","",
      f"Runner SHA-256: {script_hash}"]
    o=official_record; e=result["extended_20230103_to_latest"]
    report += ["## Results", "", "| Window | 60/40 net | 50/50 Top2 net | Four-bank hold net | vs Top2 | MDD | Fees |", "|---|---:|---:|---:|---:|---:|---:|"]
    report.append(f"| Official lifecycle | {o['strategy']['cumulative_net_return']:.2%} | {o['equal_weight_60d_top2']['cumulative_net_return']:.2%} | {o['four_bank_equal_weight_hold']['cumulative_net_return']:.2%} | {o['net_gain_pp_vs_equal_weight_top2']:+.2f} pp | {o['strategy']['max_drawdown']:.2%} | ¥{o['strategy']['fees']:.2f} |")
    report.append(f"| Extended | {e['strategy']['cumulative_net_return']:.2%} | {e['equal_weight_60d_top2']['cumulative_net_return']:.2%} | {e['four_bank_equal_weight_hold']['cumulative_net_return']:.2%} | {e['net_gain_pp_vs_equal_weight_top2']:+.2f} pp | {e['strategy']['max_drawdown']:.2%} | ¥{e['strategy']['fees']:.2f} |")
    report += ["", "## Quarterly net returns", "", "| Window | Quarter | 60/40 net | Same-hold matched return | Friction |", "|---|---|---:|---:|---:|"]
    for row in official["quarterly"]:
        report.append(f"| Official | {row['quarter']} | {row['net_return']:.2%} | {row['matched_return']:.2%} | {row['matched_friction']:.2%} |")
    for row in extended["quarterly"]:
        report.append(f"| Extended | {row['quarter']} | {row['net_return']:.2%} | {row['matched_return']:.2%} | {row['matched_friction']:.2%} |")
    report += ["", "## Extended annual net returns", "", "| Year | 60/40 net | Same-hold matched return | Friction |", "|---|---:|---:|---:|"]
    for row in extended["annual"]:
        report.append(f"| {row['year']} | {row['net_return']:.2%} | {row['matched_return']:.2%} | {row['matched_friction']:.2%} |")
    report += ["", f"Official vs four-bank hold: {o['net_gain_pp_vs_four_bank_hold']:+.2f} pp; extended vs four-bank hold: {e['net_gain_pp_vs_four_bank_hold']:+.2f} pp.", "The return advantage comes with a larger maximum drawdown than both comparison portfolios in these windows; treat this as a candidate, not a stable improvement claim.", ""]
    (OUT/"REPORT.md").write_text("\n".join(report)+"\n",encoding="utf-8")
    print(json.dumps({"official_net":official["strategy"]["cumulative_net_return"],
      "official_vs_top2_pp":official["net_gain_pp_vs_equal_weight_top2"],
      "official_vs_four_pp":official["net_gain_pp_vs_four_bank_hold"],
      "extended_net":extended["strategy"]["cumulative_net_return"],
      "extended_vs_top2_pp":extended["net_gain_pp_vs_equal_weight_top2"],
      "extended_vs_four_pp":extended["net_gain_pp_vs_four_bank_hold"],
      "runtime_seconds":result["runtime_seconds"]},indent=2),flush=True)

if __name__=="__main__": main()
