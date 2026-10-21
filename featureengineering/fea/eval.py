"""统一日截面评价：全部期限、成对有效值、逐日证据、日期 HAC 不确定性。

纯数组 API 供项目内研究复用。生产评价按年读盘，只保留逐日标量，
不聚合所有历史因子面板，也不根据统一 IC 阈值自动淘汰。无需 scipy。
"""
from __future__ import annotations
import datetime as dt
import json
import math
import os
from pathlib import Path
import time
import uuid
import numpy as np
import pandas as pd
from .targets import HORIZONS, LABEL_CONTRACT, open_to_open_labels

EVALUATION_CONTRACT = "daily_complete_pairs_all_horizons_v2"
MODEL_VALUE_CONTRACT = "zscore_win1_99_v1"


def _pearson(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    good = np.isfinite(a) & np.isfinite(b)
    a, b = a[good], b[good]
    if a.size < 2:
        return np.nan
    a, b = a-a.mean(), b-b.mean()
    den = math.sqrt(float(a@a)*float(b@b))
    return float(a@b/den) if den > 0 else np.nan


def _spearman(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    good = np.isfinite(a) & np.isfinite(b)
    return _pearson(pd.Series(a[good]).rank(method="average"),
                    pd.Series(b[good]).rank(method="average"))


def _spearman_from_rank(a, b):
    """兼容旧调用：两个已取秩数组的 Pearson。"""
    return _pearson(a, b)


def model_value_panel(values, higher_is_better=True, winsor=(.01, .99), clip=10.):
    """逐日缩尾→zscore→截幅→方向；保留缺失。"""
    v = np.asarray(values, dtype=np.float64)
    if v.ndim != 2:
        raise ValueError("values must have (T,C) shape")
    out = np.full_like(v, np.nan)
    for t, row in enumerate(v):
        good = np.isfinite(row)
        if not good.any():
            continue
        lo, hi = np.quantile(row[good], winsor)
        x = np.clip(row[good], lo, hi)
        sd = x.std(ddof=0)
        out[t, good] = np.clip((x-x.mean())/sd, -clip, clip) if sd > 0 else 0.
    return out if higher_is_better else -out


def _bins(x, count=10):
    """不拆并列值；保存真实箱数。"""
    unique = np.unique(x)
    if unique.size <= count:
        groups = np.searchsorted(unique, x)
    else:
        raw = pd.qcut(pd.Series(x), count, labels=False, duplicates="drop").to_numpy()
        if not np.isfinite(raw).all():
            return np.zeros(len(x), dtype=int), int(bool(len(x)))
        _, groups = np.unique(raw, return_inverse=True)
    return np.asarray(groups, dtype=int), int(groups.max())+1 if len(groups) else 0


def evaluate_panel(values, labels_by_horizon, dates, min_n=100, bins=10,
                   factor="candidate", eligible=None):
    """输入须对齐完整交易日和股票轴；eligible 只可由信号日已知条件构造。"""
    values, dates = np.asarray(values, dtype=float), np.asarray(dates)
    if values.ndim != 2 or dates.size != values.shape[0]:
        raise ValueError("values/dates axis mismatch")
    if len(np.unique(dates)) != dates.size or (dates.size > 1 and np.any(dates[1:] <= dates[:-1])):
        raise ValueError("dates must be unique and strictly increasing")
    if min_n < 3 or bins < 2:
        raise ValueError("min_n>=3 and bins>=2 required")
    elig = np.ones(values.shape, bool) if eligible is None else np.asarray(eligible, bool)
    if elig.shape != values.shape:
        raise ValueError("eligible shape mismatch")
    records = []
    autocorrelation=[]
    for t in range(len(values)):
        common=(np.isfinite(values[t]) & np.isfinite(values[t-1]) & elig[t] & elig[t-1]) if t else np.zeros(values.shape[1],bool)
        autocorrelation.append(_spearman(values[t,common],values[t-1,common]) if common.sum()>=min_n else np.nan)
    for h, lab in sorted(labels_by_horizon.items()):
        lab = np.asarray(lab, dtype=float)
        if lab.shape != values.shape:
            raise ValueError(f"horizon {h}: labels shape mismatch")
        for t, date in enumerate(dates):
            v, y = values[t], lab[t]
            finite = np.isfinite(v) & elig[t]
            valid = finite & np.isfinite(y)
            n, nv = int(valid.sum()), int(finite.sum())
            x, target = v[valid], y[valid]
            ic = ric = spread = mono = np.nan
            n_bins, counts, returns = 0, [], []
            reason = "insufficient_pairs" if n < min_n else ""
            if n >= min_n:
                ic, ric = _pearson(x, target), _spearman(x, target)
                if not np.isfinite(ic):
                    reason = "constant_value_or_target"
                group, n_bins = _bins(x, bins)
                for k in range(n_bins):
                    keep = group == k
                    counts.append(int(keep.sum()))
                    returns.append(float(target[keep].mean()))
                if n_bins >= 2:
                    spread, mono = returns[-1]-returns[0], _spearman(np.arange(n_bins), returns)
            records.append({"factor": str(factor), "trade_date": str(date), "horizon": int(h),
                "primary": int(h) in (1,5), "score_head": None, "n_pool": values.shape[1],
                "n_eligible": int(elig[t].sum()), "pairs": n,
                "coverage": nv/max(values.shape[1],1),
                "eligible_coverage": nv/max(int(elig[t].sum()),1),
                "label_coverage": float(np.isfinite(y).mean()) if y.size else 0.,
                "ic": ic, "rankic": ric, "unique_values": int(np.unique(v[finite]).size),
                "zero_fraction": float(np.mean(v[finite] == 0)) if nv else np.nan,
                "inf_count": int(np.isinf(v).sum()), "actual_bins": n_bins,
                "bin_counts": counts, "bin_returns": returns, "spread": spread,
                "monotonicity": mono, "ac1":autocorrelation[t], "reason": reason})
    return pd.DataFrame(records)


def _daily_corr(val, lab, ts, min_n):
    daily = evaluate_panel(val, {1:lab}, ts, min_n)
    good = np.isfinite(daily.ic)
    return daily.loc[good,"ic"].to_numpy(), daily.loc[good,"rankic"].to_numpy(), int(good.sum())


def _deciles(val, lab, min_n):
    acc, cnt = np.zeros(10), np.zeros(10)
    for v, y in zip(val, lab):
        good = np.isfinite(v) & np.isfinite(y)
        if good.sum() < min_n:
            continue
        groups, n = _bins(v[good], 10)
        for k in range(n):
            acc[k] += y[good][groups==k].mean()
            cnt[k] += 1
    return acc, cnt


def mean_uncertainty(values, block_days=20, horizon=1):
    """Newey-West 均值标准误；缺失日期保留位置，最少考虑标签重叠。"""
    x = np.asarray(values, dtype=float)
    good, n = np.isfinite(x), int(np.isfinite(x).sum())
    mean = float(x[good].mean()) if n else np.nan
    lag = min(max(0,x.size-1), max(int(horizon)-1,int(block_days)-1))
    result = {"mean":mean,"days":n,"hac_lag":lag,"hac_se":None,"ci95_low":None,"ci95_high":None}
    if n < 3:
        return result
    z = np.where(good,x-mean,0.)
    variance = float(z@z)/n
    for k in range(1,lag+1):
        variance += 2*(1-k/(lag+1))*float(z[k:]@z[:-k])/n
    se = math.sqrt(max(variance,0.)/n)
    return {**result,"hac_se":se,"ci95_low":mean-1.96*se,"ci95_high":mean+1.96*se}


def summarize_daily(daily, block_days=20):
    out = []
    if daily.empty:
        return out
    for (factor,h), g in daily.groupby(["factor","horizon"], sort=True):
        g = g.sort_values("trade_date",kind="stable")
        row = {"name":str(factor),"horizon":int(h),"primary":int(h) in (1,5),
            "dates":len(g),"first_date":str(g.trade_date.iloc[0]),"last_date":str(g.trade_date.iloc[-1]),
            "cov":float(g.coverage.mean()),"coverage_median":float(g.coverage.median()),
            "coverage_min":float(g.coverage.min()),"pairs_min":int(g.pairs.min()),
            "pairs_median":float(g.pairs.median()),"sparse_days":int((g.unique_values<=20).sum()),
            "inf_count":int(g.inf_count.sum())}
        for metric in ("ic","rankic"):
            values = g[metric].to_numpy(dtype=float)
            stats = mean_uncertainty(values,block_days,h)
            valid = values[np.isfinite(values)]
            sd = valid.std(ddof=1) if valid.size>1 else np.nan
            ratio = stats["mean"]/sd if sd>0 else np.nan
            row.update({metric:stats["mean"],f"{metric}_uncertainty":stats,
                        f"{metric}ir":ratio,f"{metric}ir_annualized_244":ratio*math.sqrt(244)})
        row["days"] = row["ic_uncertainty"]["days"]
        for metric in ("spread","monotonicity","ac1"):
            v = g[metric].to_numpy(dtype=float)
            row[metric] = float(np.nanmean(v)) if np.isfinite(v).any() else None
        row["actual_bins_median"] = float(g.actual_bins.median())
        mature=g[g.label_coverage>0]
        row["last_mature_signal_date"] = str(mature.trade_date.iloc[-1]) if len(mature) else None
        row["by_year"] = [{"year":str(year),"dates":len(part),
                           "ic_days":int(part.ic.notna().sum()),"rankic_days":int(part.rankic.notna().sum()),
                           "ic":float(part.ic.mean()),"rankic":float(part.rankic.mean()),
                           "coverage_mean":float(part.coverage.mean())}
                          for year,part in g.groupby(g.trade_date.str[:4],sort=True)]
        row["flags"] = ["ERROR:INF"] if row["inf_count"] else []
        if row["coverage_median"] < .3:
            row["flags"].append("REVIEW:LOW_COVERAGE")
        if row["sparse_days"]:
            row["flags"].append("DIAGNOSTIC:SPARSE_OR_DISCRETE")
        row["note"] = "Equal-weight dates; complete pairs; HAC normal CI; no automatic alpha qualification"
        out.append(row)
    return out


def paired_comparison(candidate_daily, baseline_daily, metric="rankic", block_days=20):
    keys = ["trade_date","horizon"]
    for frame in (candidate_daily,baseline_daily):
        if frame.duplicated(keys).any():
            raise ValueError("paired comparison requires one factor with unique day/horizon keys")
    joined = candidate_daily[keys+[metric]].merge(baseline_daily[keys+[metric]],on=keys,
                             suffixes=("_candidate","_baseline"),validate="one_to_one")
    return [{"horizon":int(h),"metric":metric,"paired_dates":len(g),
             "difference":mean_uncertainty((g.sort_values("trade_date")[f"{metric}_candidate"]-
                                            g.sort_values("trade_date")[f"{metric}_baseline"]),block_days,int(h)),
             "stock_mask":"caller must ensure common stock pairs"}
            for h,g in joined.groupby("horizon",sort=True)]


def evaluate_pair(values_a, values_b, labels_by_horizon, dates, min_n=100,
                  factor_a="candidate", factor_b="baseline", block_days=20):
    a,b = np.asarray(values_a,dtype=float),np.asarray(values_b,dtype=float)
    if a.shape != b.shape:
        raise ValueError("candidate/baseline shape mismatch")
    common = np.isfinite(a)&np.isfinite(b)
    aa = evaluate_panel(np.where(common,a,np.nan),labels_by_horizon,dates,min_n,factor=factor_a)
    bb = evaluate_panel(np.where(common,b,np.nan),labels_by_horizon,dates,min_n,factor=factor_b)
    comparison = paired_comparison(aa,bb,block_days=block_days)
    for row in comparison:
        row["stock_mask"] = "same finite candidate/baseline and mature label pairs per day/horizon"
    return aa,bb,comparison


def _json_safe(value):
    if isinstance(value,dict):
        return {str(k):_json_safe(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value,np.integer):
        return int(value)
    if isinstance(value,(float,np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


def _write_json(path,payload):
    path = Path(path)
    tmp = path.with_suffix(f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(_json_safe(payload),ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
    os.replace(tmp,path)


def _new_run(outdir):
    run_id = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")+"_"+uuid.uuid4().hex[:10]
    dest = Path(outdir)/"runs"/run_id
    dest.mkdir(parents=True,exist_ok=False)
    return run_id,dest


def write_evaluation(daily,outdir,run_metadata=None,block_days=20):
    run_id,dest = _new_run(outdir)
    metadata = {**(run_metadata or {}),"run_id":run_id,"evaluation_contract":EVALUATION_CONTRACT,
                "label_contract":LABEL_CONTRACT,"block_days":int(block_days)}
    daily = daily.copy()
    daily["run_id"] = run_id
    daily.to_parquet(dest/"daily.parquet",index=False,compression="zstd")
    _write_json(dest/"summary.json",{"run":metadata,"records":summarize_daily(daily,block_days)})
    _write_json(dest/"run.json",{**metadata,"status":"complete","daily_rows":len(daily)})
    _write_json(Path(outdir)/"latest.json",{"run_id":run_id,"summary":str(dest/"summary.json")})
    return dest


def _place_frame(frame,panel,codes):
    from .dates import series_to_int
    if frame.duplicated(["trade_date","stock_code"]).any():
        raise ValueError("duplicate factor date/stock keys")
    index = pd.Index(codes).get_indexer(frame.stock_code)
    if (index<0).any():
        raise ValueError("factor contains out-of-universe stocks")
    dates=series_to_int(frame.trade_date)
    if not np.isin(dates,panel.dates).all():
        raise ValueError("factor contains dates outside requested trading-day panel")
    return panel.place(index,dates,frame.value.to_numpy(dtype=float))


def validate_frame_grid(frame,dates,codes):
    """评价前拒绝缺行/重复键/错轴；NaN 值允许，缺股票行不允许。"""
    from .dates import series_to_int
    dates,codes=np.asarray(dates),np.asarray(codes)
    if len(frame)!=len(dates)*len(codes):
        raise ValueError("factor row count does not match task date/stock grid")
    if frame.duplicated(["trade_date","stock_code"]).any():
        raise ValueError("duplicate factor date/stock keys")
    if not frame.stock_code.isin(codes).all():
        raise ValueError("out-of-universe factor stock")
    days=series_to_int(frame.trade_date)
    if not np.array_equal(np.sort(np.unique(days)),dates):
        raise ValueError("factor date axis does not match task grid")
    if not frame.groupby("trade_date",observed=True).size().eq(len(codes)).all():
        raise ValueError("factor stock axis incomplete on some dates")


def evaluate_all(args,cfg):
    """main.py eval；年块写逐日 Parquet，再按一个因子读标量汇总。"""
    import pyarrow as pa
    import pyarrow.parquet as pq
    from . import store
    from .dates import int_to_str
    from .engine import Engine
    from .panel import Panel
    from .prices import PriceLayer
    from .release import production_specs
    from .upstream import Upstream
    started = time.monotonic()
    specs = [s for s in production_specs(cfg) if not s.is_label and not s.is_market]
    if getattr(args,"factors",None):
        specs = [s for s in specs if s.name in set(args.factors)]
    if getattr(args,"group",None):
        specs = [s for s in specs if s.group in set(args.group)]
    horizons = sorted(set(int(x) for x in str(getattr(args,"horizons","1,3,5,10,20")).split(",")))
    if not horizons or any(h<=0 for h in horizons):
        raise ValueError("horizons must contain positive integers")
    years = sorted({y for s in specs for y in store.factor_years(cfg.factors_dir,s.name)})
    if getattr(args,"years",None):
        years = [y for y in years if args.years[0]<=y<=args.years[-1]]
    if not specs or not years:
        print("没有匹配的股票因子产物；市场标量需状态/交互验证。",flush=True)
        return 1
    engine,end = Engine(cfg),None
    end = engine.baseline_last_day()
    outdir = Path(getattr(args,"out",None) or cfg.state_dir/"eval")
    if not outdir.is_absolute():
        outdir = cfg.root/outdir
    if not outdir.resolve().is_relative_to(cfg.root.resolve()):
        raise ValueError("evaluation output must remain within project")
    block_days = int(getattr(args,"block_days",20))
    run_id,dest = _new_run(outdir)
    writer,label_sources = None,{}
    try:
        for year in years:
            days = engine.cal.between(max(year*10000+101,int(cfg.default_start.replace("-",""))),min(year*10000+1231,end))
            if not days.size:
                continue
            pos = min(int(np.searchsorted(engine.cal.days,days[-1]))+max(horizons)+1,len(engine.cal.days)-1)
            panel = Panel(engine.cal.between(int(days[0]),int(engine.cal.days[pos])),engine.codes)
            rows = panel.day_mask(days)
            labels,px = {},None
            for h in horizons:
                name = f"label_ret_{h}d"
                if year in store.factor_years(cfg.factors_dir,name):
                    label_frame=store.read_year(cfg.factors_dir,name,year)
                    validate_frame_grid(label_frame,days,engine.codes)
                    labels[h] = _place_frame(label_frame,panel,engine.codes)[rows]
                    label_sources[f"{year}:{h}"] = "formal_store"
                else:
                    if px is None:
                        px = PriceLayer(Upstream(cfg.upstream,cfg.root),cfg,engine.cal,engine.codes)
                    labels[h] = open_to_open_labels(px.panel(panel,"hfq_open"),px.panel(panel,"traded"),h)[rows]
                    label_sources[f"{year}:{h}"] = "shared_formal_contract_fallback"
            for spec in specs:
                frame = store.read_year(cfg.factors_dir,spec.name,year)
                if frame.empty:
                    continue
                validate_frame_grid(frame,days if spec.align_fill is not None else days[days>=spec.start_int(cfg)],engine.codes)
                values = _place_frame(frame,panel,engine.codes)[rows]
                active = days>=spec.start_int(cfg)
                daily = evaluate_panel(values[active],{h:l[active] for h,l in labels.items()},
                                       [int_to_str(int(d)) for d in days[active]],cfg.min_cross_section,factor=spec.name)
                if daily.empty:
                    continue
                daily["run_id"] = run_id
                table = pa.Table.from_pandas(daily,preserve_index=False)
                for col,typ in (("bin_counts",pa.list_(pa.int64())),("bin_returns",pa.list_(pa.float64()))):
                    table = table.set_column(table.schema.get_field_index(col),col,pa.array(daily[col].tolist(),type=typ))
                if writer is None:
                    writer = pq.ParquetWriter(dest/"daily.parquet",table.schema,compression="zstd")
                writer.write_table(table.cast(writer.schema))
            del panel,labels,px
            print(f"评价 {year} 年完成；期限 {horizons}",flush=True)
        if writer is not None:
            writer.close()
            writer = None
        summaries = []
        if (dest/"daily.parquet").exists():
            for spec in specs:
                daily = pd.read_parquet(dest/"daily.parquet",filters=[("factor","==",spec.name)])
                records = summarize_daily(daily,block_days)
                for record in records:
                    record.update(group=spec.group,start=spec.resolved_start(cfg),recipe=engine._recipe(spec))
                summaries.extend(records)
        metadata = {"run_id":run_id,"evaluation_contract":EVALUATION_CONTRACT,"label_contract":LABEL_CONTRACT,
                    "label_sources":label_sources,"horizons":horizons,"primary_horizons":[h for h in horizons if h in (1,5)],
                    "years":years,"as_of":int_to_str(end),"n_pool":len(engine.codes),"block_days":block_days,
                    "value_representation":"stored_raw_value"}
        _write_json(dest/"summary.json",{"run":metadata,"records":summaries})
        _write_json(dest/"run.json",{**metadata,"status":"complete","elapsed_seconds":time.monotonic()-started})
        _write_json(outdir/"latest.json",{"run_id":run_id,"summary":str(dest/"summary.json")})
        for record in summaries:
            if record["primary"]:
                print(f"{record['name']} h={record['horizon']} days={record['days']} coverage={record['cov']:.1%} RankIC={record['rankic']}",flush=True)
        print(f"不可变评价报告：{dest/'summary.json'}",flush=True)
        return int(any(r["inf_count"] for r in summaries))
    except BaseException as exc:
        _write_json(dest/"run.json",{"run_id":run_id,"status":"failed","error":repr(exc)})
        raise
    finally:
        if writer is not None:
            writer.close()
