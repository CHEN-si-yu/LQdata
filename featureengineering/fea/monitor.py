"""日更结束后的有限内存质量闸门。

历史分区仅查页脚合同/行数，最近交易日逐键检查。历史主键唯一性仍由
完整 check 验证；本工具不把页脚正确误称为全历史数值/PIT/收益证明。
"""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
import datetime as dt
import json
from pathlib import Path
import time
import uuid
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from .dates import int_to_str
from .eval import _json_safe, _write_json, evaluate_panel, summarize_daily
from .store import COLUMNS, MARKET_COLUMNS, factor_years, year_path


def inspect_tail(frame, expected_days, codes, spec, cfg):
    """纯函数，按固定股票池检查尾部的键、数值和已声明对齐区间。"""
    errors,warnings=[],[]
    expected_dates=[int_to_str(int(d)) for d in expected_days]
    columns=MARKET_COLUMNS if spec.is_market else COLUMNS
    result={"errors":errors,"warnings":warnings,"daily":[]}
    if list(frame.columns)!=columns:
        errors.append("schema_columns")
        return result
    keys=["trade_date"] if spec.is_market else ["trade_date","stock_code"]
    if frame.duplicated(keys).any():
        errors.append("duplicate_keys")
    if frame.trade_date.isna().any() or (not spec.is_market and frame.stock_code.isna().any()):
        errors.append("null_keys")
    dates=frame.trade_date.astype(str)
    if sorted(dates.unique())!=expected_dates:
        errors.append("tail_date_axis")
    if not spec.is_market and not frame.stock_code.isin(codes).all():
        errors.append("out_of_universe")
    sizes=frame.groupby("trade_date",observed=True).size()
    expected_width=1 if spec.is_market else len(codes)
    if not sizes.eq(expected_width).all():
        errors.append("tail_stock_axis" if not spec.is_market else "market_not_one_row_per_day")
    value=frame.value.to_numpy(dtype=float)
    if np.isinf(value).any():
        errors.append("infinite_value")
    if not spec.is_market:
        ranks=frame["rank"].to_numpy(dtype=float)
        if np.isinf(ranks).any() or ((ranks<0)|(ranks>1)).any():
            errors.append("rank_bounds")
        if spec.is_label and np.isfinite(ranks).any():
            errors.append("label_rank_not_missing")
        prefix=dates<spec.resolved_start(cfg)
        if prefix.any() and np.isfinite(ranks[prefix]).any():
            errors.append("alignment_rank_not_missing")
    for date,group in frame.groupby("trade_date",sort=True,observed=True):
        v=group.value.to_numpy(dtype=float)
        good=np.isfinite(v)
        n=int(good.sum())
        record={"trade_date":str(date),"rows":len(group),"finite":n,
                "coverage":n/max(expected_width,1),"unique_values":int(np.unique(v[good]).size),
                "zero_fraction":float(np.mean(v[good]==0)) if n else None,
                "q01":float(np.quantile(v[good],.01)) if n else None,
                "q50":float(np.median(v[good])) if n else None,
                "q99":float(np.quantile(v[good],.99)) if n else None}
        result["daily"].append(record)
        aligned=str(date)<spec.resolved_start(cfg)
        if aligned:
            fill=spec.align_fill
            if fill is None:
                errors.append("unexpected_alignment_prefix")
            elif np.isnan(float(fill)) and n:
                errors.append("alignment_value_expected_missing")
            elif not np.isnan(float(fill)) and not np.all(v==float(np.float32(fill))):
                errors.append("alignment_value_mismatch")
        elif spec.is_market and not n:
            errors.append(f"market_active_value_missing:{date}")
        elif not spec.is_label and not n:
            warnings.append(f"all_missing_active_day:{date}")
        elif not spec.is_label and not spec.is_market and record["unique_values"]<=1:
            warnings.append(f"constant_cross_section:{date}")
    active=[r for r in result["daily"] if r["trade_date"]>=spec.resolved_start(cfg)]
    if len(active)>=3 and not spec.is_label:
        baseline=np.median([r["coverage"] for r in active[:-1]])
        if baseline>.05 and active[-1]["coverage"]<baseline*.5:
            warnings.append("coverage_drop_vs_previous_tail")
    result["errors"]=sorted(set(errors))
    result["warnings"]=sorted(set(warnings))
    return result


def inspect_footer(path,expected_rows,market=False):
    """读取一份页脚；不声称页脚可证明历史主键唯一。"""
    errors=[]
    pf=pq.ParquetFile(path)
    schema=pf.schema_arrow
    names=MARKET_COLUMNS if market else COLUMNS
    if schema.names!=names:
        errors.append("schema_columns")
    else:
        for name in names:
            typ=schema.field(name).type
            if name in ("trade_date","stock_code") and not (pa.types.is_string(typ) or pa.types.is_large_string(typ)):
                errors.append(f"schema_type:{name}")
            if name in ("value","rank") and typ!=pa.float32():
                errors.append(f"schema_type:{name}")
    if pf.metadata.num_rows!=expected_rows:
        errors.append("annual_row_count")
    for rg in range(pf.metadata.num_row_groups):
        for column in ("value","rank"):
            if column not in schema.names:
                continue
            statistics=pf.metadata.row_group(rg).column(schema.get_field_index(column)).statistics
            if statistics and statistics.has_min_max:
                try:
                    if np.isinf(float(statistics.min)) or np.isinf(float(statistics.max)):
                        errors.append(f"historical_footer_infinite:{column}")
                except (ValueError,TypeError):
                    pass
    return {"rows":pf.metadata.num_rows,"expected_rows":int(expected_rows),"errors":sorted(set(errors))}


def _rolling_reports(cfg,engine,specs,end,settings):
    """显式启用的有限窗口 IC；仅读本项目正式成熟标签，缺失不填零。"""
    from .eval import _place_frame
    from .panel import Panel
    from . import store
    windows=[int(w) for w in settings.get("rolling_windows",[60,120])]
    if not windows or max(windows)>504 or min(windows)<3:
        raise ValueError("rolling windows must be within [3,504] trading days")
    pos=int(np.searchsorted(engine.cal.days,end,side="right"))
    days=engine.cal.days[max(0,pos-max(windows)):pos]
    if not days.size:
        return []
    panel=Panel(days,engine.codes)
    labels={}
    years=sorted(set(int(d)//10000 for d in days))
    first,last=int_to_str(int(days[0])),int_to_str(int(days[-1]))
    for h in (1,3,5,10,20):
        frame=store.read_factor(cfg.factors_dir,f"label_ret_{h}d",years)
        if not frame.empty:
            frame=frame[(frame.trade_date>=first)&(frame.trade_date<=last)]
        labels[h]=_place_frame(frame,panel,engine.codes) if not frame.empty else np.full((panel.T,panel.C),np.nan)
    reports=[]
    # 默认需要明确指定股票因子；避免日更暗中评价全部历史。
    wanted=set(settings.get("rolling_factors",[]))
    for spec in specs:
        if spec.name not in wanted or spec.is_label or spec.is_market:
            continue
        frame=store.read_factor(cfg.factors_dir,spec.name,years)
        if frame.empty:
            continue
        frame=frame[(frame.trade_date>=first)&(frame.trade_date<=last)]
        values=_place_frame(frame,panel,engine.codes)
        for window in windows:
            daily=evaluate_panel(values[-window:],{h:l[-window:] for h,l in labels.items()},
                                 [int_to_str(int(d)) for d in days[-window:]],cfg.min_cross_section,factor=spec.name)
            records=summarize_daily(daily)
            for record in records:
                record["window_trading_days"]=window
            reports.extend(records)
    return reports


def inspect_object(cfg,engine,spec,tail,all_days,pos):
    """一个对象的只读监控；每次最多保留一个尾部年块，异常归属本对象。"""
    from .manifest import Manifest
    errors,warnings=[],[]
    try:
        root=cfg.factor_root(spec)
        wanted=all_days if spec.align_fill is not None else all_days[all_days>=spec.start_int(cfg)]
        expected_years=sorted(set(int(d)//10000 for d in wanted))
        actual_years=factor_years(root,spec.name)
        if actual_years!=expected_years:
            errors.append("annual_partition_set")
        partitions=[]
        manifest=Manifest.load(cfg.state_dir,spec.name)
        if hasattr(engine,"_recipe") and manifest.recipe!=engine._recipe(spec):
            errors.append("manifest_recipe")
        if set(manifest.partitions)!={str(y) for y in actual_years}:
            errors.append("manifest_partition_set")
        for year in actual_years:
            path=year_path(root,spec.name,year)
            ndays=int(np.sum(wanted//10000==year))
            rec=inspect_footer(path,ndays*(1 if spec.is_market else len(engine.codes)),spec.is_market)
            partitions.append({"year":year,**rec})
            errors.extend(f"{year}:{e}" for e in rec["errors"])
            meta=manifest.partitions.get(str(year),{})
            if meta.get("rows")!=rec["rows"]:
                errors.append(f"{year}:manifest_rows")
            stat=path.stat()
            if meta.get("file_identity") and meta["file_identity"]!=[stat.st_size,stat.st_mtime_ns]:
                errors.append(f"{year}:manifest_file_identity")
        expected_tail=tail[tail>=int_to_int(spec.align_start(cfg) if spec.align_fill is not None else spec.resolved_start(cfg))]
        frames=[]
        if expected_tail.size:
            first,last=int_to_str(int(expected_tail[0])),int_to_str(int(expected_tail[-1]))
            for year in sorted(set(int(d)//10000 for d in expected_tail)):
                path=year_path(root,spec.name,year)
                if path.exists():
                    frames.append(pd.read_parquet(path,engine="pyarrow",use_threads=False,
                                  filters=[("trade_date",">=",first),("trade_date","<=",last)]))
        frame=pd.concat(frames,ignore_index=True) if frames else pd.DataFrame(columns=MARKET_COLUMNS if spec.is_market else COLUMNS)
        checked=inspect_tail(frame,expected_tail,engine.codes,spec,cfg)
        errors.extend(checked["errors"])
        warnings.extend(checked["warnings"])
        record={"factor":spec.name,"kind":"market" if spec.is_market else "label" if spec.is_label else "stock",
                "partitions":partitions,"tail":checked["daily"],"recipe":manifest.recipe}
        if spec.is_label:
            horizon=spec.forward_days-1
            mature_pos=pos-horizon-2
            record["last_mature_signal_date"]=int_to_str(int(engine.cal.days[mature_pos])) if mature_pos>=0 else None
            if expected_tail.size and len(frame):
                dates=frame.trade_date.astype(str)
                frontier=record["last_mature_signal_date"]
                premature=dates>frontier if frontier else np.ones(len(frame),bool)
                if np.isfinite(frame.value.to_numpy(dtype=float)[premature]).any():
                    errors.append("immature_label_finite")
        return {**record,"errors":sorted(set(errors)),"warnings":sorted(set(warnings))}
    except Exception as exc:
        return {"factor":spec.name,"errors":[f"monitor_exception:{type(exc).__name__}:{exc}"],"warnings":warnings}


def monitor_jobs(cfg):
    """监控最多四个读线程，自动夹在项目/实际 CPU 上限内。"""
    from .resources import available_cpus,resource_policy,safe_jobs
    requested=int(cfg.raw.get("monitor",{}).get("jobs",4))
    if requested<1:
        raise ValueError("monitor.jobs must be a positive integer")
    policy=resource_policy(cfg.raw.get("resources"))
    cap=min(4,int(policy["max_jobs"]),available_cpus())
    return safe_jobs(min(requested,cap),policy)


def run_monitor(cfg,engine=None,run_id=None,end=None):
    """返回 status=ok/failed；硬错误阻止发布，覆盖/稀疏等诊断仅警告。"""
    from .release import production_specs
    started=time.monotonic()
    if engine is None:
        from .engine import Engine
        engine=Engine(cfg)
    end=int_to_int(end if end is not None else engine.baseline_last_day())
    settings=cfg.raw.get("monitor",{})
    tail_days=max(1,min(60,int(settings.get("tail_days",5))))
    pos=int(np.searchsorted(engine.cal.days,end,side="right"))
    tail=engine.cal.days[max(0,pos-tail_days):pos]
    all_days=engine.cal.between(int(cfg.default_start.replace("-","")),end)
    specs=production_specs(cfg)
    jobs=monitor_jobs(cfg)
    if jobs==1:
        records=[inspect_object(cfg,engine,spec,tail,all_days,pos) for spec in specs]
    else:
        # map preserves production_specs order even when individual I/O finishes
        # out of order. Object reads are independent; Engine/cal/codes are readonly.
        with ThreadPoolExecutor(max_workers=jobs,thread_name_prefix="fea-monitor") as pool:
            records=list(pool.map(lambda spec:inspect_object(cfg,engine,spec,tail,all_days,pos),specs))
    identifier=(str(run_id)+"_" if run_id else "")+dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")+"_"+uuid.uuid4().hex[:8]
    outroot=cfg.project_path("monitor","artifacts/monitor") if hasattr(cfg,"project_path") else cfg.root/"artifacts"/"monitor"
    dest=outroot/"runs"/identifier
    dest.mkdir(parents=True,exist_ok=False)
    report={"run_id":identifier,"status":"failed" if any(r["errors"] for r in records) else "ok",
            "as_of":int_to_str(end),"n_pool":len(engine.codes),"tail_trading_days":tail_days,
            "objects":len(records),"error_objects":sum(bool(r["errors"]) for r in records),
            "warning_objects":sum(bool(r["warnings"]) for r in records),"records":records,
            "scope":"all annual footers; exact keys and numerical checks on recent tail; not full historical uniqueness/PIT/profitability proof"}
    if settings.get("rolling_enabled",False):
        try:
            report["rolling"]= _rolling_reports(cfg,engine,specs,end,settings)
        except Exception as exc:
            report["rolling_error"]=repr(exc)
            report["status"]="failed"
    report["elapsed_seconds"]=time.monotonic()-started
    report["report_path"]=str(dest/"report.json")
    _write_json(dest/"report.json",report)
    _write_json(outroot/"latest.json",{"run_id":identifier,"report":str(dest/"report.json"),"status":report["status"]})
    return _json_safe(report)


def int_to_int(value):
    return int(str(value).replace("-",""))
