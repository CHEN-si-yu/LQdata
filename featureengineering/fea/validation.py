"""逐年读取产物并核对完整日期轴、股票轴和状态，不一次加载全历史。"""
import datetime
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .dates import int_to_str
from .engine import Engine
from .manifest import Manifest
from .spec import all_specs
from .store import COLUMNS,DTYPES,MARKET_COLUMNS,MARKET_DTYPES,factor_years,year_path

def check(cfg):
    print("基础格式/轴验收，PIT 未执行；需独立截断输入复算验证因果性。",flush=True)
    eng=Engine(cfg); end=eng.baseline_last_day()
    records=[]; bad=0
    for spec in all_specs():
        if not spec.enabled:continue
        issues=[]; rows=nonnull=0; coverage=[]
        man=Manifest.load(cfg.state_dir,spec.name)
        root=cfg.factor_root(spec)
        # ★★ 日期轴的下界是**分区下界**（`align_start` = default_start），不是真实
        #   起点：对齐区间也要有分区（见 FactorSpec.align_fill）。用 start_int 会让
        #   每个晚起点因子都被判「全历史日期轴不完整」。
        wanted=eng.cal.between(spec.start_int(cfg) if spec.align_fill is None
                               else int(spec.align_start(cfg).replace("-","")), end)
        actual_dates=[]
        years=factor_years(root,spec.name)
        if man.recipe!=eng._recipe(spec):issues.append("逻辑指纹不匹配")
        for year in years:
            path=year_path(root,spec.name,year)
            try:df=pd.read_parquet(path)
            except Exception as exc:
                issues.append(f"{year}:读取失败:{exc}");continue
            columns,dtypes=(MARKET_COLUMNS,MARKET_DTYPES) if spec.is_market else (COLUMNS,DTYPES)
            if list(df.columns)!=columns:
                issues.append(f"{year}:列契约错误");continue
            if any(str(df[c].dtype)!=t for c,t in dtypes.items()):issues.append(f"{year}:dtype错误")
            keys=["trade_date"] if spec.is_market else ["trade_date","stock_code"]
            if df.duplicated(keys).any():issues.append(f"{year}:重复主键")
            dates=df.trade_date.astype(str)
            expected=[int_to_str(int(d)) for d in wanted[wanted//10000==year]]
            got=sorted(dates.unique())
            actual_dates.extend(got)
            if got!=expected:issues.append(f"{year}:交易日轴缺失或越界")
            vals=df.value.to_numpy(dtype=float)
            if np.isinf(vals).any():issues.append(f"{year}:无穷值")
            valid=np.isfinite(vals)
            finite=int(valid.sum())
            good_dates=sorted(dates[valid].unique())
            missing_value_dates=sorted(set(got)-set(good_dates))
            coverage.append({"year":year,"rows":len(df),"finite":finite,
                "finite_fraction":finite/max(len(df),1),
                "first_valid":good_dates[0] if good_dates else None,
                "last_valid":good_dates[-1] if good_dates else None,
                "all_missing_dates":missing_value_dates})
            if spec.is_market and missing_value_dates:
                # ★★ 2026-09-25：**对齐区间**里的 NaN 是契约要求的（`align_fill`），
                #   不是缺陷 —— 上游根本没有那几年的数据。不排除的话，7 个
                #   `mkt_limit_*` 会因 2018/2019 的填充分区被恒久报错。见 README
                #   「时间轴对齐」。与 `backfill._scan_thin` 同一处置。
                _rs = int(spec.resolved_start(cfg).replace("-", ""))
                missing_value_dates = [d for d in missing_value_dates
                                       if int(str(d).replace("-", "")) >= _rs]
                if missing_value_dates:
                    issues.append(f"{year}:市场有效日值缺失{len(missing_value_dates)}天")
            if spec.is_market:
                if len(df)!=len(got):issues.append(f"{year}:每天不是一行")
            else:
                if not df.stock_code.isin(eng.codes).all():issues.append(f"{year}:股票池外代码")
                if not df.groupby("trade_date",observed=True).size().eq(len(eng.codes)).all():
                    issues.append(f"{year}:股票轴缺失")
                ranks=df["rank"].to_numpy(dtype=float)
                if np.isinf(ranks).any() or ((ranks<0)|(ranks>1)).any():issues.append(f"{year}:rank越界")
                if spec.is_label and np.isfinite(ranks).any():issues.append(f"{year}:标签rank应为空")
            meta=man.partitions.get(str(year),{})
            if meta.get("rows")!=len(df) or meta.get("nonnull")!=finite:issues.append(f"{year}:状态行数不一致")
            if len(df) and (meta.get("min_date")!=min(got) or meta.get("max_date")!=max(got)):
                issues.append(f"{year}:状态日期不一致")
            ident=meta.get("file_identity")
            st=path.stat()
            if ident and ident!=[st.st_size,st.st_mtime_ns]:issues.append(f"{year}:状态文件指纹不一致")
            rows+=len(df);nonnull+=finite
        if sorted(actual_dates)!=[int_to_str(int(d)) for d in wanted]:issues.append("全历史日期轴不完整")
        if set(man.partitions)!={str(y) for y in years}:issues.append("状态分区目录不一致")
        if rows and not nonnull and not spec.is_label:issues.append("全历史没有有效值")
        bad+=bool(issues)
        rec={"factor":spec.name,"kind":"market" if spec.is_market else "label" if spec.is_label else "stock",
             "years":len(years),"rows":rows,"nonnull":nonnull,"coverage":coverage,"issues":issues}
        records.append(rec)
        print(f"{spec.name}: {rows:,}行，非空{nonnull/max(rows,1):.2%}，"+(";".join(issues) if issues else "通过"),flush=True)
    # check 输出是运行产物：放项目内 artifacts/checks/（原来在 ../CodeX/featureengineering_checks）。
    dest=Path(cfg.root)/"artifacts"/"checks"/datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    dest.mkdir(parents=True,exist_ok=True)
    (dest/"report.json").write_text(json.dumps({"cutoff":int_to_str(end),"bad_factors":bad,"records":records},ensure_ascii=False,indent=2))
    print(f"逐分区检查：{len(records)}项，问题{bad}项；{dest/'report.json'}",flush=True)
    return int(bool(bad))
