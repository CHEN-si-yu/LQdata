#!/usr/bin/env python3
"""V11 walk-forward LambdaRank model for four large banks."""
from __future__ import annotations
import os
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[_name] = "1"
import json, resource, sys, time
from pathlib import Path
import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT.parents[1] / "trainingdata"
CODES = ("601288.SH", "601398.SH", "601939.SH", "601988.SH")
HORIZON, SEED = 5, 17
QUARTERS = tuple(str(q) for q in pd.period_range("2020Q1", "2026Q2", freq="Q"))
MODEL_RECIPE = {"name":"V11-LambdaRank-four-bank","objective":"lambdarank",
 "label":"within-date ascending rank of label_ret_5d (integer relevance 0..3; ties share rank)",
 "n_estimators":180,"learning_rate":0.03,"num_leaves":7,"max_depth":3,"min_child_samples":80,
 "feature_fraction":0.80,"bagging_fraction":0.80,"bagging_freq":1,"reg_alpha":2.0,"reg_lambda":20.0,
 "label_gain":[0,1,2,3],"seed":SEED,"threads":1,"early_stopping":False,"sample_weight":None}
def atomic_json(path:Path,obj)->None:
 path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_suffix(path.suffix+".tmp")
 tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8"); os.replace(tmp,path)
def _rss_gib()->float:
 try:
  import psutil
  return round(psutil.Process().memory_info().rss/(1024**3),3)
 except Exception: return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024**3),3)
def _read(kind:str,year:int,columns:list[str])->pd.DataFrame:
 return pq.read_table(DATA_ROOT/kind/f"year={year}"/"data.parquet",columns=columns,
                      filters=[("stock_code","in",list(CODES))]).to_pandas()
class Panel: pass
def load_panel(quarter:str)->Panel:
 """Load only the four-bank factor/market panel needed through this quarter."""
 meta=json.loads((DATA_ROOT/"meta.json").read_text(encoding="utf-8"))
 if meta.get("semantics")!="zscore_win1_99_v1": raise RuntimeError(f"unexpected factor semantics: {meta.get('semantics')}")
 years=sorted(int(y) for y in meta["built_years"] if int(y)<=int(quarter[:4]))
 if not years: raise RuntimeError("no factor years available")
 fs=pq.ParquetFile(DATA_ROOT/"factors"/f"year={years[0]}"/"data.parquet").schema_arrow.names
 factor_cols=[c for c in fs if c not in ("trade_date","stock_code")]
 ms=pq.ParquetFile(DATA_ROOT/"market_factors"/f"year={years[0]}"/"data.parquet").schema_arrow.names
 market_cols=[c for c in ms if c!="trade_date"]; fset=set(factor_cols)
 market_names=[c if c not in fset else "market__"+c for c in market_cols]
 pieces=[]
 for year in years:
  f=_read("factors",year,["trade_date","stock_code",*factor_cols]); f["trade_date"]=f["trade_date"].astype(str).str[:10]; f["stock_code"]=f["stock_code"].astype(str)
  m=pq.read_table(DATA_ROOT/"market_factors"/f"year={year}"/"data.parquet",columns=["trade_date",*market_cols]).to_pandas()
  m["trade_date"]=m["trade_date"].astype(str).str[:10]; m=m.rename(columns=dict(zip(market_cols,market_names)))
  pieces.append(f.merge(m,on="trade_date",validate="many_to_one"))
 frame=pd.concat(pieces,ignore_index=True); frame["trade_date"]=frame["trade_date"].astype(str); frame["stock_code"]=frame["stock_code"].astype(str)
 days=np.asarray(sorted(frame["trade_date"].unique()),dtype=str); codes=np.asarray(CODES,dtype=str)
 ix=pd.MultiIndex.from_product([days,codes],names=["trade_date","stock_code"]); frame=frame.set_index(["trade_date","stock_code"]).reindex(ix)
 if frame.index.has_duplicates or frame.shape[0]!=len(days)*len(codes): raise RuntimeError("four-bank factor axis is incomplete or duplicated")
 xcols=factor_cols+market_names; x_num=frame[xcols].to_numpy(np.float32); x_num=np.nan_to_num(x_num,nan=0.0,posinf=0.0,neginf=0.0)
 onehot=np.tile(np.eye(len(codes),dtype=np.float32),(len(days),1))
 panel=Panel(); panel.days=days; panel.codes=codes; panel.features=tuple(xcols+[f"asset_{i}" for i in range(len(codes))])
 panel.X=np.column_stack((x_num,onehot)).reshape(len(days),len(codes),-1); panel.factor_semantics=meta["semantics"]; panel.data_built_at=meta.get("built_at")
 qp=pd.Period(quarter,freq="Q"); qstart=qp.start_time.strftime("%Y-%m-%d")
 test_ix=np.flatnonzero((days>=qstart)&(days<=qp.end_time.strftime("%Y-%m-%d")))
 if not len(test_ix): raise RuntimeError(f"{quarter} has no test dates")
 train_ix=np.flatnonzero(np.arange(len(days))+HORIZON<int(test_ix[0]))
 if not len(train_ix): raise RuntimeError(f"{quarter} has no training dates after purge")
 train_end=days[int(train_ix[-1])]; label_years=sorted(int(y) for y in meta["built_years"] if int(y)<=int(train_end[:4]))
 label_pieces=[]
 for year in label_years:
  y=_read("target",year,["trade_date","stock_code","label_ret_5d"]); y["trade_date"]=y["trade_date"].astype(str).str[:10]; y["stock_code"]=y["stock_code"].astype(str)
  label_pieces.append(y[y["trade_date"]<=train_end])
 labels=pd.concat(label_pieces,ignore_index=True).set_index(["trade_date","stock_code"]).reindex(ix)["label_ret_5d"]
 panel.Y=labels.to_numpy(np.float32).reshape(len(days),len(codes)); panel.test_ix=test_ix; panel.train_ix=train_ix; panel.train_end=train_end
 panel.quarter=quarter; panel.feature_count=len(panel.features); return panel
def _train_rows(panel:Panel):
 x_groups=[]; rel_groups=[]; groups=[]; rows=0
 for d in panel.train_ix:
  y=panel.Y[d]; pos=np.flatnonzero(np.isfinite(y))
  if len(pos)<2: continue
  vals=y[pos].astype(np.float64); rel=np.searchsorted(np.unique(vals),vals).astype(np.int32)
  x_groups.append(panel.X[d,pos]); rel_groups.append(rel); groups.append(len(pos)); rows+=len(pos)
 if not groups: raise RuntimeError(f"{panel.quarter}: no daily query has >=2 finite labels")
 return np.concatenate(x_groups),np.concatenate(rel_groups),groups,rows
def train_quarter(quarter:str)->dict:
 out=ROOT/"quarters"/quarter; out.mkdir(parents=True,exist_ok=True); status=out/"status.json"; start=time.time()
 atomic_json(status,{"quarter":quarter,"state":"running","pid":os.getpid(),"started_at":time.strftime("%Y-%m-%d %H:%M:%S"),"rss_start_gib":_rss_gib()})
 try:
  panel=load_panel(quarter); load_rss=_rss_gib(); xtrain,relevance,groups,rows=_train_rows(panel)
  model=lgb.LGBMRanker(objective="lambdarank",metric="ndcg",eval_at=[2],n_estimators=180,learning_rate=.03,
   num_leaves=7,max_depth=3,min_child_samples=80,colsample_bytree=.80,subsample=.80,subsample_freq=1,
   reg_alpha=2.,reg_lambda=20.,label_gain=[0,1,2,3],random_state=SEED,n_jobs=1,verbosity=-1,
   deterministic=True,force_col_wise=True)
  fit0=time.time(); model.fit(xtrain,relevance,group=groups); fit_s=time.time()-fit0
  xt=panel.X[panel.test_ix]; scores=model.predict(xt.reshape(-1,xt.shape[-1]),num_threads=1).reshape(len(panel.test_ix),len(CODES))
  days=panel.days[panel.test_ix]; order=np.argsort(-scores,axis=1,kind="stable"); ranks=np.empty_like(order)
  ranks[np.arange(len(order))[:,None],order]=np.arange(1,len(CODES)+1)[None,:]
  signal=np.zeros(len(days),dtype=bool); bymonth={}
  for i,d in enumerate(days): bymonth.setdefault(d[:7],[]).append(i)
  for ids in bymonth.values(): signal[ids[0]]=True
  chosen=np.zeros(scores.shape,dtype=bool)
  for i in np.flatnonzero(signal): chosen[i,order[i,:2]]=True
  pred=pd.DataFrame({"trade_date":np.repeat(days,4),"stock_code":np.tile(CODES,len(days)),
   "score":scores.reshape(-1),"daily_rank":ranks.reshape(-1),"monthly_signal":np.repeat(signal,4),
   "selected_top2":chosen.reshape(-1),"model_quarter":quarter})
  tmp=out/"predictions.csv.tmp"; pred.to_csv(tmp,index=False,float_format="%.10g"); os.replace(tmp,out/"predictions.csv")
  model.booster_.save_model(str(out/"ranker.txt")); gains=model.booster_.feature_importance(importance_type="gain")
  top=[{"feature":panel.features[i],"gain":float(gains[i])} for i in np.argsort(-gains)[:20] if gains[i]>0]
  result={"quarter":quarter,"state":"complete","recipe":MODEL_RECIPE,"data_built_at":panel.data_built_at,
   "factor_semantics":panel.factor_semantics,"n_features":panel.feature_count,"n_train_days_available":int(len(panel.train_ix)),
   "train_first":str(panel.days[panel.train_ix[0]]),"train_last":panel.train_end,"first_test":str(days[0]),"last_test":str(days[-1]),
   "purge_rule":"train_day_index + 5 < first_test_day_index","n_train_queries":int(len(groups)),
   "n_train_rows_after_label_filter":int(rows),"train_group_size_min":int(min(groups)),"train_group_size_max":int(max(groups)),
   "n_test_days":int(len(days)),"monthly_signal_dates":days[signal].tolist(),"fit_seconds":round(fit_s,3),
   "rss_gib":{"after_panel_load":load_rss,"rss_at_completion":_rss_gib()},"top_features_by_gain":top,
   "model_file":"ranker.txt","predictions_file":"predictions.csv","worker_pid":os.getpid(),"seconds_total":round(time.time()-start,3)}
  atomic_json(out/"result.json",result); atomic_json(status,result|{"state":"complete","finished_at":time.strftime("%Y-%m-%d %H:%M:%S")})
  return result
 except Exception as exc:
  atomic_json(status,{"quarter":quarter,"state":"failed","pid":os.getpid(),"error":repr(exc),"seconds_total":round(time.time()-start,3)})
  raise
if __name__=="__main__":
 import argparse
 p=argparse.ArgumentParser(); p.add_argument("--quarter",choices=QUARTERS,required=True); a=p.parse_args()
 print(json.dumps(train_quarter(a.quarter),ensure_ascii=False,indent=2))
