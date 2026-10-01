#!/usr/bin/env python3
"""V33 walk-forward LambdaRank using twenty factors frozen before 2024Q1."""
from __future__ import annotations
import os
for _name in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS","VECLIB_MAXIMUM_THREADS","BLIS_NUM_THREADS"):
    os.environ[_name]="1"
import json, resource, time
from pathlib import Path
import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT=Path(__file__).resolve().parent
DATA_ROOT=ROOT.parents[1]/"trainingdata"
CODES=("601288.SH","601398.SH","601939.SH","601988.SH")
PROTOCOL=json.loads((ROOT/"protocol.json").read_text(encoding="utf-8"))
FEATURES=tuple(PROTOCOL["model_recipe"]["factor_features"])
HORIZON=5
SEED=17
QUARTERS=tuple(PROTOCOL["holdout_quarters"])
if PROTOCOL.get("status")!="frozen_before_any_V33_training" or len(FEATURES)!=20:
 raise RuntimeError("V33 protocol must be frozen with exactly 20 factors before training")
selection=json.loads((ROOT/"selected_feature_list.json").read_text(encoding="utf-8"))
if FEATURES!=tuple(selection["selected_features"]):
 raise RuntimeError("protocol features do not match the frozen selection artifact")
MODEL_RECIPE=PROTOCOL["model_recipe"]
def atomic_json(path:Path,obj)->None:
 path.parent.mkdir(parents=True,exist_ok=True)
 tmp=path.with_suffix(path.suffix+".tmp")
 tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
 os.replace(tmp,path)
def rss_gib()->float:
 try:
  import psutil
  return round(psutil.Process().memory_info().rss/(1024**3),3)
 except Exception:
  value=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
  return round(value/(1024**3),3)
class Panel: pass
def load_panel(quarter:str)->Panel:
 meta=json.loads((DATA_ROOT/"meta.json").read_text(encoding="utf-8"))
 if meta.get("semantics")!="zscore_win1_99_v1":
  raise RuntimeError(f"unexpected factor semantics: {meta.get('semantics')}")
 qp=pd.Period(quarter,freq="Q")
 qstart=qp.start_time.strftime("%Y-%m-%d")
 qend=qp.end_time.strftime("%Y-%m-%d")
 years=sorted(int(y) for y in meta["built_years"] if int(y)<=int(qend[:4]))
 if not years: raise RuntimeError("no factor years available")
 schema=pq.ParquetFile(DATA_ROOT/"factors"/f"year={years[0]}"/"data.parquet").schema_arrow.names
 missing=[name for name in FEATURES if name not in schema]
 if missing: raise RuntimeError(f"required frozen features missing: {missing}")
 pieces=[]
 for year in years:
  part=pq.read_table(DATA_ROOT/"factors"/f"year={year}"/"data.parquet",
   columns=["trade_date","stock_code",*FEATURES],
   filters=[("stock_code","in",list(CODES)),("trade_date","<=",qend)]).to_pandas()
  part["trade_date"]=part["trade_date"].astype(str).str[:10]
  part["stock_code"]=part["stock_code"].astype(str)
  pieces.append(part)
 frame=pd.concat(pieces,ignore_index=True)
 if frame.duplicated(["trade_date","stock_code"]).any():
  raise RuntimeError("duplicate factor date/security rows")
 days=np.asarray(sorted(frame.trade_date.unique()),dtype=str)
 codes=np.asarray(CODES,dtype=str)
 ix=pd.MultiIndex.from_product([days,codes],names=["trade_date","stock_code"])
 frame=frame.set_index(["trade_date","stock_code"]).reindex(ix)
 if frame.shape[0]!=len(days)*len(codes): raise RuntimeError("four-bank factor axis is incomplete")
 x_num=frame[list(FEATURES)].to_numpy(np.float32)
 x_num=np.nan_to_num(x_num,nan=0.0,posinf=0.0,neginf=0.0)
 onehot=np.tile(np.eye(len(codes),dtype=np.float32),(len(days),1))
 x=np.column_stack((x_num,onehot)).reshape(len(days),len(codes),-1)
 test_ix=np.flatnonzero((days>=qstart)&(days<=qend))
 if not len(test_ix): raise RuntimeError(f"{quarter} has no test dates")
 train_ix=np.flatnonzero(np.arange(len(days))+HORIZON<int(test_ix[0]))
 if not len(train_ix): raise RuntimeError(f"{quarter} has no pre-test training dates after purge")
 train_end=str(days[int(train_ix[-1])])
 label_years=sorted(int(y) for y in meta["built_years"] if int(y)<=int(train_end[:4]))
 label_parts=[]
 for year in label_years:
  y=pq.read_table(DATA_ROOT/"target"/f"year={year}"/"data.parquet",
   columns=["trade_date","stock_code","label_ret_5d"],
   filters=[("stock_code","in",list(CODES)),("trade_date","<=",train_end)]).to_pandas()
  y["trade_date"]=y["trade_date"].astype(str).str[:10]
  y["stock_code"]=y["stock_code"].astype(str)
  label_parts.append(y)
 labels=pd.concat(label_parts,ignore_index=True).set_index(["trade_date","stock_code"]).reindex(ix)["label_ret_5d"]
 panel=Panel()
 panel.days=days
 panel.codes=codes
 panel.features=tuple(list(FEATURES)+[f"asset_{i}" for i in range(len(codes))])
 panel.X=x
 panel.Y=labels.to_numpy(np.float32).reshape(len(days),len(codes))
 panel.test_ix=test_ix
 panel.train_ix=train_ix
 panel.train_end=train_end
 panel.quarter=quarter
 panel.factor_semantics=meta["semantics"]
 panel.data_built_at=meta.get("built_at")
 panel.purge_gap_trading_sessions=int(test_ix[0]-train_ix[-1]-1)
 if panel.purge_gap_trading_sessions!=HORIZON:
  raise RuntimeError(f"{quarter}: expected exactly {HORIZON} purged trading sessions, found {panel.purge_gap_trading_sessions}")
 if str(days[int(test_ix[0])])<=train_end:
  raise RuntimeError("train end is not before first test date")
 return panel
def train_rows(panel:Panel):
 xs=[]; rels=[]; groups=[]; rows=0
 for d in panel.train_ix:
  y=panel.Y[d]
  pos=np.flatnonzero(np.isfinite(y))
  if len(pos)<2: continue
  vals=y[pos].astype(np.float64)
  rel=np.searchsorted(np.unique(vals),vals).astype(np.int32)
  xs.append(panel.X[d,pos]); rels.append(rel); groups.append(len(pos)); rows+=len(pos)
 if not groups: raise RuntimeError(f"{panel.quarter}: no daily query has >=2 finite labels")
 if max(max(np.searchsorted(np.unique(panel.Y[d,np.flatnonzero(np.isfinite(panel.Y[d]))]),panel.Y[d,np.flatnonzero(np.isfinite(panel.Y[d]))])) for d in panel.train_ix if np.isfinite(panel.Y[d]).sum()>=2)>3:
  raise RuntimeError("relevance labels exceed fixed 0..3 gain table")
 return np.concatenate(xs),np.concatenate(rels),groups,rows
def train_quarter(quarter:str)->dict:
 out=ROOT/"quarters"/quarter
 out.mkdir(parents=True,exist_ok=True)
 status=out/"status.json"
 started=time.time()
 atomic_json(status,{"quarter":quarter,"state":"running","pid":os.getpid(),"started_at":time.strftime("%Y-%m-%d %H:%M:%S"),"rss_start_gib":rss_gib()})
 try:
  panel=load_panel(quarter)
  after_panel=rss_gib()
  xtrain,relevance,groups,rows=train_rows(panel)
  model=lgb.LGBMRanker(objective="lambdarank",metric="ndcg",eval_at=[2],
   n_estimators=180,learning_rate=.03,num_leaves=7,max_depth=3,min_child_samples=80,
   colsample_bytree=.80,subsample=.80,subsample_freq=1,reg_alpha=2.,reg_lambda=20.,
   label_gain=[0,1,2,3],random_state=SEED,n_jobs=1,verbosity=-1,
   deterministic=True,force_col_wise=True)
  fit_start=time.time()
  model.fit(xtrain,relevance,group=groups)
  fit_seconds=time.time()-fit_start
  xt=panel.X[panel.test_ix]
  scores=model.predict(xt.reshape(-1,xt.shape[-1]),num_threads=1).reshape(len(panel.test_ix),len(CODES))
  days=panel.days[panel.test_ix]
  order=np.argsort(-scores,axis=1,kind="stable")
  ranks=np.empty_like(order)
  ranks[np.arange(len(order))[:,None],order]=np.arange(1,len(CODES)+1)[None,:]
  signal=np.zeros(len(days),dtype=bool)
  bymonth={}
  for i,d in enumerate(days): bymonth.setdefault(str(d)[:7],[]).append(i)
  for ids in bymonth.values(): signal[ids[0]]=True
  chosen=np.zeros(scores.shape,dtype=bool)
  for i in np.flatnonzero(signal): chosen[i,order[i,:2]]=True
  pred=pd.DataFrame({"trade_date":np.repeat(days,4),"stock_code":np.tile(CODES,len(days)),
   "score":scores.reshape(-1),"daily_rank":ranks.reshape(-1),
   "monthly_signal":np.repeat(signal,4),"selected_top2":chosen.reshape(-1),
   "model_quarter":quarter})
  tmp=out/"predictions.csv.tmp"
  pred.to_csv(tmp,index=False,float_format="%.10g")
  os.replace(tmp,out/"predictions.csv")
  model.booster_.save_model(str(out/"ranker.txt"))
  gains=model.booster_.feature_importance(importance_type="gain")
  ranked=sorted(({"feature":panel.features[i],"gain":float(gains[i])} for i in range(len(gains))),key=lambda z:(-z["gain"],z["feature"]))
  train_first=str(panel.days[int(panel.train_ix[0])])
  test_first=str(days[0]); test_last=str(days[-1])
  q_dates=days[signal].tolist()
  result={"quarter":quarter,"state":"complete","recipe":MODEL_RECIPE,
   "protocol_sha256":__import__("hashlib").sha256((ROOT/"protocol.json").read_bytes()).hexdigest(),
   "feature_selection_sha256":__import__("hashlib").sha256((ROOT/"selected_feature_list.json").read_bytes()).hexdigest(),
   "lightgbm_version":lgb.__version__,
   "factor_semantics":panel.factor_semantics,"data_built_at":panel.data_built_at,
   "features":list(panel.features),"n_features":len(panel.features),
   "n_train_days_before_label_filter":int(len(panel.train_ix)),
   "train_first":train_first,"train_last":panel.train_end,"first_test":test_first,"last_test":test_last,
   "purge_rule":"train_day_index + 5 < first_test_day_index",
   "purged_trading_sessions":panel.purge_gap_trading_sessions,
   "max_label_date_used":panel.train_end,"n_train_queries":int(len(groups)),
   "n_train_rows_after_label_filter":int(rows),"train_group_size_min":int(min(groups)),
   "train_group_size_max":int(max(groups)),"n_test_days":int(len(days)),
   "monthly_signal_dates":q_dates,"fit_seconds":round(fit_seconds,3),
   "rss_gib":{"after_panel_load":after_panel,"at_completion":rss_gib()},
   "top_features_by_gain":ranked,"model_sha256":__import__("hashlib").sha256((out/"ranker.txt").read_bytes()).hexdigest(),
   "predictions_sha256":__import__("hashlib").sha256((out/"predictions.csv").read_bytes()).hexdigest(),
   "model_file":"ranker.txt","predictions_file":"predictions.csv","worker_pid":os.getpid(),
   "seconds_total":round(time.time()-started,3)}
  atomic_json(out/"result.json",result)
  atomic_json(status,result|{"state":"complete","finished_at":time.strftime("%Y-%m-%d %H:%M:%S")})
  return result
 except Exception as exc:
  atomic_json(status,{"quarter":quarter,"state":"failed","pid":os.getpid(),"error":repr(exc),"seconds_total":round(time.time()-started,3)})
  raise
if __name__=="__main__":
 import argparse
 parser=argparse.ArgumentParser()
 parser.add_argument("--quarter",choices=QUARTERS,required=True)
 args=parser.parse_args()
 print(json.dumps(train_quarter(args.quarter),ensure_ascii=False,indent=2))
