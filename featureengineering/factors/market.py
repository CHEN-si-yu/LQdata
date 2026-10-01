"""30 个日频市场标量：固定池、缺失感知分母、无横截面排名；不宣称已验证预测收益。"""
import warnings
import numpy as np
from fea.spec import FactorSpec, register
from fea.market_morning import morning

def _div(a,b):
    a,b = np.broadcast_arrays(np.asarray(a,dtype=float),np.asarray(b,dtype=float))
    out=np.full(a.shape,np.nan)
    np.divide(a,b,out=out,where=np.isfinite(b)&(b>0))
    return out

def _mean(x):
    return _div(np.nansum(x,axis=1),np.isfinite(x).sum(axis=1))

def _quantile(x,q):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore",RuntimeWarning)
        return np.nanquantile(x,q,axis=1)

def _ratio(test,valid,need=1):
    n=valid.sum(axis=1)
    return np.where(n>=need,_div((test&valid).sum(axis=1),n),np.nan)

def _stats(x):
    mean=_mean(x)
    z=x-mean[:,None]
    sd=np.sqrt(_mean(z*z))
    return mean,sd,_div(_mean(z**3),sd**3)

# 日线/上午比较共用极小容差，避免恰好5%或恰好均线被二进制舍入改分类。
# 容差远小于原始报价精度；价格水平按基准绝对值缩放。
def _gt(a,b):
    a,b=np.broadcast_arrays(np.asarray(a,dtype=float),np.asarray(b,dtype=float))
    return np.isfinite(a)&np.isfinite(b)&((a-b)>(1e-12+1e-12*np.abs(b)))

def _ge(a,b):
    a,b=np.broadcast_arrays(np.asarray(a,dtype=float),np.asarray(b,dtype=float))
    return np.isfinite(a)&np.isfinite(b)&((a-b)>=-(1e-12+1e-12*np.abs(b)))

def _lt(a,b):return _gt(-np.asarray(a),-np.asarray(b))
def _le(a,b):return _ge(-np.asarray(a),-np.asarray(b))

def _base(ctx):
    ratio=lambda test,valid:_ratio(test,valid,ctx.cfg.min_cross_section)
    key=(int(ctx.panel.dates[0]),int(ctx.panel.dates[-1]),tuple(ctx.panel.codes))
    hit=getattr(ctx.up,"_market_daily",None)
    if hit is not None and hit[0]==key:return hit[1]
    ret=ctx.ret_clean(1)
    valid=ctx.universe & np.isfinite(ret)
    r=np.where(valid,ret,np.nan)
    amount=np.where(valid,ctx.px("amount"),np.nan)
    mean,sd,skew=_stats(r)
    close=ctx.hfq("close")
    price_valid=ctx.universe & ctx.traded() & np.isfinite(close)
    ma20=ctx.roll_mean(close,20); ma60=ctx.roll_mean(close,60)
    hi=ctx.shift(ctx.roll_max(close,60),1); lo=ctx.shift(ctx.roll_min(close,60),1)
    totals=np.nansum(amount,axis=1)
    totals=np.where(np.isfinite(amount).sum(axis=1)>=ctx.cfg.min_cross_section,totals,np.nan)
    ref=ctx.shift(ctx.roll_mean(totals[:,None],20),1)[:,0]
    up=ratio(_gt(r,0),valid)
    illiq=_div(np.abs(r)*1e8,amount)
    out={
        "up_ratio":up,
        "strong_up_ratio":ratio(_ge(r,0.05),valid),
        "strong_down_ratio":ratio(_le(r,-0.05),valid),
        "ew_return":mean,
        "dispersion":sd,
        "downside_rms":np.sqrt(_mean(np.where(valid,np.minimum(r,0.)**2,np.nan))),
        "skewness":skew,
        "return_iqr":_quantile(r,0.75)-_quantile(r,0.25),
        "above_ma20_ratio":ratio(_gt(close,ma20),price_valid&np.isfinite(ma20)),
        "above_ma60_ratio":ratio(_gt(close,ma60),price_valid&np.isfinite(ma60)),
        "new_high60_ratio":ratio(_gt(close,hi),price_valid&np.isfinite(hi)),
        "new_low60_ratio":ratio(_lt(close,lo),price_valid&np.isfinite(lo)),
        "amount_ratio20":_div(totals,ref),
        "amount_hhi":_div(np.nansum(amount**2,axis=1),totals**2),
        "illiquidity_median":np.where(np.isfinite(illiq).sum(axis=1)>=ctx.cfg.min_cross_section,_quantile(illiq,0.5),np.nan),
        "trend_breadth_change5":up-ctx.shift(up[:,None],5)[:,0],
    }
    sufficient=valid.sum(axis=1)>=ctx.cfg.min_cross_section
    out={k:np.where(sufficient,v,np.nan) for k,v in out.items()}
    ctx.up._market_daily=(key,out)
    return out

def _am(ctx):
    ratio=lambda test,valid:_ratio(test,valid,ctx.cfg.min_cross_section)
    key=(int(ctx.panel.dates[0]),int(ctx.panel.dates[-1]),tuple(ctx.panel.codes))
    hit=getattr(ctx.up,"_market_am",None)
    if hit is not None and hit[0]==key:return hit[1]
    close=morning(ctx,"am_close"); op=morning(ctx,"am_open")
    amount=morning(ctx,"am_amount")
    # 前收盘参考价在开盘已确定；落地源是日终日线，实际供数时间仍为收盘后。
    prev=ctx.px_raw("pre_close")
    valid=ctx.universe & np.isfinite(close) & (close>0) & np.isfinite(prev) & (prev>0) & np.isfinite(amount) & (amount>0)
    r=np.where(valid,_div(close,prev)-1.,np.nan)
    mean,sd,skew=_stats(r)
    a=np.where(valid,amount,np.nan)
    total=np.nansum(a,axis=1)
    total=np.where((np.isfinite(a)&(a>=0)).sum(axis=1)>=ctx.cfg.min_cross_section,total,np.nan)
    ref=ctx.shift(ctx.roll_mean(total[:,None],20),1)[:,0]
    gap=_div(op,prev)-1.; intraday=_div(close,op)-1.
    ranges=_div(morning(ctx,"am_high")-morning(ctx,"am_low"),prev)
    out={
        "up_ratio":ratio(_gt(r,0),valid),
        "strong_up_ratio":ratio(_ge(r,0.05),valid),
        "strong_down_ratio":ratio(_le(r,-0.05),valid),
        "ew_return":mean,
        "median_return":_quantile(r,0.5),
        "dispersion":sd,
        "downside_rms":np.sqrt(_mean(np.where(valid,np.minimum(r,0.)**2,np.nan))),
        "skewness":skew,
        "up_amount_share":_div(np.nansum(np.where(_gt(r,0),a,0.),axis=1),total),
        "amount_ratio20":_div(total,ref),
        "reversal_ratio":ratio((_gt(gap,0)&_lt(intraday,0))|(_lt(gap,0)&_gt(intraday,0)),valid&np.isfinite(gap)&np.isfinite(intraday)&(_gt(gap,0)|_lt(gap,0))),
        "range_mean":_mean(np.where(valid,ranges,np.nan)),
        "return_iqr":_quantile(r,0.75)-_quantile(r,0.25),
    }
    sufficient=valid.sum(axis=1)>=ctx.cfg.min_cross_section
    out={k:np.where(sufficient,v,np.nan) for k,v in out.items()}
    ctx.up._market_am=(key,out)
    return out

DAILY={
"up_ratio":("上涨比例","count(r>0)/count(valid_r)"),
"strong_up_ratio":("涨幅至少5%的比例","count(r>=0.05)/count(valid_r)"),
"strong_down_ratio":("跌幅至少5%的比例","count(r<=-0.05)/count(valid_r)"),
"ew_return":("等权市场收益","mean(r)"),
"cap_weighted_return":("前日流通市值加权收益","sum(r*lag(close*float_share,1))/sum(lag(close*float_share,1))"),
"dispersion":("收益横截面离散度","std(r,ddof=0)"),
"downside_rms":("下行收益均方根","sqrt(mean(min(r,0)^2))"),
"skewness":("收益横截面偏度","mean((r-mean(r))^3)/std(r)^3"),
"return_iqr":("收益四分位距","q75(r)-q25(r)"),
"above_ma20_ratio":("位于20日均线上方的比例","count(hfq_close>MA20)/count(valid)"),
"above_ma60_ratio":("位于60日均线上方的比例","count(hfq_close>MA60)/count(valid)"),
"new_high60_ratio":("突破前60日新高的比例","count(hfq_close>lag(MAX60,1))/count(valid)"),
"new_low60_ratio":("跌破前60日新低的比例","count(hfq_close<lag(MIN60,1))/count(valid)"),
"amount_ratio20":("市场成交额相对前20日均值","sum(amount)/lag(MA20(sum(amount)),1)"),
"amount_hhi":("成交额集中度","sum(amount^2)/sum(amount)^2"),
"illiquidity_median":("每亿元成交额价格冲击中位数","median(abs(r)*1e8/amount)"),
"trend_breadth_change5":("上涨比例5日变化","up_ratio-lag(up_ratio,5)"),
}
AM={k:DAILY[k] for k in ("up_ratio","strong_up_ratio","strong_down_ratio","ew_return","dispersion","downside_rms","skewness","return_iqr")}
AM.update({
"median_return":("上午收益中位数","median(r_am)"),
"up_amount_share":("上涨股票上午成交额占比","sum(am_amount where r_am>0)/sum(am_amount)"),
"amount_ratio20":("上午成交额相对前20日均值","sum(am_amount)/lag(MA20(sum(am_amount)),1)"),
"reversal_ratio":("上午反向回补跳空比例","count(gap*intraday_return<0)/count(valid and gap!=0)"),
"range_mean":("上午平均振幅","mean((am_high-am_low)/pre_close)"),
})

def _register(key,desc,formula,am=False):
    name=("mkt_am_" if am else "mkt_")+key
    def compute(ctx):
        return (_am(ctx) if am else _base(ctx))[key]
    compute.__name__=name
    deps=("stock_daily","stock_history_5min") if am else ("stock_daily","stock_adj_factor")
    register(FactorSpec(name=name,group="market",is_market=True,desc=("11:30 " if am else "收盘 ")+desc,
        formula=formula+("；r_am=am_close/pre_close-1" if am else "；r=单日后复权收益"),
        deps=deps,version=(4 if am else 3),warmup_days=500,start="2018-01-01",
        note="固定2115只主板池；有效样本分母，缺测/停牌不计，少于100只为NaN。"+
        ("上午要求09:35至11:30完整24根5分钟棒且上午成交额为正；零成交报价不计入样本。截面时点11:30；上游为日终文件，生产可用时点为收盘后。" if am else "当日收盘后可用。")+
        "离散比较容差为1e-12+1e-12*abs(基准)：阈值相等计入至少/至多，不计入严格高于/低于。市场标量不做截面rank；属于市场状态候选特征，未做收益有效性承诺。"))(compute)

DAILY = {k: DAILY[k] for k in (
    "up_ratio", "strong_up_ratio", "strong_down_ratio", "dispersion",
    "above_ma20_ratio", "above_ma60_ratio", "new_high60_ratio",
    "new_low60_ratio", "amount_hhi", "illiquidity_median")}
AM = {k: AM[k] for k in (
    "up_ratio", "strong_up_ratio", "strong_down_ratio", "ew_return", "dispersion",
    "downside_rms", "up_amount_share", "amount_ratio20", "reversal_ratio", "range_mean")}
for _key,(_desc,_formula) in DAILY.items(): _register(_key,_desc,_formula)
for _key,(_desc,_formula) in AM.items(): _register(_key,_desc,_formula,True)


INDEX_CODES = ("000300.SH", "000905.SH", "000001.SH", "399001.SZ")
def _index(ctx):
    key=(int(ctx.panel.dates[0]),int(ctx.panel.dates[-1]))
    hit=getattr(ctx.up,"_market_index",None)
    if hit is not None and hit[0]==key:return hit[1]
    import pandas as pd
    from fea.dates import series_to_int
    df=ctx.dataset("index_daily",columns=["ts_code","trade_date","close","amount"],
                   years=(key[0]//10000,key[1]//10000))
    df=df[df.ts_code.isin(INDEX_CODES)].copy()
    if df.duplicated(["ts_code","trade_date"]).any():
        raise ValueError("指数日线含重复主键")
    df["day"]=series_to_int(df.trade_date)
    close=df.pivot(index="day",columns="ts_code",values="close").reindex(index=ctx.panel.dates,columns=INDEX_CODES).to_numpy(float)
    amount=df.pivot(index="day",columns="ts_code",values="amount").reindex(index=ctx.panel.dates,columns=INDEX_CODES).to_numpy(float)
    r=_div(close,ctx.shift(close,1))-1.
    r20=_div(close,ctx.shift(close,20))-1.
    mean20=ctx.roll_mean(close,20)
    rv=ctx.roll_std(r,20)*np.sqrt(252.)
    hi60=ctx.roll_max(close,60)
    out={
        "csi300_ret1":r[:,0],
        "csi300_momentum20":r20[:,0],
        "csi300_volatility20":rv[:,0],
        "csi300_drawdown60":(_div(close,hi60)-1.)[:,0],
        "csi300_ma20_gap":(_div(close,mean20)-1.)[:,0],
        "small_large_momentum20":r20[:,1]-r20[:,0],
        "sz_sh_momentum20":r20[:,3]-r20[:,2],
        "csi500_csi300_vol_ratio20":_div(rv[:,1],rv[:,0]),
        "csi300_amount_ratio20":_div(amount,ctx.shift(ctx.roll_mean(amount,20),1))[:,0],
        "sh_sz_return_corr60":ctx.roll_corr(r[:,2:3],r[:,3:4],60)[:,0],
    }
    ctx.up._market_index=(key,out)
    return out

INDEX={
"csi300_ret1":("沪深300单日收益","close/lag(close,1)-1"),
"csi300_momentum20":("沪深300的20日动量","close/lag(close,20)-1"),
"csi300_volatility20":("沪深300的20日年化波动","std(ret1,20)*sqrt(252)"),
"csi300_drawdown60":("沪深300距60日高点回撤","close/rolling_max(close,60)-1"),
"csi300_ma20_gap":("沪深300相对20日均线偏离","close/MA20(close)-1"),
"small_large_momentum20":("中证500相对沪深300的20日强弱","ret20(000905.SH)-ret20(000300.SH)"),
"sz_sh_momentum20":("深证成指相对上证综指的20日强弱","ret20(399001.SZ)-ret20(000001.SH)"),
"csi500_csi300_vol_ratio20":("中盘相对大盘的波动比","vol20(000905.SH)/vol20(000300.SH)"),
"csi300_amount_ratio20":("沪深300成交额相对前20日均值","amount/lag(MA20(amount),1)"),
"sh_sz_return_corr60":("沪深市场60日联动性","corr(ret1(000001.SH),ret1(399001.SZ),60)"),
}
def _register_index(key,desc,formula):
    name="mkt_idx_"+key
    def compute(ctx):return _index(ctx)[key]
    compute.__name__=name
    register(FactorSpec(name=name,group="market",is_market=True,desc=desc,formula=formula,
        deps=("index_daily",),warmup_days=500,start="2018-01-01",
        note="指数按官方指数编制范围，不限制为固定股票池；均是价格指数，非全收益指数。"+
             "当日收盘后可用；缺失保持NaN；不做截面rank；2018之前数据只作预热。"))(compute)
for _key,(_desc,_formula) in INDEX.items():_register_index(_key,_desc,_formula)
