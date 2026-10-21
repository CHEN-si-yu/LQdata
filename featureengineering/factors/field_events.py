"""未覆盖的日频和事件字段；仅作候选暴露，收益方向由下游研究。"""
import numpy as np
import pandas as pd
from fea.spec import FactorSpec, register

SOURCE_FIELDS = {}
# 全历史数据质量验收的保留/起点决定；不依据收益筛选。
REJECTED_CANDIDATES = {
    "efx_annual_sales_yield",   # 本轮质检瘦身（工具：scripts/prune_factors.py）
    "efx_forecast_revision_delay",   # 本轮质检瘦身（工具：scripts/prune_factors.py）
    "efx_top_net_rate",   # 本轮质检瘦身（工具：scripts/prune_factors.py）
    "efx_top_participation",   # 本轮质检瘦身（工具：scripts/prune_factors.py）
}
LATE_STARTS = {'efx_lu_board_density': '2019-08-14'}

# ★★ 2026-09-25 时间轴对齐（见 FactorSpec.align_fill）：
#   对齐区间（2018-01-01 ~ 真实起点）填什么。
#   · 这里列出的因子填 **0**：上游对该区间"确实不支持"，且 0 有明确含义。
#     `efx_lu_board_density` 吃 `stock_limit_up.boards`（文本 "3天2板"），
#     实测该字段 2018/2019/2020 各年**全为空**，2019-08-14 才出现 ⇒ 序列不存在。
#   · 未列出的因子填 **NaN**（默认）：上游干脆没有这张表/这段数据，
#     保持"不知道"的语义，不用 0 冒充观测。
ALIGN_FILL_ZERO = {'efx_lu_board_density'}


def ratio(a,b):
    a,b=np.asarray(a,dtype=float),np.asarray(b,dtype=float)
    out=np.full(a.shape,np.nan)
    np.divide(a,b,out=out,where=np.isfinite(b)&(b>0))
    return np.where(np.isfinite(out),out,np.nan)

def clock_minutes(series):
    """交易时钟：09:25集合竞价至09:30=0，11:30/13:00=120，15:00=240。午休缺失。"""
    s=series.astype('string').str.replace(':','',regex=False).str.zfill(6)
    valid=s.str.fullmatch(r'\d{6}',na=False).to_numpy(bool)
    h=pd.to_numeric(s.str[:2],errors='coerce').to_numpy(float)
    m=pd.to_numeric(s.str[2:4],errors='coerce').to_numpy(float)
    sec=pd.to_numeric(s.str[4:6],errors='coerce').to_numpy(float)
    t=h*60+m+sec/60
    valid &= (m<60)&(sec<60)&(((t>=565)&(t<=690))|((t>=780)&(t<=900)))
    return np.where(valid,np.where(t<=690,np.maximum(0,t-570),t-660),np.nan)

def fraction_text(s,pattern):
    a=s.astype('string').str.extract(pattern)
    return ratio(pd.to_numeric(a[0],errors='coerce'),pd.to_numeric(a[1],errors='coerce'))

def grid(ctx, ds, fields, fn, date_field='trade_date'):
    # 公告表按报告期分区；上一年报告可能在本年公告，不能仅按公告窗口裁年份。
    years=(int(ctx.panel.dates[0])//10000-(3 if date_field=='ann_date' else 0),int(ctx.panel.dates[-1])//10000)
    df=ctx.dataset(ds,columns=list(dict.fromkeys(['stock_code',date_field,*fields])),years=years)
    if df.empty:return ctx.panel.empty()
    day=ctx.date_col(df[date_field])
    cutoff=min(int(ctx.panel.dates[-1]),int((getattr(ctx.up,'audit_cutoff',None) or ctx.panel.dates[-1])))
    keep=(day>=ctx.panel.dates[0])&(day<=cutoff)&df.stock_code.isin(ctx.panel.codes).to_numpy()
    df=df.loc[keep].copy()
    if df.empty:return ctx.panel.empty()
    val=np.asarray(fn(df),float)
    effective=day[keep]
    if date_field=='ann_date':
        effective=ctx.panel.dates[np.searchsorted(ctx.panel.dates,effective,side='left')]
    d=pd.DataFrame({'stock_code':df.stock_code.to_numpy(),'day':effective,
                    'value':np.where(np.isfinite(val),val,np.nan)})
    # 龙虎榜同股同日有不同上榜原因；取记录中位数，不能把同一营业额重复相加。
    d=d.groupby(['stock_code','day'],sort=True,as_index=False).value.median()
    code=d.stock_code.map(ctx._pos).to_numpy(np.int32)
    return ctx.panel.place(code,d.day.to_numpy(np.int32),d.value.to_numpy())

def event_mean(values,window):
    """独立按每个窗口求和，避免累计和相减的旧历史残差破坏零值并列排名。"""
    values=np.asarray(values,dtype=np.float64)
    valid=np.isfinite(values)
    filled=np.where(valid,values,0.)
    total=np.zeros_like(values)
    count=np.zeros(values.shape,dtype=np.int32)
    for lag in range(min(window,len(values))):
        stop=len(values)-lag
        total[lag:]+=filled[:stop]
        count[lag:]+=valid[:stop]
    out=np.full(values.shape,np.nan)
    np.divide(total,count,out=out,where=count>0)
    return out


def add(name,ds,fields,fn,desc,formula,window=60,start=None,date_field='trade_date'):
    if name in REJECTED_CANDIDATES:return
    start=LATE_STARTS.get(name,start)
    SOURCE_FIELDS[name]={ds:list(fields)}
    @register(FactorSpec(name=name,group='field_events',deps=(ds,),desc=desc,version=3,
        formula=f'mean_{window}({formula}, observed rows only)',warmup_days=220,start=start,
        # ★ 对齐区间填充值：见 ALIGN_FILL_ZERO 与 FactorSpec.align_fill
        align_fill=(0.0 if name in ALIGN_FILL_ZERO else float("nan")),
        note='本地候选定义；D日收盘及当晚数据供D+1使用。无事件和字段缺失保持NaN；'
             '同股同日重复原因记录取中位数，再对窗口内有效事件取均值，至少1条；不是每日事件发生率。'
             '来源未保存逐次抓取时刻，历史回补可得性局限沿用平台约定；不宣称收益方向。'))
    def compute(ctx):return event_mean(grid(ctx,ds,fields,fn,date_field),window)
    globals()[name]=compute

SF='stock_finance';LL='stock_limit_list';LU='stock_limit_up';TL='stock_top_list';DT='stock_dragon_tiger'
add('efx_volume_ratio',SF,['volume_ratio'],lambda d:np.where(d.volume_ratio>0,np.log1p(d.volume_ratio),np.nan),'量比平滑','log1p(volume_ratio)',20)
add('efx_annual_earnings_yield',SF,['pe'],lambda d:ratio(np.ones(len(d)),d.pe),'静态正盈利收益率','1/pe where pe>0',20)
add('efx_annual_sales_yield',SF,['ps'],lambda d:ratio(np.ones(len(d)),d.ps),'静态销售收益率','1/ps where ps>0',20)
# up_stat 的来源口径为「次数/天数」，而非自然日字符串。
add('efx_top_participation',TL,['l_amount','amount'],lambda d:ratio(d.l_amount,d.amount),'龙虎榜成交参与度','l_amount/amount')
add('efx_top_net_rate',TL,['net_rate'],lambda d:d.net_rate,'龙虎榜净流入占比','net_rate')

def forecast_delay(d):
    ann=pd.to_datetime(d.ann_date,errors='coerce');first=pd.to_datetime(d.first_ann_date,errors='coerce')
    gap=(ann-first).dt.days.to_numpy(float)
    return np.where((first>=pd.Timestamp('1990-01-01'))&(gap>=0),gap,np.nan)

add('efx_forecast_revision_delay','stock_forecast',['ann_date','first_ann_date'],forecast_delay,
    '业绩预告距首次披露的间隔','calendar_days(ann_date-first_ann_date), available at ann_date',window=120,date_field='ann_date')
