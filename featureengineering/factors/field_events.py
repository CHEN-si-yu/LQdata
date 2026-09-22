"""未覆盖的日频和事件字段；仅作候选暴露，收益方向由下游研究。"""
import numpy as np
import pandas as pd
from fea.spec import FactorSpec, register

SOURCE_FIELDS = {}
# 全历史数据质量验收的保留/起点决定；不依据收益筛选。
REJECTED_CANDIDATES = {
    "efx_annual_sales_yield",   # 2026-09-22 去冗余：|ρ|≥0.95 簇的重复项，见 scripts/prune_factors.py 清单
    "efx_top_net_rate",   # 2026-09-22 去冗余：|ρ|≥0.95 簇的重复项，见 scripts/prune_factors.py 清单
    "efx_top_participation",   # 2026-09-22 去冗余：|ρ|≥0.95 簇的重复项，见 scripts/prune_factors.py 清单
}
LATE_STARTS = {'efx_lu_board_density': '2019-08-14'}


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
    @register(FactorSpec(name=name,group='field_events',deps=(ds,),desc=desc,version=2,
        formula=f'mean_{window}({formula}, observed rows only)',warmup_days=220,start=start,
        note='本地候选定义；D日收盘及当晚数据供D+1使用。无事件和字段缺失保持NaN；'
             '同股同日重复原因记录取中位数，再对窗口内有效事件取均值，至少1条；不是每日事件发生率。'
             '来源未保存逐次抓取时刻，历史回补可得性局限沿用平台约定；不宣称收益方向。'))
    def compute(ctx):return event_mean(grid(ctx,ds,fields,fn,date_field),window)
    globals()[name]=compute

SF='stock_finance';LL='stock_limit_list';LU='stock_limit_up';TL='stock_top_list';DT='stock_dragon_tiger'
add('efx_volume_ratio',SF,['volume_ratio'],lambda d:np.where(d.volume_ratio>0,np.log1p(d.volume_ratio),np.nan),'量比平滑','log1p(volume_ratio)',20)
add('efx_free_turnover',SF,['turnover_rate_f','turnover_rate'],lambda d:d.turnover_rate_f-d.turnover_rate,'自由流通口径换手溢价','turnover_rate_f-turnover_rate',20)
add('efx_annual_earnings_yield',SF,['pe'],lambda d:ratio(np.ones(len(d)),d.pe),'静态正盈利收益率','1/pe where pe>0',20)
add('efx_annual_sales_yield',SF,['ps'],lambda d:ratio(np.ones(len(d)),d.ps),'静态销售收益率','1/ps where ps>0',20)
add('efx_dividend_gap',SF,['dv_ratio','dv_ttm'],lambda d:d.dv_ratio-d.dv_ttm,'静态与滚动股息率差','dv_ratio-dv_ttm',20)
def ll(name,fields,fn,desc,formula):add(name,LL,fields,fn,desc,formula,start='2020-01-01')
ll('efx_limit_trade_share',['limit_amount','amount'],lambda d:ratio(d.limit_amount,d.amount),'涨跌停价成交金额占比','limit_amount/amount')
ll('efx_seal_float',['fd_amount','float_mv'],lambda d:ratio(d.fd_amount,d.float_mv),'封单金额流通市值比','fd_amount/float_mv')
ll('efx_limit_float_fraction',['float_mv','total_mv'],lambda d:ratio(d.float_mv,d.total_mv),'触板股票流通市值比例','float_mv/total_mv')
ll('efx_limit_turnover',['turnover_ratio'],lambda d:d.turnover_ratio,'触板日换手率','turnover_ratio')
ll('efx_limit_first',['first_time'],lambda d:clock_minutes(d.first_time)/240,'首次触板时刻','trading_minutes(first_time)/240')
ll('efx_limit_reseal_span',['first_time','last_time'],lambda d:np.where(clock_minutes(d.last_time)>=clock_minutes(d.first_time),(clock_minutes(d.last_time)-clock_minutes(d.first_time))/240,np.nan),'首次至最后触板交易时间差','(last_time-first_time)/240 trading minutes')
ll('efx_limit_openings',['open_times'],lambda d:np.log1p(d.open_times.where(d.open_times>=0)),'触板后开板次数','log1p(open_times)')
ll('efx_limit_streak',['limit_times'],lambda d:d.limit_times,'连板天数暴露','limit_times')
# up_stat 的来源口径为「次数/天数」，而非自然日字符串。
ll('efx_limit_win_fraction',['up_stat'],lambda d:fraction_text(d.up_stat,r'^(\d+)/(\d+)$'),'近期涨停次数密度','up_stat numerator / denominator')
ll('efx_limit_signed_move',['pct_chg','limit'],lambda d:np.where(d['limit'].isin(['U','D','Z']),d.pct_chg,np.nan),'触板日涨跌幅','pct_chg for known U/D/Z event types')
add('efx_lu_first',LU,['first_limit_time'],lambda d:clock_minutes(d.first_limit_time)/240,'涨停首次封板时刻','trading_minutes(first_limit_time)/240')
add('efx_lu_final',LU,['final_limit_time'],lambda d:clock_minutes(d.final_limit_time)/240,'涨停最终封板时刻','trading_minutes(final_limit_time)/240')
add('efx_lu_seal_amount',LU,['sealed_amount'],lambda d:np.log1p(d.sealed_amount.where(d.sealed_amount>=0)),'封单金额规模','log1p(sealed_amount in source units)')
add('efx_lu_seal_volume',LU,['sealed_volume'],lambda d:np.log1p(d.sealed_volume.where(d.sealed_volume>=0)),'封单数量规模','log1p(sealed_volume in source units)')
add('efx_lu_board_density',LU,['boards'],lambda d:fraction_text(d.boards,r'^(\d+)天(\d+)板$')**-1,'多日连板密度','boards_count / days parsed from boards; first-board text missing')
add('efx_lu_one_price',LU,['limit_type'],lambda d:np.where(d.limit_type.fillna('').str.len()>0,d.limit_type.fillna('').str.contains('一字').astype(float),np.nan),'一字涨停类型比例','contains_one_price(limit_type) among known text events')
add('efx_lu_move',LU,['change_percent'],lambda d:d.change_percent,'涨停日涨幅暴露','change_percent')
add('efx_top_imbalance',TL,['l_buy','l_sell'],lambda d:ratio(d.l_buy-d.l_sell,d.l_buy+d.l_sell),'龙虎榜买卖不平衡','(l_buy-l_sell)/(l_buy+l_sell)')
add('efx_top_participation',TL,['l_amount','amount'],lambda d:ratio(d.l_amount,d.amount),'龙虎榜成交参与度','l_amount/amount')
add('efx_top_net_rate',TL,['net_rate'],lambda d:d.net_rate,'龙虎榜净流入占比','net_rate')
add('efx_top_amount_rate',TL,['amount_rate'],lambda d:d.amount_rate,'龙虎榜成交占比原始口径','amount_rate')
add('efx_top_float',TL,['float_values'],lambda d:np.log1p(d.float_values.where(d.float_values>=0)),'上榜股票流通市值暴露','log1p(float_values)')
add('efx_top_turnover',TL,['turnover_rate'],lambda d:d.turnover_rate,'上榜股票换手暴露','turnover_rate')
add('efx_top_move',TL,['pct_change'],lambda d:d['pct_change'],'上榜日价格变化','pct_change')
add('efx_dragon_ratio_balance',DT,['buy_ratio','sell_ratio'],lambda d:d.buy_ratio-d.sell_ratio,'营业部买卖占比差','median(buy_ratio-sell_ratio across disclosed seats)')
add('efx_dragon_net_intensity',DT,['net_buy_amount','buy_amount','sell_amount'],lambda d:ratio(d.net_buy_amount,d.buy_amount+d.sell_amount),'营业部净买入强度','median(net_buy_amount/(buy_amount+sell_amount))')

def forecast_delay(d):
    ann=pd.to_datetime(d.ann_date,errors='coerce');first=pd.to_datetime(d.first_ann_date,errors='coerce')
    gap=(ann-first).dt.days.to_numpy(float)
    return np.where((first>=pd.Timestamp('1990-01-01'))&(gap>=0),gap,np.nan)

add('efx_forecast_revision_delay','stock_forecast',['ann_date','first_ann_date'],forecast_delay,
    '业绩预告距首次披露的间隔','calendar_days(ann_date-first_ann_date), available at ann_date',window=120,date_field='ann_date')
