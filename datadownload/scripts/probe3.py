import requests, json, time
from pathlib import Path as _P  # ★ 密钥不进仓库：从项目根 APIKey.txt 读（同 lingqi/client.py）
H={'apiKey':(_P(__file__).resolve().parents[1]/'APIKey.txt').read_text(encoding='utf-8').strip(),'Content-Type':'application/json'}
B='https://data.diemeng.chat/api'
D='2026-09-10'; S='2026-08-01'; E='2026-09-10'
R=[]
def call(tag, path, payload=None, method='POST'):
    t=time.time()
    try:
        if method=='GET': r=requests.get(B+path, headers=H, params=payload or {}, timeout=120)
        else: r=requests.post(B+path, headers=H, json=payload or {}, timeout=120)
        el=time.time()-t; ct=r.headers.get('content-type','')
        desc=''
        try:
            j=r.json()
            d=j.get('data')
            if isinstance(d,dict):
                desc=f"total={d.get('total')} n={len(d.get('list') or [])} keys={sorted(d.keys())}"
            elif isinstance(d,list):
                desc=f"list n={len(d)}; top-level keys={sorted(j.keys())}"
            else:
                desc=f"data={type(d).__name__}"
            if j.get('code')!=200: desc += f" CODE={j.get('code')} msg={j.get('msg')}"
        except Exception:
            desc=f"non-json ct={ct} head={r.content[:60]!r}"
        print(f"{tag:34s} {r.status_code} {el:5.2f}s {len(r.content):>10,}B  {desc}", flush=True)
    except Exception as e:
        print(f"{tag:34s} ERROR {e}", flush=True)

# --- basic ---
call('calendar','/basic/calendar',{'start_time':S,'end_time':E})
call('stock_list','/stock/list',None,'GET')
# --- financial ---
call('financial_indicator','/stock/financial_indicator',{'stock_code':'600000.SH'})
call('income','/stock/income',{'stock_code':'600000.SH'})
call('balancesheet','/stock/balancesheet',{'stock_code':'600000.SH'})
call('cashflow','/stock/cashflow',{'stock_code':'600000.SH'})
call('finance(daily)','/stock/finance',{'stock_code':'600000.SH','start_time':S,'end_time':E})
call('forecast','/stock/forecast',{'stock_code':'600000.SH'})
# --- quotes ---
call('daily_1stock','/stock/daily',{'stock_code':'600000.SH','start_time':'2020-01-01','end_time':E})
call('market_distribution','/stock/market_distribution_history',{'date':D})
call('kline_weekly','/stock/kline',{'period':'weekly','stock_code':'600000.SH','start_time':'2020-01-01','end_time':E})
call('daily_adj','/stock/daily_adj',{'stock_code':'600000.SH','start_time':'2024-01-01','end_time':E})
call('min_adj','/stock/min_adj',{'stock_code':'600000.SH','level':'5min','start_time':D+' 09:00:00','end_time':D+' 15:30:00'})
call('adj_factor','/stock/adj_factor',{'stock_code':'600000.SH','start_time':'2024-01-01','end_time':E})
call('adj_factor_changes','/stock/adj_factor/changes',{'date':D})
call('history_min','/stock/history',{'stock_code':'600000.SH','level':'5min','start_time':D+' 09:00:00','end_time':D+' 15:30:00'})
call('suspension','/stock/suspension',{'trade_date':D},'GET')
call('st_info','/stock/st_info',{'start_time':S,'end_time':E})
call('macd','/stock/macd',{'stock_code':'600000.SH','start_time':S,'end_time':E})
call('kdj','/stock/kdj',{'stock_code':'600000.SH','start_time':S,'end_time':E})
call('rsi','/stock/rsi',{'stock_code':'600000.SH','start_time':S,'end_time':E})
call('boll','/stock/boll',{'stock_code':'600000.SH','start_time':S,'end_time':E})
call('ma','/stock/ma',{'stock_code':'600000.SH','start_time':S,'end_time':E})
call('mavol','/stock/mavol',{'stock_code':'600000.SH','start_time':S,'end_time':E})
# --- independent ---
call('limit_up','/stock/limit_up',{'start_time':S,'end_time':E})
call('limit_list','/stock/limit_list',{'start_time':S,'end_time':E})
call('main_fund_flow','/stock/main_fund_flow',{'stock_code':'600000.SH','start_time':S,'end_time':E})
# ths_hot / ths_block_fund_flow / dc_block_fund_flow 已于 2026-09-19 随数据集一并删除
call('cyq_chips','/stock/cyq_chips',{'stock_code':'600000.SH','start_time':S,'end_time':E})
call('holder_number','/stock/holder_number',{'stock_code':'600000.SH'})
call('pledge_stat','/stock/pledge_stat',{'stock_code':'600000.SH'})
call('margin_detail','/stock/margin_detail',{'stock_code':'600000.SH','start_time':S,'end_time':E})
call('dragon_tiger','/stock/dragon_tiger',{'date':D})
call('top_list','/stock/top_list',{'trade_date':D})
# --- index ---
call('index_history','/index/history',{'index_code':'000001.SH','level':'5min','start_time':D+' 09:00:00','end_time':D+' 15:30:00'})
call('index_daily','/index/daily',{'stock_code':'000001.SH','start_date':'2020-01-01','end_date':E})
call('index_macd','/index/macd',{'index_code':'000001.SH','start_time':S,'end_time':E})
call('index_kdj','/index/kdj',{'index_code':'000001.SH','start_time':S,'end_time':E})
call('index_rsi','/index/rsi',{'index_code':'000001.SH','start_time':S,'end_time':E})
call('index_boll','/index/boll',{'index_code':'000001.SH','start_time':S,'end_time':E})
call('index_ma','/index/ma',{'index_code':'000001.SH','start_time':S,'end_time':E})
call('tdx_blocks','/tdx/blocks',{'block_type':2},'GET')
# tdx_block_stocks 已于 2026-09-19 随数据集一并删除（快照表）
call('tdx_daily','/tdx/daily',{'board_code':'880471','start_date':'2026-08-01','end_date':E},'GET')
call('tdx_minute','/tdx/minute',{'board_code':'880201','level':'5min','start_time':D+' 09:00:00','end_time':D+' 15:30:00'})
call('dc_blocks','/dc/blocks',{})
call('dc_daily','/dc/daily',{'block_code':'BK0428.DC','trade_date':D})
call('ths_sector_categories','/index/ths_sector_categories',{})
call('ths_constituent_stocks','/index/ths_constituent_stocks',{'index_code':'882001.TI'})
call('ths_daily','/index/ths_daily',{'ths_code':'881123.TI','start_time':'2026-08-01','end_time':E})
