import requests, json, time
from pathlib import Path as _P  # ★ 密钥不进仓库：从项目根 APIKey.txt 读（同 lingqi/client.py）
H={'apiKey':(_P(__file__).resolve().parents[1]/'APIKey.txt').read_text(encoding='utf-8').strip(),'Content-Type':'application/json'}
B='https://data.diemeng.chat/api'
def q(tag,path,payload,method='POST'):
    t=time.time()
    try:
        r=requests.get(B+path,headers=H,params=payload,timeout=180) if method=='GET' else requests.post(B+path,headers=H,json=payload,timeout=180)
        el=time.time()-t; j=r.json(); d=j.get('data')
        if isinstance(d,dict): n=len(d.get('list') or []); tot=d.get('total')
        elif isinstance(d,list): n=len(d); tot='-'
        else: n=tot=0
        print(f"  {tag:10s} n={n:<7} total={tot} {el:.1f}s", flush=True)
    except Exception as e: print(f"  {tag:10s} ERR {e}", flush=True)

print("=== 各数据集覆盖起点 (采样) ===")
years=['2005-01-04','2010-01-04','2015-01-05','2018-01-02','2020-01-02','2022-01-04','2024-01-02']
for y in years:
    print(f"-- {y} --")
    q('daily',   '/stock/daily',{'start_time':y,'end_time':y})
    q('finance', '/stock/finance',{'start_time':y,'end_time':y})
    q('mflow',   '/stock/main_fund_flow',{'start_time':y,'end_time':y})
    q('limit_up','/stock/limit_up',{'start_time':y,'end_time':y})
    q('limit_ls','/stock/limit_list',{'start_time':y,'end_time':y})
    q('mmargin', '/stock/margin_detail',{'start_time':y,'end_time':y})
    q('cyq',     '/stock/cyq_chips',{'start_time':y,'end_time':y})
    q('toplist', '/stock/top_list',{'trade_date':y})
    q('holder',  '/stock/holder_number',{'start_time':y,'end_time':y})
    q('pledge',  '/stock/pledge_stat',{'start_time':y,'end_time':y})
    q('mktdist', '/stock/market_distribution_history',{'date':y})
    q('thshot',  '/ths/hot',{'trade_date':y},'GET')
    q('dtiger',  '/stock/dragon_tiger',{'date':y})
