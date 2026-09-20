"""扫描各数据集的数据起始年份：每年取首个交易日采样，带重试以排除瞬时空响应。"""
import requests, json, time, sys
from pathlib import Path as _P  # ★ 密钥不进仓库：从项目根 APIKey.txt 读（同 lingqi/client.py）
H={'apiKey':(_P(__file__).resolve().parents[1]/'APIKey.txt').read_text(encoding='utf-8').strip(),'Content-Type':'application/json'}
B='https://data.diemeng.chat/api'
S=requests.Session(); S.headers.update(H)
cal=S.post(B+'/basic/calendar',json={'start_time':'1990-01-01','end_time':'2026-12-31'},timeout=60).json()['data']
open_days=[d['date'] for d in cal if d['is_open']]
first_of_year={}
for d in open_days:
    y=d[:4]
    if y not in first_of_year: first_of_year[y]=d
YEARS=sorted(first_of_year)
# 每组: (名称, path, params生成器, method)
GROUPS=[
 ('daily',        '/stock/daily',            lambda d:{'start_time':d,'end_time':d}),
 ('daily_adj',    '/stock/daily_adj',        lambda d:{'start_time':d,'end_time':d}),
 ('adj_factor',   '/stock/adj_factor',       lambda d:{'start_time':d,'end_time':d}),
 ('finance',      '/stock/finance',          lambda d:{'start_time':d,'end_time':d}),
 ('main_fund_flow','/stock/main_fund_flow',  lambda d:{'start_time':d,'end_time':d}),
 ('limit_up',     '/stock/limit_up',         lambda d:{'start_time':d,'end_time':d}),
 ('limit_list',   '/stock/limit_list',       lambda d:{'start_time':d,'end_time':d}),
 ('margin_detail','/stock/margin_detail',    lambda d:{'start_time':d,'end_time':d}),
 ('cyq_chips',    '/stock/cyq_chips',        lambda d:{'start_time':d,'end_time':d}),
 ('holder_number','/stock/holder_number',    lambda d:{'start_time':d,'end_time':d}),
 ('pledge_stat',  '/stock/pledge_stat',      lambda d:{'start_time':d,'end_time':d}),
 ('top_list',     '/stock/top_list',         lambda d:{'trade_date':d}),
 ('dragon_tiger', '/stock/dragon_tiger',     lambda d:{'date':d}),
 ('st_info',      '/stock/st_info',          lambda d:{'start_time':d,'end_time':d}),
 # dc_block_flow / ths_block_flow / ths_hot 已于 2026-09-19 随数据集一并删除（时间覆盖不足）
]
def probe(path, params):
    for attempt in range(4):
        try:
            r=S.post(B+path,json=params,timeout=120)
            if r.status_code!=200: time.sleep(1.5); continue
            j=r.json()
            if j.get('code')!=200: time.sleep(1.5); continue
            d=j.get('data')
            if isinstance(d,dict): return d.get('total') or len(d.get('list') or [])
            if isinstance(d,list): return len(d)
            return 0
        except Exception: time.sleep(1.5)
    return None
out={}
for name,path,pf in GROUPS:
    row={}
    for y in YEARS:
        d=first_of_year[y]
        v=probe(path, pf(d))
        row[y]=v
        if v: 
            # 找到首个有数据的年份后，继续扫到 2026 以记录完整覆盖
            pass
        time.sleep(0.15)
    out[name]=row
    nz=[y for y,v in row.items() if v]
    print(f"{name:16s} 首个有数据年份={nz[0] if nz else 'NONE':>5s}  逐年后值={ {y:row[y] for y in YEARS[::3]} }", flush=True)
json.dump(out, open('probe/coverage_by_year.json','w'), ensure_ascii=False, indent=1)
print("saved probe/coverage_by_year.json")
