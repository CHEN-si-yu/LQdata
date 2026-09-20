import requests, json, time
from pathlib import Path as _P  # ★ 密钥不进仓库：从项目根 APIKey.txt 读（同 lingqi/client.py）
H={'apiKey':(_P(__file__).resolve().parents[1]/'APIKey.txt').read_text(encoding='utf-8').strip(),'Content-Type':'application/json'}
B='https://data.diemeng.chat/api'
def call(payload, tag, path='/stock/daily'):
    t=time.time()
    r=requests.post(B+path, headers=H, json=payload, timeout=180)
    el=time.time()-t
    print(f"[{tag}] {r.status_code} {el:.2f}s bytes={len(r.content):,}", end=' ')
    try:
        j=r.json()
        d=j.get('data')
        if isinstance(d,dict): print(f"total={d.get('total')} n={len(d.get('list') or [])} keys={list(d.keys())}")
        elif isinstance(d,list): print(f"list n={len(d)}")
        else: print("data:", str(d)[:200])
    except Exception as e: print("parse err", e); print(r.content[:300])

# month range no code
call({'start_time':'2015-01-01','end_time':'2015-01-31','page':0,'page_size':10000}, 'month-2015-01')
# year range no code
call({'start_time':'2015-01-01','end_time':'2015-12-31','page':0,'page_size':10000}, 'year-2015')
# does page work (should be empty if all returned)
call({'start_time':'2015-01-05','end_time':'2015-01-05','page':1,'page_size':10000}, 'page1-single-day')
# small page_size - does it cap?
call({'start_time':'2015-01-05','end_time':'2015-01-05','page':0,'page_size':10}, 'smallpage')
# 100 stocks array multi-day
call({'stock_code':['600000.SH','000001.SZ'],'start_time':'2020-01-01','end_time':'2020-12-31','page':0,'page_size':10000}, 'codes2-year')
