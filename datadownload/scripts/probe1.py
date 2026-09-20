import requests, json, time
from pathlib import Path as _P  # ★ 密钥不进仓库：从项目根 APIKey.txt 读（同 lingqi/client.py）
H={'apiKey':(_P(__file__).resolve().parents[1]/'APIKey.txt').read_text(encoding='utf-8').strip(),'Content-Type':'application/json'}
B='https://data.diemeng.chat/api'
def call(path, payload=None, method='POST', tag='', show=1):
    t=time.time()
    try:
        if method=='GET': r=requests.get(B+path, headers=H, params=payload or {}, timeout=60)
        else: r=requests.post(B+path, headers=H, json=payload or {}, timeout=60)
        el=time.time()-t; ct=r.headers.get('content-type','')
        print(f"[{tag}] {method} {path} -> {r.status_code} {el:.2f}s ct={ct} len={len(r.content)}", flush=True)
        if 'json' in ct:
            j=r.json()
            if isinstance(j.get('data'), dict):
                d=j['data']; print(f"   dict keys={list(d.keys())[:8]} total={d.get('total')} n={len(d.get('list') or [])}", flush=True)
            elif isinstance(j.get('data'), list):
                print(f"   list n={len(j['data'])}", flush=True)
            else: print(f"   data type={type(j.get('data'))} val={str(j.get('data'))[:100]}", flush=True)
            if show and isinstance(j.get('data'),dict) and j['data'].get('list'):
                print("   sample:", json.dumps(j['data']['list'][0], ensure_ascii=False)[:400], flush=True)
            return j
        else:
            print("   raw head:", r.content[:100], flush=True); return r.content
    except Exception as e:
        print(f"[{tag}] {path} ERROR {e}", flush=True); return None

print("="*70)
call('/basic/calendar', {'start_time':'1990-01-01','end_time':'2026-12-31'}, tag='calendar')
call('/stock/list', None, 'GET', tag='list')
call('/stock/daily', {'start_time':'2015-01-05','end_time':'2015-01-05','page':0,'page_size':10000}, tag='daily2015')
call('/stock/daily', {'start_time':'2026-09-10','end_time':'2026-09-10','page':0,'page_size':10000}, tag='daily20260910')
call('/stock/daily', {'start_time':'2005-01-04','end_time':'2005-01-04','page':0,'page_size':10000}, tag='daily2005')
call('/stock/daily', {'start_time':'1995-01-03','end_time':'1995-01-03','page':0,'page_size':10000}, tag='daily1995')
