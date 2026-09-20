import requests, json, time
from pathlib import Path as _P  # ★ 密钥不进仓库：从项目根 APIKey.txt 读（同 lingqi/client.py）
H={'apiKey':(_P(__file__).resolve().parents[1]/'APIKey.txt').read_text(encoding='utf-8').strip(),'Content-Type':'application/json'}
B='https://data.diemeng.chat/api'
def call(tag, path, payload=None, method='POST', show=None):
    t=time.time()
    r=requests.get(B+path, headers=H, params=payload or {}, timeout=180) if method=='GET' else requests.post(B+path, headers=H, json=payload or {}, timeout=180)
    el=time.time()-t
    try:
        j=r.json(); d=j.get('data')
        if isinstance(d,dict): n=len(d.get('list') or []); tot=d.get('total')
        elif isinstance(d,list): n=len(d); tot=None
        else: n=tot=None
        print(f"{tag:32s} {r.status_code} {el:6.2f}s {len(r.content):>11,}B total={tot} n={n}", flush=True)
        if show and isinstance(d,dict) and d.get('list'): print("    >", json.dumps(d['list'][0],ensure_ascii=False)[:300], flush=True)
        return j
    except Exception as e: print(f"{tag:32s} {r.status_code} parse-err {r.content[:120]!r}", flush=True)

print("--- 数据起点 (daily) ---")
for d in ['1990-12-19','1991-01-02','1992-01-02','1993-01-04','1995-01-03','2000-01-04']:
    call(f'daily {d}','/stock/daily',{'start_time':d,'end_time':d})
print("--- 全市场维度 (无 stock_code) ---")
call('finance 全市场 40天','/stock/finance',{'start_time':'2026-08-01','end_time':'2026-09-10'})
call('cyq_chips 全市场1天','/stock/cyq_chips',{'start_time':'2026-09-10','end_time':'2026-09-10'})
call('main_fund_flow 全市场1天','/stock/main_fund_flow',{'start_time':'2026-09-10','end_time':'2026-09-10'})
call('holder_number 全市场','/stock/holder_number',{'start_time':'2026-01-01','end_time':'2026-09-10'})
call('pledge_stat 全市场','/stock/pledge_stat',{'start_time':'2026-01-01','end_time':'2026-09-10'})
call('margin_detail 全市场1天','/stock/margin_detail',{'start_time':'2026-09-10','end_time':'2026-09-10'})
call('st_info 全历史','/stock/st_info',{'start_time':'1990-01-01','end_time':'2026-09-10'})
call('suspension 全历史','/stock/suspension',{})
print("--- 指数 ---")
call('index_daily 全历史','/index/daily',{'stock_code':'000001.SH','start_date':'1990-01-01','end_date':'2026-09-10'})
call('tdx_blocks type3 指数','/tdx/blocks',{'block_type':3},'GET')
print("--- 慢接口重测 ---")
call('market_dist 2026-09-10','/stock/market_distribution_history',{'date':'2026-09-10'})
call('market_dist 2026-09-09','/stock/market_distribution_history',{'date':'2026-09-09'})
