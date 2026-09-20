import requests, json, time, concurrent.futures as cf
from pathlib import Path as _P  # ★ 密钥不进仓库：从项目根 APIKey.txt 读（同 lingqi/client.py）
H={'apiKey':(_P(__file__).resolve().parents[1]/'APIKey.txt').read_text(encoding='utf-8').strip(),'Content-Type':'application/json'}
B='https://data.diemeng.chat/api'
S=requests.Session(); S.headers.update(H)
# 生成 2023 年交易日列表
cal=requests.post(B+'/basic/calendar',headers=H,json={'start_time':'2023-01-01','end_time':'2023-12-31'},timeout=60).json()['data']
days=[d['date'] for d in cal if d['is_open']]
print("2023 交易日:",len(days))
def one(d):
    t=time.time()
    try:
        r=S.post(B+'/stock/daily',json={'start_time':d,'end_time':d},timeout=120)
        j=r.json()
        n=len((j.get('data') or {}).get('list') or [])
        return d,n,j.get('code'),r.status_code,time.time()-t
    except Exception as e:
        return d,-1,str(e)[:40],0,time.time()-t
for conc in (1,6,12):
    sub=days[:36]
    t0=time.time()
    with cf.ThreadPoolExecutor(conc) as ex:
        res=list(ex.map(one,sub))
    el=time.time()-t0
    empty=[r for r in res if r[1]==0]
    err=[r for r in res if r[1]<0]
    counts=[r[1] for r in res if r[1]>0]
    print(f"conc={conc:2d} n={len(sub)} elapsed={el:6.1f}s rate={len(sub)/el*60:6.1f}/min "
          f"empty={len(empty)} err={len(err)} min_rows={min(counts) if counts else 0} max_rows={max(counts) if counts else 0}")
    if empty: print("   empty days:", [e[0] for e in empty][:12])
    if err: print("   errs:", [(e[0],e[2]) for e in err][:5])
