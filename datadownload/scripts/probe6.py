import requests, json, time
from pathlib import Path as _P  # ★ 密钥不进仓库：从项目根 APIKey.txt 读（同 lingqi/client.py）
H={'apiKey':(_P(__file__).resolve().parents[1]/'APIKey.txt').read_text(encoding='utf-8').strip(),'Content-Type':'application/json'}
B='https://data.diemeng.chat/api'
def q(tag,path,payload,method='POST',show_raw=False):
    t=time.time()
    try:
        r=requests.get(B+path,headers=H,params=payload,timeout=180) if method=='GET' else requests.post(B+path,headers=H,json=payload,timeout=180)
        el=time.time()-t
        try:
            j=r.json(); d=j.get('data')
            if isinstance(d,dict): n=len(d.get('list') or []); tot=d.get('total')
            elif isinstance(d,list): n=len(d); tot='-'
            else: n=tot=0
            extra='' if j.get('code')==200 else f"  <<< code={j.get('code')} msg={j.get('msg')!r}"
            print(f"  {tag:12s} HTTP{r.status_code} code={j.get('code')} n={n:<6} {el:5.2f}s{extra}", flush=True)
        except Exception:
            print(f"  {tag:12s} HTTP{r.status_code} NONJSON ct={r.headers.get('content-type')} len={len(r.content)} head={r.content[:200]!r} {el:.2f}s", flush=True)
    except Exception as e:
        print(f"  {tag:12s} EXC {type(e).__name__} {e}", flush=True)

# 重复同一请求 8 次，间隔 0.3s —— 看是否限流
print("=== 同一请求连发8次 (2020-01-02 daily) ===")
for i in range(8):
    q(f'try{i}','/stock/daily',{'start_time':'2020-01-02','end_time':'2020-01-02'}); time.sleep(0.3)
print("=== market_distribution 连发3次 ===")
for i in range(3):
    q(f'mkt{i}','/stock/market_distribution_history',{'date':'2020-01-02'}); time.sleep(0.5)
print("=== 高频压测: 60次快速请求 (测 280/min 限流) ===")
ok=0; bad=0; t0=time.time()
for i in range(60):
    try:
        r=requests.post(B+'/stock/daily',headers=H,json={'start_time':'2020-01-02','end_time':'2020-01-02'},timeout=60)
        j=r.json()
        if r.status_code==200 and j.get('code')==200: ok+=1
        else: bad+=1; print(f"   #{i} HTTP{r.status_code} code={j.get('code')} msg={j.get('msg')!r}", flush=True)
    except Exception as e:
        bad+=1; print(f"   #{i} EXC {e}", flush=True)
el=time.time()-t0
print(f"  => ok={ok} bad={bad} elapsed={el:.1f}s rate={60/el*60:.0f}/min")
