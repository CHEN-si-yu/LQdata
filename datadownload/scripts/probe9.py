import requests, json, time, gzip
from pathlib import Path as _P  # ★ 密钥不进仓库：从项目根 APIKey.txt 读（同 lingqi/client.py）
H={'apiKey':(_P(__file__).resolve().parents[1]/'APIKey.txt').read_text(encoding='utf-8').strip(),'Content-Type':'application/json'}
B='https://data.diemeng.chat/api'
def dump(date, level):
    t=time.time()
    r=requests.post(B+'/stock/daily_dump',headers=H,json={'date':date,'level':level},timeout=300)
    raw=r.content; el=time.time()-t
    print(f"[{date} {level}] HTTP{r.status_code} {el:.1f}s bytes={len(raw):,} ct={r.headers.get('content-type')} ce={r.headers.get('content-encoding')}", flush=True)
    body=raw
    if raw[:2]==b'\x1f\x8b':
        try: body=gzip.decompress(raw); print(f"   gunzip -> {len(body):,}B", flush=True)
        except Exception as e: print("   gunzip fail", e, flush=True); return
    try:
        j=json.loads(body); d=j.get('data')
        if isinstance(d,list): print(f"   code={j.get('code')} data=list n={len(d)} sample={json.dumps(d[0],ensure_ascii=False)[:180]}", flush=True)
        elif isinstance(d,dict):
            ks=list(d.keys()); print(f"   code={j.get('code')} data=dict stocks={len(ks)} first={ks[0]}", flush=True)
            print(f"     {ks[0]} n_bars={len(d[ks[0]])} sample={json.dumps(d[ks[0]][:2],ensure_ascii=False)}", flush=True)
        else: print(f"   code={j.get('code')} msg={j.get('msg')!r} data={str(d)[:120]}", flush=True)
    except Exception as e: print("   parse fail:",e, body[:200], flush=True)

for lv in ['daily','5min']:
    dump('2026-09-10', lv)
