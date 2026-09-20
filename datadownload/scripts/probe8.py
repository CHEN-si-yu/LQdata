import requests, json, time, gzip, io, os
from pathlib import Path as _P  # ★ 密钥不进仓库：从项目根 APIKey.txt 读（同 lingqi/client.py）
H={'apiKey':(_P(__file__).resolve().parents[1]/'APIKey.txt').read_text(encoding='utf-8').strip(),'Content-Type':'application/json'}
B='https://data.diemeng.chat/api'
def dump(date, level):
    t=time.time()
    r=requests.post(B+'/stock/daily_dump',headers=H,json={'date':date,'level':level},timeout=300, stream=True)
    raw=r.raw.read() if hasattr(r,'raw') else r.content
    el=time.time()-t
    hdr={k:v for k,v in r.headers.items() if k.lower() in ('content-type','content-encoding','content-length','content-disposition')}
    print(f"[{date} {level}] HTTP{r.status_code} {el:.1f}s bytes={len(raw):,} hdr={hdr}", flush=True)
    print("   head:", raw[:80], flush=True)
    body=raw
    # 尝试多种解压
    for name,fn in (('gzip',lambda b: gzip.decompress(b)), ('zlib',None)):
        if fn is None: continue
        try:
            body=fn(raw); print(f"   {name} decompressed -> {len(body):,} bytes", flush=True); break
        except Exception as e: print(f"   {name} fail: {type(e).__name__}", flush=True)
    if body[:2]==b'\x1f\x8b':
        print("   still gzip, try 2nd layer", flush=True)
        try: body=gzip.decompress(body); print("   ->",len(body), flush=True)
        except Exception as e: print("   fail",e, flush=True)
    try:
        j=json.loads(body)
        d=j.get('data')
        if isinstance(d,list):
            print(f"   JSON: code={j.get('code')} data=list n={len(d)} sample={json.dumps(d[0],ensure_ascii=False)[:200]}", flush=True)
        elif isinstance(d,dict):
            ks=list(d.keys())
            print(f"   JSON: code={j.get('code')} data=dict keys={len(ks)} first3={ks[:3]}", flush=True)
            k0=ks[0]; print(f"     {k0} -> n={len(d[k0])} sample={json.dumps(d[k0][:2],ensure_ascii=False)[:200]}", flush=True)
        else:
            print(f"   JSON top keys={list(j.keys())} data type={type(d)}", flush=True)
    except Exception as e:
        print(f"   JSON parse fail: {e}; body head={body[:300]!r}", flush=True)
    return len(raw)

print("=== daily_dump 测试 (2020-01-02) ===")
dump('2020-01-02','daily')
