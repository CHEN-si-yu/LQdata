import sys,json,time,subprocess
from pathlib import Path
a=Path(__file__).resolve().parent
while True:
 p=a/'status.json'
 if p.exists():
  o=json.loads(p.read_text())
  if o.get('phase')=='failed':sys.exit(1)
  if o.get('phase')=='built_and_checked':break
 time.sleep(5)
raise SystemExit(subprocess.call([sys.executable,'-B','-u',str(a/'verify_snapshot_alignment.py')]))
