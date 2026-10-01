import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

root = Path('/root/autodl-fs/model').resolve()
audit = Path(__file__).resolve().parent
live = root / 'trainingdata'
stage = audit / 'new_trainingdata'
backup = audit / 'previous_trainingdata'
assert audit.is_relative_to(root / 'history')
assert live.resolve() == root / 'trainingdata'
assert stage.resolve() == audit / 'new_trainingdata'
assert backup.resolve() == audit / 'previous_trainingdata'
assert not live.is_symlink() and not stage.is_symlink()
assert live.is_dir() and stage.is_dir() and not backup.exists()
status = json.loads((audit / 'status.json').read_text())
verification = json.loads((audit / 'alignment_verification.json').read_text())
assert status['phase'] == 'built_and_checked' and status['source_stable']
assert verification['ok'] and verification['phase'] == 'verified'
assert (live / 'meta.json').read_text() == (audit / 'original_meta.json').read_text(), 'Original snapshot changed during construction'

before = json.loads((audit / 'source_before.json').read_text())
current = {}
for name in ['factors', 'market_factors']:
    for path in (root.parent / 'featureengineering/data' / name).glob('*/year=*/data.parquet'):
        st = path.stat()
        current[str(path)] = [st.st_size, st.st_mtime_ns]
assert current == before, 'Factor source changed after verification'

ps = subprocess.run(['ps', '-eo', 'pid,args'], capture_output=True, text=True, check=True)
busy = []
for line in ps.stdout.splitlines():
    if 'bash -c' in line:
        continue
    if any('/model/' + segment + '/' in line for segment in ['experiments', 'experiments2', 'releases']) and any(token in line for token in ['model.py', 'analysis.py', 'run.py']):
        busy.append(line)
    if '/model/preparingdata.py' in line:
        busy.append(line)
assert not busy, 'A model process is active: ' + repr(busy)

meta_text = (stage / 'meta.json').read_text()
meta = json.loads(meta_text)
assert meta['axis']['end'] == verification['cutoff']
assert meta['built_years'] == [str(y) for y in range(2018, 2027)]
identities = {}
for path in stage.glob('*/year=*/data.parquet'):
    st = path.stat()
    identities[str(path.relative_to(stage))] = [st.st_size, st.st_mtime_ns]
assert len(identities) == 54

os.rename(live, backup)
try:
    os.rename(stage, live)
except BaseException:
    os.rename(backup, live)
    raise

assert (live / 'meta.json').read_text() == meta_text
for relative, identity in identities.items():
    st = (live / relative).stat()
    assert [st.st_size, st.st_mtime_ns] == identity
publication = {
    'published_at': datetime.now().isoformat(timespec='seconds'),
    'current_snapshot': str(live), 'previous_snapshot': str(backup),
    'source_stable': True, 'checked_year_partition_files': len(identities),
    'source_objects': 550, 'stock_factors': len(meta['columns']['features']),
    'labels': len(meta['columns']['labels']),
    'market_factors': meta['market_factors']['n_factors'],
    'axis': {k: v for k, v in meta['axis'].items() if k != 'codes'},
    'semantics': meta['semantics'], 'alignment_verification_ok': True,
    'inherited_upstream_market_gap_warnings': verification['inherited_upstream_market_gap_warnings'],
    'factor_source_modified': False, 'model_code_modified': False,
}
(audit / 'publication.json').write_text(json.dumps(publication, ensure_ascii=False, indent=2))
print(json.dumps(publication, ensure_ascii=False, indent=2))
