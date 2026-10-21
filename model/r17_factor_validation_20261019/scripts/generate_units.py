#!/usr/bin/env python3
"""Create isolated A/B experiment units from the four release recipes.

A = the feature set each release model used for its 2026-10-08 live retrain.
B = A + the 50 R17 stock factors.
Only the appended patch at the end of each copied model.py differs.
"""
from __future__ import annotations
import hashlib, json, shutil
from pathlib import Path

WORK = Path(__file__).resolve().parent.parent
MODEL = WORK.parent
RELEASES = MODEL / 'releases'
SNAP = WORK / 'snapshots'
UNITS = WORK / 'units'

sets = json.loads((WORK/'scripts/feature_sets.json').read_text())
recipes = json.loads((WORK/'scripts/release_recipes_live.json').read_text())
new = sets['new']
fam = sets['family']
by_fam = {}
for n in new:
    by_fam.setdefault(fam[n], []).append(n)
for k in by_fam: by_fam[k].sort()
NEW_FLOW = by_fam.get('chip_geometry', []) + by_fam.get('chip_event_C', []) + by_fam.get('cross_network', [])
NEW_FUND = by_fam.get('asset_composition', []) + by_fam.get('liability_composition', []) + by_fam.get('reopen', [])
NEW_TECH = []
assert len(NEW_FLOW) == 20 and len(NEW_FUND) == 30 and not set(NEW_FLOW) & set(NEW_FUND)
assert len(NEW_FLOW)+len(NEW_FUND)+len(NEW_TECH) == 50

def meta_hash(path):
    return hashlib.sha256((path/'meta.json').read_bytes()).hexdigest()

def meta_info(path):
    m = json.loads((path/'meta.json').read_text())
    return meta_hash(path), m['axis']['end'], len(m['columns']['features'])

def patch_code(batch, model, snap_hash, snap_end):
    nf, nu = (NEW_FLOW, NEW_FUND) if batch == 'B' else ([], [])
    nt = NEW_TECH if batch == 'B' else []
    base = recipes[model + '_live']
    bf = list(base['feature_names'])
    bfl = list(base['flow_feature_names'])
    bfu = list(base['fund_feature_names'])
    bte = list(base['technical_feature_names'])
    return f'''
# ================= R17 factor-batch validation patch (generated) =================
_R17_BATCH = {batch!r}
_R17_NEW_FLOW = {nf!r}
_R17_NEW_FUND = {nu!r}
_R17_NEW_TECH = {nt!r}
_R17_BASE_FEATURES = {bf!r}
_R17_BASE_FLOW = {bfl!r}
_R17_BASE_FUND = {bfu!r}
_R17_BASE_TECH = {bte!r}

def _r17_extend(names, extra):
    out = list(names)
    for n in extra:
        if n not in out:
            out.append(n)
    return out

# The release model defines the 260/146-feature migration only under MX_LIVE.
# We keep experiment-mode geometry (fixed 25Q3-26Q2 quarters) but use the
# release model's current live feature migration as batch A, then append R17.
RECIPE['feature_names'] = _r17_extend(_R17_BASE_FEATURES, _R17_NEW_FLOW + _R17_NEW_FUND + _R17_NEW_TECH)
RECIPE['flow_feature_names'] = _r17_extend(_R17_BASE_FLOW, _R17_NEW_FLOW)
RECIPE['fund_feature_names'] = _r17_extend(_R17_BASE_FUND, _R17_NEW_FUND)
RECIPE['technical_feature_names'] = _r17_extend(_R17_BASE_TECH, _R17_NEW_TECH)
RECIPE['flow_indices'] = [RECIPE['feature_names'].index(n) for n in RECIPE['flow_feature_names']]
RECIPE['fund_indices'] = [RECIPE['feature_names'].index(n) for n in RECIPE['fund_feature_names']]
RECIPE['technical_indices'] = [RECIPE['feature_names'].index(n) for n in RECIPE['technical_feature_names']]
RECIPE['snapshot_cutoff'] = {snap_end!r}
RECIPE['snapshot_meta_sha256'] = {snap_hash!r}
RECIPE['name'] = {batch + '_' + model!r}
RECIPE['r17_validation'] = dict(batch={batch!r}, base_model={model!r}, new_flow=len(_R17_NEW_FLOW),
                               new_fund=len(_R17_NEW_FUND), new_technical=len(_R17_NEW_TECH),
                               stock_feature_count=len(RECIPE['feature_names']))
# ================= end R17 patch =================

'''

def build_unit(batch, model):
    snap = SNAP / ('A_260' if batch == 'A' else 'B_310')
    h, end, nfeat = meta_info(snap)
    dst = UNITS / f'{batch}_{model}'
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    for name in ('run.py', 'model.py', 'analysis.py'):
        shutil.copy2(RELEASES/model/name, dst/name)
    p = dst/'model.py'
    text = p.read_text()
    marker = "if __name__ == '__main__':"
    idx = text.rfind(marker)
    assert idx != -1, p
    text = text[:idx] + patch_code(batch, model, h, end) + text[idx:]
    p.write_text(text)
    return dict(batch=batch, model=model, unit=str(dst.relative_to(MODEL)), snapshot=str(snap.relative_to(MODEL)),
                snapshot_meta_sha256=h, snapshot_end=end, snapshot_features=nfeat,
                new_flow=len(NEW_FLOW) if batch=='B' else 0, new_fund=len(NEW_FUND) if batch=='B' else 0)

if __name__ == '__main__':
    units = []
    for model in ('V220','V230','V240','V252'):
        units.append(build_unit('A', model))
    for model in ('V220','V230','V240','V252'):
        units.append(build_unit('B', model))
    (WORK/'scripts/unit_manifest.json').write_text(json.dumps(units, ensure_ascii=False, indent=1))
    print(json.dumps(units, ensure_ascii=False, indent=1))
