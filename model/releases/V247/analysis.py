"""Canonical reload and actual information boundaries, including native single-stage units."""
import os
for k in ['MX_THREADS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import hashlib,json
from pathlib import Path
import numpy as np
def main():
    import model as m
    m.torch.set_num_threads(1);p=m.Panel('all');px=m.Prices(p);checks=[];total=0
    for f in range(1,5):
        folder=m.UNIT_ROOT/f'model_train/2026Q4/fold{f}';done=json.loads((folder/'complete.json').read_text());assert done['recipe_signature']==m.recipe_signature();sp=m.splits(p.days,quarter='2026Q4')[f-1]
        assert max(sp['train'][-1],sp['valid'][-1])+21<sp['test'][0]
        if m.RECIPE['name']!='V247':assert sp['train'][-1]+21<sp['valid'][0]
        paired=bool(done.get('delivery_refit_complete'));delivery=m.delivery_train_days(p.days,sp) if paired else sp['train'];assert str(p.days[delivery[-1]])<='2026-08-31'
        phases=[('delivery','best.pt',sp['test'],'test_predictions.npy')]
        if paired: phases.insert(0,('selector','selector_best.pt',sp['valid'],'valid_predictions.npy'))
        else:assert m.RECIPE['name'] in ['V259','V247']
        errors={};sources={}
        for phase,fn,ix,stored in phases:
            ck=m.torch.load(folder/fn,map_location='cpu',weights_only=False);assert ck['recipe_signature']==m.recipe_signature() and ck['features']==p.features
            net=m.PredictModel(len(p.features),p.market_dim);net.load_state_dict(ck['model'],strict=True);net.ridge=ck.get('ridge');net.eval();actual=m.predict(net,p,px,ix,m.torch.device('cpu'));expected=np.load(folder/stored);expected=expected[:,:,0] if expected.ndim==3 else expected
            assert actual.shape==expected.shape and np.array_equal(np.isfinite(actual),np.isfinite(expected));err=float(np.nanmax(np.abs(actual-expected)));assert err<=1e-6,(phase,f,err);errors[phase]=err;total+=1
            end=str(p.days[sp['train'][-1] if phase=='selector' else delivery[-1]]);teacher=ck.get('teacher')
            if teacher:
                assert teacher['train_end']==end and teacher['phase']==('selector' if phase=='selector' else 'delivery_refit');assert hashlib.sha256((m.PROJECT_ROOT/teacher['file']).read_bytes()).hexdigest()==teacher['sha256']
            sources[phase]=dict(train_end=end,weight_sha256=hashlib.sha256((folder/fn).read_bytes()).hexdigest(),teacher=teacher)
            del net
        checks.append(dict(fold=f,errors=errors,sources=sources,paired_selector_delivery=paired,split=json.loads((folder/'split.json').read_text())));print('Q4_CPU_AUDIT',m.RECIPE['name'],m.RECIPE['seed'],f,errors,flush=True)
    out=dict(model=m.RECIPE['name'],seed=m.RECIPE['seed'],quarter='2026Q4',folds_checked=4,phases_checked=total,max_cpu_reload_error=max(v for z in checks for v in z['errors'].values()),audits=checks,snapshot_meta_sha256=m.RECIPE['snapshot_meta_sha256'],source_sha256=hashlib.sha256((m.UNIT_ROOT/'model.py').read_bytes()).hexdigest(),strict_time_and_teacher_passed=True,legacy_validation_caveat='V247 original pre-quarter blocked validation; train may follow validation within pre-quarter history, all before Q4. Not strict chronological selector-refit.')
    m.atomic_json(m.UNIT_ROOT/'model_info/live_validation.json',out);m.validate_layout(allow_missing=True);print('Q4_UNIT_AUDIT_PASSED',flush=True)
if __name__=='__main__':main()
