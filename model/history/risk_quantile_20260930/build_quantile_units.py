"""Fresh conditional quantile experiments; parent scripts only, no imported units or weights."""
import ast
from pathlib import Path
import json

BASE=Path('/root/autodl-fs/model/experiments/V50')
OUT=BASE.parent


def node(source,name):
    for n in ast.parse(source).body:
        if getattr(n,'name',None)==name:return n
        if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in n.targets):return n
    raise KeyError(name)


def replace(source,name,body):
    n=node(source,name);lines=source.splitlines(keepends=True)
    start=min([n.lineno]+[d.lineno for d in getattr(n,'decorator_list',[])])-1
    return ''.join(lines[:start])+body.rstrip()+'\n'+''.join(lines[n.end_lineno:])


ARCH='''class PredictModel(nn.Module):
    """Ordered conditional quantiles of 5d peer-relative returns; lower-tail ranking."""
    def __init__(self,input_dim,market_dim,horizons=None):
        super().__init__()
        self.horizons=list(horizons or RECIPE['label_horizons'])
        if self.horizons != [5]:
            raise ValueError('this unit predicts one 5d distribution, not multiple horizon point heads')
        width,latent,forecast=RECIPE['hidden']
        self.encoder=nn.Sequential(nn.Linear(input_dim,width),nn.LayerNorm(width),nn.GELU())
        self.recurrent=nn.GRU(width,latent,batch_first=True)
        market_width=RECIPE['market_embedding'] if market_dim else 0
        self.market_encoder=(nn.Sequential(nn.Linear(market_dim,market_width),nn.GELU()) if market_dim else None)
        self.output_layer=nn.Sequential(nn.LayerNorm(latent+market_width),nn.Linear(latent+market_width,forecast),nn.GELU(),nn.Linear(forecast,3))

    def quantiles(self,tsdata,market=None):
        if tsdata.ndim!=3 or tsdata.shape[1]!=RECIPE['sequence_days']:
            raise ValueError('expected stock x causal-session x feature input')
        hidden,_=self.recurrent(self.encoder(tsdata.float().clamp(-RECIPE['input_clip'],RECIPE['input_clip'])))
        last=hidden[:,-1]
        if self.market_encoder is not None:
            if market is None:raise ValueError('current-date market vector required')
            state=self.market_encoder(market.float()).expand(len(last),-1)
            last=torch.cat([last,state],dim=-1)
        raw=self.output_layer(last)
        center=raw[:,0]
        lower=center-RECIPE['quantile_width_scale']*torch.nn.functional.softplus(raw[:,1])
        upper=center+RECIPE['quantile_width_scale']*torch.nn.functional.softplus(raw[:,2])
        return torch.stack([lower,center,upper],dim=-1)

    def forward(self,tsdata,market=None):
        q=self.quantiles(tsdata,market)
        score=q[:,1]-RECIPE['risk_lambda']*(q[:,1]-q[:,0])
        return score[:,None]
'''

LOSS='''def quantile_loss(preds,y):
    """Proper pinball loss on the raw peer-relative return scale, with ordered outputs."""
    levels=torch.as_tensor(RECIPE['quantiles'],device=preds.device,dtype=preds.dtype)
    residual=y.reshape(-1,1)-preds
    return torch.maximum(levels*residual,(levels-1)*residual).mean()
'''

DIST='''@torch.no_grad()
def predict_quantiles(model,panel,days,device):
    model.eval()
    out=np.full((len(days),len(panel.codes),3),np.nan,np.float32)
    for i,d in enumerate(days):
        mask=panel.mask(d)
        if mask.sum():
            x=torch.from_numpy(panel.inputs(d,mask)).to(device)
            market=torch.from_numpy(panel.M[d]).to(device) if panel.market_dim else None
            out[i,mask]=model.quantiles(x,market).cpu().numpy()
    return out
'''

PINBALL='''pinball_days=[]
coverage={str(q):[] for q in RECIPE['quantiles']}
for i,d in enumerate(days):
    y=panel.Y['label_ret_5d'][d]
    ok=np.isfinite(qhat[i]).all(axis=1)&np.isfinite(y)
    if ok.sum()<3:continue
    target=y[ok].astype(np.float64)
    target=target-np.median(target)
    error=target[:,None]-qhat[i,ok]
    levels=np.asarray(RECIPE['quantiles'])
    pinball_days.append(float(np.maximum(levels*error,(levels-1)*error).mean()))
    for j,q in enumerate(RECIPE['quantiles']):
        coverage[str(q)].append(float(np.mean(target<=qhat[i,ok,j])))
pinball=float(np.mean(pinball_days)) if pinball_days else float('inf')
primary=(-pinball if criterion=='quantile_pinball' else ret if criterion=='tradable_topn' else RECIPE['valid_ic_scale']*ic_term)
'''


def create(unit,seed):
    d=OUT/unit
    if d.exists():raise FileExistsError(d)
    model=(BASE/'model.py').read_text();analysis=(BASE/'analysis.py').read_text();run=(BASE/'run.py').read_text()
    original=node(model,'RECIPE')
    fields={kw.arg:ast.unparse(kw.value) for kw in original.value.keywords}
    fields.update({
        'name':repr(unit),'architecture':repr('8-session GRU + current market context + ordered5d quantiles; proper pinball early stopping; lower-tail score'),
        'family':repr('flow_quantile_risk'),'seed':str(seed),'label_horizons':'[5]',
        'label_weights':'{5:1.0}','valid_ic_weights':'{5:1.0}','score_blend':repr('head:label_ret_5d'),
        'market_gate':'True','market_embedding':'16','hidden':'(24,32,32)','dropout':'(0.0,0.0)',
        'input_clip':'5.0','quantiles':'[0.25,0.5,0.75]','quantile_width_scale':'0.05','risk_lambda':'0.5',
        'risk_lambda_candidates':'[0.0,0.25,0.5,0.75]',
        'target_center':repr('per-date median of the finite training5d labels; raw snapshot targets remain unchanged'),
        'valid_criterion':repr('quantile_pinball'),'valid_metric':repr('negative mean daily pinball loss on peer-relative5d returns; no test selection'),
        'training_target':repr('raw5d return minus same-date eligible-label median; ordered25/50/75 percent quantiles'),
        'bag_recalibrate':'False','linear_weight':'0.0','prediction_horizon':'5',
        'weights_origin':repr('fresh independent weights; V50 scripts copied, no checkpoints'),
        'research_protocol':repr('four seeds3253/4253/5253/6253;4quarters x4folds; unchanged21-day purge; quantile calibration and downside ranking assessed on all clean blocks; fixed242-day test only after freezing; no release promotion'),
    })
    model=replace(model,'RECIPE','RECIPE=dict(\n'+''.join(f'    {k}={v},\n' for k,v in fields.items())+')')
    model=replace(model,'PredictModel',ARCH)
    model=replace(model,'wpcc',LOSS)
    old_predict=ast.unparse(node(model,'predict_heads'))
    model=replace(model,'predict_heads',DIST+'\n\n'+old_predict)
    train=ast.unparse(node(model,'train_fold'))
    assert train.count('pred = model(x, m)')==1
    train=train.replace('pred = model(x, m)','pred = model.quantiles(x, m)')
    assert train.count('yt = (yt - yt.mean()) / yt.std()')==1
    train=train.replace('yt = (yt - yt.mean()) / yt.std()','yt = yt - torch.quantile(yt,0.5)')
    assert train.count('wpcc(pred[rows, j], yt)')==1
    train=train.replace('wpcc(pred[rows, j], yt)','quantile_loss(pred[rows], yt)')
    train=train.replace("raise FloatingPointError('WPCC 非有限')", "raise FloatingPointError('quantile pinball loss is non-finite')")
    model=replace(model,'train_fold',train)
    fn=node(model,'validation')
    fn.body[0]=ast.Expr(value=ast.Constant(value='Proper quantile calibration chooses epochs; raw1/5d IC and tradable proxies are separate diagnostics.'))
    for n in ast.walk(fn):
        if isinstance(n,ast.DictComp) and ast.unparse(n.key)=="f'IC_{h}d'":
            n.generators[0].iter=ast.parse('(1,3,5,10,20)',mode='eval').body
    valid=ast.unparse(ast.fix_missing_locations(fn))
    valid=valid.replace('heads = predict_heads(model, panel, days, device)',"qhat = predict_quantiles(model,panel,days,device)\n    heads = (qhat[:,:,1]-RECIPE['risk_lambda']*(qhat[:,:,1]-qhat[:,:,0]))[:,:,None]")
    valid=valid.replace("criterion not in ('rankic', 'tradable_topn')", "criterion not in ('rankic','tradable_topn','quantile_pinball')")
    primary="primary = ret if criterion == 'tradable_topn' else RECIPE['valid_ic_scale'] * ic_term"
    assert primary in valid
    valid=valid.replace(primary,'\n    '.join(PINBALL.strip().splitlines()))
    valid=valid.replace('return dict(val_wei=primary,','return dict(quantile_pinball_loss=pinball,quantile_empirical_coverage={k:float(np.mean(v)) for k,v in coverage.items()},val_wei=primary,')
    model=replace(model,'validation',valid)
    # Source contract remains three implementation scripts; the unit imports only its own model.
    prepared={}
    for name,source in [('model.py',model),('analysis.py',analysis),('run.py',run)]:
        first=ast.parse(source).body[0]
        if isinstance(first,ast.Expr) and isinstance(first.value,ast.Constant) and isinstance(first.value.value,str):
            lines=source.splitlines(keepends=True)
            source=''.join(lines[:first.lineno-1])+repr(f'{unit}: fresh conditional quantile risk family; seed{seed}; fixed project contract.')+'\n'+''.join(lines[first.end_lineno:])
        compile(source,name,'exec');prepared[name]=source
    d.mkdir()
    for name,source in prepared.items():(d/name).write_text(source)


if __name__=='__main__':
    for unit,seed in zip(['V54','V55','V56','V57'],[3253,4253,5253,6253]):create(unit,seed)
    print(json.dumps(dict(units=['V54','V55','V56','V57'],status='fresh scripts only; not trained')))
