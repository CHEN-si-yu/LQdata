"""Source fragments copied into each standalone experiment, never imported by unit runtime."""

TECHNICAL_FEATURES = [
    'adx_14','atr_14_ratio','drawdown_duration_120','kdj_k_minus_d','macd_daily_hist_5d',
    'momentum_120','momentum_20','momentum_250','momentum_60','price_distance_from_52w_low',
    'price_to_52w_high','ret_autocorr_1d_20','ret_efficiency_20','ret_ind_rel_1d','ret_kurt_20',
    'ret_skew_20','reversal_2d','roc_12','rsi_14','rsi_spread_6_14',
    'amihud_daily_5','aroon_down_25','aroon_up_25','beta_60','bollinger_squeeze',
    'bollinger_width_20','downside_upside_vol_60','downside_vol_ratio_20','dpo_20',
    'gain_loss_asymmetry_60','gap_down_recover_freq_20d','gap_fill_tendency_10d',
    'gap_up_fade_freq_20d','idio_vol_60','liquidity_shock_20','log_mv','max_drawdown_120',
    'max_drawdown_60','new_high_60_event','new_low_60_event','overnight_gap_vol_20',
    'overnight_ma_5d','overnight_ma_20d','overnight_std_5d',
]

MODEL = '''class PredictModel(nn.Module):
    """Absolute clipped 1/5d open-to-open returns; fixed basis, closed-form weights."""
    def __init__(self,input_dim,market_dim,horizons=None):
        super().__init__()
        self.horizons=list(horizons or RECIPE['label_horizons'])
        self.input_dim=input_dim
        self.market_dim=market_dim
        self.basis_dim=3*input_dim+market_dim+1
        self.register_buffer('coef',torch.zeros(len(self.horizons),self.basis_dim))

    def basis_numpy(self,x,market=None):
        x=np.clip(np.asarray(x,np.float64),-RECIPE['input_clip'],RECIPE['input_clip'])
        parts=[x,np.tanh(x),x*x/(1+x*x)]
        if self.market_dim:
            if market is None:raise ValueError('current-date market context required')
            parts.append(np.broadcast_to(np.asarray(market,np.float64),(len(x),self.market_dim)))
        parts.append(np.ones((len(x),1),np.float64))
        return np.concatenate(parts,axis=1)

    def forward(self,x,market=None):
        x=x.float().clamp(-RECIPE['input_clip'],RECIPE['input_clip'])
        parts=[x,torch.tanh(x),x*x/(1+x*x)]
        if self.market_dim:
            if market is None:raise ValueError('current-date market context required')
            parts.append(market.float().expand(len(x),-1))
        parts.append(torch.ones((len(x),1),dtype=x.dtype,device=x.device))
        return torch.cat(parts,dim=1)@self.coef.T
'''

SOLVER = '''def closed_form_stages(model,panel,train_days,seed):
    """Nested stock samples span the same complete date set; missing labels never become zero."""
    caps=list(RECIPE['sample_caps'])
    count=len(caps);dim=model.basis_dim;heads=len(model.horizons)
    xx=np.zeros((count,dim,dim),np.float64)
    xy=np.zeros((count,dim,heads),np.float64)
    yy=np.zeros((count,heads),np.float64)
    date_counts=np.zeros(count,np.int64);sample_rows=np.zeros(count,np.int64)
    for d in train_days:
        d=int(d)
        mask=panel.mask(d).copy()
        for h in model.horizons:mask&=np.isfinite(panel.Y[f'label_ret_{h}d'][d])
        candidates=np.flatnonzero(mask)
        if len(candidates)<3:continue
        rng=np.random.default_rng(np.random.SeedSequence([int(seed),d]))
        selected=candidates[rng.permutation(len(candidates))[:caps[-1]]]
        phi=model.basis_numpy(panel.X[d,selected],panel.M[d] if model.market_dim else None)
        target=np.column_stack([np.clip(panel.Y[f'label_ret_{h}d'][d,selected],
            -RECIPE['target_clip'][h],RECIPE['target_clip'][h]) for h in model.horizons]).astype(np.float64)
        a=np.zeros((dim,dim),np.float64);b=np.zeros((dim,heads),np.float64)
        c=np.zeros(heads,np.float64);previous=0
        for stage,cap in enumerate(caps):
            n=min(cap,len(selected));block=phi[previous:n];y=target[previous:n]
            a+=block.T@block;b+=block.T@y;c+=(y*y).sum(axis=0)
            xx[stage]+=a/n;xy[stage]+=b/n;yy[stage]+=c/n
            date_counts[stage]+=1;sample_rows[stage]+=n;previous=n
    if np.any(date_counts==0):raise ValueError('no eligible training dates')
    penalty=np.full(dim,RECIPE['stock_ridge'],np.float64)
    if model.market_dim:penalty[3*model.input_dim:-1]=RECIPE['market_ridge']
    penalty[-1]=RECIPE['intercept_ridge']
    results=[]
    for stage in range(count):
        a=xx[stage]/date_counts[stage];b=xy[stage]/date_counts[stage];c=yy[stage]/date_counts[stage]
        normal=a+np.diag(penalty)
        coef=np.linalg.solve(normal,b)
        residual=float(np.linalg.norm(normal@coef-b)/(np.linalg.norm(b)+1e-20))
        if not np.isfinite(coef).all() or residual>1e-8:raise FloatingPointError('normal equation residual invalid')
        mse=np.maximum(np.einsum('dh,de,eh->h',coef,a,coef)-2*(coef*b).sum(axis=0)+c,0.)
        loss=sum(RECIPE['label_weights'][h]*mse[j]/RECIPE['loss_scale'][h]**2
                 for j,h in enumerate(model.horizons))/sum(RECIPE['label_weights'].values())
        results.append(dict(coef=coef.T.astype(np.float32),train_loss=float(loss),
            fit_dates=int(date_counts[stage]),sample_rows=int(sample_rows[stage]),
            stock_sample_cap=int(caps[stage]),normal_equation_relative_residual=residual))
    return results
'''

TRAIN = '''def train_fold(fold,device='auto',panel=None,prices=None,quarter=None):
    """Five closed-form sample-coverage stages; fresh weights and the same fixed fold artifacts."""
    recipe=copy.deepcopy(RECIPE);recipe['device']=device
    quarter=quarter or quarters(axis()[0])[0];recipe['quarter']=quarter
    info=training_info(recipe);info['fit_algorithm']='closed-form ridge on daily weighted sufficient statistics'
    out=RUN_ROOT/'model_train'/quarter/f'fold{fold}';output_file(out/'complete.json')
    out.mkdir(parents=True,exist_ok=True);done=out/'complete.json'
    if done.exists():
        old=json.loads(done.read_text())
        if not all((out/name).is_file() for name in ['best.pt','last.pt','test_predictions.npy','score_predictions.npy']):
            raise RuntimeError('complete fold is missing required artifacts')
        print(f'fold{fold} already complete',flush=True);return old
    torch.set_num_threads(recipe['threads']);dev=device_for(device)
    torch.use_deterministic_algorithms(True)
    seed=recipe['seed']+fold-1
    panel=panel or Panel(recipe['features']);prices=prices or Prices(panel)
    split=splits(panel.days,recipe['purge_horizon'],recipe['folds'],quarter)[fold-1]
    atomic_json(out/'training_info.json',info)
    atomic_json(out/'split.json',split_summary(panel.days,split))
    model=PredictModel(len(panel.features),panel.market_dim).to(dev);model.ridge=None
    fit_spec=dict(seed=seed,features=panel.features,horizons=model.horizons,
        caps=recipe['sample_caps'],target_clip=recipe['target_clip'],loss_scale=recipe['loss_scale'],
        stock_ridge=recipe['stock_ridge'],market_ridge=recipe['market_ridge'],
        train_dates=panel.days[split['train']].tolist())
    last=out/'last.pt';history=[];bank=[];start=0;best=-float('inf');stale=0
    t0=time.time()
    if last.exists():
        ck=torch.load(last,map_location='cpu',weights_only=False)
        if ck['fit_spec']!=fit_spec:raise ValueError('fit recipe changed; create a new unit')
        stages=ck['closed_form_stages'];history=ck['history'];bank=ck['bank']
        start=ck['epoch']+1;best=ck['best'];stale=ck['stale']
        print(f'fold{fold} resume stage {start+1}',flush=True)
    else:
        stages=closed_form_stages(model,panel,split['train'],seed)
    assert len(stages)==recipe['max_epochs']==len(recipe['sample_caps'])
    for epoch in range(start,recipe['max_epochs']):
        if stale>=recipe['early_stop_patience']:break
        t=time.time();stage=stages[epoch]
        with torch.no_grad():model.coef.copy_(torch.from_numpy(stage['coef']).to(dev))
        metrics=validation(model,panel,prices,split['valid'],dev);score=metrics['val_wei']
        if not np.isfinite(score):raise FloatingPointError('validation criterion is non-finite')
        improved=score>best+1e-10
        if improved:
            best=score;stale=0
            save_torch(out/'best.pt',dict(model=model.state_dict(),input_dim=len(panel.features),
                market_dim=panel.market_dim,horizons=model.horizons,features=panel.features,
                epoch=epoch,metrics=metrics,ridge=None))
        else:stale+=1
        bank.append((score,epoch,{k:v.detach().to('cpu',copy=True) for k,v in model.state_dict().items()}))
        bank.sort(key=lambda row:-row[0]);del bank[recipe['bag_topk']:]
        record=dict(epoch=epoch,train_loss=stage['train_loss'],lr=0.,
            seconds=round(time.time()-t,2),stock_sample_cap=stage['stock_sample_cap'],
            fit_dates=stage['fit_dates'],sample_rows=stage['sample_rows'],
            normal_equation_relative_residual=stage['normal_equation_relative_residual'],**metrics)
        history.append(record);atomic_json(out/'history.json',history)
        save_torch(last,dict(model=model.state_dict(),optimizer=None,scheduler=None,
            epoch=epoch,best=best,stale=stale,history=history,bank=bank,
            closed_form_stages=stages,fit_spec=fit_spec))
        print(f'fold{fold} closed-form stage {epoch+1}/{len(stages)} stocks/day≤{stage["stock_sample_cap"]} '
            f'val_wei={score:.6g} normal_residual={stage["normal_equation_relative_residual"]:.3g}',flush=True)
    ck=torch.load(out/'best.pt',map_location=dev,weights_only=False)
    argmax_epoch,argmax_metrics=ck['epoch'],ck['metrics']
    if recipe['bag_topk'] and len(bank)>1:
        model.load_state_dict(average_state_dicts([state for _,_,state in bank]))
        bagged_metrics=validation(model,panel,prices,split['valid'],dev)
        save_torch(out/'best.pt',dict(model=model.state_dict(),input_dim=len(panel.features),
            market_dim=panel.market_dim,horizons=model.horizons,features=panel.features,epoch=argmax_epoch,
            metrics=bagged_metrics,ridge=None,model_argmax=ck['model'],metrics_argmax=argmax_metrics))
    else:model.load_state_dict(ck['model']);bagged_metrics=argmax_metrics
    pred=predict(model,panel,prices,split['score'],dev)
    save_npy(out/'score_predictions.npy',pred);save_npy(out/'test_predictions.npy',pred[:len(split['test'])])
    result=dict(quarter=quarter,fold=fold,best_epoch=argmax_epoch+1,epochs=len(history),
        test_dates=panel.days[split['test']].tolist(),score_dates=panel.days[split['score']].tolist(),
        best_validation=bagged_metrics,argmax_validation=argmax_metrics,
        bag=dict(topk=recipe['bag_topk'],used=len(bank),epochs=[e+1 for _,e,_ in bank],scores=[float(s) for s,_,_ in bank]),
        peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2,
        seconds=round(time.time()-t0,2),device=str(dev),
        weights_origin='fresh closed-form fit; no reused checkpoint',
        stage_semantics='same complete training date set; nested per-date stock sample coverage; parameter bagging retained')
    atomic_json(done,result);print(f'fold{fold} EXIT:0 closed-form bag={len(bank)}',flush=True)
    return result
'''
