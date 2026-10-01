"""Source fragments copied into independent experiment scripts by the builder."""

MODEL = r'''
class PredictModel(nn.Module):
    """Fresh flow GRU plus a separately fitted absolute technical branch."""
    def __init__(self, input_dim, market_dim, horizons=None):
        super().__init__()
        self.horizons = list(horizons or RECIPE['output_horizons'])
        self.market_dim = market_dim
        self.flow_indices = RECIPE['flow_indices']
        self.technical_indices = RECIPE['technical_indices']
        if input_dim != len(RECIPE['feature_names']) or self.horizons != RECIPE['output_horizons']:
            raise ValueError('frozen composite feature/output contract changed')
        self.encoder = nn.Sequential(nn.Linear(len(self.flow_indices),24),nn.LayerNorm(24),nn.GELU())
        self.recurrent = nn.GRU(24,32,batch_first=True)
        self.output_layer = nn.Sequential(nn.LayerNorm(32),nn.Linear(32,len(RECIPE['label_horizons'])))
        self.absolute_basis_dim = 3*len(self.technical_indices)+market_dim+1
        self.register_buffer('absolute_coef',torch.zeros(2,self.absolute_basis_dim))

    def absolute_basis_numpy(self, current, market=None):
        x=np.clip(np.asarray(current,np.float64)[:,self.technical_indices],-5.,5.)
        parts=[x,np.tanh(x),x*x/(1+x*x)]
        if self.market_dim:
            if market is None:raise ValueError('absolute branch requires current causal market inputs')
            parts.append(np.broadcast_to(np.asarray(market,np.float64),(len(x),self.market_dim)))
        parts.append(np.ones((len(x),1),np.float64))
        return np.concatenate(parts,axis=1)

    def absolute_forward(self, current, market=None):
        x=current[:,self.technical_indices].float().clamp(-5.,5.)
        parts=[x,torch.tanh(x),x*x/(1+x*x)]
        if self.market_dim:
            if market is None:raise ValueError('absolute branch requires current causal market inputs')
            parts.append(market.float().expand(len(x),-1))
        parts.append(torch.ones((len(x),1),dtype=x.dtype,device=x.device))
        return torch.cat(parts,dim=1)@self.absolute_coef.T

    def forward(self, tsdata, market=None):
        if tsdata.ndim!=3 or tsdata.shape[1]!=RECIPE['sequence_days']:
            raise ValueError('expected causal eight-session stock input')
        flow=tsdata[:,:,self.flow_indices].float().clamp(-5.,5.)
        hidden,_=self.recurrent(self.encoder(flow))
        rank_heads=self.output_layer(hidden[:,-1])
        absolute=self.absolute_forward(tsdata[:,-1],market)
        return torch.cat([rank_heads,absolute[:,:1]],dim=1)


def fit_absolute_component(model,panel,train_days,seed):
    """All five predeclared nested sampling fits; date equal, training labels only."""
    caps=RECIPE['absolute_sample_caps'];dim=model.absolute_basis_dim
    xx=np.zeros((len(caps),dim,dim),np.float64)
    xy=np.zeros((len(caps),dim,2),np.float64)
    yy=np.zeros((len(caps),2),np.float64)
    dates=np.zeros(len(caps),np.int64);samples=np.zeros(len(caps),np.int64)
    for day in train_days:
        d=int(day);known=panel.technical_mask(d)
        for h in [1,5]:known&=np.isfinite(panel.Y[f'label_ret_{h}d'][d])
        candidates=np.flatnonzero(known)
        if len(candidates)<3:continue
        rng=np.random.default_rng(np.random.SeedSequence([int(seed),d]))
        selected=candidates[rng.permutation(len(candidates))[:caps[-1]]]
        phi=model.absolute_basis_numpy(panel.X[d,selected],panel.M[d] if model.market_dim else None)
        y=np.column_stack([np.clip(panel.Y[f'label_ret_{h}d'][d,selected],
            -RECIPE['absolute_target_clip'][h],RECIPE['absolute_target_clip'][h]) for h in [1,5]])
        gram=np.zeros((dim,dim));cross=np.zeros((dim,2));sq=np.zeros(2);previous=0
        for stage,cap in enumerate(caps):
            n=min(cap,len(selected));xblock=phi[previous:n];yblock=y[previous:n]
            gram+=xblock.T@xblock;cross+=xblock.T@yblock;sq+=(yblock*yblock).sum(axis=0)
            xx[stage]+=gram/n;xy[stage]+=cross/n;yy[stage]+=sq/n
            dates[stage]+=1;samples[stage]+=n;previous=n
    if np.any(dates==0):raise ValueError('absolute branch has no valid training dates')
    penalty=np.full(dim,RECIPE['absolute_stock_ridge'],np.float64)
    penalty[3*len(model.technical_indices):-1]=RECIPE['absolute_market_ridge']
    penalty[-1]=RECIPE['absolute_intercept_ridge']
    stages=[]
    for i,cap in enumerate(caps):
        gram=xx[i]/dates[i];cross=xy[i]/dates[i];normal=gram+np.diag(penalty)
        coef=np.linalg.solve(normal,cross)
        residual=float(np.linalg.norm(normal@coef-cross)/(np.linalg.norm(cross)+1e-20))
        if not np.isfinite(coef).all() or residual>1e-8:
            raise FloatingPointError('absolute normal equation failure')
        mse=np.maximum(np.einsum('dh,de,eh->h',coef,gram,coef)-2*(coef*cross).sum(axis=0)+yy[i]/dates[i],0.)
        normalized_loss=sum(RECIPE['absolute_label_weights'][h]*mse[j]/RECIPE['absolute_loss_scale'][h]**2 for j,h in enumerate([1,5]))
        normalized_loss/=sum(RECIPE['absolute_label_weights'].values())
        stages.append(dict(coef=coef.T.astype(np.float32),fit_dates=int(dates[i]),sample_rows=int(samples[i]),
            stock_sample_cap=int(cap),normal_equation_relative_residual=residual,train_loss=float(normalized_loss)))
    averaged=np.mean(np.stack([x['coef'] for x in stages]),axis=0,dtype=np.float32)
    with torch.no_grad():model.absolute_coef.copy_(torch.from_numpy(averaged).to(model.absolute_coef.device))
    return dict(seed=int(seed),horizons=[1,5],sample_caps=list(caps),stages=stages,
        bag_rule='all five training fits equally; no validation/test stage choice',
        origin='fresh fits from this unit training partition; no other checkpoint read')


def recipe_signature():
    return hashlib.sha256(json.dumps(RECIPE,sort_keys=True,ensure_ascii=False).encode()).hexdigest()


@torch.no_grad()
def predict_absolute(model,panel,days,device):
    """Auxiliary1d forecast; same weights and joint coverage, no future-label reads."""
    model.eval()
    out=np.full((len(days),len(panel.codes)),np.nan,np.float32)
    for i,day in enumerate(days):
        d=int(day);known=panel.mask(d)
        if known.any():
            x=torch.from_numpy(panel.X[d,known]).to(device)
            market=torch.from_numpy(panel.M[d]).to(device) if panel.market_dim else None
            out[i,known]=model.absolute_forward(x,market)[:,0].cpu().numpy()
    return out
'''

SCORE = r'''
def score_of(heads):
    """Internal cross-family rank fusion; later fourfold ensemble still raw-sums."""
    if RECIPE['score_blend']!='flow_technical_percentile' or heads.shape[-1]!=4:
        raise ValueError('unexpected composite head schema')
    out=np.full(heads.shape[:2],np.nan,np.float32)
    weight=RECIPE['flow_fusion_weight']
    for i,row in enumerate(heads):
        known=np.isfinite(row).all(axis=1);n=known.sum()
        if n>1:
            flow=(rankdata(row[known,1])-.5)/n
            technical=(rankdata(row[known,3])-.5)/n
            out[i,known]=weight*flow+(1-weight)*technical
    return out
'''

MASK = r'''
    def technical_mask(self,day):
        if self.technical_coverage is None:raise ValueError('feature coverage not loaded')
        return self.technical_coverage[int(day)]>=RECIPE['min_feature_coverage']

    def mask(self,day,label=None):
        width=RECIPE['sequence_days'];day=int(day)
        if day<width-1:return np.zeros(len(self.codes),dtype=bool)
        if self.flow_coverage is None:raise ValueError('feature coverage not loaded')
        out=(self.flow_coverage[day-width+1:day+1]>=RECIPE['min_feature_coverage']).all(axis=0)
        out&=self.technical_mask(day)
        if label is not None:out&=np.isfinite(self.Y[label][day])
        return out
'''

WRITE_SCORES = r'''
def write_scores(pred,panel,days,out):
    """Existing yearly score files carry a reproducible separate gate forecast."""
    for year in sorted(set(str(panel.days[d])[:4] for d in days)):
        chunks=[]
        for p,d in zip(pred,days):
            if not str(panel.days[d]).startswith(year):continue
            ok=np.isfinite(p);forecast=panel.absolute_forecast[int(d)]
            if not np.isfinite(forecast[ok]).all():raise ValueError('missing gate forecast')
            chunks.append(pd.DataFrame(dict(trade_date=str(panel.days[d]),stock_code=panel.codes[ok],
                value=p[ok].astype(np.float32),rank=(rankdata(p[ok])/ok.sum()).astype(np.float32),
                absolute_forecast_1d=forecast[ok].astype(np.float32))))
        df=pd.concat(chunks,ignore_index=True)
        df['trade_date']=df['trade_date'].astype('string');df['stock_code']=df['stock_code'].astype('string')
        folder=out/f'year={year}';folder.mkdir(parents=True,exist_ok=True)
        target=output_file(folder/'data.parquet');temp=folder/'data.tmp.parquet'
        df.to_parquet(temp,index=False);os.replace(temp,target)
'''

READ_SCORES = r'''
def read_saved_scores(folder,panel):
    """Read scalar ranks and separate absolute gate from existing score parquet."""
    meta=json.loads((folder/'score_meta.json').read_text());dates=meta['score_dates']
    if dates!=sorted(set(dates)):raise ValueError('duplicate or unordered score dates')
    date_axis=pd.Index(dates);codes=pd.Index(panel.codes);days=pd.Index(panel.days).get_indexer(dates)
    if (days<0).any():raise ValueError('score dates absent from snapshot')
    pred=np.full((len(dates),len(codes)),np.nan,np.float32)
    forecast=np.full_like(pred,np.nan);files=sorted(folder.glob('year=*/data.parquet'));seen=set()
    if not files:raise FileNotFoundError('saved composite scores missing')
    for path in files:
        frame=pd.read_parquet(path,columns=['trade_date','stock_code','value','absolute_forecast_1d'])
        if frame.duplicated(['trade_date','stock_code']).any():raise ValueError('duplicate score keys')
        t=date_axis.get_indexer(frame.trade_date.astype(str));c=codes.get_indexer(frame.stock_code.astype(str))
        if (t<0).any() or (c<0).any():raise ValueError('mismatched score coordinates')
        keys=set(zip(t.tolist(),c.tolist()))
        if seen.intersection(keys):raise ValueError('overlapping score partitions')
        seen.update(keys)
        value=frame.value.to_numpy(dtype=np.float32);gate=frame.absolute_forecast_1d.to_numpy(dtype=np.float32)
        if not np.isfinite(value).all() or not np.isfinite(gate).all():raise ValueError('nonfinite saved score/gate')
        pred[t,c]=value;forecast[t,c]=gate
    panel.absolute_forecast=np.full((len(panel.days),len(panel.codes)),np.nan,np.float32)
    panel.absolute_forecast[days]=forecast
    return pred,days,meta
'''
