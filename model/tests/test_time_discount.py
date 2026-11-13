"""指数时间目标的质量归一化、重锚和成熟重训中断恢复。"""
import importlib.util
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('time_discount_test',ROOT/'experiments/V272/model.py')
model=importlib.util.module_from_spec(spec)
spec.loader.exec_module(model)


class TimeDiscountTests(unittest.TestCase):
    def test_prefix_reanchor_equals_full_history_weighting(self):
        rng=np.random.default_rng(72)
        x=rng.normal(size=(80,4))
        y=rng.normal(size=(80,1))
        w=model.time_decay_weight(np.arange(80),79)
        old=model.time_decay_weight(np.arange(40),39)
        g,c,yy,mass,square=model.reanchor_statistics(x[:40].T@(x[:40]*old[:,None]),
            x[:40].T@(y[:40]*old[:,None]),float((y[:40]**2*old[:,None]).sum()),
            float(old.sum()),float((old**2).sum()),39,79)
        g+=x[40:].T@(x[40:]*w[40:,None])
        c+=x[40:].T@(y[40:]*w[40:,None])
        mass+=w[40:].sum()
        np.testing.assert_allclose(g/mass,x.T@(x*w[:,None])/w.sum(),atol=1e-14)
        np.testing.assert_allclose(c/mass,x.T@(y*w[:,None])/w.sum(),atol=1e-14)
        with self.assertRaises(ValueError):
            model.time_decay_weight(80,79)

    def test_interrupted_delivery_keeps_anchor_and_selector_predictions(self):
        rng=np.random.default_rng(272)
        t,n,width=120,24,3
        x=rng.normal(size=(t,n,width)).astype(np.float32)
        y=.01*x[:,:,0]
        y[48:]=.025*x[48:,:,1]
        panel=SimpleNamespace(days=pd.bdate_range('2025-01-01',periods=t).strftime('%Y-%m-%d').to_numpy(dtype=str),
            codes=np.asarray([str(i) for i in range(n)]),features=['a','b','c'],X=x,
            M=rng.normal(size=(t,1)).astype(np.float32),market_dim=1,market_features=['m'],
            Y={'label_ret_1d':y,'label_ret_5d':y*2})
        panel.mask=panel.technical_mask=lambda d:np.ones(n,bool)
        panel.inputs=lambda d,mask:np.ascontiguousarray(x[d,mask,None,:])
        split=dict(quarter='2025Q3',fold=1,train=np.arange(48),valid=np.arange(70,80),
                   test=np.arange(110,120),score=np.arange(110,120))
        original_save=model.save_torch
        def stop_after_saved_block(path,data):
            original_save(path,data)
            if Path(path).name=='last.pt' and data.get('phase')=='delivery_refit' and data['processed']==80:
                raise InterruptedError('simulated after persistent weighted prefix')
        outputs=[]
        with tempfile.TemporaryDirectory() as directory:
            for resume in (False,True):
                folder=Path(directory)/str(resume)
                with patch.multiple(model,UNIT_ROOT=folder,RUN_ROOT=folder), \
                     patch.object(model,'splits',return_value=[split]), \
                     patch.object(model,'training_info',return_value={}), \
                     patch.dict(model.RECIPE,technical_indices=[0],absolute_sample_caps=[24],time_half_life=10), \
                     patch.object(model,'validation',return_value={'val_wei':0.}):
                    kwargs=dict(device='cpu',panel=panel,prices=SimpleNamespace(hot_frac=1.),quarter='2025Q3')
                    if resume:
                        with patch.object(model,'save_torch',side_effect=stop_after_saved_block),self.assertRaises(InterruptedError):
                            model.train_fold(1,**kwargs)
                        valid=np.load(folder/'model_train/2025Q3/fold1/valid_predictions.npy').copy()
                    model.train_fold(1,**kwargs)
                    fold=folder/'model_train/2025Q3/fold1'
                    if resume:
                        np.testing.assert_array_equal(valid,np.load(fold/'valid_predictions.npy'))
                    last=model.torch.load(fold/'last.pt',weights_only=False)
                    expected=model.time_decay_weight(np.arange(89),88)
                    self.assertAlmostEqual(last['weighted_mass'],float(expected.sum()),places=12)
                    self.assertEqual(last['weight_anchor'],88)
                    net=model.torch.load(fold/'best.pt',weights_only=False)['model']
                    self.assertGreater(abs(net['linear_coef'][0,1]),abs(net['linear_coef'][0,0]))
                    outputs.append(net['linear_coef'])
            model.torch.testing.assert_close(outputs[0],outputs[1],rtol=0,atol=0)


if __name__=='__main__':
    unittest.main()
