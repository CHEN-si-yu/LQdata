"""Paired stochastic SAM must preserve data contracts and resume the actual trainer."""
import ast
import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


model=load('sam_optimizer_model',ROOT/'experiments/V286/model.py')
parent=load('sam_optimizer_parent',ROOT/'experiments/V263/model.py')


class TinyNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer=torch.nn.Sequential(torch.nn.Linear(5,7),torch.nn.SiLU(),torch.nn.Dropout(.1),torch.nn.Linear(7,1))

    def teacher_score(self,x):
        return x[:,-1,0]

    def forward(self,x,market):
        value=.5*torch.tanh(self.layer(torch.cat([x[:,-1],market],dim=1)))
        return torch.cat([value,torch.zeros_like(value)],dim=1)


def fixture():
    rng=np.random.default_rng(73)
    x=rng.normal(size=(20,32,3)).astype(np.float32)
    y=(x[:,:,0]+rng.normal(size=(20,32))*.4).astype(np.float32)
    panel=SimpleNamespace(days=np.asarray([f'day{i:02}' for i in range(20)]),
        features=list('abc'),market_dim=2,M=rng.normal(size=(20,2)).astype(np.float32),Y={'label_ret_1d':y})
    panel.mask=lambda d:np.ones(32,bool)
    panel.inputs=lambda d,mask:x[d,mask,None,:]
    return panel


class SamOptimizerTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def assert_nested_equal(self,a,b):
        if isinstance(a,torch.Tensor):
            torch.testing.assert_close(a,b,rtol=0,atol=0)
        elif isinstance(a,np.ndarray):
            np.testing.assert_array_equal(a,b)
        elif isinstance(a,dict):
            self.assertEqual(set(a),set(b))
            for key in a:self.assert_nested_equal(a[key],b[key])
        elif isinstance(a,(list,tuple)):
            self.assertEqual(len(a),len(b))
            for left,right in zip(a,b):self.assert_nested_equal(left,right)
        else:self.assertEqual(a,b)

    def test_zero_radius_matches_original_adamw_step_and_rng(self):
        torch.manual_seed(12);net=TinyNet();other=copy.deepcopy(net)
        x=torch.randn(25,1,3);market=torch.randn(25,2);target=torch.randn(25)
        optimizers=[torch.optim.AdamW(n.parameters(),lr=.001,weight_decay=.0001) for n in [net,other]]
        state=torch.get_rng_state()
        for _ in range(3):
            torch.set_rng_state(state)
            closure=lambda:((net(x,market)[:,0]-target)**2).mean()
            result=model.sam_adamw_step(net,optimizers[0],closure,rho=0.)
            expected_rng=torch.get_rng_state()
            torch.set_rng_state(state)
            loss=((other(x,market)[:,0]-target)**2).mean()
            optimizers[1].zero_grad(set_to_none=True);loss.backward()
            torch.nn.utils.clip_grad_norm_(other.parameters(),5.);optimizers[1].step()
            self.assertTrue(torch.equal(expected_rng,torch.get_rng_state()))
            self.assertFalse(result['sam_applied'])
            self.assert_nested_equal(net.state_dict(),other.state_dict())
            self.assert_nested_equal(optimizers[0].state_dict(),optimizers[1].state_dict())
            state=expected_rng

    def test_two_losses_use_identical_dropout_and_advance_rng_once(self):
        torch.manual_seed(25);net=TinyNet();x=torch.randn(128,1,3);market=torch.randn(128,2)
        optimizer=torch.optim.AdamW(net.parameters(),lr=.001)
        masks=[];handle=net.layer[2].register_forward_hook(lambda mod,args,out:masks.append((out==0).detach().clone()))
        before=torch.get_rng_state();net(x,market);expected_rng=torch.get_rng_state();masks.clear()
        torch.set_rng_state(before)
        result=model.sam_adamw_step(net,optimizer,lambda:net(x,market)[:,0].square().mean(),rho=.05)
        handle.remove()
        self.assertEqual(len(masks),2);self.assertTrue(torch.equal(masks[0],masks[1]))
        self.assertTrue(torch.equal(expected_rng,torch.get_rng_state()));self.assertTrue(result['sam_applied'])

    def test_quadratic_sam_gradient_and_exact_parameter_restore_before_step(self):
        net=torch.nn.Linear(2,1,bias=False)
        with torch.no_grad():net.weight.copy_(torch.tensor([[1.234567,2.345678]]))
        other=copy.deepcopy(net);original=net.weight.detach().clone()
        first_grad=original.clone();perturbed=original+first_grad*(.05/(first_grad.norm()+model.RECIPE['sam_gradient_epsilon']))
        optimizer=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.0001)
        reference=torch.optim.AdamW(other.parameters(),lr=.001,weight_decay=.0001)
        original_step=optimizer.step
        def checked_step(*args,**kwargs):
            torch.testing.assert_close(net.weight,original,rtol=0,atol=0)
            return original_step(*args,**kwargs)
        with patch.object(optimizer,'step',side_effect=checked_step):
            model.sam_adamw_step(net,optimizer,lambda:.5*net.weight.square().sum(),rho=.05)
        other.weight.grad=perturbed.clone();torch.nn.utils.clip_grad_norm_(other.parameters(),5.);reference.step()
        self.assert_nested_equal(net.state_dict(),other.state_dict())
        self.assert_nested_equal(optimizer.state_dict(),reference.state_dict())

    def test_second_pass_failure_restores_exact_weights_rng_and_optimizer(self):
        torch.manual_seed(37);net=TinyNet();optimizer=torch.optim.AdamW(net.parameters(),lr=.001)
        x=torch.randn(24,1,3);market=torch.randn(24,2)
        weights=copy.deepcopy(net.state_dict());opt=copy.deepcopy(optimizer.state_dict())
        before=torch.get_rng_state();net(x,market);after=torch.get_rng_state();torch.set_rng_state(before)
        count=0
        def closure():
            nonlocal count
            value=net(x,market)[:,0].square().mean();count+=1
            if count==2:raise RuntimeError('second loss deliberate failure')
            return value
        with self.assertRaisesRegex(RuntimeError,'deliberate'):
            model.sam_adamw_step(net,optimizer,closure,rho=.05)
        self.assert_nested_equal(weights,net.state_dict());self.assert_nested_equal(opt,optimizer.state_dict())
        self.assertTrue(torch.equal(after,torch.get_rng_state()))
        self.assertTrue(all(p.grad is None for p in net.parameters()))

    def test_real_stage_resume_and_warmup_and_future_label_isolation(self):
        panel=fixture()
        def teacher(p,q,f,phase,train):
            return TinyNet(),{},dict(sha256='fixed-test-teacher',train_end=str(p.days[train[-1]]),absolute_fit={})
        config=dict(residual_training_window=16,batch_days=4,seed=35262)
        with tempfile.TemporaryDirectory() as tmp,patch.dict(model.RECIPE,**config),patch.object(model,'frozen_teacher',side_effect=teacher),patch.object(model,'output_file',side_effect=Path):
            folders=[Path(tmp)/name for name in ['full','resume','future']]
            for folder in folders:folder.mkdir()
            def run(folder,epochs):
                return model.fit_residual_stage(panel,None,'unit',1,np.arange(16),[],torch.device('cpu'),folder,'delivery_refit',fixed_epochs=epochs)
            _,_,fit=run(folders[0],3)
            _,_,warm=run(folders[1],1);run(folders[1],3)
            panel.Y['label_ret_1d'][16:]=1e6;run(folders[2],3)
            checkpoints=[torch.load(f/'last.pt',weights_only=False) for f in folders]
            for other in checkpoints[1:]:
                for key in ['model','optimizer','numpy_rng','torch_rng','history']:
                    self.assert_nested_equal(checkpoints[0][key],other[key])
            self.assertFalse(warm['chosen_weights_include_sam']);self.assertTrue(fit['chosen_weights_include_sam'])
            self.assertEqual([r['sam_steps'] for r in checkpoints[0]['history']],[0,4,4])

    def test_original_network_loss_teacher_splits_and_prediction_unchanged(self):
        def functions(path):
            return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(path.read_text()).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
        a=functions(ROOT/'experiments/V286/model.py');b=functions(ROOT/'experiments/V263/model.py')
        for name in ['PredictModel','ic_diversity_loss','frozen_teacher','train_fold','validation','splits','delivery_train_days','predict','predict_auxiliary']:
            self.assertEqual(a[name],b[name],name)
        with patch.dict(model.RECIPE,technical_indices=[0],residual_hidden=[8,4]),patch.dict(parent.RECIPE,technical_indices=[0],residual_hidden=[8,4]):
            torch.manual_seed(7);a=model.PredictModel(5,2);rng=torch.get_rng_state()
            torch.manual_seed(7);b=parent.PredictModel(5,2)
            self.assert_nested_equal(a.state_dict(),b.state_dict());self.assertTrue(torch.equal(rng,torch.get_rng_state()))

    def test_real_warmup_stage_matches_original_training_weights_optimizer_rng(self):
        panel=fixture()
        def teacher(p,q,f,phase,train):
            return TinyNet(),{},dict(sha256='fixed-test-teacher',train_end=str(p.days[train[-1]]),absolute_fit={})
        config=dict(residual_training_window=16,batch_days=4,seed=35262)
        with tempfile.TemporaryDirectory() as tmp,patch.dict(model.RECIPE,**config),patch.dict(parent.RECIPE,**config),patch.object(model,'frozen_teacher',side_effect=teacher),patch.object(parent,'frozen_teacher',side_effect=teacher),patch.object(model,'output_file',side_effect=Path),patch.object(parent,'output_file',side_effect=Path):
            checkpoints=[]
            for name,module in [('original',parent),('sam',model)]:
                folder=Path(tmp)/name;folder.mkdir()
                module.fit_residual_stage(panel,None,'unit',1,np.arange(16),[],torch.device('cpu'),folder,'delivery_refit',fixed_epochs=1)
                checkpoints.append(torch.load(folder/'last.pt',weights_only=False))
            for key in ['model','optimizer','numpy_rng','torch_rng']:
                self.assert_nested_equal(checkpoints[0][key],checkpoints[1][key])

    def test_nonfinite_neighborhood_loss_cannot_update_weights(self):
        net=torch.nn.Linear(2,1);optimizer=torch.optim.AdamW(net.parameters(),lr=.001)
        original=copy.deepcopy(net.state_dict());calls=0
        def closure():
            nonlocal calls
            calls+=1
            loss=net.weight.square().sum()
            return loss if calls==1 else loss*float('nan')
        with self.assertRaises(FloatingPointError):model.sam_adamw_step(net,optimizer,closure,rho=.05)
        self.assert_nested_equal(original,net.state_dict());self.assertEqual(len(optimizer.state),0)


if __name__=='__main__':
    unittest.main()
