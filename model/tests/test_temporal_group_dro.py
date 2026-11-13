"""Training-era robustness must preserve the parent model and resume its adversary exactly."""
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

model=load('temporal_group_test',ROOT/'experiments/V284/model.py')
parent=load('temporal_group_parent',ROOT/'experiments/V263/model.py')


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
    y=(x[:,:,0]*np.repeat([1.,-1.,.3,-.2,1.],4)[:,None]+rng.normal(size=(20,32))*.4).astype(np.float32)
    panel=SimpleNamespace(days=np.asarray([f'day{i:02}' for i in range(20)]),
        features=list('abc'),market_dim=2,M=rng.normal(size=(20,2)).astype(np.float32),Y={'label_ret_1d':y})
    panel.mask=lambda d:np.ones(32,bool)
    panel.inputs=lambda d,mask:x[d,mask,None,:]
    return panel


class TemporalGroupTests(unittest.TestCase):
    def test_uniform_weights_match_original_loss_and_gradient(self):
        torch.manual_seed(9)
        prediction=torch.randn(32,requires_grad=True)
        target=torch.randn(32)
        teacher=prediction.detach()+.05*torch.randn(32)
        sizes=[8]*4
        expected=parent.ic_diversity_loss(prediction,target,teacher,sizes)[0]
        actual=model.temporal_group_ic_loss(prediction,target,teacher,sizes,[0,1,2,3],np.full(4,-np.log(4)))[0]
        torch.testing.assert_close(actual,expected,rtol=1e-6,atol=1e-7)
        a=torch.autograd.grad(actual,prediction,retain_graph=True)[0]
        b=torch.autograd.grad(expected,prediction)[0]
        torch.testing.assert_close(a,b,rtol=1e-6,atol=1e-7)

    def test_adversary_emphasizes_weaker_groups_and_teacher_penalty_stays_unweighted(self):
        q=model.update_group_adversary(np.full(4,-np.log(4)),[.2,.1,-.2,-.1])
        self.assertGreater(np.exp(q)[2],np.exp(q)[3])
        self.assertGreater(np.exp(q)[3],np.exp(q)[1])
        self.assertAlmostEqual(float(np.exp(q).sum()),1.)
        torch.manual_seed(8)
        prediction=torch.randn(32,requires_grad=True);target=torch.randn(32);teacher=prediction.detach().clone()
        shifts=[]
        for weights in [np.full(4,-np.log(4)),q]:
            with patch.dict(model.RECIPE,loss_parent_penalty_weight=0.):
                without=model.temporal_group_ic_loss(prediction,target,teacher,[8]*4,[0,1,2,3],weights)[0]
            with patch.dict(model.RECIPE,loss_parent_penalty_weight=2.):
                with_penalty=model.temporal_group_ic_loss(prediction,target,teacher,[8]*4,[0,1,2,3],weights)[0]
            shifts.append(float((with_penalty-without).detach()))
        self.assertAlmostEqual(shifts[0],shifts[1],places=7)
        with self.assertRaises(ValueError):model.update_group_adversary(q,[.1,np.nan,.1,.1])
        with self.assertRaises(ValueError):model.validated_group_log_weights([0,0,0,0],4)

    def test_groups_use_only_explicit_chronological_training_dates(self):
        panel=fixture();days=np.arange(3,19)
        mapping,info=model.chronological_training_groups(panel,days)
        self.assertEqual(set(mapping),set(days));self.assertNotIn(19,mapping)
        self.assertEqual([mapping[d] for d in days],list(np.repeat(np.arange(4),4)))
        panel.Y['label_ret_1d'][:]=1e6
        other,again=model.chronological_training_groups(panel,days)
        self.assertEqual(mapping,other);self.assertEqual(info,again)
        with self.assertRaises(ValueError):model.chronological_training_groups(panel,[2,1,3,4])

    def test_real_stage_resume_preserves_model_optimizer_adversary_and_excludes_future_labels(self):
        panel=fixture()
        def teacher(p,q,f,phase,train):
            return TinyNet(),{},dict(sha256='fixed-test-teacher',train_end=str(p.days[train[-1]]),absolute_fit={})
        config=dict(residual_training_window=16,batch_days=4,seed=35262)
        with tempfile.TemporaryDirectory() as tmp,patch.dict(model.RECIPE,**config),patch.object(model,'frozen_teacher',side_effect=teacher),patch.object(model,'output_file',side_effect=Path):
            folders=[Path(tmp)/name for name in ['full','resumed','future']]
            for folder in folders:folder.mkdir()
            def run(folder,epochs):
                return model.fit_residual_stage(panel,None,'unit',1,np.arange(16),[],torch.device('cpu'),folder,'delivery_refit',fixed_epochs=epochs)
            run(folders[0],3)
            run(folders[1],1);run(folders[1],3)
            panel.Y['label_ret_1d'][16:]=1e6
            run(folders[2],3)
            states=[torch.load(f/'last.pt',map_location='cpu',weights_only=False) for f in folders]
            for state in states[1:]:
                for key,value in states[0]['model'].items():torch.testing.assert_close(state['model'][key],value,rtol=0,atol=0)
                self.assertEqual(state['temporal_group_state'],states[0]['temporal_group_state'])
                self.assertEqual(state['history'],states[0]['history'])
                for key,value in states[0]['optimizer']['state'].items():
                    for field,tensor in value.items():torch.testing.assert_close(state['optimizer']['state'][key][field],tensor,rtol=0,atol=0)
            self.assertGreater(np.ptp(np.exp(states[0]['temporal_group_state']['log_weights'])),1e-5)

    def test_parent_network_rng_and_all_forward_outputs_unchanged(self):
        config=dict(technical_indices=[0],residual_hidden=[8,4])
        with patch.dict(model.RECIPE,**config),patch.dict(parent.RECIPE,**config):
            torch.manual_seed(61);expected=parent.PredictModel(5,2).eval();rng=torch.get_rng_state()
            torch.manual_seed(61);actual=model.PredictModel(5,2).eval()
            self.assertTrue(torch.equal(rng,torch.get_rng_state()))
            for name,value in expected.state_dict().items():torch.testing.assert_close(actual.state_dict()[name],value,rtol=0,atol=0)
            x=torch.randn(20,1,5);market=torch.randn(20,2)
            torch.testing.assert_close(actual(x,market),expected(x,market),rtol=0,atol=0)


if __name__=='__main__':
    unittest.main()
