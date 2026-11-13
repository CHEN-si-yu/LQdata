"""Peer context is feature-only, permutation equivariant and isolated by signal day."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def load():
    spec=importlib.util.spec_from_file_location('set_attention_test_model',ROOT/'experiments/V277/model.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


class SetAttentionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.m=load()

    def model_and_inputs(self):
        m=self.m;torch=m.torch;torch.set_num_threads(1);torch.manual_seed(35262)
        width=len(m.RECIPE['feature_names']);net=m.PredictModel(width,3);net.eval();rng=np.random.default_rng(277)
        x=torch.from_numpy(rng.normal(size=(18,1,width)).astype(np.float32))
        state=torch.from_numpy(np.repeat([[.1,-.2,.3],[-.7,.5,.1]],[8,10],axis=0).astype(np.float32))
        with torch.no_grad():
            net.residual[-1].weight.copy_(torch.from_numpy(rng.normal(size=net.residual[-1].weight.shape).astype(np.float32)))
        return net,x,state

    def test_day_boundaries_permutation_and_checkpoint_roundtrip(self):
        m=self.m;net,x,state=self.model_and_inputs();both=net(x,state,sizes=[8,10]).detach().numpy()
        first=net(x[:8],state[:8]).detach().numpy();np.testing.assert_allclose(first,both[:8],atol=2e-6)
        changed=x.clone();changed[8:]=4.;np.testing.assert_array_equal(both[:8],net(changed,state,sizes=[8,10]).detach().numpy()[:8])
        perm=np.asarray([4,7,1,2,5,3,0,6]);np.testing.assert_allclose(net(x[:8][perm],state[:8][perm]).detach().numpy(),first[perm],atol=2e-6)
        clone=m.PredictModel(x.shape[-1],3);clone.load_state_dict(net.state_dict());clone.eval()
        np.testing.assert_array_equal(both,clone(x,state,sizes=[8,10]).detach().numpy())
        with self.assertRaises(ValueError):net(x,state,sizes=[8,9])

    def test_missing_label_peer_stays_in_inputs_and_influences_context(self):
        m=self.m;torch=m.torch;net,x,state=self.model_and_inputs()
        panel=SimpleNamespace(X=x[:8].numpy().transpose(1,0,2),Y={'label_ret_1d':np.asarray([[.01,.02,-.01,.04,-.03,.005,np.nan,.03]])},M=state[:1].numpy(),market_dim=3,
            mask=lambda d:np.ones(8,bool))
        panel.inputs=lambda d,mask:panel.X[:,mask].transpose(1,0,2).copy()
        a=m.set_training_day(panel,0);self.assertEqual(len(a[0]),8);self.assertTrue(np.isnan(a[2][6]))
        panel.Y['label_ret_1d'][0,6]=1e8;b=m.set_training_day(panel,0)
        np.testing.assert_array_equal(a[0],b[0]);np.testing.assert_array_equal(a[1],b[1])
        before=net(x[:8],state[:8])[:,0].detach().numpy();changed=x[:8].clone();changed[6]=4.
        after=net(changed,state[:8])[:,0].detach().numpy();self.assertGreater(float(np.max(np.abs(before[:6]-after[:6]))),1e-6)
        pred=net(x[:8],state[:8])[:,0];teacher=net.teacher_score(x[:8]).detach()
        loss,_,_=m.ic_diversity_loss(pred,torch.from_numpy(a[2]),teacher,[8]);self.assertTrue(torch.isfinite(loss));loss.backward()
        self.assertTrue(torch.isfinite(net.set_key.weight.grad).all());self.assertGreater(float(net.set_key.weight.grad.abs().sum()),0.)

    def test_peer_attention_leaves_raw_teacher_and_absolute_branch_unchanged(self):
        m=self.m;torch=m.torch;net,x,state=self.model_and_inputs();rng=np.random.default_rng(278)
        with torch.no_grad():
            net.linear_coef.copy_(torch.from_numpy(rng.normal(size=net.linear_coef.shape).astype(np.float32)))
            net.absolute_coef.copy_(torch.from_numpy(rng.normal(size=net.absolute_coef.shape).astype(np.float32)))
        teacher=net.teacher_score(x).detach().numpy();absolute=net(x,state,sizes=[8,10])[:,1].detach().numpy()
        with torch.no_grad():net.set_mix.weight.mul_(10.)
        np.testing.assert_array_equal(teacher,net.teacher_score(x).detach().numpy())
        np.testing.assert_array_equal(absolute,net(x,state,sizes=[8,10])[:,1].detach().numpy())


if __name__=='__main__':unittest.main()
