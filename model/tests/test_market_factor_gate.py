"""Learned market conditioning changes ranking factors while preserving raw auxiliary contracts."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def load(name):
    spec=importlib.util.spec_from_file_location(name+'_factor_gate_test',ROOT/'experiments'/name/'model.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


class MarketFactorGateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.m=load('V276');cls.parent=load('V263')

    def test_identity_is_exact_shared_initialization_without_rng_consumption(self):
        m=self.m;torch=m.torch;torch.set_num_threads(1);width=len(m.RECIPE['feature_names'])
        torch.manual_seed(35262);parent=self.parent.PredictModel(width,3);rng=torch.get_rng_state().clone()
        torch.manual_seed(35262);net=m.PredictModel(width,3)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        for k,v in parent.state_dict().items():self.assertTrue(torch.equal(v,net.state_dict()[k]),k)
        state=torch.as_tensor([.1,-.2,.3]);np.testing.assert_array_equal(net.factor_gates(state).detach().numpy(),np.ones(width))

    def test_known_state_changes_relative_ranking_but_not_teacher_or_absolute(self):
        m=self.m;torch=m.torch;torch.set_num_threads(1);width=len(m.RECIPE['feature_names'])
        net=m.PredictModel(width,3);net.eval();rng=np.random.default_rng(276)
        x=torch.from_numpy(rng.normal(size=(35,1,width)).astype(np.float32))
        states=torch.from_numpy(rng.normal(size=(35,3)).astype(np.float32))
        with torch.no_grad():
            net.residual[-1].weight.copy_(torch.from_numpy(rng.normal(size=net.residual[-1].weight.shape).astype(np.float32)))
            net.linear_coef.copy_(torch.from_numpy(rng.normal(size=net.linear_coef.shape).astype(np.float32)))
            net.absolute_coef.copy_(torch.from_numpy(rng.normal(size=net.absolute_coef.shape).astype(np.float32)))
        before=net(x,states).detach().numpy();teacher=net.teacher_score(x).detach().clone()
        with torch.no_grad():net.market_factor_weight.copy_(torch.from_numpy(rng.normal(size=net.market_factor_weight.shape).astype(np.float32)))
        after=net(x,states).detach().numpy();np.testing.assert_array_equal(before[:,1],after[:,1])
        self.assertTrue(torch.equal(teacher,net.teacher_score(x).detach()))
        self.assertGreater(float(np.std(before[:,0]-after[:,0])),1e-5)
        gates=net.factor_gates(states).detach().numpy();self.assertGreaterEqual(gates.min(),0.);self.assertLessEqual(gates.max(),2.)
        perm=rng.permutation(len(x));np.testing.assert_allclose(net(x[perm],states[perm]).detach().numpy(),after[perm],atol=2e-6)
        changed=states.clone();changed[20:]=1e6
        np.testing.assert_array_equal(after[:20],net(x,changed).detach().numpy()[:20])
        loss=net(x,states)[:,0].square().mean();loss.backward()
        self.assertTrue(torch.isfinite(net.market_factor_weight.grad).all());self.assertGreater(float(net.market_factor_weight.grad.abs().sum()),0.)
        clone=m.PredictModel(width,3);clone.load_state_dict(net.state_dict());clone.eval()
        np.testing.assert_array_equal(after,clone(x,states).detach().numpy())


if __name__=='__main__':unittest.main()
