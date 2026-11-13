"""Factor pairs exclude self-products; the model must retain raw auxiliary contracts."""
import importlib.util
from pathlib import Path
import unittest
import torch
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('factorization_model_test',ROOT/'experiments/V281/model.py')
model=importlib.util.module_from_spec(spec);spec.loader.exec_module(model)


class FactorizationModelTests(unittest.TestCase):
    def test_fast_interactions_match_explicit_pairs_and_gradients(self):
        torch.manual_seed(21)
        fast=model.FactorizationResidual(5,3,0.).double().eval()
        with torch.no_grad():fast.linear.weight.normal_();fast.linear.bias.fill_(.3);fast.factors.normal_(std=.1)
        x=torch.randn(7,5,dtype=torch.float64,requires_grad=True)
        expected=fast.linear(x)
        for i in range(5):
            for j in range(i+1,5):
                expected=expected+(fast.factors[i]*fast.factors[j]).sum()*x[:,i:i+1]*x[:,j:j+1]
        actual=fast(x)
        torch.testing.assert_close(actual,expected,rtol=1e-12,atol=1e-12)
        a=torch.autograd.grad(actual.sum(),[x,fast.factors],retain_graph=True)
        b=torch.autograd.grad(expected.sum(),[x,fast.factors])
        for left,right in zip(a,b):torch.testing.assert_close(left,right,rtol=1e-12,atol=1e-12)

    def test_reload_row_independence_and_raw_auxiliary_are_preserved(self):
        torch.manual_seed(31)
        with patch.dict(model.RECIPE,technical_indices=[0],residual_hidden=[8,4],factorization_rank=3):
            net=model.PredictModel(5,2).eval()
            net.absolute_coef.normal_(std=.02);net.linear_coef.normal_(std=.02)
            x=torch.randn(12,1,5);market=torch.randn(12,2)
            before=net(x,market).detach();absolute=net.absolute_forward(x[:,-1],market).detach()
            teacher=net.teacher_score(x).detach()
            restored=model.PredictModel(5,2).eval();restored.load_state_dict(net.state_dict(),strict=True)
            torch.testing.assert_close(restored(x,market),before,rtol=0,atol=0)
            order=torch.randperm(12)
            torch.testing.assert_close(restored(x[order],market[order]),before[order],rtol=0,atol=0)
            changed=x.clone();changed[8:]=100
            torch.testing.assert_close(net(changed,market)[:8],before[:8],rtol=0,atol=0)
            with torch.no_grad():net.residual.factors.mul_(2)
            torch.testing.assert_close(net.absolute_forward(x[:,-1],market),absolute,rtol=0,atol=0)
            torch.testing.assert_close(net.teacher_score(x),teacher,rtol=0,atol=0)

    def test_direct_ic_has_finite_nonzero_linear_and_factor_gradients(self):
        torch.manual_seed(52)
        layer=model.FactorizationResidual(5,3,.1)
        x=torch.randn(128,5);target=x[:,0]*x[:,1]+.2*x[:,2]
        prediction=.5*torch.tanh(layer(x)[:,0])
        loss,_,_=model.ic_diversity_loss(prediction,target,x[:,4],[64,64])
        loss.backward()
        for parameter in [layer.linear.weight,layer.factors]:
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(float(parameter.grad.abs().sum()),0.)


if __name__=='__main__':unittest.main()
