"""Learned activations preserve the parent initialization and raw auxiliary/teacher branches."""
import ast
import importlib.util
import io
from pathlib import Path
import unittest
from unittest.mock import patch
import torch

ROOT=Path(__file__).resolve().parents[1]

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module

model=load('local_activation_test',ROOT/'experiments/V285/model.py')
parent=load('local_activation_parent',ROOT/'experiments/V263/model.py')


class LocalSplineActivationTests(unittest.TestCase):
    def test_local_interpolation_and_silu_outside_support(self):
        activation=model.LocalSplineSiLU(1).double()
        with torch.no_grad():activation.coefficients.copy_(torch.tensor([[1.,2.,3.,4.,5.]],dtype=torch.float64))
        x=torch.tensor([[-4.],[-2.],[-1.5],[.25],[2.],[4.]],dtype=torch.float64,requires_grad=True)
        residual=activation(x)-torch.nn.functional.silu(x)
        torch.testing.assert_close(residual[:,0],torch.tensor([0.,1.,1.5,3.25,5.,0.],dtype=torch.float64),rtol=0,atol=1e-14)
        slope=torch.autograd.grad(residual[[2,3]].sum(),x)[0]
        torch.testing.assert_close(slope[[2,3],0],torch.ones(2,dtype=torch.float64),rtol=0,atol=1e-14)

    def test_initial_rng_parameters_and_nonzero_parent_forward_identical(self):
        config=dict(technical_indices=[0],residual_hidden=[8,4])
        with patch.dict(model.RECIPE,**config),patch.dict(parent.RECIPE,**config):
            torch.manual_seed(51);expected=parent.PredictModel(5,2).eval();rng=torch.get_rng_state()
            torch.manual_seed(51);actual=model.PredictModel(5,2).eval()
            self.assertTrue(torch.equal(rng,torch.get_rng_state()))
            for key,value in expected.state_dict().items():torch.testing.assert_close(actual.state_dict()[key],value,rtol=0,atol=0)
            with torch.no_grad():
                expected.residual[-1].weight.normal_(std=.1)
                actual.residual[-1].weight.copy_(expected.residual[-1].weight)
            x=torch.randn(32,1,5);market=torch.randn(32,2)
            torch.testing.assert_close(actual(x,market),expected(x,market),rtol=0,atol=0)

    def test_learning_changes_relative_only_and_ic_reaches_activation_coefficients(self):
        with patch.dict(model.RECIPE,technical_indices=[0],residual_hidden=[8,4]):
            torch.manual_seed(24);net=model.PredictModel(5,2).eval()
            with torch.no_grad():
                net.residual[-1].weight.normal_(std=.1);net.absolute_coef.normal_(std=.02);net.linear_coef.normal_(std=.02)
            x=torch.randn(24,1,5);market=torch.randn(24,2);before=net(x,market);teacher=net.teacher_score(x).detach()
            with torch.no_grad():
                net.residual[2].coefficients.normal_(std=.1)
                net.residual[6].coefficients.normal_(std=.1)
            after=net(x,market)
            torch.testing.assert_close(before[:,1],after[:,1],rtol=0,atol=0)
            torch.testing.assert_close(teacher,net.teacher_score(x),rtol=0,atol=0)
            self.assertGreater(float((before[:,0]-after[:,0]).detach().abs().max()),1e-5)
            loss=model.ic_diversity_loss(after[:,0],torch.randn(24),teacher,[12,12])[0]
            loss.backward()
            for index in [2,6]:
                grad=net.residual[index].coefficients.grad
                self.assertTrue(torch.isfinite(grad).all());self.assertGreater(float(grad.abs().sum()),0)

    def test_complete_weights_reload_and_chunk_inference(self):
        with patch.dict(model.RECIPE,technical_indices=[0],residual_hidden=[8,4]):
            torch.manual_seed(27);net=model.PredictModel(5,2).eval()
            with torch.no_grad():
                net.residual[-1].weight.normal_(std=.1)
                for i in [2,6]:net.residual[i].coefficients.normal_(std=.1)
            stream=io.BytesIO();torch.save(net.state_dict(),stream);stream.seek(0)
            other=model.PredictModel(5,2).eval();other.load_state_dict(torch.load(stream,weights_only=True),strict=True)
            x=torch.randn(31,1,5);market=torch.randn(31,2)
            expected=net(x,market)
            torch.testing.assert_close(other(x,market),expected,rtol=0,atol=0)
            chunks=torch.cat([other(x[:13],market[:13]),other(x[13:],market[13:])])
            torch.testing.assert_close(chunks,expected,rtol=1e-5,atol=1e-7)

    def test_effective_training_teacher_and_loss_code_unchanged(self):
        def functions(path):
            tree=ast.parse(path.read_text())
            return {n.name:ast.dump(n,include_attributes=False) for n in tree.body if isinstance(n,ast.FunctionDef)}
        a=functions(ROOT/'experiments/V285/model.py');b=functions(ROOT/'experiments/V263/model.py')
        for name in ['fit_residual_stage','frozen_teacher','ic_diversity_loss','train_fold','validation']:
            self.assertEqual(a[name],b[name],name)


if __name__=='__main__':
    unittest.main()
