"""A functional teacher anchor must be embedded, phase-matched and auxiliary-neutral."""
import ast
import copy
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


model=load('teacher_anchored_test',ROOT/'experiments/V287/model.py')
parent=load('teacher_anchored_parent',ROOT/'experiments/V263/model.py')


class TeacherAnchoredScoreTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_no_new_parameters_or_rng_and_initial_relative_score_exactly_embedded_teacher(self):
        config=dict(technical_indices=[0],residual_hidden=[8,4])
        with patch.dict(model.RECIPE,**config),patch.dict(parent.RECIPE,**config):
            torch.manual_seed(18);net=model.PredictModel(5,2).eval();state=torch.get_rng_state()
            torch.manual_seed(18);original=parent.PredictModel(5,2).eval()
            self.assertTrue(torch.equal(state,torch.get_rng_state()))
            self.assertEqual([(k,p.shape) for k,p in net.named_parameters()],[(k,p.shape) for k,p in original.named_parameters()])
            for key,value in original.state_dict().items():torch.testing.assert_close(net.state_dict()[key],value,rtol=0,atol=0)
            with torch.no_grad():net.linear_coef.normal_(std=.02);net.absolute_coef.normal_(std=.02)
            x=torch.randn(31,1,5);market=torch.randn(31,2)
            torch.testing.assert_close(net(x,market)[:,0],net.teacher_score(x),rtol=0,atol=0)

    def test_nonzero_correction_is_additive_and_absolute_branch_unchanged(self):
        config=dict(technical_indices=[0],residual_hidden=[8,4])
        with patch.dict(model.RECIPE,**config),patch.dict(parent.RECIPE,**config):
            torch.manual_seed(24);net=model.PredictModel(5,2).eval()
            with torch.no_grad():
                net.linear_coef.normal_(std=.02);net.absolute_coef.normal_(std=.02);net.residual[-1].weight.normal_(std=.1)
            original=parent.PredictModel(5,2).eval();original.load_state_dict(net.state_dict(),strict=True)
            x=torch.randn(31,1,5);market=torch.randn(31,2)
            expected=original(x,market);actual=net(x,market)
            torch.testing.assert_close(actual[:,0],expected[:,0]+net.teacher_score(x),rtol=0,atol=0)
            torch.testing.assert_close(actual[:,1:],expected[:,1:],rtol=0,atol=0)

    def test_original_full_ic_loss_learns_correction_without_updating_teacher_or_absolute(self):
        with patch.dict(model.RECIPE,technical_indices=[0],residual_hidden=[8,4]):
            torch.manual_seed(27);net=model.PredictModel(5,2).train()
            with torch.no_grad():net.linear_coef.normal_(std=.02);net.absolute_coef.normal_(std=.02)
            teacher=net.linear_coef.clone();absolute=net.absolute_coef.clone()
            x=torch.randn(48,1,5);market=torch.randn(48,2);target=torch.randn(48)
            loss=model.ic_diversity_loss(net(x,market)[:,0],target,net.teacher_score(x),[24,24])[0]
            self.assertTrue(torch.isfinite(loss));loss.backward()
            self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in net.parameters()))
            self.assertGreater(float(net.residual[-1].weight.grad.norm()),1e-8)
            optimizer=torch.optim.AdamW(net.parameters(),lr=.001);optimizer.step()
            torch.testing.assert_close(net.linear_coef,teacher,rtol=0,atol=0)
            torch.testing.assert_close(net.absolute_coef,absolute,rtol=0,atol=0)
            self.assertFalse(net.linear_coef.requires_grad);self.assertFalse(net.absolute_coef.requires_grad)

    def test_whole_state_is_portable_and_chunk_inference_independent_of_training_teacher_files(self):
        with patch.dict(model.RECIPE,technical_indices=[0],residual_hidden=[8,4]):
            torch.manual_seed(28);net=model.PredictModel(5,2).eval()
            with torch.no_grad():
                net.linear_coef.normal_(std=.02);net.absolute_coef.normal_(std=.02);net.residual[-1].weight.normal_(std=.1)
            stream=io.BytesIO();torch.save(dict(model=net.state_dict(),features=list('abcde')),stream);stream.seek(0)
            other=model.PredictModel(5,2).eval();other.load_state_dict(torch.load(stream,weights_only=True)['model'],strict=True)
            x=torch.randn(37,1,5);market=torch.randn(37,2)
            with patch.object(model,'frozen_teacher',side_effect=RuntimeError('training teacher unavailable')):
                expected=net(x,market);actual=other(x,market)
                chunks=torch.cat([other(x[:17],market[:17]),other(x[17:],market[17:])])
            torch.testing.assert_close(actual,expected,rtol=0,atol=0)
            torch.testing.assert_close(chunks,expected,rtol=1e-5,atol=1e-7)

    def test_training_loss_stage_matched_teacher_refit_and_prediction_code_unchanged(self):
        def functions(path):
            return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef)}
        a=functions(ROOT/'experiments/V287/model.py');b=functions(ROOT/'experiments/V263/model.py')
        for name in ['fit_residual_stage','ic_diversity_loss','frozen_teacher','train_fold','validation','splits','delivery_train_days','predict','predict_auxiliary']:
            self.assertEqual(a[name],b[name],name)
        for path in (ROOT/'experiments/V287').glob('*.py'):
            tree=ast.parse(path.read_text());guards=[n for n in tree.body if isinstance(n,ast.If) and '__name__' in ast.unparse(n.test)]
            self.assertEqual(len(guards),1);self.assertIs(tree.body[-1],guards[0])


if __name__=='__main__':
    unittest.main()
