"""下行目标应区分均值相同但风险不同的收益分布。"""
import importlib.util
from pathlib import Path
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('test_expectile_model_module',ROOT/'experiments/V266/model.py')
model=importlib.util.module_from_spec(spec)
spec.loader.exec_module(model)


class ExpectileModelTests(unittest.TestCase):
    def test_equal_mean_higher_downside_distribution_gets_lower_score(self):
        outcomes=model.torch.tensor([-2.,-1.,0.,1.,2.])
        target=model.torch.cat([outcomes,3*outcomes])
        scores=model.torch.nn.Parameter(model.torch.zeros(2))
        optimizer=model.torch.optim.SGD([scores],lr=.2)
        weight=model.torch.full((10,),.2)
        for _ in range(400):
            prediction=model.torch.repeat_interleave(scores,5)
            loss=model.downside_expectile_loss(prediction,target,weight,2)
            optimizer.zero_grad();loss.backward();optimizer.step()
        self.assertLess(float(scores[1].detach()),float(scores[0].detach()))
        self.assertLess(float(scores[0].detach()),0.)
        np.testing.assert_allclose(scores.detach().numpy(),[-1.8/2.2,-5.4/2.2],rtol=0,atol=1e-4)


if __name__=='__main__':unittest.main()
