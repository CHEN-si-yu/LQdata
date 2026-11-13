"""异方差似然把均值与风险尺度分别学习。"""
import importlib.util
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('test_student_model_module',ROOT/'experiments/V267/model.py')
model=importlib.util.module_from_spec(spec)
spec.loader.exec_module(model)


class StudentModelTests(unittest.TestCase):
    def test_symmetric_distributions_share_mean_but_learn_distinct_scales(self):
        base=model.torch.tensor([-2.,-1.,0.,1.,2.])
        target=model.torch.cat([base,3*base])
        mean=model.torch.nn.Parameter(model.torch.zeros(2))
        raw_scale=model.torch.nn.Parameter(model.torch.zeros(2))
        optimizer=model.torch.optim.Adam([mean,raw_scale],lr=.03)
        weight=model.torch.full((10,),.2)
        for _ in range(500):
            scale=model.torch.nn.functional.softplus(raw_scale)+model.RECIPE['student_scale_floor']
            loss=model.student_t_loss(model.torch.repeat_interleave(mean,5),
                model.torch.repeat_interleave(scale,5),target,weight,2)
            optimizer.zero_grad();loss.backward();optimizer.step()
        self.assertTrue(model.torch.isfinite(mean).all())
        self.assertLess(float(mean.detach().abs().max()),.01)
        scale=model.torch.nn.functional.softplus(raw_scale)+model.RECIPE['student_scale_floor']
        self.assertGreater(float(scale[1].detach()),2*float(scale[0].detach()))


if __name__=='__main__':unittest.main()
