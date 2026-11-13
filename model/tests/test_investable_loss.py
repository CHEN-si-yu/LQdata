"""Known-capacity training weights change the IC term, leaving the parent constraint fixed."""
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def load(script,label):
    spec=importlib.util.spec_from_file_location(label,ROOT/'experiments/V278'/script)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


class InvestableLossTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.m=load('model.py','investable_loss_model')

    def test_uniform_weights_reproduce_parent_loss_and_gradient(self):
        m=self.m;torch=m.torch;torch.set_num_threads(1);rng=np.random.default_rng(278)
        p=torch.tensor(rng.normal(size=35),dtype=torch.float64,requires_grad=True)
        y=torch.tensor(rng.normal(size=35),dtype=torch.float64);teacher=torch.tensor(rng.normal(size=35),dtype=torch.float64)
        old=m._UnweightedICLoss(p,y,teacher,[15,20])[0];go=torch.autograd.grad(old,p)[0]
        new=m.ic_diversity_loss(p,y,teacher,[15,20],sample_weight=torch.ones_like(p))[0];gn=torch.autograd.grad(new,p)[0]
        torch.testing.assert_close(old,new,rtol=1e-12,atol=1e-12);torch.testing.assert_close(go,gn,rtol=1e-11,atol=1e-12)

    def test_zero_weight_affects_only_ic_not_unweighted_parent_penalty(self):
        m=self.m;torch=m.torch;torch.set_num_threads(1)
        p=torch.tensor([-.4,.2,.6,.7,1.2],requires_grad=True);y=torch.tensor([.8,-.3,.2,.6,.5]);teacher=p.detach().clone()
        w=torch.tensor([1.,.5,1.,1.,0.]);a=m.ic_diversity_loss(p,y,teacher,[5],sample_weight=w)
        changed=p.detach().clone();changed[-1]=-10.;changed.requires_grad_();b=m.ic_diversity_loss(changed,y,teacher,[5],sample_weight=w)
        torch.testing.assert_close(a[1],b[1]);self.assertNotAlmostEqual(float(a[2]),float(b[2]))
        for result in [a,b]:
            expected=-result[1]+torch.relu(result[2]-m.RECIPE['loss_parent_max_correlation']).square()
            torch.testing.assert_close(result[0],expected)
        a[0].backward();self.assertTrue(torch.isfinite(p.grad).all())

    def test_capacity_weights_use_known_day_only_and_preserve_frozen_default(self):
        m=self.m;amount=np.asarray([[0.,np.nan,150e6,300e6,900e6,-3.,np.inf],[1.]*7])
        panel=SimpleNamespace(amount=amount);mask=np.ones(7,bool)
        expected=np.asarray([0,0,.5,1,1,0,0],np.float32)
        np.testing.assert_array_equal(m.capacity_sample_weights(panel,0,mask),expected)
        amount[1]=1e15;np.testing.assert_array_equal(m.capacity_sample_weights(panel,0,mask),expected)
        with patch.dict(os.environ,{}):
            for key in ['MX_DATA','MX_STOCK_LIST','MX_CALENDAR']:os.environ.pop(key,None)
            for p in [m.data_root(),m.stock_list_path(),m.calendar_path()]:
                self.assertIn('input_snapshots/20261024_d4c3dea169ea',str(p));self.assertTrue(p.exists())

    def test_duration_schedule_preserves_exact_task_set_and_records_source(self):
        runner=load('run.py','investable_run_test');tasks=[(q,f) for q in runner.QUARTERS for f in range(1,5)]
        ordered,info=runner.schedule_tasks(tasks);self.assertEqual(set(ordered),set(tasks));self.assertEqual(len(ordered),20)
        self.assertEqual(info['original_order'],tasks);self.assertEqual(info['actual_order'],ordered)
        if len(info['sources'])==20:
            costs={(Path(s['path']).parent.parent.name,int(Path(s['path']).parent.name[4:])):s['seconds'] for s in info['sources']}
            values=[costs[t] for t in ordered];self.assertEqual(values,sorted(values,reverse=True))


if __name__=='__main__':unittest.main()
