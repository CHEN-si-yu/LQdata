"""Fixed seed averaging is distinct from selecting a seed or averaging neural weights."""
import importlib.util
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'experiments/_archive/20261024_20261025/scripts'))
import seed_replica_v287_20261025 as review

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module

ensemble=load('seed_ensemble_test',ROOT/'experiments/_archive/20261024_20261025/replicas/v287_seed_review_20261025/ensemble3/model.py')
original=load('seed_member_test',ROOT/'experiments/V287/model.py')


class SeedReplicaReviewTests(unittest.TestCase):
    def test_mean_of_distinct_nonlinear_members_and_complete_reload(self):
        config=dict(technical_indices=[0],residual_hidden=[8,4])
        with patch.dict(ensemble.RECIPE,**config),patch.dict(original.RECIPE,**config):
            torch.manual_seed(814);net=ensemble.PredictModel(5,2).eval()
            with torch.no_grad():
                for i,member in enumerate(net.members):
                    member.residual[-1].weight.normal_(std=.1*(i+1))
                    member.absolute_coef.normal_(std=.02);member.linear_coef.normal_(std=.02)
            standalone=[]
            for member in net.members:
                m=original.PredictModel(5,2).eval();m.load_state_dict(member.state_dict(),strict=True);standalone.append(m)
            x=torch.randn(27,1,5);market=torch.randn(27,2)
            expected=torch.stack([m(x,market) for m in standalone]).mean(dim=0)
            torch.testing.assert_close(net(x,market),expected,rtol=0,atol=0)
            expected_aux=torch.stack([m.absolute_forward(x[:,-1],market) for m in standalone]).mean(dim=0)
            torch.testing.assert_close(net.absolute_forward(x[:,-1],market),expected_aux,rtol=0,atol=0)
            other=ensemble.PredictModel(5,2).eval();other.load_state_dict(net.state_dict(),strict=True)
            torch.testing.assert_close(other(x,market),expected,rtol=0,atol=0)
            self.assertTrue(all(f'members.{i}.residual.0.weight' in net.state_dict() for i in range(3)))

    def test_raw_score_mean_preserves_common_missing_mask_and_refuses_backfill(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths=[Path(tmp)/f'{i}.npy' for i in range(3)]
            for i,p in enumerate(paths):np.save(p,np.asarray([[i+1,np.nan],[2*(i+1),4]],np.float32))
            actual=review.equal_raw_arrays(paths)
            np.testing.assert_array_equal(actual,np.asarray([[2,np.nan],[4,4]],np.float32))
            np.save(paths[-1],np.asarray([[3,0],[6,4]],np.float32))
            with self.assertRaisesRegex(ValueError,'backfill'):review.equal_raw_arrays(paths)

    def test_cached_prevalidation_scoring_restores_functions_and_partition_guard(self):
        days=np.asarray([5,6]);panel=SimpleNamespace(codes=['a','b','c'])
        valid=np.arange(18,dtype=np.float32).reshape(2,3,3)
        old=lambda *args:None
        module=SimpleNamespace(predict_heads=old)
        def validation(net,p,px,requested,device):
            return {'sum':float(module.predict_heads(net,p,requested,device)[:,:,0].sum())}
        module.validation=validation
        result=review.cached_validation_metrics(module,panel,None,days,valid)
        self.assertEqual(result['sum'],float(valid[:,:,0].sum()));self.assertIs(module.predict_heads,old)
        def wrong(net,p,px,requested,device):
            return module.predict_heads(net,p,requested+1,device)
        module.validation=wrong
        with self.assertRaisesRegex(ValueError,'another data partition'):
            review.cached_validation_metrics(module,panel,None,days,valid)
        self.assertIs(module.predict_heads,old)

    def test_composite_cannot_be_retrained_as_one_network(self):
        with self.assertRaisesRegex(RuntimeError,'three fixed seedunits'):ensemble.train_fold(1)

    def test_nested_unit_project_bindings_and_final_main_guard(self):
        import ast
        previous=sys.modules.get('model')
        try:
            for name in ['seed45262','seed55262','ensemble3']:
                directory=review.STUDY/name
                tree=ast.parse((directory/'analysis.py').read_text())
                self.assertIsInstance(tree.body[-1],ast.If)
                self.assertIn('__name__',ast.unparse(tree.body[-1].test))
                member=review.module_at(directory/'model.py',f'scope_member_{name}')
                sys.modules['model']=member
                analysis=review.module_at(directory/'analysis.py',f'scope_analysis_{name}')
                self.assertEqual(analysis.PROJECT_ROOT.resolve(),review.ROOT.resolve())
                self.assertIn('PROJECT_ROOT',analysis.diversity_report.__globals__)
                if name!='ensemble3':
                    runner=review.module_at(directory/'run.py',f'scope_runner_{name}')
                    tasks,schedule=runner.schedule_tasks([('2025Q3',1)])
                    self.assertEqual(tasks,[('2025Q3',1)])
                    self.assertEqual(schedule['reference'],'V287')
        finally:
            if previous is None:sys.modules.pop('model',None)
            else:sys.modules['model']=previous

    def test_optional_quarter_refuses_to_overwrite_main_member_audit(self):
        with patch.object(review,'verify_sources',side_effect=AssertionError('must reject before reading or writing')):
            with self.assertRaisesRegex(ValueError,'separate audit'):
                review.assemble(quarters=('2026Q4',))
            with self.assertRaisesRegex(ValueError,'remain in archive model_info'):
                review.assemble(quarters=('2026Q4',),audit_output='/tmp/seed_audit.json')


if __name__=='__main__':
    unittest.main()
