"""The control must reuse only equivalent prevalidation selectors and freeze their masks."""
import ast
import copy
import hashlib
import importlib.util
import json
import multiprocessing as mp
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('selector_feature_freeze_test',ROOT/'experiments/V283/model.py')
model=importlib.util.module_from_spec(spec);spec.loader.exec_module(model)

def parent_recipe():
    source=(ROOT/'experiments/V282/model.py').read_text()
    tree=ast.parse(source)
    recipe=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign)
        and any(isinstance(t,ast.Name) and t.id=='RECIPE' for t in n.targets))
    return recipe,source


def cpu_training_child_probe():
    torch.set_num_threads(2)
    value=torch.randn(128,32)@torch.randn(32,16)
    if not torch.isfinite(value).all():raise AssertionError('nonfinite child CPU operation')


class SelectorFeatureFreezeTests(unittest.TestCase):
    def test_selector_equivalence_accepts_only_metadata_delivery_changes(self):
        recipe,source=parent_recipe();candidate=(ROOT/'experiments/V283/model.py').read_text()
        signature=model.verify_selector_equivalence(recipe,model.RECIPE,source,candidate)
        self.assertEqual(len(signature),64)
        changed=dict(model.RECIPE,lr=model.RECIPE['lr']*2)
        with self.assertRaisesRegex(ValueError,'配方不等价'):
            model.verify_selector_equivalence(recipe,changed,source,candidate)
        changed=candidate.replace('raw*self.feature_gate','raw')
        with self.assertRaisesRegex(ValueError,'代码不等价'):
            model.verify_selector_equivalence(recipe,model.RECIPE,source,changed)

    def test_mature_stage_reads_frozen_gate_without_accessing_new_labels(self):
        with tempfile.TemporaryDirectory() as directory,patch.object(model,'PROJECT_ROOT',Path(directory)):
            folder=Path(directory)/'fold';folder.mkdir()
            panel=SimpleNamespace(features=list('abc'))
            gate=np.asarray([1,0,1],np.float32)
            digest=hashlib.sha256(gate.tobytes()+json.dumps(panel.features).encode()).hexdigest()
            checkpoint=dict(recipe_signature=model.recipe_signature(),model={'feature_gate':torch.from_numpy(gate)},
                teacher=dict(phase='selector',relative_feature_gate=dict(mask_sha256=digest,train_end='2024-10-21',active_features=2)))
            torch.save(checkpoint,folder/'selector_best.pt')
            with patch.object(model,'fit_feature_gate',side_effect=AssertionError('new labels must not be consulted')):
                actual,info=model.stage_feature_gate(panel,np.arange(100),'delivery_refit',folder)
            np.testing.assert_array_equal(actual,gate)
            self.assertEqual(info['train_end'],'2024-10-21')
            self.assertTrue(info['mask_not_selected_on_validation'])
            checkpoint['teacher']['phase']='delivery_refit';torch.save(checkpoint,folder/'selector_best.pt')
            with self.assertRaisesRegex(ValueError,'预验证'):model.stage_feature_gate(panel,np.arange(100),'delivery_refit',folder)

    def test_real_parent_selector_copy_is_exact_and_idempotent(self):
        days,codes=model.axis()
        panel=SimpleNamespace(days=np.asarray(days),codes=np.asarray(codes),features=list(model.RECIPE['feature_names']))
        parent=ROOT/'experiments/V282/model_train/2025Q3/fold1'
        source=torch.load(parent/'selector_best.pt',map_location='cpu',weights_only=False)
        with tempfile.TemporaryDirectory(dir=ROOT/'experiments',prefix='tmp_selector_reuse_') as directory, \
             patch.object(model,'UNIT_ROOT',Path(directory)):
            torch.set_num_threads(2)
            report=model.prepare_selector_reuse(panel,[('2025Q3',1)])
            self.assertEqual(torch.get_num_threads(),1)
            target=Path(directory)/'model_train/2025Q3/fold1'
            copied=torch.load(target/'selector_best.pt',map_location='cpu',weights_only=False)
            for key,value in source['model'].items():torch.testing.assert_close(copied['model'][key],value,rtol=0,atol=0)
            self.assertEqual((target/'valid_predictions.npy').read_bytes(),(parent/'valid_predictions.npy').read_bytes())
            first=(target/'selector_best.pt').read_bytes()
            model.prepare_selector_reuse(panel,[('2025Q3',1)])
            self.assertEqual((target/'selector_best.pt').read_bytes(),first)
            self.assertTrue(report['selector_training_reused'])
            self.assertFalse(report['delivery_weights_reused'])
            self.assertFalse((target/'best.pt').exists())
            process=mp.get_context('fork').Process(target=cpu_training_child_probe)
            process.start();process.join(5)
            if process.is_alive():
                process.terminate();process.join()
                self.fail('training child blocked after parent selector tensor checks')
            self.assertEqual(process.exitcode,0)


if __name__=='__main__':unittest.main()
