"""Exercise path routing with synthetic data; do not calculate or prune production factors."""
import contextlib
import importlib.util
import io
from pathlib import Path
import sys
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from fea.dayhash import cmd_dayhash


class LayoutTests(unittest.TestCase):
    def test_post_steps_locates_project_from_another_cwd(self):
        source=Path(__file__).resolve().parents[1]/'scripts/post_steps.sh'
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)/'project';(root/'scripts').mkdir(parents=True);(root/'logs').mkdir()
            stub=Path(td)/'python-stub'
            stub.write_text('#!/bin/sh\nprintf "resolved-root=%s\\n" "$PWD"\n')
            stub.chmod(0o700)
            script=root/'scripts/post_steps.sh'
            # Stub computation only; exercise the actual project's directory and log resolution.
            script.write_text(source.read_text().replace('PY=/autodl-fs/data/miniconda3/bin/python',f'PY={stub}'))
            result=subprocess.run(['bash',str(script)],cwd=td,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            logs=list((root/'logs').glob('post_steps_*.log'))
            self.assertEqual(len(logs),1)
            self.assertEqual(logs[0].read_text().count('resolved-root='+str(root)),4)

    def test_dayhash_date_dirs_and_explicit_override(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            cfg=SimpleNamespace(root=root, factors_dir=root/'data/factors')
            for date,out,expected in [
                ('2026-09-18',None,'artifacts/dayhash/2026-09-18'),
                ('2027-09-18',None,'artifacts/dayhash/2027-09-18'),
                ('2026-09-18','custom/location','custom/location'),
            ]:
                args=SimpleNamespace(date=date,days=1,factors=None,out=out,verify=False,jobs=1)
                with patch('fea.dayhash.recent_trading_days',return_value=[int(date.replace('-',''))]), \
                     patch('fea.dayhash.all_specs',return_value=[]), contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(cmd_dayhash(args,cfg),0)
                self.assertTrue((root/expected/'dayhash.tsv').exists())
                self.assertTrue((root/expected/'meta.json').exists())
            self.assertFalse((root/'log0918').exists())

    def test_verify_does_not_borrow_old_date_baseline(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            old=root/'artifacts/dayhash/log0918'
            old.mkdir(parents=True)
            (old/'dayhash.prev.tsv').write_text('old evidence')
            args=SimpleNamespace(date='2026-09-18',days=1,factors=None,out=None,verify=True,jobs=1)
            with patch('fea.dayhash.recent_trading_days',return_value=[20260918]), \
                 patch('fea.dayhash.all_specs',return_value=[]), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(cmd_dayhash(args,SimpleNamespace(root=root)),2)
            self.assertEqual((old/'dayhash.prev.tsv').read_text(),'old evidence')
            self.assertFalse((root/'artifacts/dayhash/2026-09-18/dayhash.tsv').exists())

    def test_manifest_backup_goes_to_archive(self):
        source=Path(__file__).resolve().parents[1]/'scripts/rebuild_manifest.py'
        spec=importlib.util.spec_from_file_location('manifest_layout_test',source)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);factors=root/'data/factors';state=root/'state'
            (factors/'sample').mkdir(parents=True);state.mkdir()
            (state/'sample.json').write_text('{"original":true}')
            cfg=SimpleNamespace(root=root,factors_dir=factors,state_dir=state)
            man=SimpleNamespace(partitions={},coverage=[],save=lambda:None)
            with patch.object(module.cfg_mod,'load',return_value=cfg), \
                 patch.object(module,'scan_partitions',return_value=({},[])), \
                 patch.object(module.Manifest,'load',return_value=man), \
                 patch.object(sys,'argv',['rebuild_manifest.py']), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(module.main(),0)
            found=list((root/'artifacts/archive/backtest').glob('*/sample.json'))
            self.assertEqual(len(found),1)
            self.assertEqual(found[0].read_bytes(),(state/'sample.json').read_bytes())
            self.assertFalse((state/'backtest').exists())


if __name__=='__main__': unittest.main()
