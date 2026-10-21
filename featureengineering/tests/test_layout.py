"""Exercise path routing with synthetic data; do not calculate or prune production factors."""
import contextlib
import importlib.util
import io
from pathlib import Path
import shlex
import sys
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import pandas as pd

from fea.dayhash import cmd_dayhash


class LayoutTests(unittest.TestCase):
    def test_post_steps_locates_project_from_another_cwd(self):
        source=Path(__file__).resolve().parents[1]/'scripts/post_steps.sh'
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)/'project';(root/'scripts').mkdir(parents=True);(root/'logs').mkdir()
            stub=Path(td)/'python-stub'
            stub.write_text('#!/bin/sh\n'
                            'if [ "$1" = "scripts/_project_paths.py" ]; then\n'
                            f'  exec {shlex.quote(Path(sys.executable).as_posix())} "$@"\n'
                            'fi\n'
                            'printf "resolved-root=%s\\n" "$PWD"\n')
            stub.chmod(0o700)
            script=root/'scripts/post_steps.sh'
            (root/'scripts/_project_paths.py').write_bytes((source.parent/'_project_paths.py').read_bytes())
            # Stub computation only; exercise the actual project's directory and log resolution.
            script.write_text(source.read_text().replace('PY=/autodl-fs/data/miniconda3/bin/python',
                                                        f'PY={shlex.quote(stub.as_posix())}'))
            result=subprocess.run(['bash',str(script)],cwd=td,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            logs=list((root/'logs').glob('post_steps_*.log'))
            self.assertEqual(len(logs),1)
            log=logs[0].read_text()
            self.assertEqual(log.count('resolved-root='+str(root)),4)
            steps=('check','audit-pit','eval --years 2026 2026','dedup')
            positions=[log.index('=== main.py '+step+'  ') for step in steps]
            self.assertEqual(positions,sorted(positions))
            self.assertEqual(log.count('退出码 0'),4)
            before={path.name:path.read_bytes() for path in logs}
            outside=Path(td)/'outside'/'forbidden.log'
            for argument in (str(outside),'../outside/forbidden.log'):
                result=subprocess.run(['bash',str(script),argument],cwd=td,capture_output=True,text=True)
                self.assertNotEqual(result.returncode,0)
                self.assertIn('Output must stay inside',result.stderr)
                self.assertFalse(outside.parent.exists())
                self.assertEqual({path.name:path.read_bytes() for path in (root/'logs').glob('*')},before)

    def test_dayhash_date_dirs_and_explicit_override(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)/'project';root.mkdir()
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
            root=Path(td)/'project';root.mkdir()
            old=root/'artifacts/dayhash/log0918'
            old.mkdir(parents=True)
            (old/'dayhash.prev.tsv').write_text('old evidence')
            args=SimpleNamespace(date='2026-09-18',days=1,factors=None,out=None,verify=True,jobs=1)
            with patch('fea.dayhash.recent_trading_days',return_value=[20260918]), \
                 patch('fea.dayhash.all_specs',return_value=[]), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(cmd_dayhash(args,SimpleNamespace(root=root)),2)
            self.assertEqual((old/'dayhash.prev.tsv').read_text(),'old evidence')
            self.assertFalse((root/'artifacts/dayhash/2026-09-18/dayhash.tsv').exists())

    def test_manifest_write_is_disabled_and_dry_run_preserves_state_and_data(self):
        source=Path(__file__).resolve().parents[1]/'scripts/rebuild_manifest.py'
        spec=importlib.util.spec_from_file_location('manifest_layout_test',source)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);factors=root/'data/factors';state=root/'state'
            partition=factors/'sample/year=2026/data.parquet'
            partition.parent.mkdir(parents=True);state.mkdir()
            (state/'sample.json').write_text('{"original":true}')
            pd.DataFrame({'trade_date':['2026-01-02'],'stock_code':['600001.SH'],
                          'value':[1.25],'rank':[0.5]}).to_parquet(partition,index=False)
            cfg=SimpleNamespace(root=root,factors_dir=factors,state_dir=state)
            before={str(path.relative_to(root)):(path.read_bytes(),path.stat().st_mtime_ns)
                    for path in root.rglob('*') if path.is_file()}
            with patch.object(module.cfg_mod,'load',return_value=cfg), \
                 patch.object(module.Manifest,'save',side_effect=AssertionError('Maintenance script must not save manifests')) as save:
                for arguments in ([],['--backup']):
                    error=io.StringIO()
                    with patch.object(sys,'argv',['rebuild_manifest.py',*arguments]), \
                         contextlib.redirect_stderr(error), self.assertRaises(SystemExit) as rejected:
                        module.main()
                    self.assertEqual(rejected.exception.code,2)
                    self.assertIn('--dry-run',error.getvalue())
                for arguments in (['--dry-run'],['--dry-run','--backup']):
                    with patch.object(sys,'argv',['rebuild_manifest.py',*arguments]), \
                         contextlib.redirect_stdout(io.StringIO()):
                        self.assertEqual(module.main(),0)
                save.assert_not_called()
            after={str(path.relative_to(root)):(path.read_bytes(),path.stat().st_mtime_ns)
                   for path in root.rglob('*') if path.is_file()}
            self.assertEqual(after,before)
            self.assertFalse((root/'artifacts').exists())
            self.assertFalse((state/'backtest').exists())


if __name__=='__main__': unittest.main()
