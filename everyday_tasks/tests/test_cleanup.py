import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from data_incremental import paths
from data_incremental.tools import cleanup_backups as C


class CleanupTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.patches = [patch.object(paths, name, self.root / rel) for name, rel in (
            ('BACKTEST_DIR', 'tests'), ('DATA_ROOT', 'data'), ('STATE_ROOT', 'state'))]
        for p in self.patches: p.start()
        for d in (paths.BACKTEST_DIR, paths.DATA_ROOT, paths.STATE_ROOT): d.mkdir()
        self.old = paths.BACKTEST_DIR / 'old.parquet'
        self.old.write_bytes(b'backup content')
        self.live = paths.DATA_ROOT / 'live.parquet'
        self.live.write_bytes(b'production')

    def tearDown(self):
        for p in reversed(self.patches): p.stop()
        self.tmp.cleanup()

    def test_dry_run_does_not_mutate(self):
        result = C.run(log=lambda *_: None)
        self.assertEqual(result['files'], 1)
        self.assertTrue(self.old.exists())
        self.assertFalse((paths.BACKTEST_DIR / 'checksum_archives').exists())

    def test_hashes_retained_and_production_untouched(self):
        md5_file = self.old.with_suffix('.parquet.md5')
        md5_file.write_text('prior fingerprint')
        report = paths.BACKTEST_DIR / 'report.json'; report.write_text('{}')
        result = C.run(apply=True, log=lambda *_: None)
        items = json.loads(Path(result['manifest']).read_text())['files']
        self.assertEqual(items[0]['md5'], hashlib.md5(b'backup content').hexdigest())
        self.assertEqual(items[0]['sha256'], hashlib.sha256(b'backup content').hexdigest())
        self.assertFalse(self.old.exists())
        self.assertEqual(self.live.read_bytes(), b'production')
        self.assertTrue(report.exists()); self.assertTrue(md5_file.exists())
        self.assertEqual(C.run(apply=True, log=lambda *_: None)['files'], 0)

    def test_symlink_never_follows_production(self):
        (paths.BACKTEST_DIR / 'link.parquet').symlink_to(self.live)
        (paths.BACKTEST_DIR / 'linked_dir').symlink_to(paths.DATA_ROOT, target_is_directory=True)
        self.assertEqual(len(C.candidates()), 1)

    def test_changed_file_is_not_deleted(self):
        original = C.durable_json
        def change_after_manifest(path, data):
            original(path, data)
            if path.name == 'checksums.json': self.old.write_bytes(b'changed externally')
        with patch.object(C, 'durable_json', side_effect=change_after_manifest):
            with self.assertRaises(RuntimeError): C.run(apply=True, log=lambda *_: None)
        self.assertEqual(self.old.read_bytes(), b'changed externally')

    def test_manifest_write_failure_deletes_nothing(self):
        with patch.object(C, 'durable_json', side_effect=OSError('disk full')):
            with self.assertRaises(OSError): C.run(apply=True, log=lambda *_: None)
        self.assertTrue(self.old.exists())

    def test_unfinished_download_is_retained(self):
        full = paths.STATE_ROOT / 'cyq_perf_full'; full.mkdir()
        (full / 'report.json').write_text('{"complete":false}')
        (full / 'chunk.parquet').write_bytes(b'in progress')
        self.assertEqual(len(C.candidates()), 1)

    def test_missing_production_partition_blocks_checkpoint_cleanup(self):
        full = paths.STATE_ROOT / 'cyq_perf_full'; full.mkdir()
        (full / 'report.json').write_text('{"complete":true,"years":[{"year":2026,"stored_rows":2}]}')
        (full / 'chunk.parquet').write_bytes(b'checkpoint')
        with self.assertRaises(RuntimeError): C.run(apply=True, log=lambda *_: None)
        self.assertTrue(self.old.exists())

    def test_production_ancestor_is_rejected(self):
        with patch.object(paths, 'BACKTEST_DIR', self.root):
            with self.assertRaises(ValueError): C.candidates()

    def test_resource_cap_applies_in_child_only(self):
        code = '''from data_incremental.core.resources import configure
import resource,os,json
r=configure({'resources':{'max_address_space_gib':24,'numeric_threads':2}})
assert resource.getrlimit(resource.RLIMIT_AS)[0]==24*1024**3
assert os.environ['OPENBLAS_NUM_THREADS']=='2'
print(json.dumps(r))
'''
        run = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)
        self.assertEqual(run.returncode, 0, run.stderr)


if __name__ == '__main__':
    unittest.main(verbosity=2)
