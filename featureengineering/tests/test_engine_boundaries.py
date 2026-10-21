import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from fea.delay import load, audit_contract
from fea.derived import DerivedCache
from fea.engine import Engine, partition_content_sha256
from fea.resources import configure, memory_limits, resource_policy, safe_jobs
from fea.upstream import Upstream


class EngineBoundaryTests(unittest.TestCase):
    def frame(self):
        return pd.DataFrame({"trade_date": ["2018-01-02", "2018-01-02"],
                             "stock_code": ["000001.SZ", "600000.SH"],
                             "value": np.array([1.0, np.nan], dtype="float32"),
                             "rank": np.array([0.5, np.nan], dtype="float32")})

    def test_same_footer_but_changed_factor_value_fails_content_check(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "data.parquet"
            frame = self.frame()
            record = {"rows": 2, "min_date": "2018-01-02", "max_date": "2018-01-02",
                      "content_sha256": partition_content_sha256(frame)}
            engine = object.__new__(Engine)
            frame.to_parquet(path, index=False)
            self.assertTrue(engine._partition_matches_record(path, record))
            frame.loc[0, "value"] = 2.0
            frame.to_parquet(path, index=False)
            self.assertFalse(engine._partition_matches_record(path, record))

    def test_compression_and_row_order_do_not_change_content_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "data.parquet"
            frame = self.frame()
            frame.iloc[::-1].to_parquet(path, compression="gzip", index=False)
            self.assertEqual(partition_content_sha256(frame),
                             partition_content_sha256(pd.read_parquet(path)))
            engine = object.__new__(Engine)
            self.assertFalse(engine._partition_matches_record(path, {
                "rows": 2, "min_date": "2018-01-02", "max_date": "2018-01-02"}))

    def test_180_gib_container_does_not_expand_90_gib_project_budget(self):
        with patch.dict(os.environ, {}, clear=True):
            limits = memory_limits(180 * 2**30)
            self.assertEqual(limits["project"], 90 * 2**30)
            self.assertEqual(limits["charged"], 81 * 2**30)
            self.assertEqual(memory_limits(60 * 2**30)["effective"], 60 * 2**30)
            self.assertEqual(memory_limits(None, {"memory_budget_gib": 40})["charged"], 36 * 2**30)
            with self.assertRaises(ValueError):
                resource_policy({"memory_budget_gib": 180})

    def test_automatic_parallelism_respects_cpu_quota(self):
        with patch.dict(os.environ, {}, clear=True), patch("fea.resources.available_cpus", return_value=4):
            self.assertEqual(safe_jobs(0, {"default_jobs": 8, "max_jobs": 12}), 4)
            self.assertEqual(safe_jobs(0, {"default_jobs": 2, "max_jobs": 12}), 2)
            self.assertEqual(safe_jobs(40, {"default_jobs": 8, "max_jobs": 12}), 40)

    def test_process_memory_limit_is_optional_and_enabled_by_default(self):
        import resource
        with patch('fea.resources._POLICY', {}), patch.dict(os.environ, {}, clear=True), patch(
                'resource.getrlimit', return_value=(resource.RLIM_INFINITY, resource.RLIM_INFINITY)) as get_limit, patch(
                'resource.setrlimit') as set_limit:
            policy = {'default_jobs': 12, 'max_jobs': 20, 'memory_budget_gib': 90,
                      'process_address_space_gib': 24}
            self.assertIsNone(configure(policy, enforce_limits=False))
            get_limit.assert_not_called()
            set_limit.assert_not_called()
            self.assertEqual(configure(policy), 24*2**30)
            set_limit.assert_called_once_with(resource.RLIMIT_AS, (24*2**30, resource.RLIM_INFINITY))

    def test_cli_exempts_only_explicit_positive_jobs(self):
        from types import SimpleNamespace
        import main
        cfg = SimpleNamespace(raw={'resources': {}}, logs_dir=Path('/unused'))
        for argv, enforce in ((['--jobs', '40'], False), ([], True), (['--jobs', '0'], True)):
            with self.subTest(argv=argv), patch('fea.config.load', return_value=cfg), patch(
                    'fea.config.setup_logging'), patch('fea.resources.configure') as setup, patch(
                    'fea.progress.announce'), patch('fea.production.run', return_value=0):
                self.assertEqual(main.main(argv), 0)
                setup.assert_called_once_with({}, enforce_limits=enforce)

    def test_common_kernel_v1_preserves_legacy_recipe_and_v2_invalidates_it(self):
        from types import SimpleNamespace
        from fea.spec import FactorSpec
        engine=object.__new__(Engine)
        engine.cfg=SimpleNamespace(default_start="2018-01-01",raw={})
        engine.universe_fp="test"
        spec=FactorSpec("stock","test",formula="v")
        legacy="stock@v1|v|500|2018-01-01|deps=||test||sparse-rank=v2"
        self.assertEqual(engine._recipe(spec),legacy)
        engine.cfg.raw={"compute":{"kernel_version":1}}
        self.assertEqual(engine._recipe(spec),legacy)
        engine.cfg.raw["compute"]["kernel_version"]=2
        self.assertEqual(engine._recipe(spec),legacy+"||common-kernels=v2")
        engine.cfg.raw["compute"]["kernel_version"]=3
        self.assertEqual(engine._recipe(spec),legacy+"||common-kernels=v3")

    def test_common_kernel_version_requires_positive_integer(self):
        from types import SimpleNamespace
        from fea.spec import FactorSpec
        engine=object.__new__(Engine)
        engine.universe_fp="test"
        for invalid in (0,-1,1.5,"2",True,None):
            engine.cfg=SimpleNamespace(default_start="2018-01-01",raw={"compute":{"kernel_version":invalid}})
            with self.subTest(invalid=invalid),self.assertRaisesRegex(ValueError,"kernel_version"):
                engine._recipe(FactorSpec("stock","test"))

    def test_delay_contract_never_reads_other_projects(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "project"
            (root / "conf").mkdir(parents=True)
            (root / "conf/frequency.yaml").write_text(
                "datasets:\n  stock_margin_detail:\n    delay_days: 1\n  stock_daily:\n    delay_days: 0\n")
            upstream = Path(temporary) / "datadownload/data"
            upstream.mkdir(parents=True)
            (upstream.parent / "conf").mkdir()
            (upstream.parent / "conf/frequency.yaml").write_text(
                "datasets:\n  stock_margin_detail:\n    delay_days: 9\n")
            self.assertEqual(load(upstream, refresh=True, project_root=root), {"stock_margin_detail": 1})
            self.assertEqual(audit_contract(upstream, project_root=root)["unexplained"], [])
            with self.assertRaises(ValueError):
                load(project_root=root, contract_path="../datadownload/conf/frequency.yaml")

    def test_local_repeated_delay_observations_are_reported(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "conf").mkdir()
            (root / "state").mkdir()
            (root / "conf/frequency.yaml").write_text("datasets:\n  source:\n    delay_days: 0\n")
            (root / "state/delay_history.json").write_text(json.dumps({
                str(i): {"tables": {"source": {"delay_obs": 1}}} for i in range(3)}))
            self.assertEqual(load(project_root=root, refresh=True), {})
            self.assertEqual(len(audit_contract(project_root=root)["unexplained"]), 1)

    def test_upstream_dataset_path_cannot_escape_raw_root(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            up = Upstream(root / "raw", root / "project")
            with self.assertRaises(ValueError):
                up.read("../secrets")

    def test_cache_override_cannot_escape_execution_namespace(self):
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as temporary:
            cfg = SimpleNamespace(root=Path(temporary) / "project")
            with patch.dict(os.environ, {"FEA_CACHE_DIR": str(Path(temporary) / "outside")}):
                with self.assertRaises(ValueError):
                    DerivedCache(None, cfg, np.array(["000001.SZ"]))

    @unittest.skipUnless(os.name == "posix", "Linux process metadata contract")
    def test_only_open_writable_raw_descriptors_identify_a_writer(self):
        from fea.processes import upstream_writer_pids
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = root / "raw"
            raw.mkdir()
            target = raw / "annual.parquet"
            target.write_bytes(b"fixture")
            proc = root / "proc"
            for pid, flags, location in ((98761, "0100000", target),
                                         (98762, "0100002", target),
                                         (98763, "0100002", root / "other.py")):
                (proc / str(pid) / "fd").mkdir(parents=True)
                (proc / str(pid) / "fdinfo").mkdir()
                (proc / str(pid) / "fd/3").symlink_to(location)
                (proc / str(pid) / "fdinfo/3").write_text(f"flags:\t{flags}\n")
            self.assertEqual(upstream_writer_pids(raw, proc_root=proc), [98762])


if __name__ == "__main__":
    unittest.main()
