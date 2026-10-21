import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import pandas as pd

from fea.dates import Calendar
from fea.engine import Engine, partition_content_sha256
from fea.manifest import Manifest
from fea.spec import FactorSpec


class OutputRecoveryTests(unittest.TestCase):
    def fixture(self, temporary):
        root=Path(temporary)
        cfg=SimpleNamespace(root=root,factors_dir=root/"data/factors",default_start="2018-01-01",revision_days=1,
                            factor_root=lambda spec:root/"data/factors")
        spec=FactorSpec("recovery_test","test")
        engine=object.__new__(Engine)
        engine.cfg=cfg;engine._start_floor=20180101;engine.override_start=None;engine.universe_fp="test"
        engine.cal=Calendar(np.array([20180102,20180103],dtype=np.int32))
        engine.codes=np.array(["000001.SZ","600000.SH"])
        engine._run_end=20180103
        # ★ 2026-10-03：`fea.daily.run` 现在还会调闸门与派生层预建（并行执行器的
        #   driver 职责）。本 fixture 是 Engine 的最小替身，只关心恢复路径，
        #   把这两类副作用置成空操作。
        engine.state_writes=True
        engine.st_gate=lambda:{}
        engine.delay_gate=lambda:{}
        engine.build_derived_layers=lambda todo,jobs=1:[]
        frame=pd.DataFrame({"trade_date":pd.Series(["2018-01-02","2018-01-02","2018-01-03","2018-01-03"],dtype="string"),
                            "stock_code":pd.Series(["000001.SZ","600000.SH"]*2,dtype="string"),
                            "value":np.array([1,2,3,4],np.float32),"rank":np.array([.5,1,.5,1],np.float32)})
        path=cfg.factors_dir/spec.name/"year=2018/data.parquet"
        path.parent.mkdir(parents=True)
        frame.to_parquet(path,index=False)
        stat=path.stat()
        record={"rows":4,"min_date":"2018-01-02","max_date":"2018-01-03",
                "file_identity":[stat.st_size,stat.st_mtime_ns],"content_sha256":partition_content_sha256(frame)}
        manifest=Manifest(spec.name,root/"state.json",recipe=engine._recipe(spec),
                          coverage=[["2018-01-02","2018-01-03"]],partitions={"2018":record})
        return engine,spec,frame,path,manifest

    def test_unreadable_output_is_atomically_replaced_only_with_complete_year_axis(self):
        with tempfile.TemporaryDirectory() as temporary:
            engine,spec,frame,path,manifest=self.fixture(temporary)
            path.write_bytes(b"unreadable-original")
            partial=frame.iloc[2:]
            with self.assertRaisesRegex(RuntimeError,"完整年轴"):
                engine._upsert_output(spec,2018,partial,"zstd",replace_days=np.array([20180103]))
            self.assertEqual(path.read_bytes(),b"unreadable-original")
            repaired=engine._upsert_output(spec,2018,frame,"zstd",replace_days=engine.cal.days)
            pd.testing.assert_frame_equal(repaired,frame)
            pd.testing.assert_frame_equal(pd.read_parquet(path),frame)

    @unittest.skipUnless(os.name=="posix","Linux daily lock module")
    def test_same_footer_value_repair_has_explicit_change_reason_and_new_digest(self):
        from fea.daily import snapshot_hashes,output_repair_reasons
        with tempfile.TemporaryDirectory() as temporary:
            engine,spec,frame,path,manifest=self.fixture(temporary)
            changed=frame.copy();changed.loc[0,"value"]=99
            changed.to_parquet(path,index=False)
            stat=path.stat();os.utime(path,ns=(stat.st_atime_ns,stat.st_mtime_ns+1000000))
            plan=engine.plan(spec,manifest,20180103,False)
            self.assertEqual(plan[2018].tolist(),[20180102,20180103])
            reasons=output_repair_reasons(engine.output_repairs,spec.name)
            self.assertEqual(reasons,["output_repair:2018:content_changed_or_unreadable"])
            before=snapshot_hashes(engine.cfg,[spec],["2018-01-02","2018-01-03"])
            engine._upsert_output(spec,2018,frame,"zstd",replace_days=plan[2018])
            after=snapshot_hashes(engine.cfg,[spec],["2018-01-02","2018-01-03"])
            self.assertNotEqual(before["recovery_test|2018-01-02"],after["recovery_test|2018-01-02"])
            self.assertEqual(before["recovery_test|2018-01-03"],after["recovery_test|2018-01-03"])
            self.assertTrue(reasons)

    @unittest.skipUnless(os.name=="posix","Linux daily lock module")
    def test_before_hash_skips_unreadable_output_only_when_repair_was_planned(self):
        from fea.daily import snapshot_hashes
        with tempfile.TemporaryDirectory() as temporary:
            engine,spec,frame,path,manifest=self.fixture(temporary)
            path.write_bytes(b"broken")
            engine.plan(spec,manifest,20180103,False)
            diagnostics=[]
            self.assertEqual(snapshot_hashes(engine.cfg,[spec],["2018-01-02"],
                                           planned_repairs=engine.output_repairs,errors=diagnostics),{})
            self.assertEqual(diagnostics[0]["reason"],"planned_output_repair_unreadable_baseline")
            with self.assertRaises(Exception):
                snapshot_hashes(engine.cfg,[spec],["2018-01-02"])
            engine._upsert_output(spec,2018,frame,"zstd",replace_days=engine.cal.days)
            self.assertIn("recovery_test|2018-01-02",snapshot_hashes(engine.cfg,[spec],["2018-01-02"]))

    @unittest.skipUnless(os.name=="posix","Real Linux production lock contract")
    def test_daily_recovery_keeps_production_locked_and_changes_explicitly_explained(self):
        import fcntl
        import json
        from unittest.mock import patch
        from fea.config import Config
        from fea.daily import run
        from fea.release import read_release
        from fea.store import LAST_WRITE
        for corrupt_kind in ("changed_value","unreadable"):
            with self.subTest(corrupt_kind=corrupt_kind),tempfile.TemporaryDirectory() as temporary:
                engine,spec,frame,path,manifest=self.fixture(temporary)
                root=Path(temporary)
                cfg=Config(root=root,raw={"default_start":"2018-01-01",
                    "paths":{"upstream":"raw","factors":"data/factors","state":"state"},
                    "universe":{},"daily":{"monitor":False,"catalog":False,"publish":False},
                    "compute":{"revision_days":1,"kernel_version":1}})
                engine.cfg=cfg
                engine.frozen_input_changes=[]
                engine.baseline_last_day=lambda:20180103
                engine.watermark=lambda name:{"exists":False}
                manifest.path=cfg.state_dir/(spec.name+".json")
                manifest.save()
                if corrupt_kind=="unreadable":
                    path.write_bytes(b"invalid parquet")
                else:
                    changed=frame.copy();changed.loc[0,"value"]=99
                    changed.to_parquet(path,index=False)
                    stat=path.stat();os.utime(path,ns=(stat.st_atime_ns,stat.st_mtime_ns+1000000))
                args=SimpleNamespace(factors=[spec.name],group=None,end="2018-01-03",start=None,
                                     refresh=False,rebuild=False,jobs=1,sandbox=None,slice_worker=False)
                def controlled_worker(command,**kwargs):
                    # ★ 2026-10-03：这里模拟的是**coordinator**（它跑在 run_bounded 里）：
                    #   写数据分区 → 写结果文件/progress → 由单写者提交台账（含 coverage）。
                    self.assertTrue((cfg.state_dir/"daily_inflight.json").exists())
                    self.assertTrue((cfg.state_dir/"publication_dirty.json").exists())
                    with (cfg.state_dir/"daily.lock").open("a+") as reader:
                        with self.assertRaises(BlockingIOError):
                            fcntl.flock(reader,fcntl.LOCK_SH|fcntl.LOCK_NB)
                    merged=engine._upsert_output(spec,2018,frame,"zstd",replace_days=engine.cal.days)
                    state=Manifest.load(cfg.state_dir,spec.name)
                    stat=path.stat()
                    state.partitions["2018"].update(file_identity=[stat.st_size,stat.st_mtime_ns],
                        content_sha256=partition_content_sha256(merged),last_write=dict(LAST_WRITE))
                    state.add_coverage("2018-01-02","2018-01-03")
                    state.save()
                    run_dir=next((cfg.state_dir/"runs").glob("*/"))
                    (run_dir/"results.2018.json").write_text(json.dumps({"version":1,"year":2018,"factors":{
                        spec.name:{"results":[{"factor":spec.name,"year":2018,"rows":4,"nonnull":4,
                            "partition_rows":4,"min_date":"2018-01-02","max_date":"2018-01-03",
                            "content_sha256":partition_content_sha256(merged),
                            "write":dict(LAST_WRITE),"seconds":0.01}],"error":None}}}))
                    (run_dir/"progress.json").write_text(json.dumps({"completed":[{"year":2018,"returncode":0,
                        "seconds":0.02,"cpu_seconds":0.0,"tasks":1,"task_seconds":0.01,"finalized":1,
                        "failed":[]}],"streams":1,"jobs_per_stream":1,
                        "first_dispatch":0.0,"last_completion":0.02}))
                    return {"returncode":0,"peak_rss_bytes":0,"peak_cgroup_bytes":0,
                            "peak_shared_anon_bytes":0,"peak_pss_bytes":0,"cgroup_limit_bytes":None,
                            "memory_guard_bytes":0,"cpu_seconds":0.0,"memory_stopped":False,"stop_reason":""}
                with patch("fea.daily.Engine",return_value=engine),patch("fea.daily.REGISTRY",{spec.name:spec}),\
                     patch("fea.daily.upstream_writer_pids",return_value=[]),\
                     patch("fea.resources.run_bounded",side_effect=controlled_worker):
                    self.assertEqual(run(args,cfg,lambda names,groups:[spec]),0)
                report_path=next((cfg.state_dir/"runs").glob("*/report.json"))
                report=json.loads(report_path.read_text())
                self.assertEqual(report["unexplained_changes"],[])
                self.assertEqual(len(report["output_repairs"]),1)
                if corrupt_kind=="changed_value":
                    self.assertTrue(report["changed"])
                    self.assertTrue(any("output_repair:" in reason for reasons in report["change_reasons"].values() for reason in reasons))
                else:
                    self.assertEqual(len(report["unreadable_before_baselines"]),1)
                self.assertFalse((cfg.state_dir/"daily_inflight.json").exists())
                with self.assertRaisesRegex(RuntimeError,"Production changed"):
                    with read_release(cfg):
                        pass


if __name__=="__main__":unittest.main()
