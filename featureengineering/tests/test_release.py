import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from fea.config import Config
from fea.manifest import Manifest
from fea.release import bootstrap_library, production_specs, publish, read_release, definition, accept_common, mark_dirty, approve_candidate, registry_path
from fea.spec import FactorSpec


def sample_fn(ctx):
    return np.ones((2,2))


class ReleaseTests(unittest.TestCase):
    def fixture(self, root):
        return Config(root=Path(root),raw={"paths":{"upstream":"raw","factors":"data/factors","state":"state"},
                        "default_start":"2018-01-01","universe":{}})

    def test_new_registration_does_not_enter_approved_library(self):
        with tempfile.TemporaryDirectory() as root:
            cfg=self.fixture(root)
            s=FactorSpec(name="old",group="test",fn=sample_fn)
            new=FactorSpec(name="new",group="test",fn=sample_fn)
            with patch("fea.release.REGISTRY",{"old":s}):
                bootstrap_library(cfg)
            with patch("fea.release.REGISTRY",{"old":s,"new":new}):
                self.assertEqual([x.name for x in production_specs(cfg)],["old"])
                s.version=2
                with self.assertRaisesRegex(ValueError,"definition changed"):
                    production_specs(cfg)

    def test_complete_batch_and_inflight_refusal(self):
        with tempfile.TemporaryDirectory() as root:
            cfg=self.fixture(root)
            s=FactorSpec(name="x",group="test",fn=sample_fn)
            cfg.state_dir.mkdir(parents=True)
            p=cfg.factors_dir/"x/year=2018/data.parquet"
            p.parent.mkdir(parents=True)
            pd.DataFrame({"trade_date":["2018-01-02","2018-01-02"],"stock_code":["000001.SZ","600000.SH"],
                          "value":np.array([1,2],np.float32),"rank":np.array([0.5,1],np.float32)}).to_parquet(p,index=False)
            m=Manifest.load(cfg.state_dir,"x")
            m.mark_partition(2018,2,2,"2018-01-02","2018-01-02")
            m.add_coverage("2018-01-02","2018-01-02");m.save()
            r=publish(cfg,[s],"2018-01-02","synthetic",{}, {})
            with read_release(cfg) as read:
                self.assertEqual(read["release_id"],r["release_id"])
                self.assertEqual(len(read["objects"]),1)
                self.assertEqual(read["objects"][0]["partitions"][0]["rows"],2)
            (cfg.state_dir/"daily_inflight.json").write_text("{}")
            with self.assertRaisesRegex(RuntimeError,"unfinished"):
                with read_release(cfg):
                    pass
            with self.assertRaises(ValueError):
                publish(cfg,[s],"2018-01-02","bad",{"upstream_changed_during_run":True},{})

    def test_paths_and_sandbox_isolate_all_mutable_paths(self):
        with tempfile.TemporaryDirectory() as root:
            cfg=self.fixture(root)
            sandbox=cfg.sandbox(Path(root)/"artifacts/sandbox")
            self.assertEqual(sandbox.root,Path(root)/"artifacts/sandbox")
            self.assertEqual(sandbox.upstream,cfg.upstream)
            self.assertTrue(sandbox.state_dir.is_relative_to(sandbox.root))
            self.assertTrue(sandbox.factors_dir.is_relative_to(sandbox.root))
            self.assertTrue(sandbox.logs_dir.is_relative_to(sandbox.root))
            with self.assertRaises(ValueError):
                cfg.sandbox(Path(root).parent/"outside")
            cfg.raw["paths"]["state"]="../outside"
            with self.assertRaises(ValueError):
                _=cfg.state_dir

    def test_shared_kernel_change_requires_review_and_version(self):
        with tempfile.TemporaryDirectory() as root:
            cfg=self.fixture(root)
            source=Path(root)/"fea/mathx.py"
            source.parent.mkdir()
            source.write_text("def f(x): return x + 1\n")
            s=FactorSpec(name="old",group="test",fn=sample_fn)
            with patch("fea.release.REGISTRY",{"old":s}):
                bootstrap_library(cfg)
                source.write_text("def f(x): return x + 2\n")
                with self.assertRaisesRegex(ValueError,"Shared numerical kernel"):
                    production_specs(cfg)
                with self.assertRaisesRegex(ValueError,"Increment"):
                    accept_common(cfg,"reviewed")
                cfg.raw["compute"]={"kernel_version":2}
                accept_common(cfg,"correct division rule and rebuild all objects")
                self.assertEqual([x.name for x in production_specs(cfg)],["old"])
                self.assertTrue((cfg.state_dir/"publication_dirty.json").exists())
                with self.assertRaisesRegex(ValueError,"Sandbox"):
                    bootstrap_library(cfg.sandbox(Path(root)/"artifacts/trial"))

    def test_published_files_and_dirty_state_are_checked(self):
        with tempfile.TemporaryDirectory() as root:
            cfg=self.fixture(root)
            s=FactorSpec(name="x",group="test",fn=sample_fn)
            cfg.state_dir.mkdir(parents=True)
            p=cfg.factors_dir/"x/year=2018/data.parquet"
            p.parent.mkdir(parents=True)
            p.write_bytes(b"existing annual bytes")
            m=Manifest.load(cfg.state_dir,"x")
            m.mark_partition(2018,2,2,"2018-01-02","2018-01-02");m.save()
            publish(cfg,[s],"2018-01-02","synthetic",{}, {})
            with read_release(cfg,verify_hashes=True):
                pass
            mark_dirty(cfg,"selected maintenance")
            with self.assertRaisesRegex(RuntimeError,"changed since publication"):
                with read_release(cfg):pass
            (cfg.state_dir/"publication_dirty.json").unlink()
            p.write_bytes(b"changed annual bytes")
            with self.assertRaisesRegex(RuntimeError,"Published partition changed"):
                with read_release(cfg):pass

    def test_private_parent_is_not_implicitly_promoted_with_child(self):
        with tempfile.TemporaryDirectory() as root:
            cfg=self.fixture(root)
            child=FactorSpec(name="child_id",group="test",fn=sample_fn,deps=("private_parent_id",))
            with patch("fea.release.REGISTRY",{}):
                bootstrap_library(cfg)
                with patch("research.catalog.Catalog") as catalog, \
                     patch("research.catalog.spec_for_candidate",return_value=child), \
                     patch("research.lifecycle._check_evidence",return_value={"evaluation_id":"e","verification_id":"v"}):
                    catalog.return_value.state.return_value="shadow"
                    catalog.return_value.records.return_value=[{"candidate_id":"private_parent_id"},{"candidate_id":"child_id"}]
                    with self.assertRaisesRegex(ValueError,"independently approved"):
                        approve_candidate(cfg,"child_id",child,{"evaluation_id":"e","verification_id":"v","review_reason":"reviewed child"})
            self.assertEqual(json.loads(registry_path(cfg).read_text())["objects"],[])


if __name__=="__main__":unittest.main()
