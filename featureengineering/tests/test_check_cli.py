import io,tempfile,unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import pandas as pd
from fea.config import Config
from fea.spec import FactorSpec
from fea.dates import Calendar
from fea import validation
from fea.manifest import Manifest
from fea.store import _atomic_write

class CheckCliTests(unittest.TestCase):
    def check(self,rows,future=False,missing_partition=False,aligned=False):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/"project";root.mkdir()
            cfg=Config({"paths":{"state":"state","factors":"factors"},"default_start":"2018-01-01"},root)
            # ★★ 2026-09-25：默认 `align_fill=nan` ⇒ 所有因子都要求从 default_start
            #   起有分区（时间轴对齐）。`future=True` 的用例考的是"起点在未来、
            #   尚未有真实值"，所以显式用 `align_fill=None` 关掉对齐（逃生舱），
            #   否则"尚未可用"会与"对齐分区缺失"混在一起。对齐本身另有用例覆盖。
            spec=FactorSpec("sample","test",start="2027-01-01" if future else None,
                            align_fill=None if future else float("nan"))
            man=Manifest.load(cfg.state_dir,"sample")
            man.recipe="test"
            if rows:
                df=pd.DataFrame({"trade_date":pd.Series(["2026-09-18"],dtype="string"),
                    "stock_code":pd.Series(["000001.SZ"],dtype="string"),
                    "value":pd.Series([1.],dtype="float32"),"rank":pd.Series([.5],dtype="float32")})
                if not missing_partition:_atomic_write(df,cfg.factors_dir/"sample/year=2026/data.parquet")
                man.mark_partition(2026,1,1,"2026-09-18","2026-09-18")
            man.save()
            eng=SimpleNamespace(baseline_last_day=lambda:20260918,
                 cal=Calendar(np.array([20260918])),codes=np.array(["000001.SZ"]),_recipe=lambda s:"test")
            text=io.StringIO()
            with patch.object(validation,"Engine",return_value=eng),patch.object(validation,"all_specs",return_value=[spec]),redirect_stdout(text):
                rc=validation.check(cfg)
            return rc,text.getvalue()
    def test_missing_outputs_fail(self):
        rc,text=self.check(0);self.assertEqual(rc,1);self.assertIn("日期轴不完整",text)
    def test_not_yet_available_is_not_missing(self):
        rc,text=self.check(0,future=True);self.assertEqual(rc,0)
    def test_aligned_factor_must_cover_from_default_start(self):
        """★ 2026-09-25 新契约：默认 `align_fill=nan` ⇒ 分区必须从 default_start 起。

        起点在未来的因子也一样 —— 它只在**真实值区间**尚无数据，但**对齐区间
        （2018-01-01 ~ 起点）必须有填充分区**。上面那个用例用 `align_fill=None`
        关掉对齐才得到 rc=0；这里是不关对齐的对照组，结论相反即为正确行为。
        """
        rc,text=self.check(0,aligned=True);self.assertEqual(rc,1)
        self.assertIn("日期轴不完整",text)
    def test_manifest_cannot_hide_missing_file(self):
        rc,text=self.check(1,missing_partition=True);self.assertEqual(rc,1);self.assertIn("状态分区",text)
    def test_basic_check_does_not_claim_causal_proof(self):
        rc,text=self.check(1);self.assertEqual(rc,0);self.assertIn("PIT 未执行",text)
