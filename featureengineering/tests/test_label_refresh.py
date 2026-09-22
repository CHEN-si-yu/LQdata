import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
from fea.engine import Engine
from fea.spec import FactorSpec
from fea.manifest import Manifest
from fea.dates import Calendar

class LabelRefreshTests(unittest.TestCase):
    def engine(self):
        e=object.__new__(Engine)
        # 回填窗口给足（3000 天）：本文件的用例关心的是**回溯逻辑本身**
        # （前向依赖回退、标签成熟），窗口收口会掩盖这些行为；
        # 收口本身由 test_rebuild_contract 的专用用例固定。
        e.cfg=SimpleNamespace(default_start="2018-01-01",revision_days=10,
                              dep_backfill_days=lambda ds:3000)
        e.cal=Calendar(np.asarray([int(d.strftime("%Y%m%d")) for d in pd.bdate_range("2018-01-01","2026-09-18")],dtype=np.int32))
        e._start_floor=20180101;e.override_start=None;e.universe_fp='test'
        return e

    def test_long_label_refresh_reaches_newly_matured_day(self):
        e=self.engine();s=FactorSpec('label','test',is_label=True,forward_days=21)
        m=Manifest('label',Path('/unused'),recipe=e._recipe(s),coverage=[['2018-01-01','2026-09-18']])
        plan=e.plan(s,m,20260918,False)
        self.assertIn(int(e.cal.days[-22]),plan[2026])

    def test_january_input_revision_reaches_previous_year_labels(self):
        e=self.engine();s=FactorSpec('label','test',is_label=True,forward_days=21,deps=('source',))
        old={'exists':True,'rows':10,'max_pit':20260918,'files':{'year=2020':[10,1]}}
        new={**old,'files':{'year=2020':[10,2]}}
        e.watermark=lambda _:new
        m=Manifest('label',Path('/unused'),recipe=e._recipe(s),coverage=[['2018-01-01','2026-09-18']],input_watermark={'source':old})
        self.assertIn(2019,e.plan(s,m,20260918,False))

    def test_factor_without_forward_dependency_is_not_shifted(self):
        e=self.engine();s=FactorSpec('factor','test')
        self.assertEqual(e._rewind_forward_dependency(s,20200101,20180101),20200101)
