import io,tempfile,unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import pandas as pd
import main

class CheckCliTests(unittest.TestCase):
    def check(self,rows,frame=None,future=False,missing_partition=False):
        with tempfile.TemporaryDirectory() as d:
            cfg=SimpleNamespace(board_prefixes=['000'],state_dir=Path(d),factors_dir=Path(d))
            spec=SimpleNamespace(name='sample',deps=('stock_daily',),allow_qfq=False,
                                 resolved_start=lambda cfg:'2027-01-01' if future else '2018-01-01')
            man=SimpleNamespace(partition_rows=lambda:rows,partitions={'2026':{}} if missing_partition else {})
            engine=SimpleNamespace(baseline_last_day=lambda:20260918)
            text=io.StringIO()
            with patch('fea.engine.Engine',return_value=engine),patch.object(main,'all_specs',return_value=[spec]),patch.object(main.Manifest,'load',return_value=man),patch.object(main.store,'read_factor',return_value=frame),patch.object(main,'_check_format',return_value=0),redirect_stdout(text):
                rc=main.cmd_check(SimpleNamespace(),cfg)
            return rc,text.getvalue()

    def test_missing_outputs_fail(self):
        rc,text=self.check(0)
        self.assertEqual(rc,1);self.assertIn('尚未生成',text)

    def test_not_yet_available_is_not_missing(self):
        rc,text=self.check(0,future=True)
        self.assertEqual(rc,0);self.assertIn('尚未到有效起点',text)

    def test_manifest_cannot_hide_missing_file(self):
        rc,text=self.check(1,pd.DataFrame(),missing_partition=True)
        self.assertEqual(rc,1);self.assertIn('分区文件缺失',text)

    def test_basic_check_does_not_claim_causal_proof(self):
        frame=pd.DataFrame({'trade_date':['2026-09-18'],'stock_code':['000001.SZ'],'value':[1.0],'rank':[0.5]})
        rc,text=self.check(1,frame)
        self.assertEqual(rc,0);self.assertIn('PIT 未执行',text)
        self.assertIn('无前复权输入',text);self.assertNotIn('无价格依赖',text)
