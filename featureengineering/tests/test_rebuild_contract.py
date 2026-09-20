import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
from fea.spec import FactorSpec
from fea.engine import Engine
from fea.manifest import Manifest
from fea.dates import Calendar
from fea.chips import ChipLayer
from fea.intraday import IntradayLayer
from fea.resources import safe_jobs


class RebuildContractTests(unittest.TestCase):
    def test_missing_or_modified_output_rebuilds_old_year(self):
        with tempfile.TemporaryDirectory() as td:
            spec=FactorSpec('test','test')
            e=object.__new__(Engine)
            e.cfg=SimpleNamespace(default_start='2018-01-01',revision_days=1,factors_dir=Path(td))
            e.universe_fp='test';e._start_floor=20180101;e.override_start=None
            e.cal=Calendar(np.array([20180102,20190102,20200102],dtype=np.int32))
            man=Manifest('test',Path('/unused'),recipe=e._recipe(spec),coverage=[['2018-01-02','2020-01-02']],partitions={'2018':{'file_identity':[100,1]}})
            self.assertEqual(sorted(e.plan(spec,man,20200102,False)),[2018,2020])
            p=Path(td)/'test/year=2018/data.parquet';p.parent.mkdir(parents=True);p.write_bytes(b'changed')
            self.assertEqual(sorted(e.plan(spec,man,20200102,False)),[2018,2020])
            st=p.stat();man.partitions['2018']['file_identity']=[st.st_size,st.st_mtime_ns]
            self.assertEqual(sorted(e.plan(spec,man,20200102,False)),[2020])


    def test_snapshot_identity_uses_content_and_accepts_legacy(self):
        from fea.engine import _file_changed
        self.assertFalse(_file_changed('snapshot',[10,1,'a'],[10,2,'a']))
        self.assertTrue(_file_changed('snapshot',[10,1,'a'],[10,1,'b']))
        self.assertFalse(_file_changed('snapshot',[10,1],[10,1,'a']))
        self.assertTrue(_file_changed('snapshot',[10,1],[10,2,'a']))
        self.assertTrue(_file_changed('year=2020',[10,1],[10,2]))

    def test_forced_resume_does_not_trust_old_coverage(self):
        from scripts.backfill_history import _invalidate_coverage
        with tempfile.TemporaryDirectory() as td:
            cfg=SimpleNamespace(state_dir=Path(td))
            spec=FactorSpec('test','test')
            man=Manifest.load(cfg.state_dir,'test')
            man.coverage=[['2018-01-02','2026-09-18']];man.save()
            _invalidate_coverage(cfg,[spec],2020,2023)
            man=Manifest.load(cfg.state_dir,'test')
            self.assertEqual(man.coverage,[['2018-01-02','2019-12-31'],['2024-01-01','2026-09-18']])
            man.add_coverage('2020-01-01','2020-12-31');man.save()
            self.assertEqual(man.missing_ranges('2020-01-01','2023-12-31'),[('2021-01-01','2023-12-31')])
            self.assertEqual(len(list((cfg.state_dir/'rebuild_attempts').glob('*.json'))),1)


    def test_global_floor_applies_to_explicit_start_and_recipe(self):
        s = FactorSpec('test', 'test', start='2011-01-01')
        a = SimpleNamespace(default_start='2012-01-01')
        b = SimpleNamespace(default_start='2018-01-01')
        self.assertEqual(s.resolved_start(b), '2018-01-01')
        self.assertNotEqual(s.recipe(a), s.recipe(b))
        self.assertEqual(FactorSpec('late','test',start='2020-01-02').resolved_start(b),'2020-01-02')

    def test_refresh_preserves_other_year_coverage(self):
        s=FactorSpec('test','test')
        e=object.__new__(Engine)
        e.cfg=SimpleNamespace(default_start='2018-01-01', revision_days=10)
        e.universe_fp='test';e._start_floor=20180101;e.override_start=20200101;e.force_refresh=True
        e.cal=Calendar(np.array([20180102,20190102,20200102,20200103],dtype=np.int32))
        m=Manifest('test',Path('/unused'),recipe=e._recipe(s),coverage=[['2018-01-02','2020-01-03']])
        p=e.plan(s,m,20200103,False)
        self.assertEqual(list(p),[2020])
        self.assertEqual(p[2020].tolist(),[20200102,20200103])
        self.assertEqual(m.coverage,[['2018-01-02','2020-01-03']])

    def test_resources_reject_excessive_parallelism(self):
        self.assertEqual(safe_jobs(0),1)
        self.assertEqual(safe_jobs(2),2)
        for n in (3,8,-1):
            with self.assertRaises(ValueError):safe_jobs(n)

    def test_value_only_revision_recomputes_history(self):
        s=FactorSpec('test','test',deps=('source',))
        e=object.__new__(Engine)
        e.cfg=SimpleNamespace(default_start='2018-01-01',revision_days=1)
        e.universe_fp='test';e._start_floor=20180101;e.override_start=None
        e.cal=Calendar(np.array([20180102,20190102,20200102],dtype=np.int32))
        prev={'exists':True,'rows':3,'max_pit':20200102,'files':{'year=2018':[100,1],'year=2020':[100,1]}}
        cur={**prev,'files':{'year=2018':[100,2],'year=2020':[100,1]}}
        e.watermark=lambda _:cur
        m=Manifest('test',Path('/unused'),recipe=e._recipe(s),coverage=[['2018-01-02','2020-01-02']],input_watermark={'source':prev})
        self.assertEqual(sorted(e.plan(s,m,20200102,False)),[2018,2019,2020])

    def test_non_financial_run_skips_vintages(self):
        e=object.__new__(Engine)
        self.assertIsNone(e.deriv_for(20180101,20181231,fields=frozenset()))

    def test_coupling_inherits_parent_repair_dates(self):
        e=object.__new__(Engine);e.cfg=SimpleNamespace(default_start='2018-01-01')
        e._start_floor=20180101;e.override_start=None
        parent=FactorSpec('parent','test');child=FactorSpec('child','test',deps=('parent',))
        p={2020:np.array([20200102,20200515],dtype=np.int32)}
        c={2020:np.array([20200515],dtype=np.int32)}
        e._propagate_parent_plans([(child,None,c),(parent,None,p)],20200515)
        self.assertEqual(c[2020].tolist(),[20200102,20200515])

    def test_coupling_ignores_retired_output_years(self):
        from fea.factors_io import FactorIO
        cfg=SimpleNamespace(default_start='2018-01-01',factors_dir=Path('/unused'))
        io=FactorIO(cfg,None)
        with patch('fea.factors_io.store.read_year',return_value=pd.DataFrame()) as read:
            io._read('parent',2016,2018)
            self.assertEqual([x.args[2] for x in read.call_args_list],[2018])

    def test_prune_previews_then_removes_only_before_floor(self):
        from fea.history import run
        from fea.config import Config
        from fea import store
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            cfg=Config({'default_start':'2018-01-01','paths':{'factors':'data/factors','state':'state'},'storage':{}},root)
            for y in (2017,2018):
                df=pd.DataFrame({'trade_date':pd.Series([f'{y}-01-03'],dtype='string'),'stock_code':pd.Series(['600000.SH'],dtype='string'),'value':np.array([1],dtype=np.float32),'rank':np.array([1],dtype=np.float32)})
                store._atomic_write(df,cfg.factors_dir/f'test/year={y}/data.parquet')
            kept=(cfg.factors_dir/'test/year=2018/data.parquet').read_bytes()
            with patch('fea.history.all_specs',return_value=[FactorSpec('test','test')]):
                self.assertEqual(len(run(cfg)),1)
                self.assertTrue((cfg.factors_dir/'test/year=2017/data.parquet').exists())
                run(cfg,apply=True)
                self.assertFalse((cfg.factors_dir/'test/year=2017').exists())
                self.assertEqual((cfg.factors_dir/'test/year=2018/data.parquet').read_bytes(),kept)
                self.assertEqual(run(cfg),[])

    def test_quarter_aggregates_match_whole_year(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);cfg=SimpleNamespace(root=root,upstream=root/'raw',compression='zstd')
            codes=np.array(['600000.SH','000001.SZ'])
            days=['2020-01-02','2020-03-31','2020-04-01','2020-07-01','2020-10-09']
            chips=[];bars=[]
            for j,day in enumerate(days):
                for k,code in enumerate([*codes,'300001.SZ']):
                    for i in range(15):chips.append([day,code,10+i*.2+j+k,1+i])
                    for i in range(48):
                        minute=(9*60+35+i*5) if i<24 else 13*60+5+(i-24)*5
                        price=10+j+k+i*.01
                        bars.append([code,f'{day} {minute//60:02}:{minute%60:02}:00',price,price+.02,price-.02,price+.01,1000+i,(1000+i)*price])
            cf=pd.DataFrame(chips,columns=['trade_date','stock_code','price','percent'])
            bf=pd.DataFrame(bars,columns=['stock_code','trade_time','open','high','low','close','vol','amount'])
            for cls,ds,df in [(ChipLayer,'stock_cyq_chips',cf),(IntradayLayer,'stock_history_5min',bf)]:
                p=cfg.upstream/ds/'year=2020/data.parquet';p.parent.mkdir(parents=True);df.to_parquet(p,index=False)
                layer=cls(None,cfg,codes)
                whole=layer._aggregate_frame(df.copy(),2020).sort_values(['trade_date','stock_code']).reset_index(drop=True)
                chunked=layer.build_year(2020,None).sort_values(['trade_date','stock_code']).reset_index(drop=True)
                pd.testing.assert_frame_equal(whole,chunked,check_exact=True)

    def test_derived_detects_same_rows_revision_pool_and_missing_output(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);cfg=SimpleNamespace(root=root,upstream=root/'raw',compression='zstd')
            p=cfg.upstream/'stock_cyq_chips/year=2020/data.parquet';p.parent.mkdir(parents=True)
            pd.DataFrame({'trade_date':['2020-01-02'],'stock_code':['600000.SH'],'price':[10.],'percent':[100.]}).to_parquet(p,index=False)
            layer=ChipLayer(None,cfg,np.array(['600000.SH']))
            layer.ensure(2020,2020);self.assertTrue(layer.ready(2020,2020))
            other=ChipLayer(None,cfg,np.array(['000001.SZ']))
            self.assertFalse(other.ready(2020,2020))
            import os
            st=p.stat();os.utime(p,ns=(st.st_atime_ns,st.st_mtime_ns+1000000000))
            other=ChipLayer(None,cfg,np.array(['600000.SH']))
            self.assertFalse(other.ready(2020,2020))
            (layer.root/'year=2020/data.parquet').unlink()
            self.assertFalse(layer.ready(2020,2020))

if __name__=='__main__':unittest.main()
