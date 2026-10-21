from pathlib import Path
import tempfile
from types import SimpleNamespace
import threading
import time
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from fea.monitor import inspect_tail,inspect_footer,inspect_object,monitor_jobs,run_monitor
from fea.spec import FactorSpec
from fea.store import DTYPES,year_path
from fea.manifest import Manifest


def grid(dates=('2026-01-05','2026-01-06'),codes=('A','B','C')):
    frame=pd.DataFrame([(d,c,i+1.,(i+1)/3) for d in dates for i,c in enumerate(codes)],
                       columns=['trade_date','stock_code','value','rank'])
    return frame.astype(DTYPES)


class QualityMonitorTests(unittest.TestCase):
    def setUp(self):
        self.cfg=SimpleNamespace(default_start='2026-01-01')
        self.spec=FactorSpec('factor','technical')
        self.days=np.array([20260105,20260106])
        self.codes=np.array(['A','B','C'])

    def test_valid_tail_and_duplicate_balancing_missing_stock(self):
        frame=grid()
        self.assertEqual(inspect_tail(frame,self.days,self.codes,self.spec,self.cfg)['errors'],[])
        frame.loc[1,'stock_code']='A'
        checked=inspect_tail(frame,self.days,self.codes,self.spec,self.cfg)
        self.assertIn('duplicate_keys',checked['errors'])
        frame.loc[1,'stock_code']='Z'
        self.assertIn('out_of_universe',inspect_tail(frame,self.days,self.codes,self.spec,self.cfg)['errors'])

    def test_missing_rows_inf_and_rank_bounds_are_hard_errors(self):
        frame=grid().iloc[:-1].copy()
        frame.loc[0,'value']=np.inf
        frame.loc[1,'rank']=2
        checked=inspect_tail(frame,self.days,self.codes,self.spec,self.cfg)
        for error in ('tail_stock_axis','infinite_value','rank_bounds'):
            self.assertIn(error,checked['errors'])

    def test_alignment_missing_values_allowed_and_ranks_must_missing(self):
        spec=FactorSpec('late','event',start='2026-02-01')
        frame=grid()
        frame['value']=np.nan
        frame['rank']=np.nan
        checked=inspect_tail(frame,self.days,self.codes,spec,self.cfg)
        self.assertEqual(checked['errors'],[])
        self.assertEqual(checked['warnings'],[])
        frame.loc[0,'rank']=.5
        self.assertIn('alignment_rank_not_missing',inspect_tail(frame,self.days,self.codes,spec,self.cfg)['errors'])

    def test_sparse_or_constant_values_are_warning_not_automatic_retirement(self):
        frame=grid()
        frame['value']=0
        checked=inspect_tail(frame,self.days,self.codes,self.spec,self.cfg)
        self.assertEqual(checked['errors'],[])
        self.assertTrue(checked['warnings'])

    def test_market_scalar_is_not_a_constant_stock_cross_section(self):
        frame=pd.DataFrame({'trade_date':['2026-01-05','2026-01-06'],
                            'value':np.array([1.,2.],dtype=np.float32)})
        spec=FactorSpec('market','market',is_market=True)
        checked=inspect_tail(frame,self.days,self.codes,spec,self.cfg)
        self.assertEqual(checked['errors'],[])
        self.assertEqual(checked['warnings'],[])

    def test_footer_row_count_and_dtype(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'partition.parquet'
            frame=grid()
            frame.to_parquet(path,index=False)
            self.assertEqual(inspect_footer(path,6)['errors'],[])
            self.assertIn('annual_row_count',inspect_footer(path,7)['errors'])
            frame['value']=frame.value.astype('float64')
            frame.to_parquet(path,index=False)
            self.assertIn('schema_type:value',inspect_footer(path,6)['errors'])

    def test_monitor_immutable_runs_and_approved_scope(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            cfg=SimpleNamespace(root=root,default_start='2026-01-01',raw={'monitor':{'tail_days':2}},
                state_dir=root/'state',factor_root=lambda spec:root/'data'/'factors')
            class Calendar:
                days=np.array([20260105,20260106,20260107,20260108])
                def between(self,start,end):
                    return self.days[(self.days>=start)&(self.days<=end)]
            engine=SimpleNamespace(cal=Calendar(),codes=self.codes,baseline_last_day=lambda:20260106)
            path=year_path(cfg.factor_root(self.spec),self.spec.name,2026)
            path.parent.mkdir(parents=True)
            grid().to_parquet(path,index=False)
            man=Manifest.load(cfg.state_dir,self.spec.name)
            man.mark_partition(2026,6,6,'2026-01-05','2026-01-06')
            man.save()
            with patch('fea.release.production_specs',return_value=[self.spec]):
                first=run_monitor(cfg,engine,end='2026-01-06')
                original=Path(first['report_path']).read_bytes()
                second=run_monitor(cfg,engine)
            self.assertEqual(first['status'],'ok')
            self.assertEqual(first['objects'],1)
            self.assertNotEqual(first['report_path'],second['report_path'])
            self.assertEqual(Path(first['report_path']).read_bytes(),original)

    def test_parallel_monitor_real_partitions_matches_serial_with_corruption(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            cfg=SimpleNamespace(root=root,default_start='2018-01-01',
                raw={'monitor':{'tail_days':2,'jobs':1},'resources':{'max_jobs':12}},
                state_dir=root/'state',factor_root=lambda spec:root/'data'/('market_factors' if spec.is_market else 'factors'))
            class Calendar:
                days=np.array([20180102,20180103,20260105,20260106])
                def between(self,start,end):
                    return self.days[(self.days>=start)&(self.days<=end)]
            engine=SimpleNamespace(cal=Calendar(),codes=self.codes,
                                   baseline_last_day=lambda:20260106,_recipe=lambda spec:'known')
            # Deliberately not alphabetical: preserve the production library order.
            specs=[FactorSpec('z_stock','technical'),FactorSpec('broken','technical'),
                   FactorSpec('a_market','market',is_market=True),
                   FactorSpec('label','label',is_label=True,forward_days=2)]
            for spec in specs:
                manifest=Manifest.load(cfg.state_dir,spec.name)
                manifest.recipe='known'
                for year,dates in ((2018,('2018-01-02','2018-01-03')),
                                   (2026,('2026-01-05','2026-01-06'))):
                    path=year_path(cfg.factor_root(spec),spec.name,year)
                    path.parent.mkdir(parents=True)
                    if spec.is_market:
                        frame=pd.DataFrame({'trade_date':pd.Series(dates,dtype='string'),
                                            'value':np.array([1.,2.],dtype=np.float32)})
                    else:
                        frame=grid(dates=dates)
                        if spec.is_label:
                            frame['rank']=np.float32(np.nan)
                            if year==2026:
                                frame['value']=np.float32(np.nan)
                    frame.to_parquet(path,index=False)
                    manifest.mark_partition(year,len(frame),int(np.isfinite(frame.value).sum()),dates[0],dates[-1])
                manifest.save()
            broken=year_path(cfg.factor_root(specs[1]),specs[1].name,2018)
            broken.write_bytes(b'not a parquet file')
            files=list((root/'data').glob('**/data.parquet'))+list(cfg.state_dir.glob('*.json'))
            identities={str(path):(path.stat().st_size,path.stat().st_mtime_ns) for path in files}
            with patch('fea.release.production_specs',return_value=specs):
                serial=run_monitor(cfg,engine)
            cfg.raw['monitor']['jobs']=4
            observed=set()
            observed_lock=threading.Lock()
            def inspect_with_out_of_order_io(*args):
                with observed_lock:
                    observed.add(threading.current_thread().name)
                if args[2].name=='z_stock':
                    time.sleep(.04)
                return inspect_object(*args)
            with patch('fea.release.production_specs',return_value=specs), \
                 patch('fea.monitor.inspect_object',side_effect=inspect_with_out_of_order_io), \
                 patch('fea.resources.available_cpus',return_value=8), \
                 patch('fea.monitor.pd.read_parquet',wraps=pd.read_parquet) as reads:
                parallel=run_monitor(cfg,engine)
            ephemeral={'run_id','elapsed_seconds','report_path'}
            self.assertEqual({k:v for k,v in serial.items() if k not in ephemeral},
                             {k:v for k,v in parallel.items() if k not in ephemeral})
            self.assertEqual([r['factor'] for r in parallel['records']],[s.name for s in specs])
            self.assertEqual(parallel['status'],'failed')
            self.assertEqual(parallel['error_objects'],1)
            self.assertEqual([r['factor'] for r in parallel['records'] if r['errors']],['broken'])
            self.assertTrue(parallel['records'][1]['errors'][0].startswith('monitor_exception:'))
            self.assertGreaterEqual(len(observed),2)
            self.assertTrue(all(name.startswith('fea-monitor') for name in observed))
            self.assertTrue(reads.call_args_list)
            self.assertTrue(all(call.kwargs.get('use_threads') is False for call in reads.call_args_list))
            self.assertEqual(identities,{str(path):(path.stat().st_size,path.stat().st_mtime_ns) for path in files})

    def test_monitor_threads_respect_cpu_and_project_cap(self):
        cfg=SimpleNamespace(raw={'resources':{'max_jobs':12}})
        with patch('fea.resources.available_cpus',return_value=8):
            self.assertEqual(monitor_jobs(cfg),4)
            cfg.raw['monitor']={'jobs':1}
            self.assertEqual(monitor_jobs(cfg),1)
            cfg.raw['monitor']['jobs']=20
            self.assertEqual(monitor_jobs(cfg),4)
        with patch('fea.resources.available_cpus',return_value=2):
            self.assertEqual(monitor_jobs(cfg),2)
        cfg.raw['resources']['max_jobs']=1
        with patch('fea.resources.available_cpus',return_value=8):
            self.assertEqual(monitor_jobs(cfg),1)
        cfg.raw['monitor']['jobs']=0
        with self.assertRaises(ValueError):
            monitor_jobs(cfg)


if __name__=='__main__':unittest.main()
