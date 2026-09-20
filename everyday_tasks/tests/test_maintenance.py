"""Offline regression tests. All filesystem writes are confined to temporary directories."""
import copy,json,os,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import pandas as pd
from data_incremental import paths,registry as R,config
from data_incremental.core import state,store
from data_incremental.core.client import Client,ApiError
from data_incremental.core import client as client_mod
from data_incremental.pipeline import strategies as S,roster,runner,calendar,probe,dump_bridge

class FakeClient:
    max_rows_per_query=100000
    def __init__(self,fn):
        self.fn=fn; self.calls=[]; self.stats={'requests':0,'retries':0,'empty_retries':0,'total_mismatch':0}
    def fetch_all(self,path,payload,*args,**kwargs):
        self.calls.append((path,dict(payload))); self.stats['requests']+=1
        return self.fn(path,payload)
    def call(self,path,payload,*args,**kwargs):return self.fetch_all(path,payload)
    def close(self):pass

class MaintenanceTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='maintenance_test_'); self.root=Path(self.tmp.name)
        self.pp=[]
        for name,value in list(vars(paths).items()):
            if isinstance(value,Path):
                try: rel=value.relative_to(Path('/autodl-fs/data'))
                except ValueError:continue
                p=patch.object(paths,name,self.root/rel);p.start();self.pp.append(p)
        paths.ensure_dirs();paths.STATE_ROOT.mkdir(parents=True,exist_ok=True)
        self.cfg=config.load();self.cfg['download'].update(redundancy_days=0,revision_days=0)
        self.cfg['api']['concurrency']=1;self.cfg['dayhash']['enabled']=False
        self.cal=pd.bdate_range('2025-01-01','2026-12-31').strftime('%Y-%m-%d').tolist()
    def tearDown(self):
        for p in reversed(self.pp):p.stop()
        self.tmp.cleanup()
    def ctx(self,cli,T='2026-09-18'):return S.Ctx(cli,self.cfg,self.cal,T)
    def test_cyq_perf_daily_restores_hole_and_adds_new_row(self):
        ds=copy.deepcopy(R.get('stock_cyq_perf'));ds.start='2026-09-17'
        man=state.Manifest.load(ds.name)
        S._save(ds,man,pd.DataFrame([{'trade_date':'2026-09-17','stock_code':'a','winner_rate':85.11}]))
        man.add_coverage('2026-09-17','2026-09-18');man.save()
        rows=[{'trade_date':'2026-09-17','stock_code':'a','winner_rate':85.11},
              {'trade_date':'2026-09-18','stock_code':'a','winner_rate':23.72},
              {'trade_date':'2026-09-18','stock_code':'new','winner_rate':80.12}]
        cli=FakeClient(lambda _,p:[r for r in rows if p['start_time']<=r['trade_date']<=p['end_time']])
        for _ in range(2):
            result=S.run_one(ds,man,self.ctx(cli))
            self.assertEqual(result.status,'✔')
            df=store.read_parquet(paths.year_partition_path(ds.name,2026))
            self.assertEqual(len(df),3)
            self.assertEqual(df.loc[df.stock_code=='new','winner_rate'].iloc[0],80.12)
        self.assertEqual(R.gate_tier(ds),'wait')

    def test_range_retries_old_failure_despite_existing_coverage(self):
        ds=R.DS('audit_range','/audit','range',keys=('trade_date','stock_code'),start='2026-08-01')
        man=state.Manifest.load(ds.name);man.add_coverage(ds.start,'2026-09-18')
        man.mark_suspect('2026-08-03~2026-08-04|error')
        cli=FakeClient(lambda _,p:[{'trade_date':p['start_time'],'stock_code':'a'}])
        S.run_one(ds,man,self.ctx(cli))
        self.assertTrue(any(p['start_time']<='2026-08-03'<=p['end_time'] for _,p in cli.calls))
        self.assertNotIn('2026-08-03~2026-08-04|error',man.suspect)
    def test_by_date_entity_retries_old_failure(self):
        ds=copy.deepcopy(R.get('tdx_daily'));man=state.Manifest.load(ds.name)
        man.mark_suspect('2026-08-03|error')
        cli=FakeClient(lambda _,p:[{'trade_date':p['trade_date'],'board_code':'a'}])
        S._run_by_date(ds,man,self.ctx(cli),['a'],'2026-09-18','2026-09-18',S.Result(ds.name,ds.mode))
        self.assertIn('2026-08-03',[p['trade_date'] for _,p in cli.calls])
        self.assertNotIn('2026-08-03|error',man.suspect)
    def test_by_date_entity_empty_response_warns(self):
        ds=copy.deepcopy(R.get('tdx_daily'));man=state.Manifest.load(ds.name)
        out=S._run_by_date(ds,man,self.ctx(FakeClient(lambda *_:[])),['a'],'2026-09-18','2026-09-18',S.Result(ds.name,ds.mode))
        self.assertEqual(out.status,'⚠️')
        self.assertIn('2026-09-18|empty',man.suspect)

    def test_first_write_deduplicates_and_sorts(self):
        p=self.root/'first.parquet'
        df=pd.DataFrame([{'k':2,'v':1},{'k':1,'v':2},{'k':2,'v':3}])
        out=store.upsert(p,df,keys=('k',))
        self.assertEqual(out.to_dict('records'),[{'k':1,'v':2},{'k':2,'v':3}])
    def test_failed_data_write_never_persists_coverage(self):
        ds=R.DS('audit_range','/audit','range',keys=('trade_date','stock_code'),start='2026-09-18')
        man=state.Manifest.load(ds.name);cli=FakeClient(lambda *_:[{'trade_date':'2026-09-18','stock_code':'a'}])
        with patch.object(store,'upsert',side_effect=OSError('injected disk full')):
            result=S.run_one(ds,man,self.ctx(cli))
        self.assertEqual(result.status,'✘');man.save()
        self.assertEqual(state.Manifest.load(ds.name).coverage,[])
    def test_partial_snapshot_never_replaces_old(self):
        ds=copy.deepcopy(R.get('tdx_blocks'));man=state.Manifest.load(ds.name)
        old=pd.DataFrame([{'block_code':str(i),'block_type':0 if i<95 else 1} for i in range(100)])
        S._save(ds,man,old)
        def fetch(_,p):
            if p['block_type']==1:raise ApiError('injected failure')
            return old.iloc[:95].to_dict('records') if p['block_type']==0 else []
        result=S.run_snapshot(ds,man,self.ctx(FakeClient(fetch)))
        self.assertNotEqual(result.status,'✔')
        self.assertEqual(len(store.read_parquet(paths.flat_path(ds.name))),100)
    def test_empty_required_snapshot_is_failure(self):
        ds=copy.deepcopy(R.get('stock_list'));man=state.Manifest.load(ds.name)
        result=S.run_snapshot(ds,man,self.ctx(FakeClient(lambda *_:[])))
        self.assertIn(result.status,('✘','⚠️'))
    def test_snapshot_missing_key_rejected(self):
        ds=copy.deepcopy(R.get('stock_list'));man=state.Manifest.load(ds.name)
        with self.assertRaises((ValueError,KeyError)):S._save(ds,man,pd.DataFrame([{'wrong_key':'x'}]))
    def test_new_entity_missing_listing_date_uses_global_start(self):
        ds=copy.deepcopy(R.get('stock_cyq_chips'));ds.start='2025-01-01';ds.entity_codes=['old','new']
        man=state.Manifest.load(ds.name);cli=FakeClient(lambda *_:[])
        with patch.object(S,'_entity_list_dates',return_value={'new':'2026-09-17'}):
            S.run_per_entity(ds,man,self.ctx(cli))
        self.assertEqual(min(p['start_time'] for _,p in cli.calls),'2025-01-01')
    def test_missing_entity_source_fails(self):
        ds=copy.deepcopy(R.get('tdx_minute'));ds.entity_codes=None
        result=S.run_per_entity(ds,state.Manifest.load(ds.name),self.ctx(FakeClient(lambda *_:[])))
        self.assertIn(result.status,('✘','⚠️'))
    def test_per_date_resumes_after_long_outage(self):
        ds=copy.deepcopy(R.get('stock_top_list'));man=state.Manifest.load(ds.name)
        man.mark_partition(2026,1,'2026-08-20','2026-08-20')
        cli=FakeClient(lambda *_:[])
        with patch.object(probe,'trailing_nonzero_ratio',return_value=0):S.run_per_date(ds,man,self.ctx(cli))
        dates=[p[ds.date_param] for _,p in cli.calls]
        self.assertIn('2026-08-21',dates)
    def test_per_date_retries_old_failed_day(self):
        ds=copy.deepcopy(R.get('stock_top_list'));man=state.Manifest.load(ds.name)
        man.mark_partition(2026,1,'2026-09-18','2026-09-18');man.mark_suspect('2026-08-21|error')
        cli=FakeClient(lambda *_:[])
        with patch.object(probe,'trailing_nonzero_ratio',return_value=0):S.run_per_date(ds,man,self.ctx(cli))
        self.assertIn('2026-08-21',[p[ds.date_param] for _,p in cli.calls])
    def test_sweep_failure_does_not_advance_success_watermark(self):
        p=self.root/'sweep.json';ds=copy.deepcopy(R.get('index_ths_daily'));ds.entity_codes=['a']
        with patch.object(roster,'SWEEP_STATE',p),patch.object(R,'get',return_value=ds):
            result=roster.sweep(self.cfg,FakeClient(lambda *_:(_ for _ in ()).throw(ApiError('injected failure'))))
            self.assertFalse(result['ok']);self.assertTrue(roster._due()[0])
    def test_snapshot_dependencies_precede_daily_consumers(self):
        order=[]
        selected=[R.get('stock_list'),R.get('stock_cyq_chips')]
        cli=FakeClient(lambda *_:[])
        class Quiet:
            def note(self,*args):pass
        def group(ds,*args):
            order.extend(d.name for d in ds);return []
        with patch.object(R,'enabled',return_value=selected),patch.object(R,'daily_tables',return_value=[selected[1]]),patch.object(calendar,'update_calendar',return_value={'ok':True}),patch.object(state,'effective_calendar',return_value=self.cal),patch.object(probe,'resolve_T',return_value=type('E',(),{'T':'2026-09-18'})()),patch.object(runner,'_run_group',side_effect=group),patch.object(runner.report_mod,'build',return_value={}),patch.object(runner.report_mod,'save'):
            runner.run(self.cfg,Quiet(),cli,skip_gate=True)
        self.assertLess(order.index('stock_list'),order.index('stock_cyq_chips'))
    def test_partial_gate_excludes_unready_minute_table(self):
        ds=copy.deepcopy(R.get('tdx_minute'));ctx=self.ctx(FakeClient(lambda *_:[]));ctx.ready=set()
        self.assertFalse(ctx.selected(ds,ignore_ready=True))
    def test_store_read_failure_preserves_file(self):
        p=self.root/'bad.parquet';p.write_bytes(b'corrupt but existing')
        with self.assertRaises(store.StoreReadError):store.upsert(p,pd.DataFrame([{'k':1}]),keys=('k',))
        self.assertEqual(p.read_bytes(),b'corrupt but existing')
    def test_upsert_retains_vendor_withdrawn_history(self):
        p=self.root/'keep.parquet';old=pd.DataFrame([{'trade_date':'2026-09-18','code':'a','v':1},{'trade_date':'2026-09-18','code':'b','v':2}])
        store.upsert(p,old,keys=('trade_date','code'))
        out=store.upsert(p,old.iloc[:1].assign(v=3),keys=('trade_date','code'),date_field='trade_date',replace_suffix=True)
        self.assertEqual(len(out),2);self.assertEqual(out.loc[out.code=='b','v'].iloc[0],2)
    def test_short_pagination_never_marks_entity_done(self):
        ds=copy.deepcopy(R.get('stock_cyq_chips'));ds.start='2026-09-18';ds.entity_codes=['600000.SH']
        c=Client.__new__(Client);c.stats={};c.max_rows_per_query=100000
        c.call=lambda *_args,**_kw:{'total':3,'list':[{'stock_code':'600000.SH','trade_date':'2026-09-18','price':1.0}]}
        rows=c.fetch_all(ds.path,{},10000)
        man=state.Manifest.load(ds.name)
        with patch.object(S,'_entity_list_dates',return_value={}):
            result=S.run_per_entity(ds,man,self.ctx(FakeClient(lambda *_:rows)))
        self.assertFalse(man.is_done('entities','600000.SH'))
        self.assertIn(result.status,('✘','⚠️'))
        self.assertEqual(len(store.read_parquet(paths.year_partition_path(ds.name,2026))),1)
    def test_short_snapshot_pagination_does_not_replace(self):
        ds=copy.deepcopy(R.get('dc_blocks'));man=state.Manifest.load(ds.name)
        old=pd.DataFrame([{'block_code':str(i)} for i in range(100)]);S._save(ds,man,old)
        c=Client.__new__(Client);c.stats={};c.max_rows_per_query=100000
        c.call=lambda *_args,**_kw:{'total':100,'list':old.iloc[:95].to_dict('records')}
        rows=c.fetch_all(ds.path,{},10000)
        result=S.run_snapshot(ds,man,self.ctx(FakeClient(lambda *_:rows)))
        self.assertIn(result.status,('✘','⚠️'))
        self.assertEqual(len(store.read_parquet(paths.flat_path(ds.name))),100)
    def test_pagination_without_total_stops_at_limit(self):
        c=Client.__new__(Client);c.stats={};c.max_rows_per_query=4;requests=[]
        def call(_,p,**kw):requests.append(p['page']);return {'list':[{'x':1},{'x':2}]} if p['page']<3 else {'list':[]}
        c.call=call
        with self.assertRaises(ApiError):c.fetch_all('/audit',{},2)
        self.assertEqual(requests,[0,1])
    def test_pagination_without_total_gets_all_pages(self):
        c=Client.__new__(Client);c.stats={};c.max_rows_per_query=100
        c.call=lambda _,p,**kw:{'list':[{'x':p['page']*2},{'x':p['page']*2+1}]} if p['page']<2 else {'list':[{'x':4}]}
        self.assertEqual(len(c.fetch_all('/audit',{},2)),5)
    def test_invalid_http_success_not_empty_success(self):
        c=Client.__new__(Client);c.max_retries=1;c.empty_retries=0;c.base_url='https://invalid.test';c.api_key='test';c.timeout=1;c.retry_backoff=[];c._throttle=lambda:None
        import threading
        c._gate=threading.BoundedSemaphore(1);c.stats={'requests':0,'bytes_in':0,'errors':0,'retries':0}
        class Response:
            status_code=200;content=b'<html>bad</html>';text='<html>bad</html>'
            def json(self):raise ValueError('not JSON')
        c.session=type('Session',(),{'post':lambda *_a,**_kw:Response()})()
        with self.assertRaises(client_mod.RetryableError):c.call('/audit',{},expect_rows=False)
    def test_old_failed_entity_window_retried_and_cleared(self):
        ds=copy.deepcopy(R.get('tdx_minute'));ds.entity_codes=['880201'];man=state.Manifest.load(ds.name);man.mark_done('entities','880201')
        man._extra['pending_entity_windows']=[{'codes':['880201'],'start':'2026-08-20','end':'2026-08-21'}]
        cli=FakeClient(lambda *_:[]);S.run_per_entity(ds,man,self.ctx(cli))
        self.assertIn('2026-08-20',[p['start_time'] for _,p in cli.calls]);self.assertEqual(man._extra['pending_entity_windows'],[])
    def test_partial_gate_is_actionable(self):
        rep=runner.report_mod.build([],self.ctx(FakeClient(lambda *_:[])),gate_info={'timed_out':True,'blocking':['stock_st_info']})
        self.assertTrue(rep['alerts_actionable'])
    def test_sweep_failure_enters_cli_report_and_exit(self):
        import main
        args=main.build_parser().parse_args(['run','--no-wait','--only','index_ths_daily'])
        rep={'groups':{'daily':[{'name':'index_ths_daily'}]},'alerts':[],'alerts_actionable':[]}
        self.cfg['viz']['summary']=False
        with patch.object(main,'_new_client',return_value=FakeClient(lambda *_:[])),patch.object(roster,'sync',return_value={'ok':True,'added':0}),patch.object(roster,'sweep_if_due',return_value={'ok':False,'failed':1,'note':'injected'}),patch.object(runner,'run',return_value=rep),patch.object(runner.report_mod,'save'):
            rc=main.cmd_run(args,self.cfg)
        self.assertEqual(rc,1);self.assertTrue(rep['alerts_actionable'])
    def test_strict_gate_does_not_execute_sweep(self):
        import main
        args=main.build_parser().parse_args(['run'])
        rep={'T':None,'groups':{},'alerts':['strict'],'alerts_actionable':['strict']}
        self.cfg['viz']['summary']=False
        with patch.object(main,'_new_client',return_value=FakeClient(lambda *_:[])),patch.object(roster,'sync',return_value={'ok':True,'added':0}),patch.object(roster,'sweep_if_due') as sweep,patch.object(runner,'run',return_value=rep),patch.object(runner.report_mod,'save'):
            main.cmd_run(args,self.cfg)
        sweep.assert_not_called()
    def test_old_failed_financial_announcement_retries(self):
        ds=copy.deepcopy(R.get('stock_income'));man=state.Manifest.load(ds.name)
        man.mark_suspect('ann|2025-01-05|error');cli=FakeClient(lambda *_:[])
        result=S.run_per_stock(ds,man,self.ctx(cli))
        self.assertIn('2025-01-05',[p.get('ann_date') for _,p in cli.calls]);self.assertNotIn('ann|2025-01-05|error',man.suspect)
    def test_dump_retries_respect_remaining_quota(self):
        import threading
        day='2026-09-18';q=state.DumpQuota();q.record(day,'5min',n=9)
        c=Client.__new__(Client);c.max_retries=6;c.empty_retries=0;c.base_url='https://invalid.test';c.api_key='test';c.timeout=1;c.retry_backoff=[0]*5;c._throttle=lambda:None;c._retry_delay=lambda _:0
        c._gate=threading.BoundedSemaphore(1);c.stats={'requests':0,'bytes_in':0,'errors':0,'retries':0}
        calls=[]
        class Response:
            status_code=502;content=b'bad';text='bad'
            def json(self):return {'code':502,'data':None}
        def post(*a,**kw):calls.append(1);return Response()
        c.session=type('Session',(),{'post':post})()
        with patch.object(client_mod.time,'sleep'):
            raw,source,note=dump_bridge._load_or_fetch(c,day,log=lambda *_:None)
        self.assertIsNone(raw);self.assertEqual(len(calls),1);self.assertEqual(state.DumpQuota().count_on(day),10)
    def test_oversized_new_entity_group_splits_and_recovers(self):
        ds=copy.deepcopy(R.get('stock_cyq_chips'));ds.start='2026-09-18';ds.entity_codes=['a','b'];man=state.Manifest.load(ds.name)
        def fetch(_,p):
            if isinstance(p['stock_code'],list):
                cls=getattr(client_mod,'QueryLimitError',ApiError)
                raise cls('too many rows')
            return [{'trade_date':'2026-09-18','stock_code':p['stock_code'],'price':1.0}]
        with patch.object(S,'_entity_list_dates',return_value={}):res=S.run_per_entity(ds,man,self.ctx(FakeClient(fetch)))
        self.assertEqual(res.status,'✔');self.assertTrue(all(man.is_done('entities',c) for c in ds.entity_codes))
        self.assertEqual(len(store.read_parquet(paths.year_partition_path(ds.name,2026))),2)

if __name__=='__main__':unittest.main(verbosity=2)
