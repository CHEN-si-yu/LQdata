import unittest
from unittest.mock import patch
from pathlib import Path
import tempfile
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pyarrow as pa
from fea.context import FactorContext
from fea.panel import Panel
from factors import field_events as ev
from factors import field_markets as mk

class FieldMarketEventTests(unittest.TestCase):
    def test_event_mean_is_independent_of_expired_history(self):
        values=np.zeros((100,3));values[:30]=1e18
        values[50:60,1]=np.nan;values[60:,2]=np.nan
        full=ev.event_mean(values,20)
        cropped=ev.event_mean(values[40:],20)
        np.testing.assert_array_equal(full[59:],cropped[19:])
        np.testing.assert_array_equal(full[80:,:2],0.)
        self.assertTrue(np.isnan(full[80:,2]).all())

    def test_event_mean_matches_explicit_finite_window(self):
        values=np.array([[np.nan,2.],[4.,np.nan],[np.nan,6.],[8.,np.nan],[0.,10.]])
        result=ev.event_mean(values,3)
        expected=np.array([[np.nan,2.],[4.,2.],[4.,4.],[6.,6.],[4.,8.]])
        np.testing.assert_array_equal(result,expected)

    def test_trade_clock_formats_and_lunch(self):
        s=pd.Series(['93000','09:30:00','113000','13:00:00','14:29:13','150000','120000','','250000','09:65:00','09:25:00'])
        np.testing.assert_allclose(ev.clock_minutes(s),[0,0,120,120,209+13/60,240,np.nan,np.nan,np.nan,np.nan,0],equal_nan=True)

    def test_event_grid_deduplicates_and_keeps_missing_days(self):
        p=Panel(np.array([20250102,20250103,20250106],dtype=np.int32),np.array(['A','B']))
        raw=pd.DataFrame({'stock_code':['A','A','A','B'],'trade_date':['2025-01-02','2025-01-02','2025-01-06','2025-01-03'],'l_buy':[20,40,999,10],'l_sell':[10,20,1,10]})
        up=SimpleNamespace(audit_cutoff=20250103,read=lambda ds,columns,**kw:raw[columns])
        ctx=FactorContext(p,None,up,None,np.ones(p.shape,bool))
        out=ev.grid(ctx,'stock_top_list',['l_buy','l_sell'],lambda d:ev.ratio(d.l_buy-d.l_sell,d.l_buy+d.l_sell))
        np.testing.assert_allclose(out,[[1/3,np.nan],[np.nan,0],[np.nan,np.nan]],equal_nan=True)

    def test_string_ratios_do_not_invent_values(self):
        a=ev.fraction_text(pd.Series(['3/5','0/0','bad']),r'^(\d+)/(\d+)$')
        np.testing.assert_array_equal(a,[.6,np.nan,np.nan])
        np.testing.assert_array_equal(ev.ratio([1,2,3],[0,-1,2]),[np.nan,np.nan,1.5])

    def test_daily_market_uses_historical_rows_and_source_cutoff(self):
        rows=[]
        for day in ['2025-01-02','2025-01-03','2025-01-06']:
            for i in range(6):rows.append(dict(board_code=str(i),trade_date=day,open=10,high=12,low=9,close=11,vol=100+i,amount=200+i))
        raw=pd.DataFrame(rows)
        up=SimpleNamespace(read=lambda ds,columns,**kw:raw[columns],audit_cutoff=20250103)
        p=Panel(np.array([20250102,20250103,20250106],np.int32),np.array(['A']))
        c=FactorContext(p,None,up,None,np.ones(p.shape,bool))
        x=mk.market(c,'tdx_daily')
        self.assertEqual(list(x.index),[20250102,20250103])
        np.testing.assert_allclose(x.pressure,1/3)
        self.assertIs(x,mk.market(c,'tdx_daily'))

    def test_market_correlation_is_stock_specific_and_prefix_invariant(self):
        dates=pd.bdate_range('2024-01-01',periods=150)
        days=dates.strftime('%Y%m%d').astype(int).to_numpy(np.int32)
        signal=np.sin(np.arange(150)/5)+np.cos(np.arange(150)/13)
        frame=pd.DataFrame({'pressure':signal},index=days)
        def ctx(n):
            panel=Panel(days[:n],np.array(['A','B']))
            # ★ 2026-09-22 去冗余：`mfx_tdx_pressure`（与 `mfx_dc_pressure` |ρ|=0.988）已删，
            #   改用同族的 `mfx_dc_pressure`（数据集相应从 tdx_daily 换成 dc_daily）。
            up=SimpleNamespace(audit_cutoff=None,_expansion_market_cache={('dc_daily',int(days[0]),int(days[n-1])):frame.iloc[:n]})
            return SimpleNamespace(panel=panel,up=up,ret_clean=lambda:np.stack([signal[:n],-signal[:n]],axis=1))
        full=mk.mfx_dc_pressure(ctx(150));prefix=mk.mfx_dc_pressure(ctx(100))
        np.testing.assert_allclose(full[:100],prefix,atol=1e-12,equal_nan=True)
        np.testing.assert_allclose(full[-1],[1,-1],atol=1e-12)

    def test_minute_chunk_aggregation_and_cutoff(self):
        rows=[]
        for date in ['2025-01-02','2025-01-03']:
            for board in range(30):
                for k in range(48):
                    rows.append(dict(board_code=str(board),trade_time=date+' 10:00:00',open=10,high=12,low=9,close=11,vol=100,amount=500,pct_change=.1,amplitude=.3))
        batches=pa.Table.from_pandas(pd.DataFrame(rows)).to_batches(max_chunksize=101)
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'tdx_minute/year=2025/data.parquet';p.parent.mkdir(parents=True);p.touch()
            up=SimpleNamespace(root=Path(temp),audit_cutoff=20250102)
            panel=Panel(np.array([20250102,20250103],np.int32),np.array(['A']))
            ctx=FactorContext(panel,None,up,None,np.ones(panel.shape,bool))
            with patch.object(mk.pq,'ParquetFile',return_value=SimpleNamespace(iter_batches=lambda **kw:iter(batches))):
                result=mk._minute(ctx)
        self.assertEqual(list(result.index),[20250102])
        np.testing.assert_allclose(result.pressure,[1/3])
        np.testing.assert_allclose(result.weighted_pressure,[1/3])
        np.testing.assert_allclose(result.volume_concentration,[1])
        np.testing.assert_allclose(result.realized_vol,[np.sqrt(48)*.1])

    def test_forecast_disclosure_on_weekend_maps_to_next_trading_day(self):
        raw=pd.DataFrame({'stock_code':['A'],'ann_date':['2025-01-04'],'first_ann_date':['2024-12-25']})
        seen=[]
        def read(ds,columns,years):seen.append(years);return raw[columns]
        panel=Panel(np.array([20250103,20250106],np.int32),np.array(['A']))
        ctx=FactorContext(panel,None,SimpleNamespace(read=read,audit_cutoff=None),None,np.ones(panel.shape,bool))
        out=ev.grid(ctx,'stock_forecast',['ann_date','first_ann_date'],ev.forecast_delay,'ann_date')
        np.testing.assert_array_equal(out[:,0],[np.nan,10])
        self.assertEqual(seen,[(2022,2025)])

    def test_board_minute_does_not_rebuild_stock_minute_layer(self):
        from fea.engine import Engine
        from fea.spec import FactorSpec
        with tempfile.TemporaryDirectory() as temp:
            for dataset,expected in [('tdx_minute',(0,-1)),('stock_history_5min',(2017,2018))]:
                calls=[]
                engine=object.__new__(Engine)
                engine.cfg=SimpleNamespace(state_dir=Path(temp))
                engine.plan=lambda *args:{2018:np.array([20180102,20181228],np.int32)}
                engine._propagate_parent_plans=lambda *args:None
                engine.deriv_for=lambda *args,**kwargs:None
                engine._st=();engine._prices=None
                engine.prices_for=lambda *args:None
                engine.intraday_layer=lambda:SimpleNamespace(ensure=lambda *args:calls.append(args))
                engine.chips_layer=lambda:SimpleNamespace(ensure=lambda *args:None)
                # 2026-09-21 新加的 open5 派生层同属「兄弟层」，与本用例无关，一并打桩。
                engine.open5_layer=lambda:SimpleNamespace(ensure=lambda *args:None)
                engine.factor_io=lambda:None
                engine.up=SimpleNamespace(clear_cache=lambda:None)
                # S-01（2026-09-21）在 prebuild 开头加了两道闸门（ST 内容 / 可用时点契约）——
                # 那是**另一个关注点**，要真的读上游 ST 表；本用例只验证日内层的依赖规划，
                # 故把两道闸门打成空操作（它们自己的路径由 main.py run 的集成流程覆盖）。
                engine.st_gate=lambda:{}
                engine.delay_gate=lambda:{}
                engine.prebuild([FactorSpec('fixture','fixture',deps=(dataset,),warmup_days=220)],20181228)
                self.assertEqual(calls,[expected])

if __name__=='__main__':unittest.main()
