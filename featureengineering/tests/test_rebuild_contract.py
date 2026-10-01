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
    @staticmethod
    def _engine(td):
        e=object.__new__(Engine)
        e.cfg=SimpleNamespace(default_start='2018-01-01',revision_days=1,factors_dir=Path(td))
        e.universe_fp='test';e._start_floor=20180101;e.override_start=None
        e.cal=Calendar(np.array([20180102,20190102,20200102],dtype=np.int32))
        return e

    # ★★ 2026-09-30（用户硬约定）改写：本条原先固化的是**旧契约**
    #   「历史分区缺失/被改 ⇒ 自动整年重建」。那是错的：因子必须具有
    #   「不被未来数据增量改变历史值」的固有属性 ⇒ 自动逻辑**永远不得**重算
    #   已覆盖的历史，只能记账，由人看过台账后用显式操作决定。
    def test_covered_history_is_never_replanned_by_an_incremental_run(self):
        with tempfile.TemporaryDirectory() as td:
            spec=FactorSpec('test','test')
            e=self._engine(td)
            man=Manifest('test',Path('/unused'),recipe=e._recipe(spec),
                         coverage=[['2018-01-02','2020-01-02']],partitions={'2018':{'file_identity':[100,1]}})
            # 2018 分区**文件根本不存在**（最"该修"的情形）—— 仍然不进计划，只记账
            self.assertEqual(sorted(e.plan(spec,man,20200102,False)),[2020])
            self.assertEqual([(b['factor'],b['year'],b['days'],b['earliest'],b['freeze_floor'])
                              for b in e.history_recompute],
                             [('test',2018,1,20180102,20200101)])
            # 文件被改坏成非法内容 —— 同样只记账
            p=Path(td)/'test/year=2018/data.parquet';p.parent.mkdir(parents=True);p.write_bytes(b'changed')
            e2=self._engine(td); man.partitions['2018']['file_identity']=[100,1]
            self.assertEqual(sorted(e2.plan(spec,man,20200102,False)),[2020])
            self.assertEqual(len(e2.history_recompute),1)

    def test_explicit_operations_are_the_only_way_to_rewrite_history(self):
        with tempfile.TemporaryDirectory() as td:
            spec=FactorSpec('test','test')
            e=self._engine(td)
            man=Manifest('test',Path('/unused'),recipe=e._recipe(spec),
                         coverage=[['2018-01-02','2020-01-02']],partitions={'2018':{'file_identity':[100,1]}})
            # ① `--rebuild`
            self.assertEqual(sorted(e.plan(spec,man,20200102,True)),[2018,2019,2020])
            self.assertEqual(e.history_recompute,[])
            # ② `--start`：显式放行 → 不被剔出
            e2=self._engine(td); e2.override_start=20180101
            man2=Manifest('test',Path('/unused'),recipe=e2._recipe(spec),
                          coverage=[['2018-01-02','2020-01-02']],
                          partitions={'2018':{'file_identity':[100,1]}})
            self.assertIn(2018,sorted(e2.plan(spec,man2,20200102,False)))
            self.assertEqual(e2.history_recompute,[])

    def test_identity_mismatch_with_intact_content_refreshes_without_rebuild(self):
        """身份不符 ≠ 损坏（§11.2 的教训落进代码）。

        判据：内容（行数 + 日期上下界）与台账一致 ⇒ **只刷新身份、不重建**。
        2026-09-30 实测：正是这条误判让 550 个因子的 year=2026 整年重建，
        99591 个任务里 97% 逐格零变化（② 从 ~7 min 涨到 12.1 min）。
        这里断言 `history_recompute == []` 是关键 —— 内容核对若没生效，
        2018 会被判"损坏"从而进 history_recompute。
        """
        import pandas as pd
        with tempfile.TemporaryDirectory() as td:
            spec=FactorSpec('test','test'); e=self._engine(td)
            df=pd.DataFrame({'trade_date':pd.Series(['2018-01-02'],dtype='string'),
                             'stock_code':pd.Series(['000001.SZ'],dtype='string'),
                             'value':pd.Series([1.0],dtype='float32'),
                             'rank':pd.Series([0.5],dtype='float32')})
            p=Path(td)/'test/year=2018/data.parquet';p.parent.mkdir(parents=True)
            df.to_parquet(p,index=False)
            man=Manifest('test',Path('/unused'),recipe=e._recipe(spec),
                         coverage=[['2018-01-02','2020-01-02']],
                         partitions={'2018':{'file_identity':[1,1],'rows':1,
                                             'min_date':'2018-01-02','max_date':'2018-01-02'}})
            self.assertEqual(sorted(e.plan(spec,man,20200102,False)),[2020])
            st=p.stat()
            self.assertEqual(man.partitions['2018']['file_identity'],[st.st_size,st.st_mtime_ns])
            self.assertEqual(e.identity_refreshed,[{'factor':'test','year':2018}])
            self.assertEqual(e.history_recompute,[])


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

    def test_explicit_plan_keeps_align_years(self):
        """★ 2026-09-25：显式计划（`--plan-file`，切片 worker 走的就是这条）里
        的对齐年份必须**不被真实值起点滤掉**。

        实测事故：`plan()` 的 explicit 分支用 `start_i`（真实值起点）过滤交易日，
        而 `plan.json` 已经把对齐年份（2018/2019）的交易日写进去了 ⇒ 26 个晚起点
        因子在 2018/2019 块"跑了但什么都没写"，**退出码还是 0**，静默丢数据。
        """
        s = FactorSpec('late', 'test', start='2020-01-01')     # 默认 align_fill=nan
        e = object.__new__(Engine)
        e.cfg = SimpleNamespace(default_start='2018-01-01', revision_days=10,
                                raw={'_explicit_plan': {'late': {2018: [20180102, 20181228],
                                                                 2020: [20200102]}}})
        e.universe_fp = 'test'; e._start_floor = 20180101
        e.override_start = None; e.force_refresh = False
        e.cal = Calendar(np.array([20180102, 20181228, 20200102], dtype=np.int32))
        m = Manifest('late', Path('/unused'), recipe=e._recipe(s), coverage=[])
        p = e.plan(s, m, 20201231, False)
        self.assertIn(2018, p, "对齐年份被真实值起点滤掉了 —— 又会静默丢数据")
        self.assertEqual(p[2018].tolist(), [20180102, 20181228])
        # 对照组：显式关掉对齐（逃生舱）后，2018 不应再出现。
        s2 = FactorSpec('late2', 'test', start='2020-01-01', align_fill=None)
        e.cfg.raw['_explicit_plan'] = {'late2': {2018: [20180102, 20181228]}}
        m2 = Manifest('late2', Path('/unused'), recipe=e._recipe(s2), coverage=[])
        self.assertNotIn(2018, e.plan(s2, m2, 20201231, False))

    def test_prune_cutoff_follows_align_contract(self):
        """★ 2026-09-25：裁剪下界必须跟着**对齐契约**走，否则会删掉合法产物。

        实测事故：`main.py rebuild` 成功后无条件调 `history.run(apply=True)`，而它
        当时用 `resolved_start`（真实值起点）当裁剪下界 ⇒ 会把晚起点因子的
        2018/2019 **对齐分区整年删掉**，并同步收紧台账 coverage（账实一起消失）。
        枚举结果：35 个分区 / 18 个因子。
        """
        from fea.history import cutoff_of
        cfg = SimpleNamespace(default_start='2018-01-01')
        # 声明了对齐（默认 nan）⇒ 下界是 default_start，不能按真实起点裁
        self.assertEqual(cutoff_of(FactorSpec('a', 'g', start='2020-01-01'), cfg), '2018-01-01')
        self.assertEqual(cutoff_of(FactorSpec('b', 'g', start='2019-08-14'), cfg), '2018-01-01')
        # 显式关掉对齐（逃生舱）⇒ 维持旧行为
        self.assertEqual(cutoff_of(FactorSpec('c', 'g', start='2020-01-01',
                                              align_fill=None), cfg), '2020-01-01')

    def test_finalize_refuses_planned_but_unproduced_years(self):
        """★ 2026-09-25：`_finalize` 不得把「计划了但没产出」的年份标成已完成。

        这是「静默零产出」那一族的守门人：原来无条件按 `plan` 打 coverage，
        于是任务没产出也记成成功，`missing_ranges` 以后看不到缺口 ⇒ **永不自愈**。
        """
        with tempfile.TemporaryDirectory() as td:
            e = object.__new__(Engine)
            e.cfg = SimpleNamespace(state_dir=Path(td))
            e._run_end = None
            e._wm_cache = {}
            plan = {2018: np.array([20180102, 20180103], dtype=np.int32)}
            m = Manifest('x', Path(td) / 'x.json', recipe='r', coverage=[])
            with self.assertRaises(RuntimeError) as cm:
                e._finalize(FactorSpec('x', 'g', deps=()), m, plan, [], 0.0)
            self.assertIn('计划了', str(cm.exception))
            # 对照组：结果与计划对齐时不应抛
            e._trading_runs = lambda days: [(int(days[0]), int(days[-1]))]
            e.watermark = lambda dep: {}
            m2 = Manifest('x', Path(td) / 'x2.json', recipe='r', coverage=[])
            e._finalize(FactorSpec('x', 'g', deps=()), m2, plan,
                        [{"factor": 'x', "year": 2018, "rows": 0, "nonnull": 0,
                          "partition_rows": 0, "min_date": "", "max_date": "", "seconds": 0.0}], 0.0)
            self.assertEqual(m2.coverage, [["2018-01-02", "2018-01-03"]])

    def test_resources_reject_excessive_parallelism(self):
        """`safe_jobs` 只接受 1..cap；0 归一为串行。cap 由 `FEA_MAX_JOBS` 控制。

        ★ 2026-09-25：原来写死「cap=2，所以 3/8 必须抛」。默认 cap 改成 12 之后
          （见 `fea/resources.py::safe_jobs` 的阶梯实测）这条就假失败了。改成
          **验证机制本身**：显式钉住 env 再断言边界，这样以后调 cap 不会再误伤。
        """
        import os
        with patch.dict(os.environ, {"FEA_MAX_JOBS": "2"}):
            self.assertEqual(safe_jobs(0), 1)      # 0 → 串行
            self.assertEqual(safe_jobs(1), 1)
            self.assertEqual(safe_jobs(2), 2)
            for n in (3, 8, -1):
                with self.assertRaises(ValueError):
                    safe_jobs(n)
        with patch.dict(os.environ, {"FEA_MAX_JOBS": "12"}):
            self.assertEqual(safe_jobs(12), 12)
            for n in (13, 16, -1):
                with self.assertRaises(ValueError):
                    safe_jobs(n)

    def test_value_only_revision_in_history_is_frozen_to_the_tail(self):
        """★★ 2026-09-29 契约变更：**上游的旧年修订不再回填历史，只记录**。

        旧契约（2026-09-21 起）：年分区被重写（哪怕字节数没变、只有 mtime 变）
          ⇒ 判"历史修订" ⇒ 从最早被改的年份年初一路重算。
        旧契约的代价（2026-09-29 实测）：厂商新增 225 行「报告期在 2018–2024、
          ann_date 在 2026 年」的财务数据 ⇒ **217 个因子重算 2018–2025、
          约 28 分钟 CPU、窗口内逐格 0 变化**（纯空转）。

        新契约：引擎的 as-of 口径是 `pit = max(ann_date, f_ann_date, end_date)`
          （`fea/deriv.py`，"保留所有版本，as-of 时取最新可见版本"）——
          新公告对"它公告之前"的任何一天都不可见 ⇒ 那些天的值**不可能变**。
          所以回填下界一律夹到尾部回刷窗；被夹掉的部分记进
          `Engine.frozen_input_changes`（进 `report.json`）供观测。
        """
        s=FactorSpec('test','test',deps=('source',))
        e=object.__new__(Engine)
        e.cfg=SimpleNamespace(default_start='2018-01-01',revision_days=1,
                              dep_backfill_days=lambda ds:2000)
        e.universe_fp='test';e._start_floor=20180101;e.override_start=None
        e.cal=Calendar(np.array([20180102,20190102,20200102],dtype=np.int32))
        prev={'exists':True,'rows':3,'max_pit':20200102,'files':{'year=2018':[100,1],'year=2020':[100,1]}}
        cur={**prev,'files':{'year=2018':[100,2],'year=2020':[100,1]}}
        e.watermark=lambda _:cur
        m=Manifest('test',Path('/unused'),recipe=e._recipe(s),coverage=[['2018-01-02','2020-01-02']],input_watermark={'source':prev})
        # 2018 那次重写落在冻结下界（当前年）之前 ⇒ 只重算 2020，且留下一条冻结记录
        self.assertEqual(sorted(e.plan(s,m,20200102,False)),[2020])
        self.assertEqual(len(e.frozen_input_changes),1)
        self.assertEqual(e.frozen_input_changes[0]['dep'],'source')
        self.assertEqual(e.frozen_input_changes[0]['source'],'changed_files')
        self.assertEqual(e.frozen_input_changes[0]['earliest_changed'],'2018-01-01')

    def test_frozen_history_change_is_recorded_not_silently_dropped(self):
        """冻结 ≠ 丢弃：被挡掉的历史修订必须**留下可观测的记录**。

        这是"厂商是否原地改写了历史值"的唯一观测通道（用户 2026-09-29：
        「如果是数据厂商偷偷替换数据这个没辙，只能观测」）。
        记录进 `Engine.frozen_input_changes` → `fea/daily.py` 汇总进
        `report.json.frozen_input_changes`。

        ★ 与上一条用例的分工：上一条固定"会被冻"，这一条固定"冻了要留痕"。
        """
        s=FactorSpec('test','test',deps=('source',))
        e=object.__new__(Engine)
        e.cfg=SimpleNamespace(default_start='2018-01-01',revision_days=1,
                              dep_backfill_days=lambda ds:1)
        e.universe_fp='test';e._start_floor=20180101;e.override_start=None
        e.cal=Calendar(np.array([20180102,20190102,20200102],dtype=np.int32))
        prev={'exists':True,'rows':3,'max_pit':20200102,'files':{'year=2018':[100,1],'year=2020':[100,1]}}
        cur={**prev,'files':{'year=2018':[100,2],'year=2020':[100,1]}}
        e.watermark=lambda _:cur
        m=Manifest('test',Path('/unused'),recipe=e._recipe(s),coverage=[['2018-01-02','2020-01-02']],input_watermark={'source':prev})
        self.assertEqual(sorted(e.plan(s,m,20200102,False)),[2020])
        rec=e.frozen_input_changes[0]
        # `freeze` 取的是 `minus_days(end, revision_days)` 的**日历日**（forward_days=0
        # 时 `_rewind_forward_dependency` 直接原样返回，不对齐到交易日）
        self.assertEqual(rec['freeze_floor'],'2020-01-01')
        self.assertEqual(rec['frozen_days'],2)      # 2018-01-02 与 2019-01-02 两天被冻住

    def test_explicit_refresh_still_rebuilds_history(self):
        """冻结**不许**挡住人工重建：`--refresh` / `--rebuild` 仍是全历史。

        （`main.py rebuild` → `scripts/backfill_history.py` →
          `main.py run --start … --refresh`，走 `plan()` 的 `force_refresh` 分支。）
        """
        s=FactorSpec('test','test',deps=('source',))
        e=object.__new__(Engine)
        e.cfg=SimpleNamespace(default_start='2018-01-01',revision_days=1,
                              dep_backfill_days=lambda ds:1)
        e.universe_fp='test';e._start_floor=20180101;e.override_start=None
        e.force_refresh=True
        e.watermark=lambda _:{'exists':False,'rows':0,'max_pit':0}   # L2 循环直接跳过
        e.cal=Calendar(np.array([20180102,20190102,20200102],dtype=np.int32))
        m=Manifest('test',Path('/unused'),recipe=e._recipe(s),coverage=[['2018-01-02','2020-01-02']])
        self.assertEqual(sorted(e.plan(s,m,20200102,False)),[2018,2019,2020])
        self.assertEqual(getattr(e,'frozen_input_changes',[]),[])

    def test_non_financial_run_skips_vintages(self):
        e=object.__new__(Engine)
        self.assertIsNone(e.deriv_for(20180101,20181231,fields=frozenset()))

    def test_coupling_inherits_parent_repair_dates(self):
        e=object.__new__(Engine);e.cfg=SimpleNamespace(default_start='2018-01-01')
        e._start_floor=20180101;e.override_start=None
        parent=FactorSpec('parent','test');child=FactorSpec('child','test',deps=('parent',))
        e.cal=Calendar(np.array([20200102,20200302,20200515],dtype=np.int32))
        p={2020:np.array([20200102,20200515],dtype=np.int32)}
        c={2020:np.array([20200515],dtype=np.int32)}
        e._propagate_parent_plans([(child,None,c),(parent,None,p)],20200515)
        self.assertEqual(c[2020].tolist(),[20200102,20200302,20200515])

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
