"""Actual native cash loop must honor injected policies and strictly lag decisions."""
from pathlib import Path
from types import SimpleNamespace
import ast,importlib.util,json,unittest
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('market_cash_research_helpers',ROOT/'experiments/_archive/market_cash_20261030/scripts/research.py');research=importlib.util.module_from_spec(spec);spec.loader.exec_module(research)
class MarketCashBindingTests(unittest.TestCase):
    def setUp(self):
        source=ROOT/'experiments/V292/analysis.py';tree=ast.parse(source.read_text());ns=dict(np=np,pd=pd,json=json,fees=lambda gross,sell=False:max(5.,gross*.00025)+gross*.00001+(gross*.0005 if sell else 0.))
        for name in ['market_budget','strategy_backtest']:
            node=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name][-1];exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),ns)
        self.analysis=SimpleNamespace(**ns);self.analysis.__file__=str(source);self.small,_=research.small_with_holdings(self.analysis);self.base=ns['market_budget'];self.p=SimpleNamespace(days=np.array(['2025-01-02','2025-01-03','2025-01-06','2025-01-07','2025-01-08']),codes=np.array(['A','B','C','D','E']),amount=np.full((5,5),1e8));self.px=SimpleNamespace(open=np.full((5,5),100.),close=np.full((5,5),100.),adj=np.ones((5,5)),limit_up=np.full((5,5),110.),limit_down=np.full((5,5),90.),entry=np.ones((5,5),bool),exit=np.ones((5,5),bool));self.context={k:np.zeros(5) for k in research.RAWMARKET};self.context['mkt_above_ma20_ratio'][:]=.6
    def run_policy(self,arm):
        research.bind_budget(self.analysis,self.small,research.policy(arm,self.base));return self.small(np.tile(np.arange(5.),(4,1)),np.zeros((4,5,2)),np.zeros(4),self.p,self.px,np.arange(4),rule='market_cash',context=self.context,liquidate=False)
    def test_cash_negative_control_has_no_trades_or_loss(self):
        stat,curve,trades=self.run_policy('cash');self.assertEqual(trades,[]);self.assertEqual(stat['return_value'],0);self.assertTrue(all(r['target_exposure']==0 and r['positions']==0 and r['equity']==100000 for r in curve))
    def test_fixed40_is_really_reduced_and_has_atmost_two_names(self):
        stat,curve,trades=self.run_policy('fixed40');self.assertTrue(trades);self.assertLessEqual(stat['max_positions_observed'],2);self.assertTrue(all(r['target_exposure']==.4 for r in curve))
    def test_breadth_filter_is_not_silently_the_original_event_filter(self):
        self.context['mkt_above_ma20_ratio'][:]=.3;event,_,trades=self.run_policy('event_cash');self.assertTrue(trades);breadth,curve,trades=self.run_policy('breadth_cash');self.assertEqual(trades,[]);self.assertTrue(all(r['target_exposure']==0 for r in curve))
    def test_recovery_requires_two_consecutive_eligible_signals(self):
        state,recovery=self.base(.5,0,self.context,1.,0);self.assertEqual(state,0.)
        state,recovery=self.base(.1,1,self.context,state,recovery);self.assertEqual((state,recovery),(0.,1))
        state,recovery=self.base(.1,2,self.context,state,recovery);self.assertEqual((state,recovery),(1.,0))
    def test_future_market_signal_and_close_do_not_change_earlier_orders(self):
        _,before,trades_before=self.run_policy('breadth_cash');self.context['mkt_above_ma20_ratio'][3]=.2;self.px.close[4]=101.;_,after,trades_after=self.run_policy('breadth_cash');self.assertEqual(before[:3],after[:3]);self.assertEqual([t for t in trades_before if t['date']<'2025-01-08'],[t for t in trades_after if t['date']<'2025-01-08']);self.assertEqual(after[-1]['target_exposure'],0.)
if __name__=='__main__':unittest.main()
