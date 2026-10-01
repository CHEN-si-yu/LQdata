"""测试交易时序和资金守恒，不使用测试窗市场表现。"""
from types import SimpleNamespace
from dataclasses import replace
import unittest
import numpy as np
from top5_strategy import Spec,simulate,gate

def market():
    T,C=12,10
    price=np.full((T,C),100.,dtype=float)
    px=SimpleNamespace(open=price.copy(),close=price.copy(),adj=np.ones((T,C)),entry=np.ones((T,C),bool),exit=np.ones((T,C),bool),raw={'high':price.copy(),'low':price.copy()})
    panel=SimpleNamespace(days=np.array([f'2025-01-{d+1:02d}' for d in range(T)]),codes=np.array([f'S{c}' for c in range(C)]),amount=np.full((T,C),1e9))
    features={'index':np.full(T,100.),'atr':np.full((T,C),.03)}
    pred=np.tile(np.arange(C,0,-1),(8,1)).astype(float)
    return panel,px,features,pred,np.arange(8)

class ExecutionTests(unittest.TestCase):
    def test_t_plus_one_and_stop_first(self):
        panel,px,f,p,d=market();px.raw['low'][1:3]=80;px.raw['high'][1:3]=130
        spec=Spec(mode='intraday',gate_window=1)
        m,c,t=simulate(p,d,panel,px,spec,f,details=True)
        sells=[x for x in t if x['side']=='sell']
        self.assertEqual(sells[0]['date'],panel.days[2])
        self.assertEqual(sells[0]['reason'],'stop_loss')
        self.assertLess(sells[0]['price'],100)
        self.assertEqual(c[1]['positions'],0) # no reinvestment before this day's open
        self.assertTrue(all(x['cash']>=0 and x['positions']<=5 for x in c))
    def test_gap_stop(self):
        panel,px,f,p,d=market();px.open[2]=70;px.raw['low'][2]=65
        m,c,t=simulate(p,d,panel,px,Spec(mode='intraday',gate_window=1),f,details=True)
        sells=[x for x in t if x['reason']=='stop_loss']
        self.assertTrue(sells);self.assertLessEqual(sells[0]['price'],70)
    def test_blocked_sales_preserve_slots(self):
        panel,px,f,p,d=market();p[1:]=p[1:,::-1];px.exit[2:,:5]=False
        m,c,t=simulate(p,d,panel,px,Spec(exit_rank=5,min_hold=1,gate_window=1),f,details=True)
        self.assertGreater(m['blocked_exits'],0)
        self.assertTrue(all(x['positions']<=5 for x in c))
    def test_cooldown(self):
        panel,px,f,p,d=market();px.raw['low'][2]=80
        m,c,t=simulate(p,d,panel,px,Spec(mode='intraday',cooldown=3,gate_window=1),f,details=True)
        buys=[x for x in t if x['side']=='buy' and x['code']=='S0']
        self.assertTrue(all(x['date']>=panel.days[5] for x in buys[1:]))
    def test_gate_prefix_causal(self):
        f={'index':np.arange(100,150,dtype=float)};s=Spec(gate_window=5,hysteresis=.01)
        before=gate(f,s);f['index'][25:]=1
        np.testing.assert_array_equal(before[:25],gate(f,s)[:25])
    def test_gate_exit_overrides_min_hold(self):
        panel,px,f,p,d=market();f['index'][1]=50
        m,c,t=simulate(p,d,panel,px,Spec(gate_window=2,min_hold=20),f,details=True)
        # Explicit falling regime after first actual entry.
        f['index']=np.array([100,101,102,10,10,10,10,10,10,10,10,10.])
        m,c,t=simulate(p,d,panel,px,Spec(gate_window=2,min_hold=20),f,details=True)
        self.assertTrue(any(x['reason']=='market_gate' for x in t))
    def test_flat_prices_only_charge_costs(self):
        panel,px,f,p,d=market()
        m,c,t=simulate(p,d,panel,px,Spec(gate_window=1),f,slippage=0,details=True)
        self.assertAlmostEqual(c[-1]['equity'],100000-m['fees'],places=7)
        self.assertEqual(m['ending_positions'],0)
        self.assertTrue(all(x['quantity']%100==0 for x in t if x['side']=='buy'))
    def test_corporate_split_preserves_wealth(self):
        panel,px,f,p,d=market()
        normal=simulate(p,d,panel,px,Spec(gate_window=1),f,slippage=0,details=True)
        px.open[3:]/=2;px.close[3:]/=2;px.raw['high'][3:]/=2;px.raw['low'][3:]/=2;px.adj[3:]*=2
        split=simulate(p,d,panel,px,Spec(gate_window=1),f,slippage=0,details=True)
        np.testing.assert_allclose([x['equity'] for x in normal[1]],[x['equity'] for x in split[1]])
        self.assertEqual(split[0]['corporate_actions'],5)
        self.assertFalse(any(x['reason']=='stop_loss' for x in split[2]))
    def test_close_stop_waits_until_following_open(self):
        panel,px,f,p,d=market();px.close[2]=80
        m,c,t=simulate(p,d,panel,px,Spec(gate_window=1),f,details=True)
        exits=[x for x in t if x['reason']=='stop_loss']
        self.assertEqual(len(exits),5)
        self.assertTrue(all(x['date']==panel.days[3] for x in exits))
    def test_terminal_blocked_position_is_marked_not_liquidated(self):
        panel,px,f,p,d=market();px.exit[9]=False
        m,c,t=simulate(p,d,panel,px,Spec(gate_window=1),f,details=True)
        self.assertEqual(m['ending_positions'],5)
        self.assertEqual(m['blocked_exits'],5)
        self.assertFalse(any(x['side']=='sell' for x in t))
        self.assertGreater(c[-1]['equity'],c[-1]['cash'])
    def test_liquidity_budget_uses_previous_day(self):
        panel,px,f,p,d=market();panel.amount[0]=500000
        m,c,t=simulate(p,d,panel,px,Spec(gate_window=1),f,details=True)
        self.assertEqual(c[0]['positions'],0)
        self.assertEqual(t[0]['date'],panel.days[2])
    def test_atr_stop_is_frozen_at_entry(self):
        panel,px,f,p,d=market();f['atr'][1:]=.2;px.close[2]=90
        m,c,t=simulate(p,d,panel,px,Spec(gate_window=1,atr_stop=2.5),f,details=True)
        exits=[x for x in t if x['reason']=='stop_loss']
        self.assertEqual(len(exits),5)
        self.assertTrue(all(x['date']==panel.days[3] for x in exits))

if __name__=='__main__':unittest.main()
