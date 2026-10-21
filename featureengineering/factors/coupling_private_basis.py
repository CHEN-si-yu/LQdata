"""耦合子因子的私有数学基元（未注册、不发布）。

为什么存在：失败父因子 momentum_20 / bp 的独立注册与产物在清理中删除；两个保留子因子
（cp_momentum_lowvol_20 / cp_value_momentum_div）不能再读父产物。本模块按父因子原函数、
原年度锚点、原 warmup、原股票池、原 float32 舍入在子进程内重建父值（V2/V3 只读可行性
已证 9 年 18/18 逐格 value/rank 精确一致）。这里没有任何 register 副作用：_PRIVATE_*
规格 enabled=False、不属于正式库、不会发布任何数据。
"""
import numpy as np,gc
from fea.spec import FactorSpec


_BASIS_EQUITY_FIELD = 'total_hldr_eqy_exc_min_int'

def _basis_market_value(ctx, share: str) -> np.ndarray:
    """市值 = 未复权收盘价 × 股本（元）。

    `close` 与 `*_share` 都是**状态量**，价格层已做前向填充（停牌期间沿用最后一个
    成交价与最新股本）—— 这正是市值该有的语义。停牌日不是 0，也不会是 NaN。
    """
    px = np.asarray(ctx.px('close'), dtype=np.float64)
    sh = np.asarray(ctx.px(share), dtype=np.float64)
    return px * sh

def _basis_equity_value(ctx):
    return ctx.safe_div(ctx.point(_BASIS_EQUITY_FIELD), _basis_market_value(ctx, 'total_share'), min_abs_den=1000000.0)

def _basis_compound_return(ctx, k: int) -> np.ndarray:
    """k 个交易日的后复权收益 —— 与 `ctx.ret(k)` 等价，但把「脏数据」判定改回**单日**口径。

    ★ 为什么本家族不能直接用 `ctx.ret(k)`（实测，2013-09~2015-12 面板 561×3484）：

      `fea/prices.py:ret()` 的清洗是 `|r| > 60% → NaN`，它**自己的注释**写的是
      「**单日** |收益| > 60% 认定是复权因子脏数据」（实测脏数据确实是 223 个**日频**格子）。
      但这条阈值被直接套在了 **k 日累计收益** 上，于是 k 越大误杀越狠：

        k=5    误杀   2,019 格（被误杀者的 k 日收益中位 +61%）
        k=20   误杀  15,592 格（+74%）
        k=60   误杀  94,562 格（+81%）
        k=120  误杀 167,413 格（+89%）
        k=250  误杀 317,223 格 = 该面板有效格的 48%（中位 +109%，最大 +2367%）

      被误杀的全是**涨得最多的股票**（2014~2015 牛市）。对动量因子这是最坏方向的截尾：
      2015-06 泡沫顶点那些天，momentum_250 的日均截面从 ~2100 掉到 76（只剩银行、石化）。

    本实现的构造：`r_k = Π(1 + ret(1)_i) − 1`，用 `log1p` + `roll_sum`（NaN 毒化）。

      * **无单日异常时与 `ctx.ret(k)` 逐格相等**（实测最大差 3.6e-15，就是浮点误差）；
      * 任何**单日**异常毒化整个窗口 —— 这才是 prices.py 注释想表达的口径；
      * 停牌语义完全不变（停牌日 `ret(1)=0`，即 1 倍，复牌跳空自然计入）；
      * `ret(1)` 已把 `|r|>60%` 置 NaN，所以 `log1p` 的入参恒在 [0.4, 1.6]，不会出 ±inf。

    ★★ 另一处必须自己挡的坑：**窗口内一次都没成交**的格子。

      `ctx.ret(1)` 对停牌日返回的是 **0.0 而不是 NaN**（实测：`fea/prices.py:ret()` 走
      `hfq_close`，而价格水平是**前向填充**量 → 停牌期间 `px(T) == px(T−1)` → 收益恰好 0）。
      于是一只**整个窗口都没交易**的股票会得到「动量 = 0」这个**凭空造出来的值**，
      而不是「不知道」：

        实测 000023.SZ / 000033.SZ 在 2015-12-21：20 日内成交天数 = 0，
        20 日 hfq 收盘 80.4466 → 80.4466（全常数），momentum_20 = 0.000000。

      这个 0.0 会**混进截面**（正好落在动量分布的正中间，排名既不前也不后），
      比缺失更危险 —— 契约 §2 的「补 0 会让滚动均值、截面排名全部失真」说的就是它。
      更麻烦的是**引擎两条路径会给出不同结果**（全量重建 = 0.0；增量尾部重算时面板起点
      在最后成交日之后，前向填充没有种子 → NaN → 整行被丢掉），同一格因子值取决于
      走哪条路径，这破坏了「同输入同输出」。

      处置：窗口内成交天数 == 0 ⇒ NaN（本函数与家族内其它价格窗口因子统一用
      `_zero_trade_mask`）。**部分停牌不挡**（如 000048.SZ 20 日内成交 10 天，
      仍按价格层的前向填充语义给值）—— 那只影响新鲜度，不影响可算性。

    价格层把阈值改成 k 相关（或按单日判定）、并让停牌日的 `ret()` 返回 NaN 之后，
    本函数可直接换回 `ctx.ret(k)`（`_zero_trade_mask` 届时也可一并去掉）。
    """
    r1 = ctx.ret(1)
    r = np.expm1(ctx.roll_sum(np.log1p(r1), int(k)))
    return np.where(_basis_no_trade(ctx, k), np.nan, r)

def _basis_no_trade(ctx, n: int) -> np.ndarray:
    """窗口 `n` 个交易日内**一次都没成交**的格子 → True。

    这类格子上的价格全是前向填充出来的陈旧值，任何由它算出的收益/位置都是
    「0 或常数」的假值（见 `_ret_k` 的 ★★）。`traded()` 是 bool（`vol` 有限），
    没有 NaN，所以 `roll_sum` 不会毒化。
    """
    tr = np.asarray(ctx.traded(), dtype=np.float64)
    return ctx.roll_sum(tr, int(n)) <= 0.0

def _basis_twenty_day_return(ctx):
    return _basis_compound_return(ctx, 20)

"""Pure parent-value reconstruction prototype. No factor registration or writes."""
import numpy as np,gc
from fea.engine import Engine,minus_days
from fea.context import FactorContext
from fea.panel import Panel
def _private_basis_values(ctx,parent_spec):
 cache=ctx.__dict__.setdefault('_private_coupling_basis_cache',{})
 if parent_spec.name in cache:return cache[parent_spec.name]
 child=ctx.panel;out=np.full(child.shape,np.nan,dtype=np.float32)
 for year in np.unique(child.dates//10000):
  days=child.dates[(child.dates//10000==year)&(child.dates>=parent_spec.start_int(ctx.cfg))]
  if not len(days):continue
  engine=Engine(ctx.cfg);assert np.array_equal(engine.codes,child.codes)
  anchor=engine._anchor_for(parent_spec,days);wlo=minus_days(anchor,parent_spec.warmup_days)
  pdays=engine.cal.between(wlo,int(days[-1]));panel=Panel(pdays,engine.codes);uni=engine.universe_for(panel)
  financial=any(d in ('stock_income','stock_balancesheet','stock_cashflow','stock_financial_indicator') for d in parent_spec.deps)
  fields=(frozenset(parent_spec.fin_fields) if parent_spec.fin_fields else None) if financial else frozenset()
  parent_ctx=FactorContext(panel,engine.deriv_for(wlo,int(days[-1]),fields=fields),engine.up,ctx.cfg,uni,prices=engine.prices_for(wlo,int(days[-1])),cal=engine.cal)
  raw=np.asarray(parent_spec.fn(parent_ctx),dtype=np.float64)
  assert raw.shape==panel.shape and not np.isinf(raw).any()
  assert parent_ctx.accessed<=set(parent_spec.deps)
  rows=panel.day_mask(days)
  value=np.where(uni[rows],raw[rows],np.nan).astype(np.float32)
  out[child.rows_of(days)]=value
  ctx.accessed.update(parent_spec.deps)
  del engine,parent_ctx,panel,uni,raw,value;gc.collect()
 cache[parent_spec.name]=out
 return out



_PRIVATE_EQUITY = FactorSpec(name='cp_internal_book_value', group='internal', deps=('stock_balancesheet', 'stock_daily', 'stock_finance'), start=None, warmup_days=700, fin_fields=(), enabled=False, fn=_basis_equity_value)

_PRIVATE_RETURN = FactorSpec(name='cp_internal_short_return', group='internal', deps=('stock_daily', 'stock_adj_factor'), start=None, warmup_days=56, fin_fields=(), enabled=False, fn=_basis_twenty_day_return)
