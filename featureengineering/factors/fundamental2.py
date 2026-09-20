"""财务口径补漏（12 个）—— 估值×质量 6 / 股息 1 / 季度变化 4 / 复合 1。

本文件补齐**其它财务家族没有覆盖**的口径：PEG 系（5 年增长 / 1 年增长）、
现金流与 EBITDA 口径的市值比、扣非口径、3 年股息均值、报告期环比的三个变化率、
以及一个截面合成的质量因子。

## 口径总纲（与 `DEVELOPING.md` §1 的六条硬约束一致）

1. **一律 `ctx.ttm()` / `ctx.point()`**，不直接读财报 parquet。
   累计制科目走 TTM、时点科目走 point，追溯修正 / `ann_date` 前向填充 / PIT 对齐
   全部由 `fea/deriv.py` 一次做完。
2. **市值一律自算**：`未复权 close × 当期已披露 total_share`。
   ★ 本文件按任务约定取 **`ctx.point("total_share")`（资产负债表口径）**，
   与 `valuation.py` / `quality.py` 用价格层 `ctx.px("total_share")` 不同 ——
   实测差异（2024-06-28 截面 vs 2024Q1 资产负债表，5331 只可比）：
   中位相对差 0.00，但 **8.55% 的股票差 >1%、5.70% 差 >20%**
   （日频快照 14.4% 大于报表值、5.8% 小于），即日频快照**更及时**。
   之所以仍取 point 口径：① 任务约定；② 与分母里同报告期的财务字段（EPS 的股本、
   权益）**同一版本**，PEG 的分子分母口径自洽。**下游若要统一，需主 Agent 拍板**
   （见汇报）。
3. **除法一律 `ctx.safe_div(..., min_abs_den=...)`**，货币口径地板 **100 万元**。
   上游财报单位是**元**（实测 2024FY `revenue` 中位 1.66e9）。
4. **增长率一律走报告期粒度**（`ctx.lag_ttm` / `ctx.point(lag=)`），
   **绝不**对日频序列 `.shift(252)` / `.shift(63)`。
   唯一的例外是 `roe_ttm_lag63d`（见其 note：它本身就是「63 个交易日前的水平量」，
   不是用日频位移去算增长率）。
5. **报告期环比（qoq）= lag 1 期**；本文件的三个 `delta_*` 都是 qoq（见其 note：
   与参考库的 `t-252` 有意的偏离）。
6. **`start=None`**（跟随 `conf/config.yaml: default_start`，当前 **2012-01-01**）。
   ✅ **2026-09-17 已处置**（放开全历史时）：`pegh5` / `etp5` 要 `lag_ttm(NP, 20/16)`，
   上证三表最早只到 2010Q1，所以全历史口径下它们早年会是**整年全 NaN** ——
   沙箱实测首个有效日 **2016-01-18** / **2015-06-16**，已按实测值显式写死起点
   （`FIN_START_DEEP20` / `FIN_START_ETP5`，见常量区的长说明）。
   其余因子仍留 `start=None` 跟随配置；本文件不动 `conf/**`。
7. **滚动均值一律用本文件的 `_roll_mean_causal`**，不用 `ctx.roll_mean`。
   `ctx.roll_mean` 内部为数值中心化先减**逐列全样本均值**（`fea/mathx.py`
   的 `_col_mean`，把**面板右端之后的行**也算进去），于是同一个 T 的值会随
   「面板右端还有多长」而变 —— `main.py audit-pit` 的截断复算实测在
   `dividend_yield_3y_avg` 上抓到 **1.644e-04 的相对差**（> 1e-4 容差），
   已在 `etp5` / `dividend_yield_3y_avg` 两处换成 float64 前缀和差分版本。
   ★ 框架级建议（`fea/**` 本文件无权改）：`_col_mean` 若要保留，应改用
   **因果**的中心化常数（如窗口起点之前的均值），否则任何用 `roll_*` 的因子
   都在严格意义上「T 的值依赖 T 之后的面板长度」。

## 参考库对照（`学习资料/因子库.md`）

| 本文件 | 参考库 | 说明 |
|:--|:--|:--|
| `pegh5` | §9 Value #2 `pegh5` | 公式逐字；EPS 序列改为「归母净利 TTM ÷ 当期股本」重算，见 note |
| `etp5` | §9 Value #3 `etp5` | 公式逐字（5 年滚动均值口径），见 note |
| `ncf_to_market` | §9 Value #4 `ncf_to_market` | 公式逐字（三项现金流之和 ÷ 市值） |
| `ebitda_to_market` | §9 Value #6 `ebitda_to_market` | 公式逐字；EBITDA 用白名单字段近似（实测差 −4.0%） |
| `earnings_cut_to_market` | §9 Value #9 `earnings_cut_to_market` | ★ **口径偏离**：上游无扣非水平值，用「归母 −（营业外净收支 + 公允价值变动）」近似，见 note |
| `peg_252d` | §2 Growth #1 `peg_252d` | 公式逐字；window=252 天 → 4 个报告期 |
| `dividend_yield_3y_avg` | §9 Value #1 `dividend_yield_3y_avg` | ★ **数据源偏离**：上游无分红明细，用 `dv_ttm` 反推每股分红，见 note |
| `delta_roe` | §5 Quality #51 `delta_roe` | ★ **有意偏离**：参考库 t−252（同比），本实现 t−1（环比），见 note |
| `delta_roa` | §5 Quality #33 `delta_roa` | 同上 |
| `delta_gpm` | §5 Quality #25 `delta_gpm` | 同上 |
| `roe_ttm_lag63d` | §5 Quality #1 `roe_ttm_lag63d` | 逐字（lag=63 个交易日） |
| `quality_composite` | §5 Quality #10 `quality_composite` | ★ **有意偏离**：参考库 6 项 AQR QMJ 之和，本实现 3 项**截面标准化**等权，见 note |

**本文件刻意不建的候选**（契约 §8：同号重复会被下游当成两个独立特征）：
- `ocf_to_market`：与既有 `valuation.cfp_ttm` **逐格同值**（都是 OCF_TTM / 总市值），
  只是命名不同 → 不建（`ncf_to_market` 是三项之和，才是新信息）。
- `earnings_to_price`：与既有 `valuation.ep_ttm` 同值 → 不建
  （`earnings_cut_to_market` 才是扣非口径的新信息）。
- `fcf_to_market` / `sales_to_market` / `book_to_market`：已由 `quality.py`
  （`fcf_to_market`）与 `valuation.py`（`sp_ttm` = sales_to_market、`bp` = book_to_market）持有。
- `delta_opm` / `delta_npm`：与既有 `growth.net_margin_change`（净利率同比差分）/
  `growth.gross_margin_change` 的信息高度重叠，且毛利率与净利率的环比变化
  在本文件已由 `delta_gpm` 代表 → 不建。
"""

from __future__ import annotations

import numpy as np

from fea.spec import FactorSpec, register

# ★ 起点留 None = 跟随 conf/config.yaml 的 default_start（当前 2026-01-01）
FIN_START = None
# 财务 / TTM / 同比：4 季 TTM + ann_date 最长滞后 15 个月 → 700（契约 §2）
FIN_WARMUP = 700
# ★ 需要**深版本表**的三个因子（见文件头「已知限制」）：
#   pegh5 要 lag_ttm(NP, 20)（5 年 = 20 个报告期），且三行式 TTM 还要再往前 4 期
#     → 版本表必须覆盖「当期报告期 − 24」，即 wlo 要落到 6 年前
#     （引擎按 `years_for_window(wlo, hi, lag_years=3)` 定版本表年份区间）。
DEEP20_WARMUP = 2600
#   etp5 要 lag_ttm(NP, 16) + **1260 个交易日**的市值滚动均值
#     → 面板本身就必须有 1260 个交易日（1260 × 1.8 + 20 ≈ 2288），
#       这个长度同时把版本表推到足够深，两个约束取大者。
ETP5_WARMUP = 2300
#   dividend_yield_3y_avg 要 735 个交易日（245 × 3）的 dv_ttm×price 滚动均值
#     → 735 × 1.8 + 20 ≈ 1343，留到 1400（面板头部还要覆盖窗口起点）
DIV_WARMUP = 1400
# ★★ 深回看的两个因子必须**显式写起点**（2026-09-17 放开全历史时实测）。
#   否则早年是**整年全 NaN** 的空分区：不会崩（2026-09-15「NaN 也落盘」之后引擎
#   已经不在全 NaN 年上崩了），但产出的是纯垃圾分区，`main.py check` 还会报「口径太稀」。
#   实测首个有效日（沙箱重算，default_start=2012-01-01，逐年扫非空率）：
#     pegh5 → 2016-01-18（2012~2015 四个整年全空，2016 非空 18.5%）
#     etp5  → 2015-06-16（2012~2014 三个整年全空，2015 非空 20.6%）
#   根因：两者要 `lag_ttm(NP, 20/16)`，而三表最早只到 2010Q1 —— 20 个报告期
#   要等到 2015Q4/2016Q1 的报告期才凑得齐。
#   ★ 起点取**实测的首个有效日**：再往前一天都必然全空，再往后就是白丢数据。
#   ★ 以后若再把 `default_start` 往前推（如 2010），仍以实测为准重测一遍。
FIN_START_DEEP20 = "2016-01-18"     # pegh5（lag_ttm 20 期）
FIN_START_ETP5 = "2015-06-16"       # etp5（lag_ttm 16 期 + 1260 交易日市值均值）
#   dividend_yield_3y_avg 要 **735 个交易日**的 dv_ttm×price 滚动均值，而 dv_ttm 来自
#   `stock_financial_indicator`（上游 2010 起）→ 首个有效日实测 **2013-01-11**；
#   2012 整年全 NaN（回填时由 `scripts/backfill_history.py` 的空分区监测抓到）。
DIV3Y_START = "2013-01-11"

# 货币口径的分母地板：**100 万元**（上游财报单位是元）
MIN_CUR = 1e6
# PEG 分母（增速 × EPS）的地板：低于此视为「零增长 / 零盈利」，比率无定义。
# 同时是**值域保护**：PEGH5 = P / (g×EPS)，地板 1e-3 时最坏值 ≈ 1500 / 1e-3 = 1.5e6
# （远低于引擎 1e8 的红线），不设地板时 g→0+ 会直接炸到 1e11。
PEG_FLOOR = 1e-3

# ---------------------------------------------------------------- 上游字段
# 累计制（-> TTM）
NP = "n_income_attr_p"            # 归母净利润
REV = "revenue"                   # 营业收入
COST = "oper_cost"                # 营业成本
OCF = "n_cashflow_act"            # 经营活动现金流净额
ICF = "n_cashflow_inv_act"        # 投资活动现金流净额
FFC = "n_cash_flows_fnc_act"      # 筹资活动现金流净额
EBIT = "ebit"                     # 息税前利润
DEPR = "depr_fa_coga_dpba"        # 固定资产折旧、油气资产折耗、生产性生物资产折旧
AMORT_I = "amort_intang_assets"   # 无形资产摊销
AMORT_L = "lt_amort_deferred_exp"  # 长期待摊费用摊销
NON_OP_IN = "non_oper_income"     # 营业外收入
NON_OP_EX = "non_oper_exp"        # 营业外支出
FV_CHG = "fv_value_chg_gain"      # 公允价值变动收益
# 时点制（资产负债表）
EQ = "total_hldr_eqy_exc_min_int"  # 归母股东权益
TA = "total_assets"               # 总资产
SHARE = "total_share"             # 总股本

# 上游依赖（用于输入水位失效判定）
DEP_I = ("stock_income",)
DEP_B = ("stock_balancesheet",)
DEP_C = ("stock_cashflow",)
DEP_IB = ("stock_income", "stock_balancesheet")
DEP_CB = ("stock_cashflow", "stock_balancesheet")
DEP_CIB = ("stock_cashflow", "stock_income", "stock_balancesheet")
# 价格层：close 取自 stock_daily，股本取自 stock_finance（价格层内部映射）
DEP_PX = ("stock_daily", "stock_finance")
DEP_SF = ("stock_finance",)

SF_COLS = ("stock_code", "trade_date", "dv_ttm")


# ══════════════════════════════════════════════════════════════════════
# 取数助手
# ══════════════════════════════════════════════════════════════════════

def _f64(x) -> np.ndarray:
    """统一转 float64 再算 —— float32 上做「大数相减」会丢有效位。"""
    return np.asarray(x, dtype=np.float64)


def _mktcap(ctx) -> np.ndarray:
    """总市值（元）= 未复权收盘价 × 当期已披露总股本。

    `close` 是状态量（停牌期间沿用最后一个成交价），`total_share` 取
    **资产负债表口径**（`ctx.point`，见文件头第 2 条）。停牌日市值是水平量，
    不是 NaN 也不是 0。
    """
    px = _f64(ctx.px("close"))
    sh = _f64(ctx.point(SHARE))
    return px * sh


def _eps_ttm(ctx) -> np.ndarray:
    """EPS(TTM) = 归母净利润TTM / 当期已披露总股本（元/股）。

    上游白名单没有 `basic_eps` 字段，故按定义从三表重算。
    地板 100 万元：TTM 净利低于此的公司 EPS 无意义（微利/盈亏平衡），
    返回 NaN 而不是一个近似 0 的分母。
    """
    return _f64(ctx.safe_div(ctx.ttm(NP), ctx.point(SHARE), min_abs_den=MIN_CUR))


def _yoy_ttm(ctx, field: str, k: int = 0, span: int = 4) -> np.ndarray:
    """报告期 (p − k) 上的 TTM 增长率：`TTM_{p−k} / TTM_{p−k−span} − 1`。

    ★ **两端都 > 0 才算**（与 `growth._cagr` 同一条纪律）：净利润为负时
    「增长率」会让比值的符号反转（−1 亿 → +0.5 亿会被算成 −150% 的「下降」），
    而 PEG 类因子对增长率的符号是敏感的（负增速必须判为无定义，不能当成「便宜」）。
    """
    cur = _f64(ctx.lag_ttm(field, k))
    prev = _f64(ctx.lag_ttm(field, k + span))
    r = _f64(ctx.safe_div(cur, prev, min_abs_den=1e-6))
    return np.where((cur > 0) & (prev > 0), r - 1.0, np.nan)


def _cagr_ttm(ctx, field: str, k: int, years: int) -> np.ndarray:
    """k 个报告期（= years 年）的复合增速 `(TTM_t / TTM_{t−k})^(1/years) − 1`。

    两端都 > 0 才有定义（理由同 `_yoy_ttm`）；严格 > 0 而不是 >= 0，
    避免「近零基数 → 任意大的复合增速」。
    """
    cur = _f64(ctx.ttm(field))
    prev = _f64(ctx.lag_ttm(field, k))
    r = _f64(ctx.safe_div(cur, prev, min_abs_den=1e-6))
    r = np.where((cur > 0) & (prev > 0), r, np.nan)
    with np.errstate(invalid="ignore"):
        return np.power(r, 1.0 / float(years)) - 1.0


def _roe_at(ctx, k: int = 0) -> np.ndarray:
    """报告期 (p − k) 上的 ROE(TTM) = 归母净利TTM / 期末归母权益。

    与 `fundamental.roe_ttm` 同定义（期末权益、不做平均），
    多一层 100 万元地板：近零净资产会把 ROE 炸成 ±1e5 量级。
    """
    return _f64(ctx.safe_div(ctx.lag_ttm(NP, k), ctx.point(EQ, lag=k), min_abs_den=MIN_CUR))


def _roll_mean_causal(mat, n: int) -> np.ndarray:
    """窗口 n 的滚动均值，**只用窗口内的数据**（没有任何全样本统计量）。

    ★ 为什么不用 `ctx.roll_mean`（这是实测踩出来的，不是洁癖）：
      `fea/mathx.py` 的 `roll_mean` 会先减**逐列全样本均值** `_col_mean(x)`
      做数值中心化 —— 那个均值是把**整个面板（含 T 之后的行）**平均出来的，
      于是同一个 T 的值会随「面板右端还有多长」而变。
      实测量级：`main.py audit-pit`（把面板截到 T 重算再逐格比对）在
      `dividend_yield_3y_avg` 上抓到 **1.644e-04 的相对差**（> 1e-4 容差）。
      数学上中心化常数会被抵消，但 float64 的抵消不精确，**近零取值的格子**
      相对误差会被放大 —— 这正是本文件两个「滚动均值」因子的取值区间
      （股息率 ~1e-2）最容易出现的形态。
    ★ 本实现：float64 前缀和差分，前缀只累积到窗口右端；窗口内出现 NaN 即 NaN
      （与 `ctx.roll_mean` 的默认 `min_count=None` 语义一致）。
      复算口径下 T 的值因此只由 `<= T` 的数据决定。
    """
    x = np.asarray(mat, dtype=np.float64)
    T = int(x.shape[0])
    n = int(n)
    out = np.full(x.shape, np.nan, dtype=np.float64)
    if n <= 0 or T < n:
        return out
    fin = np.isfinite(x)
    z = np.zeros((1, x.shape[1]), dtype=np.float64)
    cs = np.cumsum(np.where(fin, x, 0.0), axis=0)
    cn = np.cumsum(fin.astype(np.float64), axis=0)
    s = cs[n - 1:] - np.concatenate([z, cs[:-n]], axis=0)
    c = cn[n - 1:] - np.concatenate([z, cn[:-n]], axis=0)
    with np.errstate(all="ignore"):
        out[n - 1:] = np.where(c == float(n), s / float(n), np.nan)
    return out


def _roa_at(ctx, k: int = 0) -> np.ndarray:
    """报告期 (p − k) 上的 ROA(TTM) = 归母净利TTM / 期末总资产（同 `quality.roa_ttm`）。"""
    return _f64(ctx.safe_div(ctx.lag_ttm(NP, k), ctx.point(TA, lag=k), min_abs_den=MIN_CUR))


def _gpm_at(ctx, k: int = 0) -> np.ndarray:
    """报告期 (p − k) 上的毛利率(TTM) = (营收TTM − 营业成本TTM) / 营收TTM。

    `oper_cost` 在 deriv 层是 POSITIVE_ONLY（银行/券商的精确 0.0 → NaN），
    所以金融股的毛利率天然是 NaN —— 这是有意的（否则它们恒为 100%）。
    """
    rev = _f64(ctx.lag_ttm(REV, k))
    cost = _f64(ctx.lag_ttm(COST, k))
    return _f64(ctx.safe_div(rev - cost, rev, min_abs_den=MIN_CUR))


def _dv_ttm(ctx) -> np.ndarray:
    """供应商 `stock_finance.dv_ttm`（滚动股息率，**百分数**）→ (T,C)。

    与 `valuation.dp_ttm` 同源同手法（as-of 前向填充，不是补 0）：
    不分红公司的 dv_ttm 是**真实的 0**，必须保留（实测 2024 年 31.2% 的行 = 0）。
    多读一年（`d0 - 1`）：面板头部需要面板起点**之前**的最近一条记录，
    否则上市早、窗口头停牌的股票会凭空变成 NaN，增量与全量不一致。
    """
    d0 = int(ctx.panel.dates[0]) // 10000
    d1 = int(ctx.panel.dates[-1]) // 10000
    df = ctx.dataset("stock_finance", columns=list(SF_COLS), years=(d0 - 1, d1))
    if df.empty:
        return np.full(ctx.panel.shape, np.nan, dtype=np.float64)
    return _f64(ctx.asof_daily(df["stock_code"].to_numpy(),
                               ctx.date_col(df["trade_date"]),
                               df["dv_ttm"].to_numpy(dtype=np.float64)))


# ══════════════════════════════════════════════════════════════════════
# 一、估值 × 质量（6）
# ══════════════════════════════════════════════════════════════════════

@register(FactorSpec(
    name="pegh5", group="value", deps=DEP_IB + DEP_PX,
    desc="PEG 的 5 年增长版 = 未复权收盘价 / (5 年 EPS 复合增速 × EPS_TTM)",
    formula="1. EPS_Growth_5Y = (BasicEPS_Y_t / BasicEPS_Y_{t-1260})^(1/5) - 1\n"
            "2. PEGH5 = ClosePrice / (EPS_Growth_5Y * BasicEPS_TTM)\n"
            "   Factor = -CrossSectionalRank(PEGH5)",
    start=FIN_START_DEEP20, warmup_days=DEEP20_WARMUP, higher_is_better=False,
    fin_fields=(NP, SHARE),
    note="参考库 §9 Value #2 逐字，两处口径实现说明："
         "① **EPS 序列自算**（白名单无 basic_eps）：EPS_TTM = 归母净利TTM / 当期股本；"
         "5 年前的 EPS 用 `lag_ttm(NP, 20)`（20 个**报告期** = 5 年）配同一套股本口径 ——"
         "**增速因此等于归母净利 TTM 的 5 年复合增速**（股本是同一个乘数、在比值里相消）。"
         "这样做的理由：参考库的 `BasicEPS_Y` 序列不做复权，A 股高送转会把 EPS 名义值"
         "砍到 1/10，伪造出 −90% 的「EPS 崩塌」；用净利口径则送转/拆股完全无影响。"
         "**代价**：增发摊薄不体现在增速里（下游若要每股口径需另建因子）。"
         "② **符号与定义域**：增长率两端都 > 0 才算，且要求 **g5 > 0**（负增长时 PEG 无意义，"
         "参考库取 −rank(PEGH5) 会把「负增速 = 负 PEG」顶到「最便宜」的一端，方向完全反了）。"
         "分母（g5 × EPS）加 1e-3 地板，防 g5→0+ 炸出 1e11 量级的假值。"
         "③ **量纲**：按参考库字面公式，PEGH5 = PE_TTM / g5（g5 为小数），"
         "即比常见 PEG（PE / 增速百分数）**大 100 倍**，与同族 `peg_252d` 不同量纲 ——"
         "但两者都只出 rank，排序不受影响（见两个 note 的对照）。"
         "④ **实测口径代价**：要求 g5 > 0 会丢掉「5 年零增长/负增长」的样本（估值上它们是"
         "「贵」的一端，不是「便宜」的一端）——缺失与「成长性差」强相关，下游注意。"
         "⑤ 起点 = 实测首个有效日 **2016-01-18**（见 `FIN_START_DEEP20`）："
         "20 个报告期的回看要等到 2015Q4 的报告期才凑得齐，2012~2015 四个整年必然全空。",
))
def pegh5(ctx):
    g5 = _cagr_ttm(ctx, NP, 20, 5)
    g5 = np.where(g5 > 0, g5, np.nan)
    return ctx.safe_div(_f64(ctx.px("close")), g5 * _eps_ttm(ctx),
                        min_abs_den=PEG_FLOOR)


@register(FactorSpec(
    name="etp5", group="value", deps=DEP_I + DEP_PX,
    desc="五年平均净利润 / 五年平均市值（≈ 5 年平均盈利收益率）",
    formula="ETP5 = RollingMean(NetProfit_Y, 1260) / RollingMean(MarketCap, 1260)\n"
            "   Factor = CrossSectionalRank(ETP5)",
    start=FIN_START_ETP5, warmup_days=ETP5_WARMUP, higher_is_better=True,
    fin_fields=(NP, SHARE),
    note="参考库 §9 Value #3。「5 年滚动均值」在本框架的实现（两边都逐字）："
         "① **分子** = 5 个年度观测的均值 `mean(lag_ttm(NP, 0/4/8/12/16))`，"
         "即最近 5 个报告期的 TTM（每个都是滚动年度口径，含季节性已差分）；"
         "**5 期全有效**才算（NaN 传播，不做 min_count 放松）。"
         "② **分母** = **1260 个交易日**（5 年 × 252）的总市值滚动均值，严格窗口；"
         "用本文件的 `_roll_mean_causal`（而非 `ctx.roll_mean`）—— 后者的内部数值中心化"
         "用了全样本列均值，会让 T 的值随面板右端之后的行数变化（见该函数 docstring）。"
         "③ 两个约束合起来 ⇒ 本因子只覆盖「上市满 5 年 + 有 5 年连续财报」的公司，"
         "次新股系统性缺失（这是「5 年平均」的固有要求，不是 bug）。"
         "④ **量纲**：读数是「每 1 元市值对应多少元年均利润」，典型 0.02~0.10，"
         "与 `valuation.ep_ttm`（当期口径）互补：这个是 5 年平滑版，几乎不含单年噪声，"
         "但**对近两年的盈利变化反应极慢**（1/5 权重）。"
         "⑤ 分母用**市值的历史均值**而不是当期市值（参考库口径）："
         "分子分母都是 5 年平均，比值是「长期盈利 / 长期估值」，不是当期估值。"
         "⑥ 起点限制见文件头「已知限制」。",
))
def etp5(ctx):
    parts = [_f64(ctx.lag_ttm(NP, k)) for k in (0, 4, 8, 12, 16)]
    np5 = np.mean(np.stack(parts, axis=0), axis=0)          # 任一期为 NaN -> NaN
    mc5 = _roll_mean_causal(_mktcap(ctx), 1260)             # 见 _roll_mean_causal 的说明
    return ctx.safe_div(np5, mc5, min_abs_den=MIN_CUR)






@register(FactorSpec(
    name="ebitda_to_market", group="value",
    deps=DEP_CIB + DEP_PX,
    desc="EBITDA 市值比 = (EBIT + 折旧 + 无形资产摊销 + 长期待摊摊销)TTM / 总市值",
    formula="ebitda_to_market = EBITDA / (ClosePrice × TotalShares)",
    start=FIN_START, warmup_days=FIN_WARMUP, higher_is_better=True,
    fin_fields=(EBIT, DEPR, AMORT_I, AMORT_L, SHARE),
    note="参考库 §9 Value #6 的公式逐字，但 **EBITDA 用白名单字段近似**"
         "（上游 `stock_financial_indicator` 有 `ebitda` 水平值，但它是**累计 YTD**、"
         "且不在 `ctx.ind()` 的安全字段表里 —— 见汇报的「需要引擎扩字段」）：\n"
         "    EBITDA = ebit + depr_fa_coga_dpba + amort_intang_assets + lt_amort_deferred_exp\n"
         "★ **实测精度**（2024FY，4831 只可比）：与供应商 `ebitda` 的**中位相对差 −4.0%**，"
         "|差| < 5% 占 32.8%、< 20% 占 69.0% —— 自算值系统性**偏低**，因为白名单的折旧字段"
         "（固定资产折旧、油气资产折耗、生产性生物资产折旧）不含**使用权资产折旧 /"
         "投资性房地产折旧**等科目（2019 年新租赁准则后对零售、航空、餐饮影响较大）。"
         "所以本因子是「EBITDA 的下界近似」，量级与排序可用，绝对值不要与供应商口径混用。"
         "★ 四个分项在原始表里实测**非空率 100%**（2012/2016/2020/2024 年报均如此），"
         "「非零率」62%~99%（长期待摊摊销最常为 0，那是真实值）—— 故直接相加，"
         "**不做 nan→0**（缺失只可能来自报告期缺失，此时整格应为 NaN）。"
         "★ 与 `valuation.cfp_ttm` 的分工：EBITDA 加回了折旧摊销（非现金），"
         "比经营现金流更贴近「经营性盈利能力」，且不受营运资本变动的影响。",
))
def ebitda_to_market(ctx):
    eb = ctx.ttm(EBIT) + ctx.ttm(DEPR) + ctx.ttm(AMORT_I) + ctx.ttm(AMORT_L)
    return ctx.safe_div(eb, _mktcap(ctx), min_abs_den=MIN_CUR)




# ══════════════════════════════════════════════════════════════════════
# 二、股息（1）
# ══════════════════════════════════════════════════════════════════════

@register(FactorSpec(
    name="dividend_yield_3y_avg", group="value", deps=DEP_PX,
    desc="3 年平均股息率 = 近 735 个交易日的（每股分红 ÷ 当期价）均值 / 当前价（%）",
    formula="dividend_yield_3y_avg = (SUM(ActualCashDiviRMB, 735) / 3) / ClosePrice\n"
            "   = AVG(ActualCashDiviRMB, 3年) / ClosePrice",
    start=DIV3Y_START, warmup_days=DIV_WARMUP, higher_is_better=True,
    fin_fields=(SHARE,),
    note="★★ **数据源偏离**：上游**没有分红明细表**（`ActualCashDiviRMB` 无从取得），"
         "唯一的股息来源是供应商日频快照 `stock_finance.dv_ttm`（滚动 12 个月股息率，"
         "百分数；实测 2024 年非空率 100%、其中 31.2% 是精确的 0 = 不分红，真实值）。"
         "**重建口径**（逐字对齐参考库，而不是简单地对 dv_ttm 求均值）：\n"
         "    每股分红_TTM(t) = dv_ttm(t)/100 × ClosePrice(t)      ← 由定义反推\n"
         "    dividend_yield_3y_avg(T) = mean_{t∈735}(每股分红_TTM(t)) / ClosePrice(T)\n"
         "即「3 年的平均每股分红 ÷ 今天的股价」，与参考库公式一致。"
         "★ 为什么不直接 `mean(dv_ttm, 735)`（更省事、但**不等价**）："
         "那个量是「3 年里各时点股息率的均值」，分母是**当时的股价**；"
         "参考库的分母是**今天的股价**。在上涨行情里二者会系统性分叉"
         "（朴素版高估、且随行情漂移）。本实现用同一个分母，是真正的「用今天价格衡量的"
         "历史分红水平」。分子里 dv_ttm(t)×P(t) 的 P(t) 用**未复权价**"
         "（分红/送转导致的除权跳空正是每股分红的来源，用后复权价会把它抹掉）。"
         "★ 窗口 735 = 参考库的 245×3（交易日），严格窗口：**上市不满 3 年**的公司为 NaN，"
         "这是「3 年平均」的固有要求。"
         "★ 停牌日 close 是前向填充的状态量，所以停牌不产生缺口、也不产生假值。"
         "★ **窗口均值用本文件的 `_roll_mean_causal` 而不是 `ctx.roll_mean`**："
         "`audit-pit` 的截断复算在本因子上抓到 1.644e-04 的相对差（> 1e-4 容差），"
         "根因是 `ctx.roll_mean` 内部的数值中心化用了**全样本列均值**"
         "（`fea/mathx.py` 的 `_col_mean`），会让 T 的值随面板右端之后的行数变化。"
         "换成 float64 前缀和差分后，T 的值只由 `<= T` 的数据决定（见该函数 docstring）。"
         "★ `fin_fields` 说明：本因子**不读任何财报表字段**，声明 `total_share` 只是为了让"
         "引擎走「按字段子集建表」的快路径 —— 引擎约定空元组 = **全字段全建**"
         "（见 `fea/engine.py` 的 `frozenset(s.fin_fields) or None`），"
         "那会让同一次 run 退化成 80 字段版本表、且每个任务的字段子集都触发重建。",
))
def dividend_yield_3y_avg(ctx):
    px = _f64(ctx.px("close"))
    dps = _dv_ttm(ctx) * px / 100.0          # 每股分红（TTM 口径，元/股）
    return ctx.safe_div(_roll_mean_causal(dps, 735), px, min_abs_den=1e-6)


# ══════════════════════════════════════════════════════════════════════
# 三、季度变化族（4）—— 全部在**报告期粒度**上做
# ══════════════════════════════════════════════════════════════════════







@register(FactorSpec(
    name="roe_ttm_lag63d", group="quality", deps=DEP_IB,
    desc="63 个交易日（约一季度）前的 ROE(TTM) —— 刻画「季报之间的漂移」",
    formula="ROE = NetProfit_Parent / TotalEquity\n"
            "   Factor = ROE_TTM(T − 63 个交易日)   # 参考库 lag 参数默认 0",
    start=FIN_START, warmup_days=FIN_WARMUP, higher_is_better=True,
    fin_fields=(NP, EQ),
    note="参考库 §5 Quality #1 `roe_ttm_lag63d`（其 `lag` 参数是**交易日**数）。"
         "★ **为什么这里可以、也必须用日频 shift**（与文件头第 4 条不矛盾）："
         "本因子返回的是「63 个交易日前的 ROE **水平量**」，不是用日频位移去算**增长率**。"
         "ROE(TTM) 是按 `ann_date` 前向填充的**阶梯函数**（一年只跳 4 次），"
         "所以 `shift(63)` 取到的就是「约一个季度前那一版财报算出的 ROE」。"
         "★ 用途（下游怎么用）：它与 `roe_ttm` 构成一对 ——"
         "**`roe_ttm − roe_ttm_lag63d` 就是「一份新财报带来的 ROE 漂移」**，"
         "是个事件式的边际量；直接做季度环比差分在这里是做不到的，"
         "因为新的季度值出现的确切日期（ann_date）不定，而 shift(63) 是固定窗口。"
         "★ **与 `roe_ttm` 高度相关**（同一条阶梯序列平移 63 个交易日，"
         "一年里约一半时间取到的是同一个报告期的值）—— 下游建模时"
         "**不要**把它当独立因子与 `roe_ttm` 并列，它的价值在「做差」。"
         "★ 口径：与 `roe_ttm` 完全一致（归母净利 TTM / 期末归母权益），"
         "多一层 100 万元净资产地板（近零净资产会把 ROE 炸到 ±1e5）。",
))
def roe_ttm_lag63d(ctx):
    return ctx.shift(_roe_at(ctx, 0), 63)


# ══════════════════════════════════════════════════════════════════════
# 四、复合（1）
# ══════════════════════════════════════════════════════════════════════

@register(FactorSpec(
    name="quality_composite", group="quality", deps=DEP_CIB,
    desc="质量复合 = ROE(TTM)、毛利率(TTM)、经营现金流/总资产 三项截面 z 分数等权均值",
    formula="Ratio_i = ① NetProfit/Equity ② (Revenue−Cost)/Revenue ③ OCF/TotalAssets\n"
            "Factor = mean_i( cs_zscore(Ratio_i) )   # 至少 2 项有效才合成",
    start=FIN_START, warmup_days=FIN_WARMUP, higher_is_better=True,
    fin_fields=(NP, EQ, REV, COST, OCF, TA),
    note="★ **与参考库的有意偏离**：参考库 §5 Quality #10 是 AQR QMJ 的"
         "「**6 项比率直接相加**」，本实现是「**3 项比率截面标准化后等权**」。两条理由："
         "① 直接相加**量纲不可比**（ROE ~0.1、毛利率 ~0.3、(OCF+ICF)/总资产 ~0.05，"
         "相加等于给毛利率 3 倍权重）；"
         "② 参考库第 6 项 `NetProfit / (OpCashInflow + InvCashInflow)` 的分母可以为负/近零"
         "（参考库自己要用 `filter=True` 挡 ±inf），在契约 §3.4「除法一律带保护」下"
         "会被 NaN 掉大半样本。"
         "★ **合成口径**：每项先 `ctx.cs_zscore(x, mask=ctx.universe)`（**逐交易日的截面**"
         "标准化，只用当日主板股票，**绝不使用全样本统计量** —— 那会引入未来信息），"
         "再等权平均。三项是：**盈利能力**（ROE_TTM）、**定价能力**（毛利率 TTM）、"
         "**现金创造**（经营现金流 TTM / 期末总资产）—— 分别对应利润表、"
         "利润表的毛利率结构、现金流量表，三张表各取一个，刻意不重叠。"
         "★ **缺失处理**：允许**至少 2 项有效**（`nanmean`），此时用有效项的均值。"
         "理由：银行/券商没有可比的 `oper_cost`（毛利率恒为 NaN），若要求 3 项全有，"
         "整个金融板块会被排除；要求 2 项则它们由 ROE + 现金流质量代表。"
         "只有 1 项有效的格子返回 NaN（**不给单因子冒充复合因子**）。"
         "★ **与既有因子的关系（重要）**：本因子是三个已落盘因子"
         "（`roe_ttm`、`gpm_ttm`、`ocf_to_asset` 的等价物）的线性组合，"
         "**与 `roe_ttm` 的相关性最高**（ROE 的截面分散度最大）。"
         "它的价值是「一次拿到一个已经中性化的质量分」，下游做因子筛选时"
         "**不应**再同时保留全部成分因子（共线）。",
))
def quality_composite(ctx):
    ocf_to_asset = ctx.safe_div(ctx.ttm(OCF), ctx.point(TA), min_abs_den=MIN_CUR)
    zs = np.stack([_f64(ctx.cs_zscore(_roe_at(ctx, 0), mask=ctx.universe)),
                   _f64(ctx.cs_zscore(_gpm_at(ctx, 0), mask=ctx.universe)),
                   _f64(ctx.cs_zscore(ocf_to_asset, mask=ctx.universe))], axis=0)
    ok = np.isfinite(zs)
    n = ok.sum(axis=0)
    tot = np.where(ok, zs, 0.0).sum(axis=0)
    return np.where(n >= 2, tot / np.maximum(n, 1), np.nan)
