"""成长因子（16 个）—— 跨期比较**全部在报告期粒度**上完成。

## 本家族唯一的纪律：所有「跨期」都用 `lag_ttm(f, k)` / `point(f, lag=k)`

`k` 一律是**报告期个数**（一个报告期 = 一个财报季度），不是交易日个数。

    ctx.ttm(f)            = TTM @ 当日已知的最新报告期 p
    ctx.lag_ttm(f, k)     = TTM @ 报告期 (p − k)，取「as-of 当日已披露」的那一版
    ctx.point(f, lag=k)   = 时点科目（资产负债表）@ 报告期 (p − k)

**绝不对日频序列做 `.shift(252)` / `.shift(63)`。** 参考库（`学习资料/因子库.md`
的 Growth 章节、`学习资料/factors.md`）写的都是日频口径（`Value_t / Value_{t-252} - 1`、
`GM_t / GM_{t-63} - 1`）。在日频上做这个位移，一年里有 3/4 的时间分子分母**落在同一个
报告期**（因子塌成 0），剩下 1/4 的时间**横跨报告期边界**（比较两个不同的财季），
而且分母一年变 4 次 —— 全都是静默的，看起来只像「因子有点噪」。
本文件把所有 `t-252` 一律翻译成「4 个报告期」、`t-63`（环比）一律翻译成「1 个报告期」。

## 口径总纲

1. **同比（yoy）= lag 4 个报告期**；**环比（qoq）= lag 1 个报告期**（相邻两个报告期）。
2. **加速度（accel）= 本期同比 − 上期同比**：
   `ttm(t)/ttm(t−4) − ttm(t−1)/ttm(t−5)`，需要 lag 5 可达（版本表回看到数据起点，
   早期为 NaN 是正常的，会落在 warmup 区）。
3. **分母一律 `ctx.safe_div(..., min_abs_den=1e-6)` 且取 `|基期|`**：净利润 / 经营
   现金流 / 营业利润都可能为负或近零，直接相除会产出 ±1e6 的假值（与
   `fundamental.py` 的 `yoy_net_profit` 同口径：亏损收窄记为正增长）。
4. **CAGR 只对「两端都 > 0」的样本有定义**，其余置 NaN（见 `revenue_cagr_3y` 的 note）。
5. 「TTM 同比」与「TTM 环比」是**两个不同的量**，不是同一个东西缩放：
   `TTM(q) − TTM(q−1) = 单季(q) − 单季(q−4)`（中间三项相消），所以 TTM 环比恒等于
   「最新单季的同比增量 / 上年 TTM」。两者互补，都保留。
6. 财务字段名一律走 `ctx.ttm` / `ctx.point` 的白名单（见 `fea/deriv.py`）。

参考库出处：`学习资料/因子库.md` §2 Growth（15 个）+ `学习资料/factors.md`。
本文件只写公式，TTM / 单季拆分 / 追溯修正 / PIT 对齐全部由 `fea/deriv.py` 统一处理。
"""

from __future__ import annotations

import numpy as np

from fea.spec import FactorSpec, register

# ★ 起点留 None = 跟随 conf/config.yaml 的 default_start（当前 2012-01-01）
GROWTH_START = None
# ★ 需要 ≥ 11 个报告期历史的那三个因子（growth_stability / *_cagr_3y）必须显式写起点：
#   上游三张财报表最早只到 **2010Q1**（实测 stock_income/balancesheet/cashflow 的
#   最老年份分区 = 2010），而三行式 TTM 最早只能算到第 83 期（2010FY，靠
#   `_ttm_rolling4` 兜底路径），所以：
#       lag_ttm(f, 11) 最早在「当期报告期 >= 94（2013Q3）」可达
#       lag_ttm(f, 12) 最早在「当期报告期 >= 95（2013Q4）」可达
#   实测（2010-2016 全量面板）：growth_stability 首个有效日 2013-10-11、
#   两个 CAGR 首个有效日 2014-01-22。
#   不写起点的话 2012 一整年、CAGR 连 2013 一整年都是全 NaN ——
#   而引擎目前在「某年整年全 NaN」时会直接崩（`_to_frame` 把全 NaN 行丢光 →
#   空表 → `upsert_year` 返回 object dtype 的空 DataFrame → `np.isfinite` 抛
#   "ufunc 'isfinite' not supported"，见 fea/engine.py 的收尾行）。
#   取 2014-01-01：这是「上游数据允许」与「不产出整年空分区」的交点。
DEEP_LAG_START = "2014-01-01"
# 财务 / TTM / 同比：4 季 TTM + 4 季同比 + ann_date 最长滞后 15 个月 → 700（契约 §2）
FIN_WARMUP = 700

# 上游财务字段
NP = "n_income_attr_p"              # 归母净利润（累计制 → TTM）
REV = "revenue"                     # 营业收入（累计制 → TTM）
COST = "oper_cost"                  # 营业成本（累计制 → TTM）
OP = "operate_profit"               # 营业利润（累计制 → TTM）
OCF = "n_cashflow_act"              # 经营活动现金流净额（累计制 → TTM）
EQ = "total_hldr_eqy_exc_min_int"   # 归母股东权益（时点）
TA = "total_assets"                 # 总资产（时点）
SHARE = "total_share"               # 总股本（时点）

# 上游数据集
DEP_INC = ("stock_income",)
DEP_BS = ("stock_balancesheet",)
DEP_CF = ("stock_cashflow",)
DEP_SF = ("stock_finance",)         # 供应商日频快照：pe_ttm / dv_ttm（反推分红率用）


# ══════════════════════════════════════════════════════════════════════
# 取数助手
# ══════════════════════════════════════════════════════════════════════

def _f64(x) -> np.ndarray:
    """统一转 float64 再算 —— float32 上做「大数相减」会丢有效位。"""
    return np.asarray(x, dtype=np.float64)


def _yoy(ctx, cur, prev, floor: float = 1e-6) -> np.ndarray:
    """同比 = (本期 − 基期) / |基期|。

    分母取**绝对值**：净利润 / 经营现金流 / 营业利润都可能为负，
    「亏损收窄」应记成正增长（与 `fundamental.yoy_net_profit` 同口径）。
    `min_abs_den` 挡住近零分母（壳公司的营收、经营性现金流为零的年份）。
    """
    c = _f64(cur)
    p = _f64(prev)
    return _f64(ctx.safe_div(c - p, np.abs(p), min_abs_den=floor))


def _yoy_of_ttm(ctx, field: str, k: int = 0) -> np.ndarray:
    """报告期 (p − k) 上的 TTM 同比：`TTM_{p−k} / TTM_{p−k−4} − 1`。

    ★ `k` 是**报告期**个数（lag_ttm 内部按 period 做 as-of），不是交易日。
    k=0 即当期同比；k=1 即「上期同比」（用于加速度）。
    """
    return _yoy(ctx, ctx.lag_ttm(field, k), ctx.lag_ttm(field, k + 4))


def _cagr(ctx, cur, prev, years: int = 3, floor: float = 1e-6) -> np.ndarray:
    """复合增速 `(本期 / 基期)^(1/years) − 1`，**两端都 > 0 才算**。

    · 基期为负 / 为零 → NaN：`(x/y)^(1/n)` 在 x/y < 0 时无定义（numpy 给 NaN），
      而且「亏损 → 盈利」这种样本的「增速」没有可解释的量纲
      （分母的符号会让比值的符号反转：−1亿 → +0.5亿 会算出 2 倍「增长」）。
    · 严格 > 0（不是 ≥ 0）：营收为 0 的壳公司一旦有了一点收入，
      `(x/0)^(1/3)` 会炸成任意大的值。
    · 代价：CAGR 因子在亏损股上系统性缺失（置 NaN 而非编造），
      截面因此偏向盈利样本 —— 这是**有意的**，见各因子的 note。
    """
    c = _f64(cur)
    p = _f64(prev)
    r = _f64(ctx.safe_div(c, p, min_abs_den=floor))
    r = np.where((c > 0) & (p > 0), r, np.nan)
    with np.errstate(invalid="ignore"):
        return np.power(r, 1.0 / float(years)) - 1.0


def _roe_at(ctx, k: int) -> np.ndarray:
    """报告期 (p − k) 上的 ROE(TTM) = 归母净利TTM / 期末归母权益。

    与 `fundamental.roe_ttm` 完全同口径（期末权益，不做平均），
    这样 `roe_trend` 与既有的 `roe_ttm` / `yoy_roe` 三个因子是可加的。
    """
    return _f64(ctx.safe_div(ctx.lag_ttm(NP, k), ctx.point(EQ, lag=k)))


def _vendor(ctx, field: str) -> np.ndarray:
    """供应商 `stock_finance` 的日频字段 → (T,C)，按 (code, trade_date) as-of 前向填充。

    与 `valuation._sf` 同一手法：`dv_ttm` / `pe_ttm` 是**状态量**
    （停牌期间「最后一个已知估值」仍然成立），用 as-of 前向填充而不是补 0。
    ★ 面板头部（warmup 区）需要面板起点**之前**的最近一条记录，
      否则上市早、窗口头停牌久的股票会凭空变 NaN，增量与全量就不一致。
    """
    d0 = int(ctx.panel.dates[0]) // 10000
    d1 = int(ctx.panel.dates[-1]) // 10000
    df = ctx.dataset("stock_finance",
                     columns=["stock_code", "trade_date", "pe_ttm", "dv_ttm"],
                     years=(d0 - 1, d1))
    if df.empty:
        return np.full(ctx.panel.shape, np.nan, dtype=np.float64)
    return _f64(ctx.asof_daily(df["stock_code"].to_numpy(),
                               ctx.date_col(df["trade_date"]),
                               df[field].to_numpy(dtype=np.float64)))


# ══════════════════════════════════════════════════════════════════════
# 一、同比（YoY）—— 与上一年**同一个报告期**比
# ══════════════════════════════════════════════════════════════════════











# ══════════════════════════════════════════════════════════════════════
# 二、环比（QoQ）—— 相邻两个报告期
# ══════════════════════════════════════════════════════════════════════





# ══════════════════════════════════════════════════════════════════════
# 三、加速度（Accel）—— 本期同比 − 上期同比
# ══════════════════════════════════════════════════════════════════════





# ══════════════════════════════════════════════════════════════════════
# 四、可持续增长率（SGR）
# ══════════════════════════════════════════════════════════════════════



# ══════════════════════════════════════════════════════════════════════
# 五、3 年复合增速（CAGR）
# ══════════════════════════════════════════════════════════════════════

@register(FactorSpec(
    name="revenue_cagr_3y", group="growth", deps=DEP_INC,
    desc="3 年营业收入复合增速 = (营收TTM_t / 营收TTM_{t−12Q})^(1/3) − 1",
    formula="EPS_Growth_5Y = (BasicEPS_Y_t / BasicEPS_Y_{t-1260})^(1/5) - 1  "
            "（因子库.md #2 Growth 章节 PEG 系 pegh5 的复合增速式，逐字；"
            "本实现把 5 年/1260 天换成 3 年/12 个报告期，标的换成营业收入 TTM）",
    start=DEEP_LAG_START, warmup_days=FIN_WARMUP, higher_is_better=True,
    fin_fields=(REV,),
    note="★★ 起点显式写 2014-01-01（不是 default_start）：本因子要 lag_ttm(REV, 12)，"
         "上游财报最早只到 2010Q1、TTM 最早算到第 83 期（2010FY）→ lag 12 最早在"
         "当期报告期 >= 95（2013Q4）时可达，实测首个有效日 2014-01-22、"
         "2014-04 起覆盖稳定。2012/2013 两整年会是全 NaN，"
         "而引擎目前在整年全 NaN 时会崩（见文件头 DEEP_LAG_START 的说明）。"
         "★★ 符号问题（本家族第二个大坑）：`(x_t/x_{t−k})^(1/k) − 1` 在 x 为负时无定义。"
         "本实现要求**两端都 > 0**，否则置 NaN。理由："
         "① 分母为负时比值的符号会反转（−1亿 → +0.5亿 会算出 +2 倍的「增长」）；"
         "② 严格 > 0 而不是 ≥ 0：营收近零的壳公司一旦有了一点收入，"
         "`(x/0)^(1/3)` 会炸成任意大的值。"
         "★ 亏损转盈的样本**置 NaN 而非保留**：CAGR 的定义是「几何平均」，"
         "它对符号翻转型样本没有可解释的含义；保留它们（例如取 |分母|）会把"
         "「扭亏」这个事件伪装成一个巨大的正增长，在截面上系统性高估困境反转股。"
         "代价是 CAGR 因子在亏损股上缺失（截面偏向盈利样本），这是**有意的**。"
         "★ 口径：12 个**报告期**（3 年），不是日频 t-756；12 期需要更深的版本表，"
         "由引擎的 `years_for_window(lag_years=3)` 覆盖。"
         "早期年份（数据起点后的头 3 年）会有一段 NaN，属正常。",
))
def revenue_cagr_3y(ctx):
    return _cagr(ctx, ctx.ttm(REV), ctx.lag_ttm(REV, 12), years=3)




# ══════════════════════════════════════════════════════════════════════
# 六、增速的稳定性 与 利润率变化
# ══════════════════════════════════════════════════════════════════════

@register(FactorSpec(
    name="growth_stability", group="growth", deps=DEP_INC,
    desc="营收增速的稳定性 = 近 8 个报告期同比增速的均值 / 标准差",
    formula="Stability = mean(YoY_{t-i}, i=0..7) / std(YoY_{t-i}, i=0..7) , "
            "YoY_{t-i} = Revenue_TTM_{t-i} / Revenue_TTM_{t-i-4} - 1",
    start=DEEP_LAG_START, warmup_days=FIN_WARMUP, higher_is_better=True,
    fin_fields=(REV,),
    note="★★ 实现要点：8 期同比**全部由 lag_ttm 在报告期粒度上构造**（lag 0..11："
         "第 i 期同比 = lag_ttm(REV,i)/lag_ttm(REV,i+4)−1），再逐格做 mean/std。"
         "**没有**在日频上做 rolling —— 日频 rolling 会把同一个报告期的值重复计入"
         "200 多次，而且窗口边界横跨报告期，得到的既不是「8 期」也不是任何可解释的量。"
         "★ 起点显式写 2014-01-01（不是 default_start）：本因子要 lag_ttm(REV, 11)，"
         "而上游财报最早只到 2010Q1、TTM 最早算到第 83 期（2010FY）→ "
         "lag 11 最早在当期报告期 >= 94（2013Q3）时可达，实测首个有效日 2013-10-11、"
         "2014-03 起覆盖才稳定。2012 整年会是全 NaN，而引擎目前在整年全 NaN 时会崩"
         "（见文件头 DEEP_LAG_START 的说明）。"
         "★ 性质：这不是「纯稳定性」，而是成长的**信息比**（均值/标准差）——"
         "它同时奖励高增速与低波动；稳定下滑的公司（均值 −0.2、标准差 0.05）得 −4，"
         "排在最末。若要纯波动率口径请取 `-std`（本因子不含）。"
         "★ 标的选营业收入而非净利润：净利润同比在亏损时符号会翻转，"
         "均值与标准差都会被「正负跳变」主导，得到一个与经营无关的巨值。"
         "★ 分母 std 有 min_abs_den=1e-3 的地板：8 期增速高度一致时（真实存在的，"
         "如公用事业/高速公路）std 近零会让比值爆掉。"
         "★ 传播规则：8 期里任何一期为 NaN（早期数据不足），结果即 NaN"
         "（与 `roll_sum` 的 NaN 策略一致，不做 min_count 放松）。",
))
def growth_stability(ctx):
    ys = np.stack([_yoy_of_ttm(ctx, REV, k) for k in range(8)], axis=0)  # (8, T, C)
    mu = ys.mean(axis=0)
    sd = ys.std(axis=0, ddof=0)
    return ctx.safe_div(mu, sd, min_abs_den=1e-3)






# ══════════════════════════════════════════════════════════════════════
# 七、ROE 趋势（4 期斜率）
# ══════════════════════════════════════════════════════════════════════

