"""财务衍生层 —— 单季 / TTM / 追溯修正 / PIT 对齐，**全部只在这里做一次**。

因子函数不该碰财报细节，只该写 `np_ttm / equity` 这种一行公式。
所有实测到的坑都收敛在本文件：

1. **财报是累计 YTD**（实测茅台 2025：Q1=514亿 / H1=911亿 / 9M=1309亿 / FY=1721亿）。
   直接把 YTD 当季度值是最常见的错误。TTM 用**三行式**：

       TTM@q(y) = cum_q(y) + cum_FY(y-1) - cum_q(y-1)

   它比「拆单季再滚 4 期」需要更少行（3 行 vs 4 行），
   中间季度缺失也不影响，且 **Q4 时恒等于当年累计**（因为 B、C 是同一行，
   相减为 0）—— 这是个可以直接写成断言的不变量。实测可用率 90.4% vs 83.2%。

2. **`oper_cost` 对银行/券商是精确的 0.0，不是 NaN**（实测浦发 2025 全年 = 0.0）。
   不设防的话 `gpm_ttm = (revenue-0)/revenue = 100%`，所有金融股会挤在
   毛利率截面的最顶端 —— 一个又大又系统性的错误下注。故 `oper_cost<=0 -> NaN`。

3. **`revenue` != `total_revenue`**（实测茅台 FY2025：1688.4亿 vs 1720.5亿，
   差的是利息收入）。灵启文档写的是「营业收入」，所以统一用 `revenue`，
   两个因子混用会在最大的权重股上制造 ~2% 的虚假差异。

4. **`ann_date` 可以滞后一年多**（实测：2024-12-31 的报告 2026-03-20 才公告），
   且 `f_ann_date < ann_date` 是存在的（会构成前视）。故取
   `pit = max(ann_date, f_ann_date, end_date)` —— 保守、绝不前视。

5. **追溯修正**：同一 `(code, end_date)` 可能有多次公告（实测 505 组）。
   **不按 ann_date 去重**，而是保留所有版本，as-of 时取
   「pit <= 当日 的最新一版」—— 这样修正公告之前的历史区间仍保留原始披露值。
   `current_vintages()` 再用「前缀最大报告期」挑出真正「当日已知的最新报告期」，
   避免一份迟到的旧期修正把新期数据顶掉。
"""

from __future__ import annotations

from .field_expansion import ANNUAL_ALIASES, ANNUAL_SOURCES

import logging

import numpy as np
import pandas as pd

from .dates import series_to_int

log = logging.getLogger("fea.deriv")

PERIOD_BASE = 1990
PERIOD_W = 1000          # period 取值 ~0..200，留足余量
KEY_W = 100_000_000      # pit 是 YYYYMMDD < 1e8

# 累计制（YTD）字段 -> 做 TTM
TTM_SOURCES = {
    "stock_income": [
        # —— 原有 4 个
        "revenue", "oper_cost", "n_income_attr_p", "operate_profit",
        # —— 2026-09-14 扩展：覆盖质量/成长/费用率/现金流类因子
        "total_profit", "n_income", "sell_exp", "admin_exp", "fin_exp",
        "rd_exp", "assets_impair_loss", "invest_income", "non_oper_income",
        "non_oper_exp", "income_tax", "minority_gain", "total_cogs",
        "fin_exp_int_exp", "fv_value_chg_gain", "ebit",
    ],
    "stock_cashflow": [
        "n_cashflow_act",
        # —— 扩展
        "n_cashflow_inv_act", "n_cash_flows_fnc_act", "c_fr_sale_sg",
        "c_paid_goods_s", "c_paid_to_for_empl", "c_paid_for_taxes",
        "c_pay_acq_const_fiolta",      # ★ 资本开支
        "depr_fa_coga_dpba",           # ★ 折旧
        "amort_intang_assets", "lt_amort_deferred_exp", "credit_impa_loss",
        "prov_depr_assets", "free_cashflow", "c_pay_dist_dpcp_int_exp",
        "decr_inventories", "invest_loss", "n_incr_cash_cash_equ",
        "recp_tax_rends", "c_recp_borrow", "c_prepay_amt_borr",
    ],
}
# 时点字段 -> 直接用最新披露值
POINT_SOURCES = {
    "stock_balancesheet": [
        "total_assets", "total_liab", "total_hldr_eqy_exc_min_int",
        "defer_tax_assets", "total_share",
        # —— 扩展
        "total_hldr_eqy_inc_min_int", "minority_int", "money_cap",
        "accounts_receiv", "notes_receiv", "accounts_receiv_bill",
        "inventories", "prepayment", "oth_receiv", "contract_assets",
        "total_cur_assets", "total_cur_liab", "total_nca", "total_ncl",
        "fix_assets", "cip", "cip_total", "goodwill", "intan_assets",
        "r_and_d", "lt_borr", "st_borr", "non_cur_liab_due_1y",
        "bond_payable", "acct_payable", "notes_payable", "accounts_pay",
        "adv_receipts", "contract_liab", "defer_tax_liab",
        "defer_inc_non_cur_liab", "treasury_share", "trad_asset",
    ],
}
# 必须为正才有意义的字段（<=0 视为缺失）
#
# ⚠️ **刻意不扩**：`inventories` / `goodwill` / `rd_exp` / `r_and_d` / `money_cap` /
#    `accounts_receiv` / `cip` / `fix_assets` 这些字段的「0」通常是**真实值**，
#    一律 `<=0 → NaN` 会把存货周转率 / 商誉占比 / 研发强度整族因子毁掉。
#    这些字段的**分母保护**请在因子层用 `ctx.safe_div(..., min_abs_den=...)` 做。
POSITIVE_ONLY = {"total_assets", "total_hldr_eqy_exc_min_int", "total_share",
                 "oper_cost", "revenue", "total_hldr_eqy_inc_min_int"}

# ══════════════════════════════════════════════════════════════════════════
# `stock_financial_indicator` 的「安全字段」—— 只允许这两类进 `ctx.ind()`
#
# 该表 163 个比率里**大多数是累计 YTD**，直接当日频用会得到跨季锯齿（振幅 4 倍），
# 而且是**静默**的（看起来只是"因子有点噪"）。只有下面两类安全：
#   · 时点比率：`end_date` 上的瞬时量，本身不随累计口径变化
#   · 同期同比：分子分母都是同期 YTD，季节性自动抵消
# 其它字段（roe / roa / roic / grossprofit_margin / *_turn / ...）必须用
# `ctx.ttm()` 从原始三表重算。
# ══════════════════════════════════════════════════════════════════════════
IND_DIRECT = [
    "debt_to_assets", "current_ratio", "quick_ratio", "cash_ratio",
    "assets_to_eqt", "ca_to_assets", "nca_to_assets", "tbassets_to_totalassets",
    "int_to_talcap", "currentdebt_to_debt", "longdeb_to_debt", "debt_to_eqt",
    "tangibleasset_to_debt", "ebitda_to_debt", "turn_days",
]
IND_YOY = [
    "netprofit_yoy", "or_yoy", "tr_yoy", "op_yoy", "ebt_yoy", "roe_yoy",
    "ocf_yoy", "assets_yoy", "eqt_yoy", "bps_yoy", "dt_netprofit_yoy",
]
# 只解锁已核对上游字典的单季度比率；q_ 前缀并不保证质量，不能整体放行。
# 例如 q_eps 早年几乎全零，q_impair_to_gr_ttm 的窗口也不能由名字推断。
IND_QUARTERLY = ["q_roe", "q_dt_roe", "q_npta", "q_ocf_to_sales", "q_sales_yoy"]
IND_SAFE = tuple(IND_DIRECT + IND_YOY + IND_QUARTERLY)

# `stock_financial_indicator` 作为第三个数据源进版本表。
# 它的取值方式与 `point` 一致（「当日已知的最新报告期」上的原始披露值），
# 区别只在于**不做任何口径变换** —— 供应商给什么就是什么。
# 该表**没有 `f_ann_date`**（`_load_one` 已特判）。
IND_SOURCES = {"stock_financial_indicator": list(IND_SAFE)}


def period_of(end_date: np.ndarray) -> np.ndarray:
    """YYYYMMDD -> 连续季度序号。2025-03-31 -> (2025-1990)*4 + 0。"""
    y = end_date // 10000
    m = (end_date // 100) % 100
    return ((y - PERIOD_BASE) * 4 + (m // 3 - 1)).astype(np.int64)


def _load_one(up, ds: str, fields: list[str], y0: int, y1: int) -> pd.DataFrame:
    physical = [ANNUAL_ALIASES[f][1] if f in ANNUAL_ALIASES else f for f in fields]
    cols = list(dict.fromkeys(["stock_code", "end_date", "ann_date"] + physical))
    if ds != "stock_financial_indicator":
        cols.append("f_ann_date")
    try:
        df = up.read(ds, columns=cols, years=(y0, y1))
    except Exception:
        df = up.read(ds, years=(y0, y1))
        cols = [c for c in cols if c in df.columns]
        df = df[cols]
    if df.empty:
        return df

    df = df.copy()
    # 不改旧字段归属：同名 ebit/rd_exp 等可以从不同报表读取，别名只服务新因子。
    aliases = {logical: df[raw] for logical, raw in zip(fields, physical) if logical != raw}
    if aliases:
        # 一次拼接，避免上百字段逐列插入导致块碎片化。
        df = pd.concat([df, pd.DataFrame(aliases, index=df.index)], axis=1)
    df["end_date_i"] = series_to_int(df["end_date"])
    # ★ 只保留「已经是季末」的行：实测有 4 行北交所的怪日期（如 2015-02-28）。
    #   注意必须先过滤再取各列，否则 ann/fann 与 end_date 长度不一致（踩过）。
    df = df[(df["end_date_i"] % 10000).isin([331, 630, 930, 1231])].copy()
    end_i = df["end_date_i"].to_numpy()
    ann = series_to_int(df["ann_date"])
    fann = (series_to_int(df["f_ann_date"]) if "f_ann_date" in df.columns
            else np.zeros(end_i.size, np.int32))
    # ★ pit = max(ann_date, f_ann_date, end_date)：保守、绝不用未公告的数据
    pit = np.maximum(np.maximum(ann, fann), end_i)
    df["pit"] = pit.astype(np.int32)
    df = df[df["pit"] > 0]

    # ★ 脏 ann_date：实测 `stock_financial_indicator` 里有 2 行
    #   `ann_date = 1970-01-01` 而 `end_date` 合法（如 2025-12-31）。
    #   此时 max() 会把 pit 取成 end_date —— 等于让这份财报在**报告期结束当天**
    #   就可见，比真实公告日早 3~4 个月，是**真实的前视泄漏**。
    #   公告日无法考证就保守丢掉这一行（宁可缺数据，不可用未公告的数据）。
    known = (ann >= 19900101) | (fann >= 19900101)
    if not known.all():
        log.warning("%s: dropping %d rows without a valid announcement", ds, int((~known).sum()))
    df = df.loc[known].copy()
    cutoff = getattr(up, "audit_cutoff", None)
    if cutoff is not None:
        df = df[df["pit"] <= int(cutoff)].copy()

    df["period"] = period_of(df["end_date_i"].to_numpy())
    return df[["stock_code", "end_date_i", "period", "pit"] + fields]


def _source_tables():
    tables = {}
    for source in (TTM_SOURCES, POINT_SOURCES, IND_SOURCES, ANNUAL_SOURCES):
        for ds, fields in source.items():
            tables.setdefault(ds, []).extend(f for f in fields if f not in tables.get(ds, []))
    return tables


def load_vintages(up, y0: int, y1: int, fields: frozenset | None = None) -> pd.DataFrame:
    """把所有财报拼成一张「版本表」，每个 (code, end_date, pit) 一行。

    `fields` 给定时**只加载这些字段**（引擎按本次 run 的任务取 `FactorSpec.fin_fields`
    的并集传进来）。版本表有 80 个字段 × 6 年分区，单跑一个因子时全建是纯浪费。
    `None` = 全建。
    """
    tables = _source_tables()
    if fields is not None:
        tables = {ds: [f for f in fs if f in fields] for ds, fs in tables.items()}
        tables = {ds: fs for ds, fs in tables.items() if fs}
    frames = []
    for ds, fields_ in tables.items():
        t = _load_one(up, ds, fields_, y0, y1)
        if not t.empty:
            t["_src"] = ds
            frames.append(t)
    if not frames:
        return pd.DataFrame()

    out = []
    for f in frames:
        keep = ["stock_code", "end_date_i", "period", "pit"] + \
               [c for c in f.columns if c not in ("stock_code", "end_date_i", "period", "pit", "_src")]
        g = f[keep].copy()
        for c in POSITIVE_ONLY:
            if c in g.columns:
                # ★ .copy()：to_numpy() 可能返回只读视图，就地赋值会报
                #   "assignment destination is read-only"（实测踩过）
                v = np.array(g[c].to_numpy(dtype=np.float64), copy=True)
                v[~(v > 0)] = np.nan          # ★ 0.0 和负数都当缺失
                g[c] = v
        g = g.drop_duplicates(["stock_code", "end_date_i", "period", "pit"], keep="last")
        g["_available_" + str(f["_src"].iloc[0])] = True
        out.append(g)

    # 逐字段合并（不同报表的列集合不同）
    base = None
    for g in out:
        key = ["stock_code", "end_date_i", "period", "pit"]
        base = g if base is None else base.merge(g, on=key, how="outer")
    # 每个 (code, end_date, pit) 只保留一行
    base = base.sort_values(["stock_code", "end_date_i", "pit"])
    base = base.drop_duplicates(["stock_code", "end_date_i", "pit"], keep="last")
    return base.reset_index(drop=True)


FIELD_SOURCE = {field: ds for ds, fields in _source_tables().items() for field in fields}
FINANCIAL_SOURCES = frozenset(FIELD_SOURCE.values())


def financial_dependency(spec, registry, seen=None):
    seen = set() if seen is None else seen
    if spec.name in seen:
        return False
    seen.add(spec.name)
    return any(d in FINANCIAL_SOURCES or
               (d in registry and financial_dependency(registry[d], registry, seen)) for d in spec.deps)


def _revision_events(vt):
    """Reevaluate the current report at every source announcement, including old-period corrections.

    Fill row identities, never values: an explicitly missing revised field stays missing.
    """
    if vt.empty:
        return vt
    keys = ["stock_code", "end_date_i", "period", "pit"]
    original = vt.sort_values(keys).drop_duplicates(keys, keep="last").reset_index(drop=True)
    events = original.sort_values(["stock_code", "pit", "end_date_i"])
    latest = events.groupby("stock_code", sort=False)["end_date_i"].cummax()
    extra = events[keys].copy()
    extra["end_date_i"] = latest.to_numpy()
    extra["period"] = period_of(extra["end_date_i"].to_numpy())
    grid = pd.concat([original[keys], extra], ignore_index=True).drop_duplicates(keys).sort_values(keys)
    indexed = original[keys].copy()
    indexed["_origin"] = np.arange(len(original))
    grid = grid.merge(indexed, on=keys, how="left").sort_values(keys)
    grid["_origin"] = grid.groupby(["stock_code", "end_date_i"], sort=False)["_origin"].ffill()
    assert grid["_origin"].notna().all()
    payload = original.iloc[grid["_origin"].to_numpy(dtype=np.int64)].reset_index(drop=True)
    payload[keys] = grid[keys].reset_index(drop=True)
    return payload


class Derivative:
    """版本表 + TTM/滞后列，供各因子按字段取用。"""

    def __init__(self, vt: pd.DataFrame, codes: np.ndarray):
        self.codes = np.asarray(codes)
        self._sources = {}
        self._annual = None
        self._isolated = not any(c.startswith("_available_") for c in vt.columns)
        pos = {c: i for i, c in enumerate(self.codes)}
        vt = vt[vt["stock_code"].isin(pos)].copy()
        if self._isolated:
            vt = _revision_events(vt)
        vt["code_idx"] = vt["stock_code"].map(pos).to_numpy(dtype=np.int32)
        # ★ 顺序即键序：as-of 依赖 (code, period, pit) 严格升序
        self.vt = vt.sort_values(["code_idx", "period", "pit"]).reset_index(drop=True)
        # `_row` 把派生列与 self.vt 的位置绑定，供 current() 筛子集后复用
        self.vt["_row"] = np.arange(len(self.vt), dtype=np.int64)

        self._s_code = self.vt["code_idx"].to_numpy(dtype=np.int64)
        self._s_per = self.vt["period"].to_numpy(dtype=np.int64)
        self._s_pit = self.vt["pit"].to_numpy(dtype=np.int64)
        self._keys = (self._s_code * PERIOD_W + self._s_per) * KEY_W + self._s_pit
        self._cols: dict[str, np.ndarray] = {}
        self._cur: pd.DataFrame | None = None
        self._panel_cols: dict[tuple, np.ndarray] = {}
        self._panel_window: tuple[int, int] | None = None

    # ---------------------------------------------------------------- 内部
    def _asof_period(self, target_per: np.ndarray, val: np.ndarray) -> np.ndarray:
        """对每一行，取「同一股票、指定报告期、pit <= 本行 pit」的最新值。"""
        tk = (self._s_code * PERIOD_W + target_per) * KEY_W + self._s_pit
        pos = np.searchsorted(self._keys, tk, side="right") - 1
        ok = pos >= 0
        p = np.where(ok, pos, 0)
        # ★ 必须校验同股同期，否则会静默串到别的股票/报告期上
        ok &= (self._s_code[p] == self._s_code) & (self._s_per[p] == target_per)
        return np.where(ok, val[p], np.nan)

    def _raw(self, field: str) -> np.ndarray:
        if field not in self._cols:
            self._cols[field] = self.vt[field].to_numpy(dtype=np.float64)
        return self._cols[field]

    def trim_cache(self, panel) -> None:
        """丢弃属于**其它面板窗口**的缓存。

        ★ 必须做，否则会 OOM。`_panel_cols` 按 (面板日期0, 日期1, field, mode, lag) 缓存，
          而每个 (因子, 年) 任务的 warmup 不同 → 窗口不同 → 缓存**永不命中、只增不减**。
          一个 worker 要跑几百个任务，每条缓存是 (T,C) 的 float32
          （750×3484×4 ≈ 10 MB），×5 个字段 ×几百任务 = **每 worker 6 GB 以上**
          ——实测 16 个 worker 把内存顶到 108 GB/120 GB，进程被杀。
          同一个任务内部窗口是固定的，所以「窗口一变就清」既安全又够省。
        """
        for child in self._sources.values():
            child.trim_cache(panel)
        if self._annual is not None:
            self._annual.trim_cache(panel)
        if panel.T == 0:
            return
        w = (int(panel.dates[0]), int(panel.dates[-1]))
        if getattr(self, "_panel_window", None) != w:
            self._panel_cols.clear()
            self._panel_window = w

    # ---------------------------------------------------------------- 公开
    def _ttm_at(self, field: str, p: np.ndarray) -> np.ndarray:
        cum = self._raw(field)
        own = self._asof_period(p, cum)
        out = own + self._asof_period(4 * (p // 4) - 1, cum) - self._asof_period(p - 4, cum)
        # Annual YTD is already TTM; no previous annual report is needed.
        out = np.where(p % 4 == 3, own, out)
        if (~np.isfinite(out)).any():
            parts = []
            for k in range(4):
                q = p - k
                end = self._asof_period(q, cum)
                prev = self._asof_period(q - 1, cum)
                parts.append(np.where(q % 4 == 0, end, end - prev))
            out = np.where(np.isfinite(out), out, sum(parts))
        return out

    def ttm(self, field: str) -> np.ndarray:
        return self._ttm_at(field, self._s_per)

    def lag(self, col: np.ndarray, k_periods: int) -> np.ndarray:
        """把某列按「报告期」回看 k 期（as-of 本行 pit）。

        注意：**不是**日频序列减 252 天。后者会横跨报告期边界，
        在每个公告日静默地比较两个不同的财季，且分母一年变 4 次。
        """
        return self._asof_period(self._s_per - k_periods, col)

    # ---------------------------------------------------------------- 对齐
    def current(self) -> pd.DataFrame:
        """保留历次版本，筛出每个公告时点已知的最新报告期。

        用**前缀最大报告期**来挑：同一只股票按 pit 升序，
        只有 period 创了新高的那些版本才是「当时的当前报告」。
        这样一份迟到的旧期修正不会把新期数据顶掉。
        """
        if getattr(self, "_cur", None) is not None:
            return self._cur
        vt = self.vt.sort_values(["code_idx", "pit", "period"])
        pm = vt.groupby("code_idx", sort=False)["period"].cummax()
        cur = vt[vt["period"].to_numpy() == pm.to_numpy()].copy()
        cur = cur.drop_duplicates(["code_idx", "pit"], keep="last")
        # ★ 日频前向填充要求源表按 (code, pit) 升序
        self._cur = cur.sort_values(["code_idx", "pit"]).reset_index(drop=True)
        return self._cur

    # ---------------------------------------------------------------- 对齐到面板
    def to_panel(self, panel, field: str, mode: str = "ttm", lag: int = 0) -> np.ndarray:
        """把某个财务字段 as-of 前向填充成 (T, C)。

        `mode="ttm"` 取三行式 TTM；`mode="point"` 取时点值（资产负债表科目）。
        `lag=k` 表示回看 k 个报告期（同比 = lag 4）。
        """
        self.trim_cache(panel)
        if not self._isolated:
            ds = FIELD_SOURCE[field]
            if ds not in self._sources:
                marker = "_available_" + ds
                if marker not in self.vt:
                    return np.full(panel.shape, np.nan, dtype=np.float32)
                columns = ["stock_code", "end_date_i", "period", "pit"] + [f for f in FIELD_SOURCE if FIELD_SOURCE[f] == ds and f in self.vt]
                source = self.vt.loc[self.vt[marker].eq(True), columns]
                self._sources[ds] = Derivative(source, self.codes)
            return self._sources[ds].to_panel(panel, field, mode, lag)
        if mode == "annual":
            # 先限定年度报告再选择当时最新版本，季度披露绝不能把年报推走。
            # 保留旧年报修订事件，让过去年度的修订只从实际披露时点传播。
            if self._annual is None:
                annual = self.vt.loc[self.vt["end_date_i"] % 10000 == 1231].copy()
                self._annual = Derivative(annual, self.codes)
            return self._annual.to_panel(panel, field, mode="point", lag=lag)
        if self.vt.empty:
            return np.full((panel.T, len(self.codes)), np.nan, dtype=np.float32)
        # ★ 缓存键必须带上**面板的日期范围**：不同因子的 warmup_days 不同
        #   （700 vs 60），日常增量时同一因子的窗口也可能只是年末几天，
        #   所以同一个 Derivative 会被不同形状的 Panel 复用。
        #   只按 (field, mode, lag) 做键会把上一个面板的 (T,C) 数组返回给新面板
        #   —— 表现为 "shape mismatch (706, 3484) vs ..."（实测踩过）。
        key = (int(panel.dates[0]) if panel.T else 0,
               int(panel.dates[-1]) if panel.T else 0,
               field, mode, lag)
        if key in self._panel_cols:
            return self._panel_cols[key]

        target = self._s_per - lag
        col = self._ttm_at(field, target) if mode == "ttm" else self._asof_period(target, self._raw(field))
        cur = self.current()
        vals = col[cur["_row"].to_numpy()]
        out = panel.asof(
            cur["code_idx"].to_numpy(dtype=np.int32),
            cur["pit"].to_numpy(dtype=np.int32),
            vals,
        )
        self._panel_cols[key] = out
        return out
