"""执行引擎 —— 全量 / 增量 / 断点续传。

## 增量为什么这么设计

**状态是「已覆盖的日期区间集合」**，缺口 = 目标区间 − 已覆盖区间。

关键点是每个缺口区间 `[a, b]` 要**用 `[a − warmup_days, b]` 去算**，只保留 `>= a` 的行。
warmup 保证窗口起点的值与全量重算完全一致：财务因子要覆盖 TTM(4 季) + 同比(再 4 季) +
ann_date 滞后，所以是 500 个日历天，不是 20 个交易日。
warmup 给太小，每次增量都会在窗口头部产出一年左右的 NaN —— 而且看起来只像「因子有点噪」。

## 三层失效判定

  L0 因子逻辑指纹变了（version/formula 改动）      -> 全量重建
  L1 输出日期有缺口 + 最近 revision_days 回刷      -> 常规每日增量
  L2 上游输入水位变了（行数 / 最新公告日）          -> 从公告日往前回溯重算

L2 是必需的一层：增量是按**输出日期**找缺口的，但上游财报被追溯修正时，
它的 ann_date 可能远在过去（实测最长滞后 15 个月）。只按日期找缺口，
修正就永远传播不到历史，而且悄无声息 —— 那些日期已经被标成「已完成」了。

## 每年只写一次

先把缺口归一到 `{年: {交易日集合}}`，再逐年计算+写盘。
否则「缺口区间」和「回刷窗口」落在同一年时会写两次，跨年的区间更会重复写。
`store.upsert_year` 是读-改-写整文件，重复写同一个分区就是模块① 踩过的 O(n²) 坑。
"""

from __future__ import annotations

import datetime as _dt
import logging
import time
from collections import defaultdict

import numpy as np
import pandas as pd

from .config import Config
from .context import FactorContext
from .dates import Calendar, int_to_str, int_to_str_vec, today_int
from .deriv import Derivative, load_vintages, financial_dependency
from .manifest import Manifest
from .panel import Panel, cs_rank
from .spec import FactorSpec
from .store import upsert_year
from .universe import code_master, frozen_fingerprint, listed_mask, st_events
from .upstream import Upstream

log = logging.getLogger("fea.engine")

# 各上游数据集用于「水位」的时间列
DEP_PIT_COL = {
    "stock_income": "ann_date",
    "stock_balancesheet": "ann_date",
    "stock_cashflow": "ann_date",
    "stock_financial_indicator": "ann_date",
    "stock_holder_number": "ann_date",
    "stock_forecast": "ann_date",
    "stock_margin_detail": "trade_date",
    "stock_cyq_perf": "trade_date",
    "stock_adj_factor_changes": "date",
    "stock_limit_up": "trade_date",
    # ⚠️ 下面这张表**没有 trade_date 列**（只有 trade_time）。不写这一条的话
    #    水位探测会回退到读一个不存在的列 → 退化成全量读（实测 5.3s/次），
    #    而且 L2 失效判定只剩「行数变化」可用。
    "stock_history_5min": "trade_time",
    "tdx_minute": "trade_time",
}


def _file_changed(key, old, new):
    """Snapshot content hashes avoid false repairs when identical files are rewritten.

    The two-element legacy identity remains valid when size/mtime still match.
    Annual files use metadata, avoiding full reads of large source partitions.
    """
    if old is None or new is None:
        return old != new
    if key == "snapshot":
        if len(old) >= 3 and len(new) >= 3:
            return old[2] != new[2]
        return old[:2] != new[:2]
    return old != new


def minus_days(v: int, n: int) -> int:
    d = _dt.date(v // 10000, (v // 100) % 100, v % 100) - _dt.timedelta(days=int(n))
    return d.year * 10000 + d.month * 100 + d.day


def _merge_spans(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    spans = sorted((a, b) for a, b in spans if a <= b)
    out: list[list[int]] = []
    for a, b in spans:
        if out and a <= out[-1][1] + 1:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


class Engine:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.up = Upstream(cfg.upstream)
        self.up.audit_cutoff = cfg.raw.get("_audit_cutoff")
        self.cal = Calendar.load(self.up)
        self.codes = code_master(self.up, cfg)
        self._deriv: Derivative | None = None
        self._deriv_window: tuple[int, int] | None = None
        # ★ run 级财务字段集（prebuild 的并集）：见 run_year 里那段"乒乓"说明。
        #   None = 全字段（契约：空元组/未声明 = 「我全都要」）。
        self._deriv_fields_run: frozenset | None = None
        self._deriv_fields: frozenset | None = None
        # `--start` 覆盖：本次运行只产出这一天之后的因子值（None = 用 spec 自己的起点）
        self.override_start: int | None = None
        # ★ 计算范围的下界（= conf 的 `default_start`）：既用于 plan() 的区间裁剪，
        #   也用于 run_year() 的面板锚点（见那里的说明）。
        self._start_floor: int = int(str(cfg.default_start).replace("-", ""))
        # 本次运行的右端点（YYYYMMDD）+ 其字符串形式；用于裁剪分区里的陈旧尾行
        self._run_end: int | None = None
        self._end_str: str = "9999-12-31"
        # 股票池掩码只取决于 (日期范围, 股票池, universe 配置)，与因子无关。
        # 同一年的各个因子 warmup 相同 → 面板相同 → 掩码完全一样，缓存即可。
        self._uni_cache: dict[tuple[int, int, int], np.ndarray] = {}
        self._st: tuple[np.ndarray, np.ndarray] | None = None
        # 价格层 / 日内层 / 筹码层 / 耦合 IO —— 都是「按需构建 + fork 共享」
        self._prices = None
        self._prices_window: tuple[int, int] | None = None
        self._intraday = None
        self._chips = None
        self._factor_io = None
        # 上游水位按数据集缓存：watermark 要逐分区读 ann_date 列（如 stock_income 有 37 个
        # 年分区），而它在每个因子收尾时都会被调用。不缓存的话 10 个因子 × 2 个依赖
        # = 几百次重复的文件读，在共享盘上是实打实的秒级开销。
        self._wm_cache: dict[str, dict] = {}
        # ★ 股票池/截面口径也属于「因子定义」的一部分：改了 exclude_st、
        #   板块白名单或 winsor 分位，历史因子值就会变，必须触发全量重建。
        #   否则改了配置却因为「日期已覆盖」而不重算，新旧口径会混在一个面板里。
        self.universe_fp = "|".join([
            "mb=" + ",".join(cfg.board_prefixes),
            f"st={int(cfg.exclude_st)}",
            f"mld={cfg.min_listed_days}",
            f"w={cfg.winsor[0]},{cfg.winsor[1]}",
            f"mcs={cfg.min_cross_section}",
            # ★★ 2026-09-18 冻结股票池：用**内容哈希**进指纹。
            #   池子决定每个截面的行数与 rank/cs_zscore，改一只票就必须重算全部历史。
            #   未启用时是常量 "off"，不影响老口径的指纹（老因子不会因此被误判为过期）。
            "fx=" + frozen_fingerprint(cfg),
        ])

    def _recipe(self, spec: FactorSpec) -> str:
        from .spec import REGISTRY
        suffix = "||financial-vintages=v2" if financial_dependency(spec, REGISTRY) else ""
        return f"{spec.recipe(self.cfg)}||{self.universe_fp}{suffix}"

    @staticmethod
    def _layer_years(todo, dep_names: tuple) -> tuple[int, int]:
        """某个派生层这次真正需要的**年份范围**（只统计 deps 命中该层的因子）。

        没有因子用到它 → 返回 `(0, -1)`，`ensure()` 的 `range(0, 0)` 是空循环、
        一个分区都不会动。这就是「只跑 2026 就不要去重建 2018」的实现。
        """
        lo_all, hi_all = [], []
        for spec, _, plan in todo:
            if not any(d in spec.deps for d in dep_names):
                continue
            days = np.concatenate(list(plan.values()))
            lo_all.append(minus_days(int(days.min()), spec.warmup_days))
            hi_all.append(int(days.max()))
        if not lo_all:
            return 0, -1
        return int(str(min(lo_all))[:4]), int(str(max(hi_all))[:4])

    def watermark(self, dep: str) -> dict:
        w = self._wm_cache.get(dep)
        if w is None:
            from .spec import REGISTRY
            if dep in REGISTRY:
                # ★ 耦合因子：`deps` 里写的是**别的因子名**（`ctx.load_factor` 的父因子）。
                #   它的水位要看父因子自己的产出。否则父因子更新不会传播过来 ——
                #   `up.watermark` 对不存在的目录返回 exists=False，L2 直接 continue，
                #   表现为耦合因子永远停在首次算出的值上，而且**不报错**。
                w = self._factor_watermark(dep)
            else:
                w = self.up.watermark(dep, DEP_PIT_COL.get(dep))
            self._wm_cache[dep] = w
        return w

    def _factor_watermark(self, name: str) -> dict:
        """父因子的「水位」= 它自己 manifest 里的总行数 + 最新产出日期。"""
        man = Manifest.load(self.cfg.state_dir, name)
        rows = man.partition_rows()
        if not rows:
            return {"exists": False, "rows": 0, "max_pit": 0}
        mx = 0
        for v in man.partitions.values():
            d = v.get("max_date")
            if d:
                mx = max(mx, int(str(d).replace("-", "")[:8]))
        files = sorted((self.cfg.factors_dir / name).glob("year=*/data.parquet"))
        parts = {p.parent.name: [p.stat().st_size, p.stat().st_mtime_ns] for p in files if int(p.parent.name[5:]) >= int(self.cfg.default_start[:4])}
        return {"exists": True, "rows": int(rows), "max_pit": int(mx), "files": parts}

    # 因子输出的默认右端点取自这张表（`conf/frequency.yaml` 里它也是 baseline）
    BASELINE_DEP = "stock_daily"

    def baseline_last_day(self) -> int:
        """上游**日频基准表**的最新交易日 —— 因子输出的默认右端点。

        ★ 不能用「今天」。踩过的坑：`basic_calendar` 会延伸到未来
          （实测到 2026-10-14），而**财务类因子只依赖已公告的季报**，
          所以在「今天」（上游行情还没到、甚至还没收盘）也会算出一个有效值。
          实测 `accruals_ratio` 就出现了 2026-09-15 的行，而上游 `stock_daily`
          只到 2026-09-14。后果有两个：
            · 输出覆盖了一个**尚未结束**的交易日；
            · 那天晚些时候数据到了，值可能变（L2 重算），下游却已经用过了。
          正确做法：右端点 = 基准表实际覆盖到的最后一天，
          也就是「上游到齐了没」这道闸门。要强制算到更晚，显式传 `--end`。
        """
        w = self.watermark(self.BASELINE_DEP)
        last = int(w.get("max_pit") or 0)
        if last <= 0:
            raise RuntimeError("上游 stock_daily 缺少有效交易日期；请先完成数据端更新，再生成因子")
        return last

    def universe_for(self, panel: Panel) -> np.ndarray:
        """带缓存的股票池掩码（见 __init__ 里的说明）。"""
        key = (int(panel.dates[0]), int(panel.dates[-1]), panel.C)
        m = self._uni_cache.get(key)
        if m is None:
            if self._st is None:
                self._st = st_events(self.up, self.codes)   # 只算一次，与面板无关
            m = listed_mask(panel, self.up, self.cfg, st_ev=self._st)
            self._uni_cache[key] = m
        return m

    # ---------------------------------------------------------------- 价格层
    def prices_for(self, lo: int, hi: int):
        """构建（或复用）价格层。只有用到价格的因子才付这个成本。"""
        from .prices import PriceLayer
        if self._prices is None:
            t0 = time.time()
            self._prices = PriceLayer(self.up, self.cfg, self.cal, self.codes)
            log.info("价格层初始化 %.2fs", time.time() - t0)
        self._prices_window = (lo, hi)
        return self._prices

    def intraday_layer(self):
        """构建日内层。构造函数零成本（只在第一次 `panel()` 时才读/建缓存）。"""
        from .intraday import IntradayLayer
        if self._intraday is None:
            self._intraday = IntradayLayer(self.up, self.cfg, self.codes)
        return self._intraday

    def chips_layer(self):
        """构建筹码层。同上，构造函数零成本。"""
        from .chips import ChipLayer
        if self._chips is None:
            self._chips = ChipLayer(self.up, self.cfg, self.codes)
        return self._chips

    def factor_io(self):
        from .factors_io import FactorIO
        if self._factor_io is None:
            self._factor_io = FactorIO(self.cfg, self.cal)
        return self._factor_io

    # ---------------------------------------------------------------- 衍生层
    def deriv_for(self, lo: int, hi: int, fields: frozenset | None = None) -> Derivative:
        """构建（或复用）财务衍生层。整个 run 只建一次。

        `fields` 是本次任务用到的财务字段集合（来自各 `FactorSpec.fin_fields` 的并集）。
        给定时**只加载这些字段**；`None` = 全建。版本表有 80 个字段 × 6 年分区，
        单跑一个因子时全建是纯浪费。
        """
        if fields == frozenset():
            return None  # No financial consumer in this run; do not build an empty vintage table.
        need = self.up.years_for_window(lo, hi, lag_years=3)
        # ★ 缓存可复用的判据：**已有字段集必须覆盖请求的字段集**。
        #   `fields=None` 的语义是「我需要全部字段」，不是「我什么都不要」——
        #   所以它只在**已缓存的就是全字段**时才可复用。
        #   踩过的坑：把 `fields is None` 当成「与任何缓存都兼容」，
        #   于是同一个 worker 里先建了只含 `revenue` 的窄表，
        #   后面 `debt_asset_ratio` 请求全字段时复用了它 →
        #   `KeyError: 'total_liab'`（2026 那一轮 277 个因子里挂了 2 个）。
        if fields is None:
            ok_fields = self._deriv_fields is None
        else:
            ok_fields = self._deriv_fields is not None and fields <= self._deriv_fields
        if self._deriv is not None and self._deriv_window is not None \
                and self._deriv_window[0] <= need[0] and self._deriv_window[1] >= need[1] \
                and ok_fields:
            return self._deriv
        t0 = time.time()
        vt = load_vintages(self.up, need[0], need[1], fields=fields)
        self._deriv = Derivative(vt, self.codes)
        self._deriv_window = need
        self._deriv_fields = fields
        log.info("财务衍生层：%d 个版本 / %d 个字段 / 年份 %s / %.2fs",
                 len(vt), len(vt.columns), need, time.time() - t0)
        return self._deriv

    # ---------------------------------------------------------------- 计划
    def _rewind_forward_dependency(self, spec, date, floor):
        """An input at d can affect labels beginning forward_days before d."""
        if not spec.forward_days or not len(self.cal.days):
            return max(floor, date)
        pos = min(int(np.searchsorted(self.cal.days, date, side="left")), len(self.cal.days)-1)
        return max(floor, int(self.cal.days[max(0, pos-spec.forward_days)]))

    def plan(self, spec: FactorSpec, man: Manifest, end_i: int,
             rebuild: bool) -> dict[int, np.ndarray]:
        """算出 `{年份: 需要重算的交易日}`。"""
        start_i = spec.start_int(self.cfg)
        # ★★ `default_start` 同时是**计算范围的下界**（2026-09-15 加，实测踩过）。
        #   `FactorSpec.start` 的语义是「该因子受上游数据起点限制，不能早于这一天」,
        #   它常常**早于** default_start（筹码 2018、两融 2011、涨停 2015…）。
        #   如果只把 default_start 当"没写 start 时的默认值"，那么想"只算 2026"时，
        #   这些显式写了 start 的因子仍会去补 2011–2025 的全历史 —— 实测后果：
        #   一次 run 在 prebuild 阶段重建了 15 个派生层年分区（并顺带把刚删掉的
        #   历史筹码/日内缓存重新建回来）。
        #   取二者较大值 = 「上限政策(default_start) 优先，个别因子更晚的起点也尊重」。
        start_i = max(start_i, self._start_floor)
        # `--start YYYY-MM-DD`：本次运行只产出这一天之后的因子值。
        # 用于「先只跑某一段、验证无误再往前推」。不影响 warmup ——
        # 面板照旧往前多读 spec.warmup_days 个日历天，所以值与全量重建逐格一致。
        if self.override_start:
            start_i = max(start_i, int(self.override_start))
        recipe = self._recipe(spec)
        if rebuild or man.recipe != recipe:
            if man.recipe and man.recipe != recipe and not rebuild:
                log.warning("%s 逻辑/口径已变，强制全量重建", spec.name)
            man.reset(recipe)
            spans = [(start_i, end_i)]
        else:
            spans = [(start_i, end_i)] if getattr(self, "force_refresh", False) else []
            # L1a 输出日期缺口
            for a, b in man.missing_ranges(int_to_str(start_i), int_to_str(end_i)):
                spans.append((_to_int(a), _to_int(b)))
            # A manifest cannot certify a partition that was deleted or changed externally.
            for year, meta in man.partitions.items():
                lo = max(start_i, int(year) * 10000 + 101)
                hi = min(end_i, int(year) * 10000 + 1231)
                if lo > hi:
                    continue
                path = self.cfg.factors_dir / spec.name / f"year={year}" / "data.parquet"
                damaged = not path.exists()
                expected = meta.get("file_identity")
                if not damaged and expected:
                    st = path.stat()
                    damaged = expected != [st.st_size, st.st_mtime_ns]
                if damaged:
                    spans.append((lo, hi))
                    log.warning("%s output partition %s missing/changed; rebuilding its requested interval", spec.name, year)
            # L1b 最近 N 天回刷（上游当天数据可能还没定稿）
            rev = minus_days(end_i, self.cfg.revision_days)
            rev = self._rewind_forward_dependency(spec, rev, start_i)
            spans.append((rev, end_i))
            # L2 上游输入水位变化 -> 从「上次最新公告日」往前回溯
            for dep in spec.deps:
                pit_col = DEP_PIT_COL.get(dep)
                cur = self.watermark(dep)
                prev = man.input_watermark.get(dep)
                if not cur.get("exists"):
                    continue
                if prev is None:
                    continue                      # 本因子首次建（L0 已覆盖）
                # Row/date watermarks alone cannot detect value-only repairs.
                old_files, new_files = prev.get("files"), cur.get("files")
                if old_files is not None and new_files is not None and old_files != new_files:
                    changed = [key for key in set(old_files) | set(new_files) if _file_changed(key, old_files.get(key), new_files.get(key))]
                    latest_year = max(int(cur["max_pit"] or end_i), int(prev["max_pit"] or end_i)) // 10000
                    old_years = [int(key[5:]) for key in changed if key.startswith("year=") and int(key[5:]) < latest_year]
                    value_only = (cur["rows"], cur["max_pit"]) == (prev["rows"], prev["max_pit"])
                    if changed and (old_years or value_only):
                        years_changed = [int(key[5:]) for key in changed if key.startswith("year=")]
                        lo = max(start_i, min(years_changed) * 10000 + 101) if years_changed else start_i
                        lo = self._rewind_forward_dependency(spec, lo, start_i)
                        spans.append((lo, end_i))
                        log.info("  ↻ %s 上游 %s 文件发生历史/等行数修订 -> 从 %s 重算", spec.name, dep, int_to_str(lo))
                if (cur["rows"], cur["max_pit"]) != (prev["rows"], prev["max_pit"]):
                    back = self.cfg.dep_backfill_days(dep)
                    anchor = prev["max_pit"] or end_i
                    lo = max(start_i, minus_days(int(anchor), back))
                    lo = self._rewind_forward_dependency(spec, lo, start_i)
                    log.info("  ↻ %s 上游 %s 水位变化 -> 从 %s 起重算",
                             spec.name, dep, int_to_str(lo))
                    spans.append((lo, end_i))

        per_year: dict[int, np.ndarray] = {}
        for a, b in _merge_spans(spans):
            days = self.cal.between(a, b)
            if days.size == 0:
                continue
            ys = days // 10000
            for y in np.unique(ys):
                sel = days[ys == y]
                prev = per_year.get(int(y))
                per_year[int(y)] = sel if prev is None else np.union1d(prev, sel)
        return per_year

    def plan_years(self, specs: list[FactorSpec], end_i: int,
                   rebuild: bool = False) -> dict[str, list[int]]:
        """**只做规划、不建任何层**：返回 `{因子名: [有活的年份, ...]}`。

        用途是"动手之前"判断这次 run 会不会把**一个进程**的内存吃爆
        （见 `main.py` 的多年守卫）：价格层/派生层缓存只扩不缩，一个因子跨的年头越多，
        每个 worker 攒下的窗口越大 —— 实测全历史窗口 7~8 GB/worker，
        8 个 worker 直接顶到 cgroup 上限（表现是 `BrokenProcessPool`，不是 MemoryError）。

        与 `prebuild()` 的区别：这里不读价格/财务数据、不建衍生层，只跑 `plan()`
        （水位读盘有缓存），所以可以放心地在**决定是否启动**之前调用。
        ⚠️ 不要用它的结果去跑任务（缺 derived 层）；它只回答"哪几年有活"。
        ⚠️ `plan()` 对"指纹变了"的因子会就地 `man.reset()`，但**不落盘** ——
           真正落盘的是 `prebuild()`（它会把清空后的 manifest 存下来）。
        """
        out: dict[str, list[int]] = {}
        for spec in specs:
            man = Manifest.load(self.cfg.state_dir, spec.name)
            ys = sorted(int(y) for y, days in self.plan(spec, man, end_i, rebuild).items()
                        if np.size(days))
            if ys:
                out[spec.name] = ys
        return out

    def _propagate_parent_plans(self, todo, end_i):
        plans = {s.name: plan for s, _, plan in todo}
        for _ in range(len(todo)):
            changed = False
            for spec, _, plan in todo:
                floor = max(spec.start_int(self.cfg), self._start_floor, self.override_start or 0)
                for dep in spec.deps:
                    for year, parent_days in plans.get(dep, {}).items():
                        days = parent_days[(parent_days >= floor) & (parent_days <= end_i)]
                        if not days.size:
                            continue
                        old = plan.get(year, np.array([], dtype=np.int32))
                        new = np.union1d(old, days)
                        if len(new) != len(old):
                            plan[year] = new
                            changed = True
            if not changed:
                break

    # ---------------------------------------------------------------- 执行
    def _anchor_for(self, spec: FactorSpec, days: np.ndarray) -> int:
        """本次任务的面板**下界锚点** = max(因子起点, 计算下界, 本任务所在年的 1 月 1 日)。

        `days` 必然是同一年的交易日（`plan()` 已按年切分），所以看 `days[0]` 的年份就够。
        语义与代价见 `run_year` 里那段长注释。
        """
        year = int(days[0]) // 10000
        return max(spec.start_int(self.cfg), self._start_floor, year * 10000 + 101)

    def run_year(self, spec: FactorSpec, year: int, days: np.ndarray) -> dict:
        """算**一个因子的一年**并写盘。可被并行 worker 独立调用。

        为什么按 (因子, 年) 而不是按因子切并行：各因子耗时差一倍以上
        （`debt_asset_ratio` 49s vs `yoy_roe` 87s），按因子切会被最慢的那个拖住；
        按年切有 150 个任务，负载均衡得多。
        并发安全性：每个 (因子, 年) 写的是**不同文件**
        （`data/factors/<因子>/year=YYYY/data.parquet`），互不干扰；
        manifest 由父进程在最后统一写，worker 不碰。
        """
        days = np.sort(np.asarray(days, dtype=np.int32))
        hi = int(days[-1])
        # ★★ 面板下界锚在**本次任务所在年的 1 月 1 日**（2026-09-17 放开全历史时改）。
        #
        #   原则（2026-09-15 定的，没变）：**锚点必须与"本次计划的缺口日"无关**。
        #   增量只重算最近几天，若锚点跟着缺口走，同一格在全量跑与增量跑里会落在不同
        #   面板上 —— `mathx` 的滚动原语用「列首第一个有效值」做数值中心，面板下界
        #   一挪中心值就变 → 浮点漂移（实测 **49 个因子**有 0.01%~3% 的格子落在最后
        #   一个 ULP 上），且价格层/派生层缓存按窗口失效 → 重复读盘（实测重建 73 次）。
        #
        #   本次改动：锚点由「因子全局起点（= default_start）」收窄为「本年 1 月 1 日」。
        #   面板 = [锚点 − warmup, 本年最后一个待算日]，锚点决定**每一格要算多长**：
        #   · 锚在全局起点时，放开到 2012 后每个 (因子, 年) 任务都要从 2012 重算起
        #     ⇒ 15 年总计算量 ≈ 120 个「单年」而不是 15 个（**8 倍**），
        #     且每个 worker 都要把全历史价格层读进内存（实测单进程峰值 7.1~8.1 GB；
        #     本机 cgroup 上限 60 GiB ⇒ 并行度只能压到 4~5）；
        #   · 锚在本年年初后：面板 = 1 年 + warmup，总计算量 ≈ 15 × 单年，
        #     每进程内存回到 2~3 GB，**每日增量顺带从 6 分钟降到 1~2 分钟**。
        #
        #   ⚠️ 两种锚定对**已算出的值**只差「浮点中心的舍入」（末位 ULP 量级）：
        #      唯一的实质差异是「本年第一格往前能看多久」，而那由 `warmup_days` 保证
        #      （warmup 不足的因子会在每年年初暴露出来 —— 用
        #      `scripts/check_anchor_warmup.py` 做 A/B 抽检，实测见 docs/2026-09-17.md）。
        #      **2026 年的值逐位不变**：对 2026 任务两种算法都取 max(..., 2026-01-01)。
        #
        #   ⚠️ `--start` 不再参与锚点（它只截输出范围）。否则「先跑一段」的尾部跑会与
        #      全量跑落在不同面板上，恰好违背 `--start` 自己承诺的「逐格一致」。
        #
        #   ⚠️ 别改成"全局统一窗口"：那会把最大 warmup（`pegh5` 2600 天）传染给所有因子，
        #      让每个任务都去读 2018 年以来的全部价格数据（实测单任务涨到 321s）。
        anchor = self._anchor_for(spec, days)
        wlo = minus_days(anchor, spec.warmup_days)
        # ★ 逐任务计时（2026-09-15 加，用户要求"记录每个因子增量生成的时间是否高效"）。
        #   原先只记「整轮墙钟时间」，看不出单个因子快不快 —— 而增量的性能问题
        #   恰恰出在"某些因子的 warmup 很长 / 要重建派生层"上，必须逐任务可见。
        _t_task = time.time()
        # ★ 标签要看到未来：面板向后延伸 forward_days 个交易日。
        #   不延伸的话，每个年分区的最后几天会因为「未来窗口不足」恒为 NaN，
        #   而且 2026 分区会随时间推移悄悄丢尾巴。
        hi_panel = hi
        if spec.forward_days:
            pos = int(np.searchsorted(self.cal.days, hi, side="right"))
            hi_panel = int(self.cal.days[min(pos + spec.forward_days,
                                             self.cal.days.size - 1)])
        pdays = self.cal.between(wlo, hi_panel)
        if pdays.size == 0:
            return {"factor": spec.name, "year": year, "rows": 0, "nonnull": 0,
                    "seconds": round(time.time() - _t_task, 2)}

        panel = Panel(pdays, self.codes)
        uni = self.universe_for(panel)
        # 注意：worker 里必须**沿用父进程已建好的衍生层**（fork 共享）。
        # ★★ 字段集必须用**run 级**的那一个（`prebuild` 里算好的并集），不能按因子各传各的：
        #   声明了 `fin_fields` 的因子请求子集、没声明的请求全集，两者会**互相踢掉**
        #   对方的缓存 → 每个任务都重建一次衍生层。实测增量跑里重建了 **73 次**
        #   （每次 6~25s，占整轮一半以上耗时），而全量跑因为每任务覆盖 170 天被掩盖了。
        deriv = self.deriv_for(wlo, hi_panel, fields=self._deriv_fields_run)
        ctx = FactorContext(panel, deriv, self.up, self.cfg, uni,
                            prices=self.prices_for(wlo, hi_panel),
                            intraday=self._intraday, chips=self._chips,
                            cal=self.cal, factor_io=self._factor_io)

        raw = np.asarray(spec.fn(ctx), dtype=np.float64)
        if tuple(raw.shape) != tuple(panel.shape):
            # ★ 原来这里是静默 `reshape`：形状错但元素数相同（例如 (C,T)）
            #   会被悄悄转置，产出**看起来完全正常**的错面板。改成直接报错。
            raise ValueError(
                f"因子 {spec.name} 返回形状 {tuple(raw.shape)}，面板要求 {tuple(panel.shape)}。"
                f"因子函数必须返回 (T, C)：T=交易日数、C=股票数。"
                f"（特别注意别返回 (C, T) —— 元素数相同，以前会被静默转置。）")

        rows = panel.day_mask(days)
        sub = raw[rows]
        submask = uni[rows]
        if spec.is_label:
            # 标签是模块③ 的目标变量、不是因子：跳过 winsor 与截面 rank
            # （对未来收益排名没有意义），rank 列写 NaN。但仍写出与因子
            # **完全相同**的 4 列 / 同 dtype / 同分区（用户要求格式统一）。
            val = sub.astype(np.float32)
            rk = np.full(sub.shape, np.nan, dtype=np.float32)
        else:
            val = np.where(submask, sub, np.nan).astype(np.float32)
            rk = cs_rank(sub, self.cfg.winsor, submask, self.cfg.min_cross_section)

        df = _to_frame(days, self.codes, val, rk)
        merged = upsert_year(self.cfg.factors_dir, spec.name, year, df,
                             self.cfg.compression,
                             prune_after=self._run_end,
                             # ★ 起点之后才该有行：合并语义会原样保留分区里更早的旧行，
                             #   起点后移（如 dividend_yield_3y_avg 2012→2013-01-11）后会留下
                             #   起点之前的 NaN 残留，被 check 的「≥起点」判据抓到。见 store。
                             prune_before=max(spec.start_int(self.cfg), self._start_floor),
                             replace_days=days)   # ★ 清掉"本次重算但整列空"的旧行，见 store
        # 让 manifest 里的 max_date 反映裁剪后的真实末尾
        merged = merged[merged["trade_date"] <= self._end_str] if self._run_end else merged
        n_ok = int(np.isfinite(merged["value"].to_numpy()).sum())
        return {
            "factor": spec.name, "year": year, "rows": len(df), "nonnull": n_ok,
            "partition_rows": len(merged),
            "min_date": str(merged["trade_date"].min()),
            "max_date": str(merged["trade_date"].max()),
            # ★ 本任务（因子 × 年）的**实际耗时**（含读盘与写盘）—— 增量效率靠它衡量
            "seconds": round(time.time() - _t_task, 2),
        }

    def _finalize(self, spec: FactorSpec, man: Manifest, plan: dict,
                  results: list[dict], seconds: float) -> dict:
        """父进程收尾：写 manifest（worker 绝不碰状态文件）。"""
        stats = {"factor": spec.name, "years": 0, "rows": 0, "nonnull": 0,
                 "seconds": seconds,          # 整轮墙钟（从 run 开始算）
                 "task_seconds": 0.0,         # ★ 各任务实际耗时之和（并行时 > seconds）
                 "task_days": 0}              # ★ 本因子这次真正重算了多少个交易日
        for r in results:
            stats["task_seconds"] = round(stats["task_seconds"] + float(r.get("seconds") or 0), 2)
            if r.get("rows", 0) == 0 and not r.get("min_date"):
                continue
            man.mark_partition(r["year"], r["partition_rows"], r["nonnull"],
                               r["min_date"], r["max_date"])
            path = self.cfg.factors_dir / spec.name / f"year={r['year']}" / "data.parquet"
            st = path.stat()
            man.partitions[str(r["year"])]["file_identity"] = [st.st_size, st.st_mtime_ns]
            stats["years"] += 1
            stats["rows"] += r["rows"]
            stats["nonnull"] += r["nonnull"]
        stats["task_days"] = int(sum(int(np.size(v)) for v in plan.values()))

        # 覆盖区间要按「交易日连续段」记，不能简单取 [min, max]：
        # 同一年的计划里可能有洞（例如三月有个日期缺口、十二月是回刷窗口），
        # 取 min/max 会把中间几个月错误地标成"已完成"。
        for year, days in plan.items():
            for a, b in self._trading_runs(np.sort(days)):
                man.add_coverage(int_to_str(a), int_to_str(b))

        from .spec import REGISTRY
        for dep in spec.deps:
            if dep in REGISTRY:
                self._wm_cache.pop(dep, None)
        man.input_watermark = {
            dep: self.watermark(dep) for dep in spec.deps
        }
        man.last_run = {"finished_at": _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        **stats}
        man.save()
        return stats

    def _trading_runs(self, days: np.ndarray) -> list[tuple[int, int]]:
        """把交易日集合切成「日历上连续」的若干段。"""
        if days.size == 0:
            return []
        pos = np.searchsorted(self.cal.days, days)
        brk = np.flatnonzero(np.diff(pos) != 1)
        starts = np.concatenate([[0], brk + 1])
        ends = np.concatenate([brk, [days.size - 1]])
        return [(int(days[s]), int(days[e])) for s, e in zip(starts, ends)]

    def run(self, spec: FactorSpec, end_i: int | None = None,
            rebuild: bool = False) -> dict:
        """单因子串行执行（`--jobs 1` 或只有 1 个任务时走这里）。"""
        end_i = end_i or today_int()
        man = Manifest.load(self.cfg.state_dir, spec.name)
        plan = self.plan(spec, man, end_i, rebuild)
        if not plan:
            return {"factor": spec.name, "years": 0, "rows": 0, "nonnull": 0,
                    "seconds": 0.0, "note": "已是最新，无需重算"}
        t0 = time.time()
        results = [self.run_year(spec, y, plan[y]) for y in sorted(plan)]
        return self._finalize(spec, man, plan, results, time.time() - t0)

    # ---------------------------------------------------------------- 并行执行
    def prebuild(self, specs: list[FactorSpec], end_i: int,
                 rebuild: bool = False) -> list[tuple[FactorSpec, Manifest, dict]]:
        """父进程：规划所有因子，并**提前构建共享状态**（衍生层 / ST 事件）。

        这一步是并行的关键：`fork` 出来的 worker 通过写时复制直接继承这些对象，
        否则每个 worker 都要自己重读一遍上游财务数据、重建一次衍生层
        （实测衍生层 ~6s × 10 个 worker，既慢又多读 10 倍磁盘）。
        """
        self._run_end = end_i
        self._end_str = int_to_str(end_i)
        todo = []
        for spec in specs:
            man = Manifest.load(self.cfg.state_dir, spec.name)
            before = man.recipe
            plan = self.plan(spec, man, end_i, rebuild)
            if man.recipe != before:
                # 指纹变了/全量重建：先把「已清空」落盘。否则中途崩了，
                # 下次会读到旧 coverage 而以为这段已完成，新旧口径就混在一起了。
                man.save()
            if plan:
                todo.append((spec, man, plan))

        # Propagate rebuilt parent dates to coupling factors in the same run.
        self._propagate_parent_plans(todo, end_i)
        if todo:
            los, his, warms = [], [], []
            for spec, _, plan in todo:
                d = np.concatenate(list(plan.values()))
                los.append(int(d.min()))
                his.append(int(d.max()))
                warms.append(spec.warmup_days)
            # 用「最早的起点 − 最大的 warmup」做上界，保证一次构建覆盖所有任务
            wlo, whi = minus_days(min(los), max(warms)), max(his)
            # ★ 按本次任务用到的财务字段取并集，惰性建表。
            #   ⚠️ 只要**有一个**因子没声明 `fin_fields`，就必须退回「全建」——
            #      空元组的契约语义是「我全都要」，而不是「我不要字段」。
            #      按并集硬算会把没声明者的字段悄悄排除掉，表现为
            #      `KeyError: 'goodwill'` 这种莫名其妙的报错。
            fin_specs = [s for s, _, _ in todo if any(d in ("stock_income", "stock_balancesheet", "stock_cashflow", "stock_financial_indicator") for d in s.deps)]
            if any(not s.fin_fields for s in fin_specs):
                fin = None
            else:
                fin = frozenset(f for s in fin_specs for f in s.fin_fields)
            self._deriv_fields_run = fin          # ★ worker 必须用同一个字段集（见 run_year）
            self.deriv_for(wlo, whi, fields=fin)
            if self._st is None:
                self._st = st_events(self.up, self.codes)
            # 价格层 / 日内层 / 筹码层 / 耦合 IO 的构造函数都是**零成本**的
            # （只在第一次 panel() 时才读数据），所以无条件建出来；
            # 用不到的因子不会触发任何读盘。
            # ⚠️ 必须在这里建：worker 是 fork 出来的，父进程没建的话
            #    worker 里 `ctx.chip(...)` 会直接抛「引擎没有预建 ChipLayer」。
            self.prices_for(wlo, whi)
            # ★ 派生缓存必须**在父进程里建完**再 fork。
            #   `ensure()` 是「缺了就当场构建」，如果留给 worker，12 个 worker 会
            #   同时抢着重建同一份缓存（55 分钟的活干 12 遍，还会互相写坏 manifest）。
            # ★★ 但年份范围必须按「**真正用到这一层**的因子」算，不能用上面的全局窗口：
            #   全局窗口带的是所有因子的最大 warmup（本轮 `pegh5` 是 2600 天 → 回到 2018），
            #   而 pegh5 只吃财报、根本不用日内/筹码层 —— 用全局窗口会白重建
            #   2018–2024 的派生层（实测：一次 run 在 prebuild 重建 15 个年分区、约 30 分钟，
            #   还会把「只保留 2026」的裁剪结果又建回来）。
            self.intraday_layer().ensure(
                *self._layer_years(todo, ("stock_history_5min", "tdx_minute")))
            self.chips_layer().ensure(
                *self._layer_years(todo, ("stock_cyq_chips",)))
            self.factor_io()
            # ★ 清掉原始上游表缓存再 fork：见 Upstream.clear_cache 的说明。
            #   各派生层已经把这些数据"消化"完了，worker 不需要原始表。
            #   价格层/衍生层里留着的 DataFrame 同样要清 —— fork 是写时复制的，
            #   但 Python 的引用计数会在**读**对象时改头部字段，进而整页复制。
            self.up.clear_cache()
            if self._prices is not None:
                self._prices.clear_raw()
        return todo

    def run_parallel(self, tasks: list[tuple[str, int, np.ndarray]],
                     jobs: int, on_done=None) -> list[dict]:
        """按 (因子, 年) 并行执行。任务之间写的是不同文件，无冲突。"""
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor, as_completed

        global _WORKER_ENGINE
        _WORKER_ENGINE = self                       # fork 后子进程直接可见
        out: list[dict] = []
        ctx = mp.get_context("fork")
        with ProcessPoolExecutor(max_workers=jobs, mp_context=ctx) as ex:
            futs = {ex.submit(_run_year_task, t): t for t in tasks}
            for fut in as_completed(futs):
                try:
                    r = fut.result()
                except Exception as exc:            # noqa: BLE001
                    f, y, _ = futs[fut]
                    log.exception("任务失败 %s %s", f, y)
                    r = {"factor": f, "year": y, "rows": 0, "nonnull": 0,
                         "error": str(exc)[:160]}
                out.append(r)
                if on_done:
                    on_done(r)
        return out


def _to_int(s: str) -> int:
    return int(s[:4]) * 10000 + int(s[5:7]) * 100 + int(s[8:10])


# ------------------------------------------------------------------ 并行 worker
# fork 出来的子进程通过写时复制继承父进程已建好的 Engine（含衍生层与缓存）。
_WORKER_ENGINE: "Engine | None" = None


def _run_year_task(task: tuple[str, int, np.ndarray]) -> dict:
    """worker 入口：只算一个 (因子, 年) 并写盘，不碰 manifest。"""
    from .spec import get
    name, year, days = task
    eng = _WORKER_ENGINE
    if eng is None:                                 # 理论上不会发生
        raise RuntimeError("worker 没有继承到 Engine")
    return eng.run_year(get(name), year, days)


def _to_frame(days: np.ndarray, codes: np.ndarray,
              val: np.ndarray, rk: np.ndarray) -> pd.DataFrame:
    """把 (T,C) 面板转成落盘长表。

    ★ 空帧必须带**正确的 dtype**。踩过的坑：某因子整年无有效值（例如受上游起点限制
      的深滞后因子在早年）→ 这里返回一个 0 行的 object-dtype 空表 → `upsert_year`
      写出一个空的 parquet → 下次 `read_year()` 读回**全 object dtype** →
      下游 `np.isfinite()` 抛 `TypeError: ufunc 'isfinite' not supported`。
      报错点离病根十万八千里。所以在源头就把 dtype 定死。
    """
    T, C = val.shape
    if T == 0 or C == 0:
        return pd.DataFrame({"trade_date": pd.Series(dtype="string"),
                             "stock_code": pd.Series(dtype="string"),
                             "value": pd.Series(dtype="float32"),
                             "rank": pd.Series(dtype="float32")})
    df = pd.DataFrame({
        "trade_date": np.repeat(days, C),
        "stock_code": np.tile(np.asarray(codes, dtype=object), T),
        "value": val.reshape(-1),
        "rank": rk.reshape(-1),
    })
    # ★★ 2026-09-15 晚（用户拍板）：「**NaN 还是落盘的好**」——
    #    去掉原先的 `df[df["value"].notna() | df["rank"].notna()]` 过滤，**整块面板都落盘**
    #    （含 value/rank 全 NaN 的格子）。理由：只写非空行会让
    #    「这只票当天不在股票池 / 上游没数据 / 算了是 NaN」三种情况在产物里**长得一模一样**，
    #    消费方无法区分、覆盖率也没法审计。
    #    代价：行数 = 面板大小（≈3484 只 × 交易日），比原来多 ~20% 行；
    #    NaN 行压缩率高，体积涨幅远小于行数涨幅。`main.py check` 本来就断言
    #    `rank ∈ [0,1] ∪ {NaN}`，是这套口径的配套断言。
    df = df.copy()
    df["trade_date"] = int_to_str_vec(df["trade_date"].to_numpy())
    # 定死 dtype，避免空帧 / 全 NaN 帧落盘后读回来变成 object
    df["trade_date"] = df["trade_date"].astype("string")
    df["stock_code"] = df["stock_code"].astype("string")
    df["value"] = df["value"].astype("float32")
    df["rank"] = df["rank"].astype("float32")
    return df
