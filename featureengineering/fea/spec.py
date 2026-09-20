"""因子声明式注册表 —— 一个因子 = 一个 `@register(FactorSpec(...))` + 一个函数。

与模块① `lingqi/spec.py` 同一套思路：把「元数据」和「实现」放在一起，
main.py 只需遍历注册表就能列出/运行/校验所有因子，新增因子不改框架代码。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .delay import load as _load_delays


@dataclass
class FactorSpec:
    name: str
    group: str                       # quality / growth / value / risk / event / ...
    desc: str = ""
    formula: str = ""                # 逐字抄参考文档，便于日后核对口径
    deps: tuple = ()                 # 依赖的上游数据集（用于输入水位失效判定）
    # 输出起点。**留 None 表示用 conf/config.yaml 的 default_start**——
    # 这样改起点只需改一处配置，不必挨个改因子。
    # 受上游数据起点限制的因子（如 holder_number 只有 2016 起）仍显式写死。
    start: str | None = None
    # 计算窗口要往前多读多少**日历天**，才能保证窗口起点的值和全量重算一致。
    # 财务类要覆盖 TTM(4季) + 同比(再4季) + ann_date 滞后 → 500 天起；
    # 纯日频滚动（如 20 日涨停次数）40 天足够。
    warmup_days: int = 500
    higher_is_better: bool = True    # 方向语义（rank 始终是「值大在前」，方向由下游用）
    version: int = 1                 # 因子逻辑变更时 +1，会强制全量重建
    enabled: bool = True
    note: str = ""                   # 实测坑 / 与参考文档的偏离，写在这里
    allow_qfq: bool = False          # 显式放行前复权依赖，仅在有充分理由时置 True
    # 显式声明「我处理了这些滞后表的可得日」。deps 里出现 LAGGED_DATASETS 而这里没写
    # 对应名字 -> 注册期直接抛错（见 register 的守卫）。
    lagged_ok: tuple = ()
    # ★ 标签（label）：给模块③ 用的目标变量，不是因子。
    #   True 时引擎**跳过 winsor 与截面 rank**（rank 列写 NaN），
    #   但仍写出与因子完全相同的 4 列 / 同 dtype / 同分区（用户要求格式统一）。
    is_label: bool = False
    # 标签需要看到未来：面板要向后延伸这么多个交易日，否则每年最后几天恒为 NaN。
    forward_days: int = 0
    # 该因子用到的财务字段（ttm/point/ind）。引擎按并集**惰性**构建衍生层 ——
    # 不写就是"全都要"，单跑一个因子时会白建 80 个字段的版本表。
    fin_fields: tuple = ()
    fn: Callable | None = None

    def resolved_start(self, cfg) -> str:
        """实际输出起点：全局下界与数据源可用起点二者取较晚日期。"""
        return max(self.start or cfg.default_start, cfg.default_start)

    def start_int(self, cfg) -> int:
        s = self.resolved_start(cfg)
        return int(s[:4]) * 10000 + int(s[5:7]) * 100 + int(s[8:10])

    def recipe(self, cfg) -> str:
        """逻辑指纹：变了就必须全量重建，否则新旧口径会混在同一个面板里。"""
        extra = ""
        if self.is_label or self.forward_days:
            # 标签的 forward 窗口口径变了，历史值就会变，必须进指纹
            extra = f"|lbl={int(self.is_label)},fwd={self.forward_days}"
        return (f"{self.name}@v{self.version}|{self.formula}|{self.warmup_days}"
                f"|{self.resolved_start(cfg)}{extra}")


REGISTRY: dict[str, FactorSpec] = {}

# `register(spec)` 是**装饰器工厂**：必须写成
#     @register(FactorSpec(...))
#     def name(ctx): ...
# 单独写 `register(FactorSpec(...))` 是**静默 no-op** —— 不报错、也没注册。
# 实测有因子开发 Agent 就这样"通过"了守卫探测：护栏看着失效，其实是根本没跑。
# 比抛错更危险，所以在每个家族模块导入完后核对这两个计数（见 factors/__init__.py）。
_N_REGISTER_CALLS = 0


# ══════════════════════════════════════════════════════════════════════════
# ★★★ PIT 红线（用户 2026-09-13 明确的基本原则）★★★
#
#    「历史的因子不能够因为未来分红的事件出现变化。」
#
# 为什么前复权（qfq）会破坏这条原则：
#     前复权把整条价格序列按 **最新** 的复权因子缩放：
#         price_adj(t) = price_raw(t) × adj_factor(t) / adj_factor(T_latest)
#     一旦发生新的分红/送转，`adj_factor(T_latest)` 变了，
#     **全部历史价格会一起被重算** —— 于是同一段历史，今天算出的因子值
#     和昨天算出的不一样。回测里这就是前视偏差：模型「知道」了未来的分红。
#
#     后复权（hfq）锚定在序列起点，历史值不随新分红改变，是 PIT 安全的。
#
# 正确的价格因子做法（将来加估值/动量因子时照此实现）：
#     1. 价格**水平**类（市值、book-to-market）→ 用**未复权**价 × 当期已披露股本；
#     2. 收益**比率**类（动量、波动）→ 用未复权价配合 `stock_adj_factor`
#        在**当日之前**的累计复权因子还原，等价于「截至当日的后复权」；
#     3. 任何情况下都不要把 qfq 面板直接喂给因子函数。
#
# 违反会被 `register()` 直接拒绝（要放行必须显式写 allow_qfq=True 并给出理由）。
# ══════════════════════════════════════════════════════════════════════════
QFQ_DATASETS: dict[str, str] = {
    "stock_kline_adj": "周期K线（前复权 qfq）",
    "stock_daily_adj": "日K线（前复权 qfq）",
}


# ══════════════════════════════════════════════════════════════════════════
# ★★ 第二个 PIT 陷阱：数据**到达晚**同样会改写历史
#
#    上面那条 PIT 红线说的是「数据被**重算**」（前复权）；
#    这一条说的是「数据**来晚了**」—— 后果一样（同一个 T 的因子值前后不同），
#    但更隐蔽，因为**它当天看起来完全正确**。
#
#    实测（2026-09-13 首次穷举，2026-09-15 由日更工程逐日复核）：下面这些表
#    交易日的行要么当天拿不到、要么要到次日才稳定，T 日的行要等下一个交易日才有。
#
#    不处理的后果（以 margin 为例，T = 09-11）：
#        今天跑（上游只到 09-10）：T 的因子取到 09-10 的值   → 回退一格，看起来对
#        明天跑（上游到了 09-11）：T 的因子取到 09-11 自己   → **同一个 T 的值变了**
#
#    正确做法：把这些表的 (T, C) 网格整体**下移一个交易日**
#    （`ctx.lag_grid(mat, n)`，面板本身就是交易日历，移一行 = 移一个交易日），
#    等价于「T 日的因子只用到 trade_date <= T−n 的记录」。
#
#    ★ **名单不在这里硬编码**：由 `fea/delay.py` 从
#      ① 日更工程的逐日实测 `everyday_tasks/state/delay_history.json`
#      ② 模块① 的静态声明 `datadownload/conf/frequency.yaml`
#      合并而成（取较大值，保守方向）。两张表都在外面持续更新，
#      所以这里跟着自动走 —— 手工维护的常量迟早会和观测脱节。
#    ⚠️ 2026-09-15 更正：`stock_dragon_tiger` / `stock_top_list` 原先被列为
#      "别误判成滞后"，**那条结论是错的**（当时只看"上榜家数每天波动"，
#      没看最后一天是否为 0）。日更工程连续观测确认它们确实晚 1 个交易日。
#      `stock_report_rc`（研报）与 `dc_daily`（板块日线）同日获同样的确认。
#
#    ⚠️ 仍然别误判成滞后（行数本来就波动，最后一天为 0 属正常）：
#      stock_limit_up / stock_limit_list（涨停家数 40~95 波动）、
#      全部季频/月频/不定期表。
#    ★ 2026-09-16：原列表里的 `stock_kline`（周线/月线）**已彻底删除**，不在下游清单中。
#      它是全库唯一「每次抓取都改写历史值」的表：周/月线实为「抓取请求范围 ∩ 该周期」
#      内日线的聚合（非日历周期聚合），窗口下界一滑动已结束周期的值也会变；
#      且 14 个数据列与 stock_daily 完全相同、零额外信息。
#      **需要周/月频特征请从 stock_daily 现算**（用交易日历切周期，比接口那份准）。
# ══════════════════════════════════════════════════════════════════════════
LAGGED_DATASETS: dict[str, int] = _load_delays()       # ← fea/delay.py，勿在此硬编码


def register(spec: FactorSpec):
    global _N_REGISTER_CALLS
    _N_REGISTER_CALLS += 1
    validated = False

    def deco(fn):
        nonlocal validated
        validated = True
        import warnings
        if not isinstance(fn, type(lambda: None)) or getattr(fn, "__name__", "") == "<lambda>":
            # 允许用 lambda 注册（内部测试用），但**必须显式写 @register(...)**：
            # 单独 `register(FactorSpec(...))` 不会走到这里，由 _check_all_applied 兜底。
            pass
        lat = [d for d in spec.deps if d in LAGGED_DATASETS and d not in spec.lagged_ok]
        if lat:
            detail = "、".join(f"{d}（晚 {LAGGED_DATASETS[d]} 个交易日）" for d in lat)
            raise ValueError(
                f"因子 {spec.name} 依赖了可得性滞后的表：{detail}。\n"
                f"  这些表 trade_date=d 的数据要到 d 的**下一个交易日**才拿得到。\n"
                f"  不做位移的话，T 日的因子值会在数据到达后**静默改变** ——\n"
                f"  当天算出来完全正确，第二天悄悄变成另一个值，而且没有任何报错。\n"
                f"  正确做法（二选一）：\n"
                f"    1. 行位移（推荐，零框架依赖）：把结果整体下移一行\n"
                f"         return ctx.lag_grid(grid, 1)\n"
                f"    2. 日期位移：用 ctx.next_trading_day(days) 而不是 days 去做 asof。\n"
                f"  然后在 FactorSpec 里显式声明 lagged_ok=({', '.join(repr(x) for x in lat)},)。")
        bad = [d for d in spec.deps if d in QFQ_DATASETS]
        if bad and not spec.allow_qfq:
            raise ValueError(
                f"因子 {spec.name} 直接依赖前复权数据集 {bad}。\n"
                f"  前复权序列会被未来的分红/除权事件整体重算，"
                f"导致**历史因子值随未来事件变化**——这是明确禁止的。\n"
                f"  正确做法见 fea/spec.py 顶部的「PIT 红线」注释："
                f"价格水平用未复权价，收益比率用未复权价+截至当日的复权因子。\n"
                f"  确实需要用（例如口径本身就定义为复权价之比）时，"
                f"在 FactorSpec 里显式写 allow_qfq=True 并在 note 里说明理由。")
        spec.fn = fn
        if spec.name in REGISTRY:
            raise ValueError(f"因子重名：{spec.name}")
        REGISTRY[spec.name] = spec
        return fn
    return deco


def check_all_applied(strict: bool = False) -> int:
    """核对「`register()` 被调用了几次」与「注册表里有几个因子」。

    ★ 防的是这个静默失败：写成 `register(FactorSpec(...))` 而不是
      `@register(FactorSpec(...))`。前者不报错、也不注册 ——
      **护栏看起来失效了，其实是根本没跑**。比抛错危险得多。

    返回漏注册的个数；`strict=True` 时直接抛错（在 `factors/__init__.py` 的
    导入完成后调用）。允许 lambda 注册，所以只比对数量。
    """
    missing = _N_REGISTER_CALLS - len(REGISTRY)
    if missing and strict:
        raise RuntimeError(
            f"有 {missing} 个 FactorSpec 漏注册：`register(...)` 被调用了 "
            f"{_N_REGISTER_CALLS} 次，但注册表里只有 {len(REGISTRY)} 个。\n"
            f"  最常见的原因是把装饰器写成了**普通调用**：\n"
            f"      ✘ register(FactorSpec(...))        # 静默 no-op\n"
            f"      ✔ @register(FactorSpec(...))       # 正确\n"
            f"        def my_factor(ctx): ...")
    return missing


def get(name: str) -> FactorSpec:
    if name not in REGISTRY:
        raise KeyError(f"未知因子：{name}")
    return REGISTRY[name]


def all_specs() -> list[FactorSpec]:
    return [REGISTRY[k] for k in sorted(REGISTRY)]


def enabled_specs() -> list[FactorSpec]:
    return [s for s in all_specs() if s.enabled]
