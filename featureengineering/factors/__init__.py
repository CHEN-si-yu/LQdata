"""因子实现包 —— 一个文件一个家族，import 即完成注册。

新增因子只需：在本目录建/改一个模块，写 `@register(FactorSpec(...))`，
然后在下面加一行 import。**不需要改框架代码。**
"""

from . import event          # noqa: F401
from . import event2         # noqa: F401
from . import fundamental    # noqa: F401
from . import labels         # noqa: F401

# —— 第二批（2026-09-14，多 Agent 并行开发，一个家族一个文件）——
from . import chips          # noqa: F401
from . import fundflow       # noqa: F401
from . import growth         # noqa: F401
from . import intraday       # noqa: F401
from . import liquidity      # noqa: F401
from . import margin         # noqa: F401
from . import quality        # noqa: F401
from . import momentum       # noqa: F401
from . import technical      # noqa: F401
from . import valuation      # noqa: F401
from . import volatility     # noqa: F401

# —— 第三批（2026-09-15，P0：耦合 / 日内形态 / 筹码深挖 / 资金结构 / 事件 / 财务补漏）——
#   ⚠️ `coupling` 家族用 `ctx.load_factor()` 读**其它因子的产物**：
#      它在 `main.py run` 里会被排到第二趟（等父因子落盘后再算），
#      父因子名必须写在它的 `deps` 里，否则父因子更新不会触发它重算。
from . import coupling       # noqa: F401
#   ⚠️ 另外 5 个家族同批交付（多 Agent 并行开发，各自一个文件）：
#      · overnight_pattern —— 隔夜/日内收益分解 + K线形态 + 缺口行为（纯日线）
#      · chips2            —— 筹码深挖（成本位移 / 尾部风险 / 峰结构）
#      · fundflow2         —— 资金流订单分层 + 两融结构
#      · event3            —— 跌停/新低事件 + 指数衰减
#      · fundamental2      —— 财务补漏（PEG / 现金流 / EBITDA / 环比变化）
from . import overnight_pattern   # noqa: F401
from . import chips2              # noqa: F401
from . import fundflow2           # noqa: F401
from . import event3              # noqa: F401
from . import fundamental2        # noqa: F401

# —— 第四批（2026-09-18，补三个大覆盖缺口：日内深化 / 市场宽度 / 板块 / 资金流动力学）——
#   ⚠️ 每个家族文件只吃 `ctx` 白名单 API，不改 `fea/**`。逐族的「做不到什么 + 为什么」
#      写在各自模块的 docstring 里，接手前先读那一节，别再重试一遍。
#      · intraday2   —— 会话切分 / 极差型波动 / 路径集中度 / 会话 VWAP（25 个）
#      · breadth     —— 个股对市场宽度的敏感度与条件统计量（13 个）
#      · sector      —— 自建「滚动相关性动态行业」的行业相对因子（17 个）
#      · fundflow3   —— 资金流档位**动力学**（水平维已被三条恒等式挖干）（12 个）
#      · fundamental3—— 财务明细：per-unit 比率 YoY / 盈利构成 / 现金流质量（18 个）
from . import intraday2           # noqa: F401
from . import breadth             # noqa: F401
from . import sector              # noqa: F401
from . import fundflow3           # noqa: F401
from . import fundamental3        # noqa: F401

from . import cyq_perf, financial_detail, disclosure_detail  # noqa: F401

__all__ = ["cyq_perf", "financial_detail", "disclosure_detail","fundamental", "event", "labels", "chips", "fundflow", "growth",
           "intraday", "liquidity", "margin", "momentum", "quality", "technical",
           "valuation", "volatility", "coupling", "overnight_pattern", "chips2",
           "fundflow2", "event3", "fundamental2",
           "intraday2", "breadth", "sector", "fundflow3", "fundamental3"]

# ★ 兜底：核对 `register(...)` 的调用次数与注册表大小。
#   防的是「把装饰器写成普通调用」——那是个**静默 no-op**，不报错也不注册，
#   让人以为守卫通过了、其实根本没跑（实测有 Agent 这样被绕过一次）。
from fea.spec import check_all_applied as _check_all_applied   # noqa: E402
_check_all_applied(strict=True)
