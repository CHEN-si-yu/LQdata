"""V17 · analysis —— **奇数（激进）线第四版**（池内训练 + 池内打分）。约定见 docs/architecture.md。

V15 从 V13 复制，**只在策略层改两个旋钮**（都由 `tmp/vlines/strat_oos_lab.py` 的
1893 天配对定案）：`entry_rule` 由 `board` 改回 **`open`**（打板口径在全部 12 种配置下更差）、
`hold_band` 20 → **40**（年费 12.6%→8.4%）。训练配方一字未改 ⇒ 交付打分与 V12/V13/V14 一致。

V13 引入的两件事保留：

① **缓冲区换仓**（`RECIPE['hold_band']`）：`cash_backtest` 里"还在前 `hold_band` 名"的
   持仓**不卖**，空出的仓位再按排名补足。`hold_band <= topn` 时逐位退回 V11 的
   "每天无脑重排"口径。针对的是每日全额换仓 ~25%/年 的往返成本。**只作用于本单元策略**，基线不动。
② **bagged vs argmax 的配对对照**（`bag_vs_argmax`）—— 从偶数线 V12 搬来：
   交付权重改为「val_wei 前 k 名平均」之后，必须把两份权重的验证指标成对留档，
   否则"这个改动到底有没有用"就只能靠跨版本抽奖来猜。

★ 沿用 V11 的口径（一字未改）：本单元策略 `top5_1d`（每日调仓）、基线 `top1_1d`、
   可买判定走 `Prices.entry_ok()`（`RECIPE['entry_rule']='board'` 打板口径，
   **卖出侧不放开**）、`picks.md` 的买卖日按本线实际持有期写。
★ 热门池在 V13 **关掉交付口径**（`hot_pool['frac']=1.0` ⇒ `hot_filter` 是恒等），
   依据是 V11 上那张单调的扫描表（见 model.py 文件头）。

独立单元约束：除 `trainingdata` 数据块（含其中的 `prices` 价格块，2026-09-26 起）外不依赖任何外部环境
——不读 /sys、/proc，不调用 nvidia-smi 之类的外部命令，也不引用单元目录之外的路径。

注：本文件在 `import model` **之前**就导入了 numpy/pandas/scipy，而线程数环境变量是
`model.py` 导入时设的 ⇒ `MX_THREADS` 管不到分析进程的 numpy 线程。低危（分析阶段不做
重矩阵运算），留此说明以免误判。

调用方式（2026-09-26 起）：训练调度搬到了 `run.py`（简单的分季度×折循环），本文件只负责
评价与产物 —— 默认读 `model_train` 的推送打分做完整评价与回测，`--from-scores` 则直接读
集成打分，两者都不再自己起子进程。
"""

# =============================================================================
# ① 导入
#
# matplotlib 只在下标曲线的函数里按需 import：IC-only 的运行不该为它付加载开销。
# =============================================================================
import json
import os
import sys
sys.dont_write_bytecode = True
import time
import unicodedata
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import rankdata

# 路径合同与固定布局在 model.py（2026-09-26 从 run.py 迁来，run.py 只剩训练调度）。
from model import (ROOT, RECIPE, Panel, Prices, atomic_json, axis, corr, score_of,
                   blend_linear, hot_filter, predict_heads, quarters, PredictModel, predict, torch,
                   run_root, analysis_input_root, info_root, run_name, output_file, validate_layout,
                   stock_list_path, calendar_path)
RUN_ROOT = run_root()
INPUT_ROOT = analysis_input_root()

# =============================================================================
# ② 指标：逐标签 IC 表
# =============================================================================


def ic_table(pred, panel, days):
    """逐标签算 RankIC / ICIR / Pearson IC：先按日算，再在日间平均。"""
    results = []
    for label in panel.labels:
        ic = []
        pearson = []
        for p, d in zip(pred, days):
            y = panel.Y[label][d]
            ok = np.isfinite(p) & np.isfinite(y)
            if ok.sum() < 3:
                continue
            ic.append(corr(rankdata(p[ok]), rankdata(y[ok])))
            pearson.append(corr(p[ok], y[ok]))
        std = float(np.std(ic, ddof=1)) if len(ic) > 1 else 0.
        results.append(dict(label=label, days=len(ic), RankIC=float(np.mean(ic)),
                            ICIR=float(np.mean(ic) / std) if std else 0.,
                            Pearson_IC=float(np.mean(pearson))))
    return results

# =============================================================================
# ②b 逐头指标（四指标 IC/ICIR/top_return/top_return_stability 的口径出处：
#     tmp/min_test.ipynb 的 get_ret_ic / get_metrics）
#
# 用户 2026-09-24 定：每版实验在 docs/iterations.md 里记录 IC / ICIR /
# top_return / top_return_stability，另附季度表现表。这一节只做「逐日序列 →
# 指标」的换算，组合与 IC 的定义照抄 min_test：
#   · 组合：逐日调仓，按打分降序建仓，每只票建仓 min(剩余资金, 当日成交额)；
#   · 收益：Σ(1日标签 × 仓位) / 资金，单位 %，不连乘；
#   · IC：逐日截面 Pearson(打分, 1日标签)。
# ★ 这里的 IC 是 Pearson，训练侧验证与 report.json 里的 RankIC 是逐日
#   Spearman —— 同名不同源，报告里不能只写「IC」。
# 用户 2026-09-26 定：再加 1d / 5d 的 RankIC（见 `rank_ic_horizons()`）。
# ★ V3 起这两列**真的**与训练侧验证同源：`model.py:validation()` 与本节都算
#   「**打分**对 h 日标签」的逐日截面相关（V2 那边 valid 算的是"第 h 个头对自己
#   的标签"、test 算的是"5d 头对 h 日标签"，同名不同物，不能相减）。
# =============================================================================

TOP_PORTFOLIO = dict(topn=500, money=1.5e9, label='label_ret_1d')


def rank_ic_horizons():
    """评价窗与训练验证两侧都报 RankIC 的 horizon：1d 对照 + 打分主 horizon。

    ★ V3 起 `RECIPE['score_label']` **只决定报表口径**（报哪两个 horizon），不再决定
    "取哪个头"——打分是 `model.py:score_of()` 的多头秩平均。写成函数只因为
    `horizon_of()` 定义在 §⑧b（文件靠后），模块级常量会在它之前求值。
    """
    return (1, horizon_of(RECIPE['score_label']))


def top_portfolio_daily(pred, panel, days, spec=TOP_PORTFOLIO):
    """评价窗内逐日的组合日收益（%）、逐头 Pearson IC 与逐头 RankIC 序列。

    只读标签和成交额：不碰价格层、不套涨跌停/停牌掩码，也不计费用——这正是
    min_test 的无摩擦口径，成交额在这里只当资金容量上限（缺失或为 0 的票拿
    不到仓位）。标签缺失按 0 计，照抄 min_test 的 fillna(0)：本快照评价窗内
    逐日缺失率 ≤0.05%，对结果无实质影响。

    两个 IC 家族不同源，各自与训练侧同名指标同式，便于对照：
      · IC：逐日截面 Pearson，**标签缺失补 0**（min_test 口径）；
      · RankIC：逐日截面 Spearman，**只用成对有效值**（缺失标签剔除）——
        补 0 会在一堆真实收益里插进一批并列的假值，秩相关会被它们带偏。
    """
    topn, money = spec['topn'], spec['money']
    label = spec['label']
    horizons = rank_ic_horizons()
    daily_ret = np.zeros(len(days))
    daily_ic = {h: np.zeros(len(days)) for h in horizons}
    daily_rank_ic = {h: np.zeros(len(days)) for h in horizons}
    for i, d in enumerate(days):
        score = pred[i]
        known = np.isfinite(score)          # 打分缺失的票既排不了序、也进不了 IC
        for h in horizons:
            yh = panel.Y[f'label_ret_{h}d'][d]
            ok = known & np.isfinite(yh)
            # corr() 对「有效样本 <3」返回 0，与 ic_table 的跳过阈值同值。
            daily_rank_ic[h][i] = corr(rankdata(score[ok]), rankdata(yh[ok]))
            daily_ic[h][i] = corr(score[known], np.where(np.isfinite(yh[known]), yh[known], 0.))
        # 组合：打分降序取前 topn 只，仓位按「先到先得、单只不超过当日成交额」分配，
        # 也就是累计建仓额取 min(资金, 累计成交额) 后的逐只差分；资金用尽后其余为 0。
        y = panel.Y[label][d]
        order = np.flatnonzero(known)[np.argsort(-score[known], kind='stable')][:topn]
        cap = np.where(np.isfinite(panel.amount[d, order]), panel.amount[d, order], 0.)
        held = np.minimum(money, np.cumsum(cap))
        weight = np.diff(np.r_[0., held]) / money
        ret = np.where(np.isfinite(y[order]), y[order], 0.) * 100
        daily_ret[i] = float(weight @ ret)
    return daily_ret, daily_ic, daily_rank_ic


def top_metrics(daily_ret, daily_ic, daily_rank_ic):
    """逐头 PEARSON/RankIC + 三指标 + 区间总收益。

    两个「稳定性」都是均值 ÷ 标准差，未年化；标准差用样本口径（ddof=1），
    与 min_test 里的 pandas std 一致。top_return 是日均收益（%），不是区间累计。
    `total_return` 则是把同一串逐日收益**复利**到整窗的总收益（%）：
    ∏(1+r/100)−1，即「这段时间真按这个无摩擦组合跑下来赚了多少」——用户
    2026-09-26 要求表里报这个。两个都留：min_test 四指标要 top_return，
    评价表给 total_return。
    `IC` 与 `IC_1d` 是同一个数（前者是 min_test 四指标里的名字，后者是逐头表头），
    写两遍是为了两边都不必知道对方的命名习惯。
    """
    h1, h5 = rank_ic_horizons()
    ic_1d = daily_ic[h1]
    ic_std = float(np.std(ic_1d, ddof=1)) if len(ic_1d) > 1 else 0.
    ret_std = float(np.std(daily_ret, ddof=1)) if len(daily_ret) > 1 else 0.
    return dict(IC=float(np.mean(ic_1d)),
                ICIR=float(np.mean(ic_1d) / ic_std) if ic_std else 0.,
                top_return=float(np.mean(daily_ret)),
                top_return_stability=float(np.mean(daily_ret) / ret_std) if ret_std else 0.,
                total_return=float((np.prod(1. + daily_ret / 100) - 1) * 100),
                **{f'IC_{h}d': float(np.mean(daily_ic[h])) for h in (h1, h5)},
                **{f'RankIC_{h}d': float(np.mean(daily_rank_ic[h])) for h in (h1, h5)})


# =============================================================================
# ②d ★ V17：**可实现口径** —— 把"买不进的票"从 top-K 里剔掉再算一遍
#
# 执行缺口诊断（`tmp/vlines/exec_gap_lab.py`，见 docs/iterations.md）测出：
# **表观 top5 alpha 的 38% 来自买不进的票**，其中 **93% 是一字板**
# （开盘=最高=收在涨停、整日封死，任何日频口径都买不到）。于是所有"前 K 只标签均值"
# 的报表都系统性偏乐观 —— 只看它会把"一半是真的"的改进当成"完全是真的"。
#   实测例：epoch-bagging 的增益在理想口径 +0.377%（t=4.25）、可实现口径 +0.176%（t=2.38），
#   而它的 top1 增益在可实现口径下**变负**（−0.266%）。
# 两份数**都留**：理想口径与历史报表逐位可比，可实现口径才是账户够得着的那部分。
# =============================================================================

REALIZABLE_K = 5


def realizable_topk(pred, panel, prices, days, k=REALIZABLE_K, label=None):
    """前 k 名的标签均值，两个口径：**理想**（不看可买性）与**可实现**（只留 T+1 买得进的）。

    ★ 可实现口径**不回补**：买不进时回补下一名会换进别的票，那是"另一个策略"；
      回补的代价单独测过（−0.0124 %/天，很小），所以这里只用"原来的票里买得进的那些"。
    ★ 一只都买不进的日子记 **NaN**、不填 0 —— 填 0 会把"当天没有持仓"混进"持仓收益为 0"，
      而且这两件事在配对检验里的含义完全不同。
    """
    label = label or RECIPE['score_label']
    entry = prices.next_entry_ok()
    ideal, real = [], []
    for i, d in enumerate(days):
        d = int(d)
        p, y = pred[i], panel.Y[label][d]
        known = np.isfinite(p)
        ranked = np.flatnonzero(known)[np.argsort(-p[known], kind='stable')][:k]
        if len(ranked) < k:
            ideal.append(np.nan); real.append(np.nan)
            continue
        ideal.append(float(np.nanmean(y[ranked])))
        ok = entry[d][ranked] & np.isfinite(y[ranked])
        real.append(float(np.nanmean(y[ranked][ok])) if ok.sum() else np.nan)
    return np.asarray(ideal), np.asarray(real)


def realizable_summary(pred, panel, prices, eval_days, rows):
    """把两个口径的 top-K 指标汇成一块，写进 `final_audit.json` 并打日志。

    `rows` 是评价窗的四季度 + 总计行（用来按季度切段），复用同一套切段口径。
    """
    ideal, real = realizable_topk(pred, panel, prices, eval_days)
    out = dict(topk=REALIZABLE_K, label=RECIPE['score_label'],
               ideal_mean_pct=float(np.nanmean(ideal)) * 100,
               realizable_mean_pct=float(np.nanmean(real)) * 100,
               ratio=float(np.nanmean(real) / np.nanmean(ideal)),
               days=int(len(ideal)), days_with_position=int(np.isfinite(real).sum()))
    # 配对差：同一天两个口径之差 —— 这就是"买不到的票贡献了多少"
    diff = real - ideal
    se = float(np.nanstd(diff, ddof=1) / np.sqrt(np.isfinite(diff).sum()))
    out['gap_pct'] = float(np.nanmean(diff)) * 100
    out['gap_t'] = float(np.nanmean(diff) / se) if se else 0.
    text = (f'===== 可实现口径（前 {REALIZABLE_K} 名，{RECIPE["score_label"]}，'
            f'{out["days"]} 天，**评价窗**）=====\n'
            f'  ★ 只能当参照：242 天对 top5 没有统计功效（要 80% 功效检出 +0.27% 需约 37 年）。\n'
            f'    有功效的版本在 **1893 天样本外**上，见 docs/iterations.md：'
            f'那里可实现的 top5 只有理想的约 65%。\n'
            f'  理想（不看可买性） {out["ideal_mean_pct"]:+.4f} %/天\n'
            f'  可实现（只留买得进） {out["realizable_mean_pct"]:+.4f} %/天'
            f'（{out["ratio"]:.1%}，有持仓 {out["days_with_position"]} 天）\n'
            f'  差（买不到的票的贡献） {out["gap_pct"]:+.4f} %/天  t={out["gap_t"]:+.2f}')
    print('\n' + text, flush=True)
    return out, text


def display_width(text):
    """文本在等宽字体下的列数：东亚全角字符占 2 列，其余 1 列。

    pandas 的 `to_string` 按**字符数**排宽度，中文表头会整体错位；评价块要贴进
    picks.md 的 ```text 围栏里，所以自己按显示宽度铺格。
    """
    return sum(2 if unicodedata.east_asian_width(ch) in 'WF' else 1 for ch in str(text))


def summary_table_text(rows, keys, headers):
    """四季度行 + 总计行、表头下画一条横线的固定宽度表。

    `keys` 是每行取数的键，`headers` 是对应表头文字；数值统一四位小数
    （收益列由调用方先换成 %）。列宽按 `display_width` 算，中英混排也对齐。
    """
    body = [[r['quarter'], *[f'{r[k]:.4f}' for k in keys]] for r in rows]
    table = [['季度', *headers], *body]
    widths = [max(display_width(cell) for cell in col) for col in zip(*table)]
    def line(cells):
        return '  '.join(c + ' ' * (w - display_width(c)) for c, w in zip(cells, widths)).rstrip()
    return '\n'.join([line(table[0]), '-' * (sum(widths) + 2 * (len(widths) - 1)),
                      *[line(row) for row in table[1:]]])


def metric_blocks_text(rows, valid_rows=None):
    """两张同版式的五列表：评价窗（test）+ 训练验证集（valid）。

    用户 2026-09-26 定：每张表四季度行 + 总计行、五列
    `收益(%) / IC_1d / IC_5d / RankIC_1d / RankIC_5d`，不带折数；`收益` 是
    **区间总收益**（复利到整窗，见 `top_metrics` / `training_validation_rows`），
    不是日均。两表的行汇总口径不同：test 的季度行是各自季度窗、总计行是整窗 242 天
    一次复利；valid 的行是各折自己验证窗的总收益再按 4 折 / 16 折取均值。
    日志、picks.md 与报告用同一段文字，避免三处各写一份格式化代码。
    """
    h1, h5 = rank_ic_horizons()
    headers = [f'IC_{h1}d', f'IC_{h5}d', f'RankIC_{h1}d', f'RankIC_{h5}d']
    keys = [f'IC_{h}d' for h in (h1, h5)] + [f'RankIC_{h}d' for h in (h1, h5)]
    text = ('---单一模型评估---\n\n'
            f'===== test 评价窗（{RECIPE["test_start"]}～{RECIPE["test_end"]}，'
            f'{rows[-1]["days"]} 天）=====\n'
            f'{summary_table_text(rows, ["total_return", *keys], ["收益(%)", *headers])}\n')
    if valid_rows:
        text += ('\n\n===== valid 训练验证集（各折 best epoch；'
                 '季度=4 折均值，总计=16 折均值）=====\n'
                 f'{summary_table_text(valid_rows, ["total_return", *keys], ["收益(%)", *headers])}\n')
    return text


def print_metric_blocks(rows, valid_rows=None):
    """把上面那段评价文字打到日志（数字以 main_metrics_1d.csv 为准）。"""
    print('\n' + metric_blocks_text(rows, valid_rows), flush=True)


def evaluation_slice(pred, panel, days):
    """固定四季度评价窗的（打分切片, 日期索引）；打分没覆盖满窗就直接报错。

    入参 days 是全部打分日（含评价窗之后的尾段），这里只取评价窗那一段。
    """
    selected = (panel.days[days] >= RECIPE['test_start']) & (panel.days[days] <= RECIPE['test_end'])
    eval_days = days[selected]
    expected = np.flatnonzero((panel.days >= RECIPE['test_start']) & (panel.days <= RECIPE['test_end']))
    if not np.array_equal(eval_days, expected):
        raise ValueError('打分未完整覆盖固定四季度评价窗')
    return pred[selected], eval_days


def evaluation_metrics(ev, panel, eval_days):
    """评价窗的逐日序列与逐头指标行（口径见 §②b），供回测与 picks 共用。"""
    daily_ret, daily_ic, daily_rank_ic = top_portfolio_daily(ev, panel, eval_days)
    rows = []
    for q in [*quarters(panel.days), '总计']:
        loc = (np.ones(len(eval_days), dtype=bool) if q == '总计'
               else np.array([str(pd.Period(panel.days[d], freq='Q')) == q for d in eval_days]))
        rows.append(dict(quarter=q, days=int(loc.sum()),
                         **top_metrics(daily_ret[loc], {h: s[loc] for h, s in daily_ic.items()},
                                       {h: s[loc] for h, s in daily_rank_ic.items()})))
    return dict(rows=rows, daily_ret=daily_ret, daily_ic=daily_ic, daily_rank_ic=daily_rank_ic)

# =============================================================================
# ②c 训练侧验证集表现（各折 `complete.json.best_validation` 的汇总）
#
# 用户 2026-09-26 定：picks.md 除了评价窗的指标，还要给出「训练过程中 valid 的
# 表现」。数据源就是各折训练收尾时写进 `complete.json` 的 `best_validation`
# （`model.py:validation()` 在被选中那个 epoch 上算的），不重新加载权重、不重算。
# ★ 两条必须写进报告的限制：
#   ① 这是**选轮指标本身**（best epoch 就是让 val_wei 最大的那一轮），带选择偏差，
#      只能当训练侧参照；样本外口径以 test 表那五列为准。
#   ② 两边的日期不重叠：验证段是各折切分里从 2018 起、四折互不重叠、最晚到 2025-04
#      的那四块（见 `model.py:splits()`），评价窗从 2025-07-01 起。所以两边的数字
#      **不能直接相减**说「衰减了多少」。
# =============================================================================


def fold_completions():
    """按季度、折号读 16 份 `complete.json`（只读 JSON，不碰权重文件）。"""
    rows = []
    for q in quarters(axis()[0]):
        for f in range(1, 5):
            path = RUN_ROOT / 'model_train' / q / f'fold{f}' / 'complete.json'
            if not path.is_file():
                raise FileNotFoundError(path)
            rows.append(json.loads(path.read_text()))
    return rows


def fold_valid_days():
    """各折验证窗的天数（`split.json` 的 valid 段），键 = (季度, 折号)。

    `complete.json` 只存了验证指标的标量，总收益要按验证窗长度折算 5 日段，
    所以另读一次切分记录。打分元数据里的折记录不带切分，所以这里一律从
    `model_train` 读——`split.json` 是常驻产物，两条入口都读得到。
    """
    days = {}
    for q in quarters(axis()[0]):
        for f in range(1, 5):
            path = RUN_ROOT / 'model_train' / q / f'fold{f}' / 'split.json'
            days[(q, f)] = sum(span['days'] for span in json.loads(path.read_text())['valid'])
    return days


def bag_vs_argmax(records):
    """★ V12：同一批训练上「平均权重 − argmax 权重」的配对汇总。

    两份权重出自**同一次运行、同一个种子、同一批数据**，唯一的差别是"交出去哪一份"
    ⇒ 这是**干净比较**，不受 0.22%~0.39% 的跨版本「选轮抽奖」地板影响。
    重复单位是**折**（16 个季度×折）：每折在它自己的验证块上算两个口径的指标，再逐折配对。

    ★ 两条必须写在报告里的限制：
      ① 验证块就是**选轮用的那一块**（best epoch 由它的 val_wei 选出）⇒ 两边都带选择偏差，
         **绝对水平不可引用**（会偏乐观），只有**两者的差**有意义 —— 同一个偏差在两边都在；
      ② 折与折之间不独立（同一段市场历史、相邻季度训练窗高度重叠）⇒ t 值只作量级参照，
         真正的判据是**符号一致性**（win 数）与它在有功效窗口上的复核。
    """
    rows = [r for r in records if r.get('argmax_validation')]
    if not rows:
        return None
    fields = ['val_wei', 'IC_1d', 'IC_5d', 'RankIC_1d', 'RankIC_5d', 'val_return_proxy']
    out = dict(folds=len(rows), topk=rows[0].get('bag', {}).get('topk'),
               bag_epochs=[r.get('bag', {}).get('epochs') for r in rows], metrics={})
    for name in fields:
        bagged = np.array([float(r['best_validation'][name]) for r in rows])
        argmax = np.array([float(r['argmax_validation'][name]) for r in rows])
        diff = bagged - argmax
        se = float(np.std(diff, ddof=1) / np.sqrt(len(diff))) if len(diff) > 1 else 0.
        out['metrics'][name] = dict(bagged=float(bagged.mean()), argmax=float(argmax.mean()),
                                    delta=float(diff.mean()),
                                    t=float(diff.mean() / se) if se else 0.,
                                    wins=int((diff > 0).sum()))
    return out


def bag_comparison_text(summary):
    """把上面的配对汇总压成一段日志文字（数字落 final_audit.json，叙述归 Agent）。"""
    if not summary:
        return ''
    lines = [f'===== V12 epoch-bagging 对照（前 {summary["topk"]} 名权重平均 vs argmax；'
             f'{summary["folds"]} 折，配对）=====',
             f'{"指标":<20}{"bagged":>12}{"argmax":>12}{"差":>12}{"t":>8}{"bag 胜":>8}']
    for name, m in summary['metrics'].items():
        lines.append(f'{name:<20}{m["bagged"]:>12.5f}{m["argmax"]:>12.5f}{m["delta"]:>+12.5f}'
                     f'{m["t"]:>+8.2f}{m["wins"]:>5}/{summary["folds"]}')
    return '\n'.join(lines)


def training_validation_rows(records):
    """各折 `best_validation` → valid 表：季度 4 折均值 + 总计 16 折均值。

    只汇总 `model.py:validation()` 已经算好并落盘的那几个字段，不重算任何指标：
      · IC_<h>d / RankIC_<h>d：逐日截面 Pearson / Spearman 的日等权平均。
        ★ V3 起这两族是「**打分**对 h 日标签」，与评价窗逐字同式（Pearson 补 0、
        Spearman 剔缺失、无效日计 0）——两张表逐列同源，可以直接对照读；
        V2 那边算的是「第 h 个头对自己的标签」，与 test 表的「5d 头对 h 日标签」
        同名不同物，不能相减。
      · 代理收益：可执行 top5 的 5d 标签收益日平均，这里 ×100 换成 %。★ V3 起它
        **不再进 val_wei**（V2 是按 100× 混进选轮判据的），只为这张表的收益列而算；
        与 test 表的 `total_return` **同量纲但不同物**（那边是 500 只、15 亿元的无摩擦
        组合，这边是 5 只等权、无摩擦的标签代理）；
      · 总收益：把上面的 5 日代理收益按「每 5 日一段」复利到整个验证窗
        （段数 = 验证天数 / 5），与 test 表的 `total_return` 同口径；
      · val_wei：选轮用的合成分 = Σ_h v_h·RankIC_h（打分），只随行携带、不上表。
    """
    expected = {(q, f) for q in quarters(axis()[0]) for f in range(1, 5)}
    seen = [(r['quarter'], r['fold']) for r in records]
    if len(set(seen)) != len(seen) or set(seen) != expected:
        raise ValueError(f'训练验证集汇总需要四季度×四折共 {len(expected)} 条不重复记录，'
                         f'实际 {len(seen)} 条；缺 {sorted(expected - set(seen))}')
    h1, h5 = rank_ic_horizons()
    valid_days = fold_valid_days()
    fields = [f'IC_{h}d' for h in (h1, h5)] + [f'RankIC_{h}d' for h in (h1, h5)] \
             + ['val_wei', 'return_proxy_pct', 'total_return']
    collected = []
    for r in records:
        v = r['best_validation']
        proxy = float(v['val_return_proxy'])
        # 总收益：把「信号日均 5 日代理收益」当成每 5 日一段的持有期收益复利到整个
        # 验证窗（段数 = 验证天数 / 5）。这是从标量折算出来的近似，不是逐段实测。
        segments = valid_days[(r['quarter'], r['fold'])] / h5
        collected.append(dict(quarter=r['quarter'],
                              **{k: float(v[k]) for k in fields
                                 if k not in ('return_proxy_pct', 'total_return')},
                              return_proxy_pct=100 * proxy,
                              total_return=100 * ((1. + proxy) ** segments - 1)))
    frame = pd.DataFrame(collected)
    rows = []
    for q in [*quarters(axis()[0]), '总计']:
        part = frame if q == '总计' else frame[frame.quarter == q]
        mean = part[fields].mean()
        rows.append(dict(quarter=q, **{k: float(mean[k]) for k in fields}))
    return rows

# =============================================================================
# ③ 模型质量与固定策略分开评价
# =============================================================================


# =============================================================================
# ④ 费用与现金回测
# =============================================================================


def fees(notional, sell=False):
    """原参考工程/项目约定的固定费率；这是研究假设，不是历史费率复原。"""
    return max(notional * .00025, 5) + notional * .00001 + (notional * .0005 if sell else 0)


# 本单元只有一个固定交易策略及一个每日换手基线。其他策略研究使用导出的打分。
# ★ V11（激进线）：本单元策略由「Top5 每 5 日调仓」改成 **Top5 每日调仓** —— 高换手是这条
#   线的主题本身（打板/妖股的钱赚的是隔日情绪溢价，持有 5 日会把溢价还回去）。
#   基线 top1_1d 不变，仍然是最激进的每日单票对照。
STRATEGIES = [
    # ★ V13：本单元策略带**缓冲区**（`band` = 掉出前多少名才卖，取自 `RECIPE['hold_band']`）。
    #   基线 `top1_1d` **不带缓冲区**（band=0）—— 它是**跨版本固定的参照物**，
    #   给它加缓冲会把"每日单票"变成"持有到掉出前 N 名"，与 V2~V12 的基线不可比。
    #   （实测过：V14 一开始把 band 施加到基线上，成交从 362 笔掉到 104 笔 —— 那是另一个策略了。）
    dict(name='top5_1d', topn=5, period=1, band=RECIPE['hold_band']),
    dict(name='top1_1d', topn=1, period=1, band=0),
]


def cash_backtest(pred, panel, px, days, n=5, period=5, band=0):
    """固定周期TopN现金账户：T日排序，T+1开盘执行，保留费用、拒单和资金约束。

    每period日重选目标；目标相同则继续持有。仅服务本单元的两种固定策略。

    ★ V11：可买判定走 `px.entry_ok()`（由 `RECIPE['entry_rule']` 决定），不在调用处写死
    `px.entry` —— 打板口径下「开盘涨停但盘中开板」的票是可买的，而保守口径下买不进。
    卖出侧一律保持保守（跌停卖不掉照旧顺延），打板只放宽**买入**。
    """
    cash = 100000.
    holdings = {}
    buys = {}
    desired = []
    curves = []
    trades = []
    corp_events = 0
    first, last = int(days[0]) + 1, int(days[-1]) + 2
    pred_by_day = {int(d): p for d, p in zip(days, pred)}
    spec_band = band
    entry_ok = px.entry_ok() if px is not None else None
    for d in range(first, last + 1):
        # --- 公司行为：原始价按整手下单；复权比例折算经济股数，假设分红立即再投资。
        #     没有逐笔公司行为现金流，因此该部分是总收益近似，并非券商交割单。
        for c in list(holdings):
            ratio = px.adj[d, c] / px.adj[d - 1, c]
            if not np.isfinite(ratio) or ratio <= 0:
                raise ValueError('持仓复权因子缺失')
            if abs(ratio - 1) > 1e-8:
                holdings[c] *= ratio
                corp_events += 1

        # --- 目标持仓：信号日必须落在打分窗内 ---
        signal = d - 1
        p = pred_by_day.get(signal)
        if p is not None:
            candidates = np.flatnonzero(np.isfinite(p))
            # 信号日先排序；开盘只根据当时价格拒单/顺位补足，不用标签选股。
            ranked = candidates[np.argsort(-p[candidates], kind='stable')]
            if (signal - int(days[0])) % period == 0:
                # ★ V13 缓冲区：还在**前 `hold_band` 名**里的持仓不卖（继续拿着），
                #   空出来的仓位再按排名从高到低补足到 n 只。
                #   针对的是每日全额换仓 ~25%/年 的往返成本：一个每天都被模型排在前 20
                #   的票，没必要为了"重排"先卖后买、白交一趟 10.2bp。
                #   `hold_band <= n` 时 `keep` 为空 ⇒ 逐位退回 V11 的"每天无脑重排"口径。
                band = spec_band
                keep = (set(int(c) for c in ranked[:band]) & set(holdings)) if band > n else set()
                fresh = [int(c) for c in ranked if entry_ok[d, c] and c not in keep]
                desired = list(keep) + fresh[:max(0, n - len(keep))]
        if d == last:
            desired = []                 # 末尾清仓，把持仓换成现金后再估值

        # --- 先卖：不在目标里就卖；卖不出去或当日刚买的顺延到下一个开盘 ---
        blocked = []
        for c in list(holdings):
            if c in desired:
                continue
            if not px.exit[d, c] or buys[c] >= d:
                blocked.append(c)
                continue
            quantity = holdings[c]
            price = px.open[d, c] * (1 - .0003)
            gross = quantity * price
            fee = fees(gross, True)
            cash += gross - fee
            trades.append(dict(date=str(panel.days[d]), code=str(panel.codes[c]), side='sell',
                               quantity=quantity, price=float(price), fee=float(fee)))
            del holdings[c]
            del buys[c]

        # --- 再买：按 1/N 预算、100 股整手，单笔不超过信号日成交额的 1% ---
        # 只在固定调仓日买入。
        if p is not None and d < last and (signal - int(days[0])) % period == 0:
            equity = cash + sum(q * px.open[d, c] if np.isfinite(px.open[d, c]) else q * px.close[d - 1, c]
                                for c, q in holdings.items())
            for c in desired:
                if c in holdings or len(holdings) >= n:
                    continue
                price = px.open[d, c] * (1 + .0003)
                # 成交额只用信号日已知值；不偷看执行日全天成交额。
                amount = panel.amount[signal, c]
                budget = min(cash, equity / n, amount * .01 if np.isfinite(amount) else 0.)
                quantity = int(max(0, budget - 5) / (price * (1 + .00025 + .00001)) / 100) * 100
                while quantity > 0 and quantity * price + fees(quantity * price) > budget:
                    quantity -= 100
                if quantity <= 0:
                    continue
                gross = quantity * price
                fee = fees(gross)
                cash -= gross + fee
                holdings[c] = float(quantity)
                buys[c] = d
                trades.append(dict(date=str(panel.days[d]), code=str(panel.codes[c]), side='buy',
                                   quantity=quantity, price=float(price), fee=float(fee)))

        # --- 当日估值 ---
        if cash < -1e-6 or len(holdings) > n:
            raise AssertionError('资金/持仓上限被突破')
        equity = cash + sum(q * px.close[d, c] for c, q in holdings.items())
        if not np.isfinite(equity):
            raise ValueError('持仓估值缺失')
        curves.append(dict(date=str(panel.days[d]), equity=float(equity), cash=float(cash),
                           positions=len(holdings), blocked_exits=len(blocked)))

    # --- 汇总：净值曲线 → 收益 / 回撤 / 夏普 / 费用 ---
    values = np.r_[100000., [r['equity'] for r in curves]]
    returns = values[1:] / values[:-1] - 1
    dd = values / np.maximum.accumulate(values) - 1
    return dict(topn=n, rebalance_days=period,
                return_value=float(values[-1] / 100000 - 1),
                max_drawdown=float(dd.min()),
                sharpe=float(np.mean(returns) / np.std(returns) * np.sqrt(242)) if np.std(returns) > 0 else 0.,
                total_fees=float(sum(t['fee'] for t in trades)), trades=len(trades), ending_cash=cash,
                ending_positions=len(holdings), corporate_action_adjustments=corp_events,
                start=str(panel.days[first]), end=str(panel.days[last]),
                corporate_action_assumption='复权比折算经济股数，分红立即再投资近似'), curves, trades

# =============================================================================
# ⑤ 打分四列落盘
# =============================================================================


def write_scores(pred, panel, days, out):
    """按年分区写打分：trade_date / stock_code / value / rank，float32 + 原子替换。"""
    for year in sorted(set(str(panel.days[d])[:4] for d in days)):
        chunks = []
        for p, d in zip(pred, days):
            if not str(panel.days[d]).startswith(year):
                continue
            ok = np.isfinite(p)
            chunks.append(pd.DataFrame(dict(trade_date=str(panel.days[d]), stock_code=panel.codes[ok],
                                            value=p[ok].astype(np.float32),
                                            rank=(rankdata(p[ok]) / ok.sum()).astype(np.float32))))
        df = pd.concat(chunks, ignore_index=True)
        df['trade_date'] = df['trade_date'].astype('string')
        df['stock_code'] = df['stock_code'].astype('string')
        folder = out / f'year={year}'
        target = output_file(folder / 'data.parquet')
        temp = folder / 'data.tmp.parquet'
        # 新单元第一次运行时年度目录还不存在（建目录的 mkdir 在回测那一步，比这里晚）。
        folder.mkdir(parents=True, exist_ok=True)
        df.to_parquet(temp, index=False)
        os.replace(temp, target)

# =============================================================================
# ⑥ 季度归集与复合校验
# =============================================================================


def quarter_metrics(source, strategy, curve, trades, pred, panel, days):
    """把全年曲线/成交按季度切段逐季出指标，并断言季度收益能复合回全年。"""
    rows = []
    previous = 100000.
    cursor = 0

    # 执行日归属于发出指令的信号季度；最后清仓顺延归入最后一个季度。
    boundaries = []
    for q in quarters(panel.days):
        qdays = [int(d) for d in days if str(pd.Period(panel.days[d], freq='Q')) == q]
        if qdays:
            boundaries.append((q, qdays))

    for i, (q, qdays) in enumerate(boundaries):
        end_day = qdays[-1] + (2 if i == len(boundaries) - 1 else 1)
        end_date = str(panel.days[end_day])
        start = cursor
        while cursor < len(curve) and curve[cursor]['date'] <= end_date:
            cursor += 1
        segment = curve[start:cursor]
        if not segment:
            continue
        eq = np.r_[previous, [r['equity'] for r in segment]]
        ret = eq[1:] / eq[:-1] - 1
        dd = eq / np.maximum.accumulate(eq) - 1
        tx = [t for t in trades if segment[0]['date'] <= t['date'] <= segment[-1]['date']]
        pos = np.flatnonzero(np.isin(days, qdays))
        ric = next(r for r in ic_table(pred[pos], panel, days[pos]) if r['label'] == RECIPE['label'])
        rows.append(dict(source=source, quarter=q, strategy=strategy, signal_days=len(qdays),
                         execution_start=segment[0]['date'], execution_end=segment[-1]['date'],
                         start_equity=float(previous), end_equity=float(eq[-1]),
                         return_value=float(eq[-1] / previous - 1), max_drawdown=float(dd.min()),
                         daily_sharpe=float(ret.mean() / ret.std() * np.sqrt(242)) if ret.std() > 0 else 0.,
                         daily_win_rate=float(np.mean(ret > 0)), fees=float(sum(t['fee'] for t in tx)),
                         trades=len(tx), turnover=float(sum(t['quantity'] * t['price'] for t in tx) / eq.mean()),
                         RankIC_5d=ric['RankIC']))
        previous = float(eq[-1])

    assert cursor == len(curve), '季度汇总遗漏现金曲线日期'
    if rows and not np.isclose(np.prod([1 + r['return_value'] for r in rows]), curve[-1]['equity'] / 100000, atol=1e-10):
        raise AssertionError('季度收益无法复合回全年收益')
    return rows

# =============================================================================
# ⑦ 策略曲线图
# =============================================================================


def plot_results(out, source, first_day):
    """独立单元仅展示固定策略和Top1日换手基线的含费收益。"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder = out / source / 'CSV'
    plots = out
    plots.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 5.6), layout='constrained')
    for spec, color in zip(STRATEGIES, ['#17917e', '#c33f3f']):
        frame = pd.read_csv(folder / f'cash_{spec["name"]}.csv')
        equity = np.r_[100000., frame.equity.to_numpy()]
        dates = pd.to_datetime([first_day] + frame.date.tolist())
        # 图例由策略规格推出来，不写死策略名 —— 换策略（V11 把 5d 改成 1d）时不会留下
        # 一条说旧口径的图例，而这正是"静默说谎"最容易发生的地方。
        label = (f"Top{spec['topn']} | " + ('daily baseline' if spec['topn'] == 1
                                            else f"rebalance every {spec['period']} day(s)"))
        ax.plot(dates, 100 * (equity / 100000 - 1), color=color, linewidth=2,
                label=f'{label} | {100*(equity[-1]/100000-1):+.2f}%')
    ax.set_title('Baseline model | Fixed strategy vs daily baseline', fontsize=14)
    ax.set_ylabel('Cumulative net return (%)')
    ax.set_xlabel('Execution / valuation date | fees and slippage included')
    ax.axhline(0, color='#555555', linewidth=.7)
    ax.grid(alpha=.2)
    ax.legend(loc='upper left')
    ticks = ['2025-07-01', '2025-10-01', '2026-01-01', '2026-04-01', '2026-07-02']
    ax.set_xticks(pd.to_datetime(ticks), labels=ticks)
    fig.savefig(output_file(plots / f'{source}_curves.png'), dpi=160)
    plt.close(fig)

# =============================================================================
# ⑧b 最新一日推荐表（给实战 / 增量推演用，不参与任何评价指标）
# =============================================================================


def markdown_table(headers, rows):
    """四列表的极简渲染（picks.md 每日排名用）。"""
    return ('| ' + ' | '.join(headers) + ' |\n|' + '|'.join(['---'] * len(headers)) + '|\n'
            + ''.join('| ' + ' | '.join(map(str, r)) + ' |\n' for r in rows))


def stock_names(codes):
    """股票代码 → 中文名。查不到、表读不到都返回空串（只影响显示，不影响任何数值）。

    外部读取之一（用户 2026-09-25 授权）：`datadownload/data/stock_list/data.parquet`
    的 `stock_code` + `name` 两列。退市/更名的票查不到就留空，不报错——picks.md 是常驻
    产物，不能因为一张对照表而缺席。
    """
    path = stock_list_path()
    try:
        frame = pd.read_parquet(path, columns=['stock_code', 'name'])
    except (OSError, ValueError, KeyError) as exc:
        print(f'警告：读不到股票名称表 {path}（{type(exc).__name__}: {exc}），picks 只显示代码', flush=True)
        return {}
    table = dict(zip(frame.stock_code.astype(str), frame.name.astype(str)))
    if len(table) != len(frame):
        raise ValueError(f'{path}: stock_code 有重复')
    return {str(c): table.get(str(c), '') for c in codes}


def horizon_of(label):
    """`label_ret_5d` → 5。"""
    return int(label.split('_ret_')[1].rstrip('d'))


def trade_axis(panel):
    """信号日 → 交易日轴：快照内用 `panel.days`，快照之后接外部交易日历。

    标签口径是 `px[T+1+h] / px[T+1] - 1`，所以「T+1 开盘买入、T+h+1 开盘卖出」里的日期
    必须落在真实交易日上。快照内直接用 `panel.days`（与标签同源）；快照之后的日期快照里
    没有、但交易所有（日历含未来交易日），否则最新一天的买卖日只能写成「次日」这种
    相对说法。日历读不到就降级为相对说法，不中断。
    """
    days = [str(d) for d in panel.days]
    try:
        frame = pd.read_parquet(calendar_path(), columns=['date', 'is_open'])
    except (OSError, ValueError, KeyError) as exc:
        print(f'警告：读不到交易日历 {calendar_path()}（{type(exc).__name__}: {exc}），'
              'picks 的买卖日改用相对说法', flush=True)
        return days
    future = sorted(str(d) for d in frame[frame.is_open == 1].date if str(d) > days[-1])
    return days + future


def write_daily_picks(pred, panel, days, metrics=None, valid_rows=None):
    """最近10个交易日、每天前 `RECIPE['picks_topn']` 名的打分排名。

    版式（用户 2026-09-25/26 定）：**文件开头只有两张表**（评价窗 test + 训练
    验证集 valid，见 `metric_blocks_text`，用户 2026-09-26 要求删掉表与每日排名
    之间的全部说明文字），之后直接是每日一节：标题「<信号日> 收盘」，正文第一行
    写清这个打分预测的是什么、以及**具体到日期的**买卖时点，再列
    排名/代码/名称/打分 四列。回测路径已算好两张表，用 metrics / valid_rows
    传入即可，不重算。

    打分口径沿用 `RECIPE['score_label']`（V3 起它只决定**预测内容怎么写**：这里是 5d），
    与集成打分、固定策略同源；名称与买卖日期只做展示，不参与排序或任何指标。
    """
    if len(days) == 0:
        raise ValueError('没有可展示的打分日期')
    if pred.shape != (len(days), len(panel.codes)):
        raise ValueError('打分维度与日期、股票轴不一致')
    out = RUN_ROOT / 'model_pred'
    out.mkdir(parents=True, exist_ok=True)
    if metrics is None:
        ev, eval_days = evaluation_slice(pred, panel, days)
        metrics = evaluation_metrics(ev, panel, eval_days)['rows']
    if valid_rows is None:
        valid_rows = training_validation_rows(fold_completions())
    names = stock_names(panel.codes)
    axis = trade_axis(panel)
    position = {d: i for i, d in enumerate(axis)}
    horizon = horizon_of(RECIPE['score_label'])
    # ★ V11：买卖时点按**本线实际持有期**写（每日调仓 ⇒ T+1 买、T+2 卖），而不是按打分头
    #   的标签期。打分头仍然是 5d 头（两张表的第二列要用），但本线不持有 5 天 ——
    #   照抄标签期会把"预测什么"和"怎么交易"写成两回事。
    holding = RECIPE.get('picks_holding', horizon) + 1
    topn = RECIPE['picks_topn']
    first = max(0, len(days) - 10)
    text = '```text\n' + metric_blocks_text(metrics, valid_rows) + '```\n\n'
    for index in range(len(days) - 1, first - 1, -1):
        signal_day = str(panel.days[int(days[index])])
        score = pred[index]
        candidates = np.flatnonzero(np.isfinite(score))
        picks = candidates[np.argsort(-score[candidates], kind='stable')][:topn]
        at = position.get(signal_day)
        buy = axis[at + 1] if at is not None and at + 1 < len(axis) else None
        sell = axis[at + holding] if at is not None and at + holding < len(axis) else None
        text += f'## {signal_day} 收盘\n\n'
        text += (f'模型预测 {horizon} 日收益率，本线持有 {holding - 1} 个交易日：'
                 f'**{buy} 开盘买入 → {sell} 开盘卖出**\n\n'
                 if buy and sell else
                 f'模型预测 {horizon} 日收益率，本线持有 {holding - 1} 个交易日：'
                 f'次一交易日（T+1）开盘买入 → 第 {holding} 个交易日（T+{holding}）开盘卖出'
                 '（买卖日超出已知交易日历，故用相对说法）\n\n')
        if not len(picks):
            text += '当日没有有效打分。\n\n'
            continue
        rows = [[i, str(panel.codes[c]).split('.')[0], names.get(str(panel.codes[c]), ''),
                 f'{float(score[c]):.4f}'] for i, c in enumerate(picks, 1)]
        text += markdown_table(['排名', '代码', '名称', '打分'], rows) + '\n'
    output_file(out / 'picks.md').write_text(text, encoding='utf-8')
    print('最近10个交易日排名已生成:', out / 'picks.md', flush=True)


# =============================================================================
# ⑨ 命令行入口
# =============================================================================


def save_score_metadata(folder, report, dates):
    metadata = {k:v for k,v in report.items() if k not in ('variants','evaluation_mode')}
    metadata['score_dates'] = list(dates)
    metadata['columns'] = ['trade_date','stock_code','value','rank']
    atomic_json(folder/'score_meta.json',metadata)


def read_saved_scores(folder, panel):
    """独立读集成打分；不访问model_train或模型权重。"""
    meta = json.loads((folder/'score_meta.json').read_text())
    dates = meta['score_dates']
    if dates != sorted(set(dates)):
        raise ValueError('打分日期重复或乱序')
    date_axis = pd.Index(dates)
    codes = pd.Index(panel.codes)
    days = pd.Index(panel.days).get_indexer(dates)
    if (days<0).any():raise ValueError('打分日期不在当前价格数据中')
    pred = np.full((len(dates),len(codes)),np.nan,np.float32)
    files = sorted(folder.glob('year=*/data.parquet'))
    if not files:
        raise FileNotFoundError(f'{folder}: 缺年度打分文件')
    seen = set()
    for path in files:
        frame = pd.read_parquet(path,columns=['trade_date','stock_code','value'])
        if frame.duplicated(['trade_date','stock_code']).any():raise ValueError('打分主键重复')
        t=date_axis.get_indexer(frame.trade_date.astype(str));c=codes.get_indexer(frame.stock_code.astype(str))
        if (t<0).any() or (c<0).any():raise ValueError('打分坐标不匹配')
        keys=set(zip(t.tolist(),c.tolist()))
        if seen.intersection(keys):raise ValueError('年度文件含重复打分')
        seen.update(keys)
        values=frame.value.to_numpy(dtype=np.float32)
        if not np.isfinite(values).all():raise ValueError('导出打分含无效值')
        pred[t,c]=values
    return pred,days,meta


def evaluate_fixed_strategies(pred, panel, px, days, report, mode):
    out=RUN_ROOT/'model_pred';csv_root=out/'ensemble/CSV'
    csv_root.mkdir(parents=True,exist_ok=True)
    ev,eval_days=evaluation_slice(pred,panel,days)
    report={k:v for k,v in report.items() if k not in ('score_dates','columns','variants')}
    report['evaluation_mode']=mode
    report['evaluation_window']=[str(panel.days[eval_days[0]]),str(panel.days[eval_days[-1]])]
    result={'IC':ic_table(ev,panel,eval_days),'strategies':[]}
    summaries=[];quarterly=[]
    if px is not None:
        report['label_price_check']=px.verify_labels(panel,days)
        for spec in STRATEGIES:
            stat,curve,trades=cash_backtest(ev,panel,px,eval_days,n=spec['topn'],period=spec['period'],band=spec['band'])
            result['strategies'].append(dict(strategy=spec['name'],**stat))
            summaries.append(dict(source='ensemble',strategy=spec['name'],**stat))
            quarterly.extend(quarter_metrics('ensemble',spec['name'],curve,trades,ev,panel,eval_days))
            pd.DataFrame(curve).to_csv(output_file(csv_root/f'cash_{spec["name"]}.csv'),index=False)
            pd.DataFrame(trades).to_csv(output_file(csv_root/f'trades_{spec["name"]}.csv'),index=False)
    report['variants']={'ensemble':result}
    # --- 六指标与季度表现：评价窗 242 天的逐日调仓组合（口径见 §②b）---
    # 这一块不依赖价格层：只读标签与成交额，所以 --no-backtest 也能出。
    measured=evaluation_metrics(ev,panel,eval_days)
    metrics=measured['rows']
    # 训练侧验证集表现：优先用打分元数据里冻结的那份折记录（与本次打分同源），
    # 缺了才回落到直接读 model_train 的 complete.json。
    valid_rows=training_validation_rows(report.get('folds') or fold_completions())
    h1,h5=rank_ic_horizons()
    # 逐日序列随报告一并留档，便于复核指标之外的时点（不另建文件）。
    result['eval_portfolio']=dict(params=TOP_PORTFOLIO,total=metrics[-1],
        daily=dict(date=[str(panel.days[d]) for d in eval_days],
                   return_pct=[round(float(x),6) for x in measured['daily_ret']],
                   **{f'ic_{h}d':[round(float(x),6) for x in measured['daily_ic'][h]]
                      for h in (h1,h5)},
                   **{f'rank_ic_{h}d':[round(float(x),6) for x in measured['daily_rank_ic'][h]]
                      for h in (h1,h5)}))
    print_metric_blocks(metrics,valid_rows)
    # ★ V13：bagged vs argmax 的配对对照。数字进 final_audit.json、日志打一份，
    #   叙述（结论怎么写）归 Agent —— 脚本只出数，不写结论。
    bag_summary = bag_vs_argmax(report.get('folds') or fold_completions())
    if bag_summary:
        print(bag_comparison_text(bag_summary), flush=True)
    write_daily_picks(pred,panel,days,metrics,valid_rows)
    # ★ 用户 2026-09-26 定：`model_pred/tables/` 整个目录不要了（CSV 与 report.json
    #   都不再落盘）。本版结果的留档位置：`model_info/final_audit.json`（总计指标）、
    #   `model_info/two_strategy_validation.json`（两策略实现值）、
    #   `model_pred/ensemble/score_meta.json`（折记录、audit、口径与 caveats），
    #   以及 `model_pred/picks.md` 开头那两张表；逐日序列只在本次运行的 report 里，
    #   需要时重跑本文件即可复算。
    if px is None:
        return report
    plot_results(out,'ensemble',report['evaluation_window'][0])
    # 两个核验记录：推理对拍的逐折结果，以及两策略口径的自检。
    # `--from-scores` 不加载权重，此时 report 里带的 audit 是上一次训练运行的存档，
    # 不能当成本次验过，所以按来源显式区分。
    from_saved = mode == 'saved_ensemble_scores'
    audit = [] if from_saved else (report.get('audit') or [])
    # ★ V17：可实现口径 —— 与理想口径一起留档，避免只看偏乐观的那一份。
    real_block = realizable_summary(ev, panel, px, eval_days, metrics) if px is not None else (None, '')
    atomic_json(info_root()/'final_audit.json', dict(
        run_name=run_name(), checked_at=time.strftime('%Y-%m-%d %H:%M:%S'), score_source=mode,
        evaluation_window=report['evaluation_window'], score_window=report.get('score_window'),
        model_train_read=bool(audit), weights_reloaded=bool(audit), folds_checked=len(audit),
        max_cpu_reload_error=max([a['max_cpu_reload_error'] for a in audit], default=None),
        audit=audit,
        metrics_total=metrics[-1],
        # ★ V17：顶栏的 metrics_total 仍是**理想口径**（与历史报表逐位可比）；
        #   可实现口径单列一块，附差额与 t，谁都能看出"有多少是买不到的"。
        realizable=real_block[0],
        # ★ V13：交付权重是「前 k 名平均」而不是 argmax ⇒ 把两者的配对对照一并留档。
        bag_vs_argmax=bag_summary))
    # 每一项都是本次真正量到的事实，不写没验过的断言。
    atomic_json(info_root()/'two_strategy_validation.json', dict(
        run_name=run_name(), checked_at=time.strftime('%Y-%m-%d %H:%M:%S'), score_source=mode,
        reads_model_train=not from_saved, loads_model_weights=not from_saved,
        strategies=[s['name'] for s in STRATEGIES], strategy_rows=len(summaries),
        quarterly_rows=len(quarterly), evaluation_days=int(len(measured['daily_ret'])),
        evaluation_window=report['evaluation_window'], label_price_check=report.get('label_price_check'),
        # quarter_metrics 里已经断言「四段季度收益能复合回全年」，跑到这里就是通过。
        quarterly_returns_compound=True,
        date_axis_matches_price_axis=True,
        realized={s['strategy']: dict(return_value=s['return_value'], max_drawdown=s['max_drawdown'],
                                      sharpe=s['sharpe'], trades=s['trades'], total_fees=s['total_fees'])
                  for s in summaries}))
    print('两策略回测完成:',json.dumps(summaries,ensure_ascii=False),flush=True)
    return report


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--from-scores', action='store_true', help='直接读取集成打分，只计算本单元两策略')
    parser.add_argument('--no-backtest', action='store_true', help='明确只计算 IC；不输出现金策略')
    parser.add_argument('--audit', action='store_true', help='最佳模型逐日截断/重载推理对拍')
    parser.add_argument('--picks-only', action='store_true',
                        help='只出最近10个交易日排名（不跑评价/回测），给增量推演用')
    args = parser.parse_args()
    # 新单元的常驻产物由运行生成，所以起点允许缺失；多余路径仍然一律拒绝。
    validate_layout(allow_temporary=True, allow_missing=True)
    # 进程内线程数与模型侧用同一个旋钮（run.py 导出 MX_THREADS），不写死主机假设。
    torch.set_num_threads(int(os.environ.get('MX_THREADS', '3')))

    if args.from_scores and args.audit:
        parser.error('--from-scores不加载权重，不能同时使用--audit')
    panel = Panel(RECIPE['features'], load_x=args.audit)
    out = RUN_ROOT / 'model_pred'
    out.mkdir(parents=True, exist_ok=True)
    if args.from_scores:
        pred,days,report=read_saved_scores(out/'ensemble',panel)
        if args.picks_only:return write_daily_picks(pred,panel,days)
        px=None if args.no_backtest else Prices(panel)
        return evaluate_fixed_strategies(pred,panel,px,days,report,'saved_ensemble_scores')

    # ★ V11：打板口径与热门池都住在 `Prices` 里，而重载对拍要调 `predict(net, panel, px, …)`
    #   ⇒ 走训练打分这条路时必须先把价格层建出来（--no-backtest 也一样：对拍不看回测，
    #   但要看口径）。只读落盘打分的那条路不调 predict，不需要它。
    px = Prices(panel) if (args.audit or not args.no_backtest) else None

    variants = {}
    completions = []
    dates = None
    audit = []
    fold_ids = [1, 2, 3, 4]
    qs = quarters(panel.days)

    # --- 逐折拼出「季度首日 → 数据末日」的完整推演序列 ---
    # 每一天由「最新一个没见过它的季度模型组」负责：
    #   四个季度各由自己那组给（该组的训练数据止于季度首日前 6 个交易日，整个季度都是样本外）；
    #   季度之后的尾段由**最后那个季度**的组给（其余组其实也都没见过它，取最新即可）。
    for fold in fold_ids:
        chunks = []
        fold_dates = []
        for quarter in qs:
            folder = INPUT_ROOT / 'model_train' / quarter / f'fold{fold}'
            done = json.loads((folder / 'complete.json').read_text())
            st = np.load(folder / 'score_predictions.npy')       # 全推演窗：季度首日 → 数据末日
            n_test = len(done['test_dates'])
            p = st[:n_test]                                      # 本季度那一段（= 旧 test_predictions）
            qdays = np.searchsorted(panel.days, done['test_dates'])
            chunks.append(p)
            fold_dates.extend(done['test_dates'])
            if quarter == qs[-1]:
                chunks.append(st[n_test:])                        # 尾段 → 增量最新打分
                fold_dates.extend(done['score_dates'][n_test:])
            completions.append(done)

            # --- 重载对拍：从磁盘重建模型，验证当日独立推理可复现 ---
            if args.audit:
                checkpoint = torch.load(folder / 'best.pt', map_location='cpu', weights_only=False)
                net = PredictModel(len(panel.features), panel.market_dim)
                net.load_state_dict(checkpoint['model'])
                net.ridge = checkpoint.get('ridge')     # ★ V10：打分含岭回归分量
                net.eval()
                check_ix = np.unique([0, len(qdays) // 2, len(qdays) - 1])
                check_days = qdays[check_ix]
                actual = predict(net, panel, px, check_days, torch.device('cpu'))
                # 逐头再跑一遍：既是「打分 == score_of(头)」这条恒等式的自检，也给下面
                # 的逐行检查提供比对基准。★ 打分是**当日全截面**的秩，单看 10 行没有意义，
                # 所以逐行检查比的是**头输出**，不是打分。
                raw = predict_heads(net, panel, check_days, torch.device('cpu'))
                expect = hot_filter(blend_linear(score_of(raw), net.ridge, panel, check_days),
                                    px, check_days)
                if not np.allclose(actual, expect, atol=2e-5, rtol=2e-5, equal_nan=True):
                    raise AssertionError('predict() 与 blend_linear(score_of(predict_heads())) 不一致')
                expected = p[check_ix]
                finite = np.isfinite(actual) & np.isfinite(expected)
                error = float(np.max(np.abs(actual[finite] - expected[finite])))
                if not np.array_equal(np.isfinite(actual), np.isfinite(expected)) or error > 2e-4:
                    raise AssertionError(f'推理重载对拍失败 {error}')
                # 同一模型、同一日，逐行**头输出**必须与整批一致（无跨行/跨日依赖）。
                # 必须显式比二维的 (n,5)：`.squeeze(-1)` 在 (n,5) 上是 no-op，会和
                # (k,) 的 actual 靠广播比较而假通过。
                for j, d in enumerate(check_days):
                    mask = panel.mask(d)
                    with torch.no_grad():
                        m = torch.from_numpy(panel.M[d]) if panel.market_dim else None
                        one = net(torch.from_numpy(panel.X[d, mask][:10]), m).numpy()
                    if not np.allclose(one, raw[j, mask][:10], atol=2e-5, rtol=2e-5):
                        raise AssertionError('跨行/跨日推理依赖')
                audit.append(dict(quarter=quarter, fold=fold, max_cpu_reload_error=error,
                                  days=panel.days[check_days].tolist()))
        if dates is not None and dates != fold_dates:
            raise ValueError('各折测试日期不一致')
        dates = fold_dates
        if len(set(dates)) != len(dates) or dates != sorted(dates):
            raise ValueError('季度拼接重叠/乱序')
        variants[f'fold{fold}'] = np.concatenate(chunks)

    days = np.searchsorted(panel.days, dates)
    # 评价（IC / 回测 / 季度表）**只用四个季度那一段**（用户固定的评价窗）；
    # 尾段（2026-07-01 起）只进打分产物，供增量推演用。

    # --- 集成：四折打分**直接相加**，不做逐日标准化（用户 2026-09-20）---
    # 用 NaN 传播的加法：某票某天没有分数时，集成结果仍是"没有分数"，
    # 不会变成 0 分混进排序（np.nansum 会犯这个错）。
    if len(variants) > 1:
        variants['ensemble'] = sum(variants.values()).astype(np.float32)

    pred=variants['ensemble']
    if args.picks_only:return write_daily_picks(pred,panel,days)
    report = dict(evaluation_window=[RECIPE["test_start"], RECIPE["test_end"]],
                  score_window=[dates[0], dates[-1]],
                  folds=completions, analysis_mode='IC_only' if args.no_backtest else 'full',
                  variants={}, audit=audit,
                  ensemble='四折打分直接相加（不归一化）',
                  inference_rule='每天由「最新一个没见过它的季度模型组」推演；季度之后的尾段由最后一个季度那组给',
                  caveats=['固定存续且从未ST股票池含幸存者偏差', '四折共用测试窗，不是四段独立样本外收益',
                           '验证收益是5日标签代理；策略净值另算', '不据测试结果调参或晋级best',
                           '评价只用2025Q3～2026Q2四季度；之后的尾段仅供增量推演'])
    first=completions[0]
    info=json.loads((INPUT_ROOT/'model_train'/first['quarter']/f'fold{first["fold"]}'/'training_info.json').read_text())
    report['training']=dict(recipe=info['recipe'],feature_count=len(info['feature_columns']),feature_source=info['feature_source'])
    write_scores(pred,panel,days,out/'ensemble')
    result=evaluate_fixed_strategies(pred,panel,px,days,report,'training_predictions')
    save_score_metadata(out/'ensemble',result,dates)


if __name__ == '__main__':
    main()
