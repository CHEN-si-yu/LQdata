'V53: flow_temporal; seed 6253; fresh independent research unit, fixed project contract.'

# =============================================================================
# ① 环境前置 —— 必须排在 import torch 之前
#
# BLAS / OpenMP 的线程数只在库首次加载时读取一次，import torch 之后再设就无效，
# 所以这段必须放在文件最前面。线程数是**内存纪律**而不是性能偏好：并发跑 16 折时
# 每个进程的开销要乘上并发数，必须与容器内存上限一起算。
# ★ 上限**换机器就会变，一律现读**（`run.py:free_memory_gib()` 读 cgroup v2 的
#   `memory.max` 或 v1 的 `memory/memory.limit_in_bytes`）；`free` 看到的是宿主机总量，
#   **不可信**，不要把任何具体数字写死在这里。
#   参考量级（2026-09-26 那台 50 核 / 180 GiB / 2×RTX 4080 实测）：每折稳态约 9.9 GiB、
#   峰值约 11.6 GiB，8 折并发 79 GiB；**瓶颈是 GPU 不是内存**（4 折/卡即 98% 利用率）。
#
# 479 列下的内存构成（V1 注释里"306 特征 + 20 线程 = 38 GiB/折"那条已过期、且特征数与
# 当时的模型都不同，别再引用）：
#   · 主项是常驻的 `Panel.X` = 2120 × 2115 × 479 × 4B ≈ **8.00 GiB**，**与线程数无关**；
#   · 线程相关的是 BLAS/OMP 的每线程工作区，量级远小于主项；
#   · 另外每年读盘有一次约 2.7 GB 的瞬时拷贝（`nan_to_num` 等），逐年释放。
# 每折实测峰值见 `complete.json.peak_rss_gib`。
#
# ★ 两个默认值在这里统一成一个来源：原先 BLAS 侧回退 `'6'`、而 `RECIPE['threads']`
#   回退 `'2'`，于是**直接跑 `model.py`** 会得到"BLAS 6 线程 + torch 2 线程"的混合口径，
#   与 `run.py` 跑出来的不一致。现在都取 `DEFAULT_THREADS`。
# =============================================================================
import os

#: 线程数默认值：`MX_THREADS` 覆盖（`run.py --threads` 会设它）。
DEFAULT_THREADS = '2'

# 启用确定性算法时 cuBLAS 必须用固定大小的工作区，不设这个变量 torch 会直接报错。
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
for _key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[_key] = os.environ.get('MX_THREADS', DEFAULT_THREADS)

# =============================================================================
# ② 常规导入与项目根目录
# =============================================================================
import copy
import json
import random
import resource
import sys
sys.dont_write_bytecode = True
import time
import re
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from scipy.stats import rankdata
from torch import nn

ROOT = Path(__file__).resolve().parent

# =============================================================================
# ②b 单元路径与固定布局（2026-09-26 从 run.py 迁来：run.py 只剩训练调度）
#
# 独立单元的路径合同、固定文件清单与「只许写清单内路径」的硬闸都在这里；
# `model.py` 与 `analysis.py` 共用，`run.py`（调度器）不依赖它们。
# =============================================================================

UNIT_ROOT = ROOT
PROJECT_ROOT = UNIT_ROOT.parents[1]
UNIT = UNIT_ROOT.name

#: 本单元的特征块。V28 使用**当前全因子快照**：479 列全用（`docs/baseline.md` 明文要求）。
#: `model.py` 的 RECIPE 之后有一道硬闸，特征块不是它就直接报错。
DEFAULT_FEATURE_BLOCK = 'factors'


# 独立单元固定布局。开发规则：项目 docs/architecture.md。
# ★ 2026-09-26：`train.sh` 并进 run.py（本文件即可执行），源码固定为三个 Python 文件；
#   并发调度器与它写的 `model_logs/resource_summary.json`/`resource_usage.csv`、
#   `model_info/finished.json`/`pipeline.log` 一并去掉，清单同步收缩。
#   同日用户定：日志目录由 `logs/` 改名 `model_logs/`（与 `model_info/`、`model_pred/`、
#   `model_train/` 一个命名系列）。
SOURCE_FILES = ('run.py', 'model.py', 'analysis.py')
LAYOUT_QUARTERS = ('2025Q3', '2025Q4', '2026Q1', '2026Q2')
LAYOUT_FOLDS = (1, 2, 3, 4)
LAYOUT_YEARS = (2025, 2026)
LAYOUT_LOGS = 'model_logs'
FOLD_FILES = ('best.pt', 'last.pt', 'complete.json', 'history.json',
              'score_predictions.npy', 'test_predictions.npy', 'split.json', 'training_info.json')


def fixed_files():
    """当前单元的固定159个常驻文件；不新增清单文件。"""
    files = set(SOURCE_FILES)
    files.update(f'model_train/{q}/fold{f}/{name}'
                 for q in LAYOUT_QUARTERS for f in LAYOUT_FOLDS for name in FOLD_FILES)
    files.update(f'{LAYOUT_LOGS}/{q}_{f}.log' for q in LAYOUT_QUARTERS for f in LAYOUT_FOLDS)
    files.update(f'{LAYOUT_LOGS}/analysis_0.log' for _ in (0,))
    # ★ 2026-09-26：`launch.json` / `preflight.json` 从清单里去掉了 —— 它们是旧调度器
    #   （`run.py` 带 check_inputs 的那版）写的，现在 run.py 只调度、不写记录，
    #   留着会变成"清单里有、没人写"的死条目。
    files.update('model_info/' + name for name in ('final_audit.json', 'two_strategy_validation.json'))
    files.update(('model_pred/ensemble_curves.png', 'model_pred/picks.md', 'model_pred/ensemble/score_meta.json'))
    files.update(f'model_pred/ensemble/year={y}/data.parquet' for y in LAYOUT_YEARS)
    # ★ V11：本单元策略由「Top5 每 5 日调仓」改成「Top5 **每日**调仓」，所以这一档
    #   文件名跟着换成 top5_1d；基线 top1_1d 不动。清单仍是 159 个文件（只换名不增删）。
    files.update(f'model_pred/ensemble/CSV/{kind}_{strategy}.csv'
                 for kind in ('cash', 'trades') for strategy in ('top5_1d', 'top1_1d'))
    return files


# 编辑器与解释器的自动产物：不是单元文件，不参与结构核对，也不代表结构扩张。
TOOL_DIRS = ('.ipynb_checkpoints', '__pycache__')


def unit_paths():
    """扫描单元内的文件与目录（已剔除工具自动产物）。"""
    files, dirs = set(), set()
    for p in UNIT_ROOT.rglob('*'):
        rel = p.relative_to(UNIT_ROOT)
        if any(part in TOOL_DIRS for part in rel.parts):
            continue
        (files if p.is_file() else dirs).add(rel.as_posix())
    return files, dirs


def validate_layout(*, allow_temporary=False, allow_missing=False):
    """只核对文件名和目录位置；开发与运行均须保持当前结构。

    新单元按规则只初始化3个脚本文件，158个常驻产物由运行原位生成，所以起点允许缺失
    （`allow_missing`）；但**任何多余路径永远拒绝**，缺失清单交由调用方比对起终点。
    """
    expected = fixed_files()
    actual, actual_dirs = unit_paths()
    if allow_temporary:
        temporary = set()
        for name in expected:
            path = Path(name)
            if path.suffix == '.json':
                temporary.add(path.with_suffix('.json.tmp').as_posix())
            elif path.suffix in ('.pt', '.npy'):
                temporary.add(path.with_suffix('.tmp').as_posix())
            elif path.suffix == '.parquet':
                temporary.add(path.with_name('data.tmp.parquet').as_posix())
        actual -= temporary
    allowed_dirs = {parent.as_posix() for name in expected for parent in Path(name).parents if str(parent) != '.'}
    extra = sorted((actual - expected) | (actual_dirs - allowed_dirs))
    missing = sorted(expected - actual)
    if extra or (missing and not allow_missing):
        raise ValueError(f'独立单元文件树必须保持固定；额外路径={extra}；缺失文件={missing}；规则见docs/architecture.md')
    return dict(files=len(actual), directories=len(actual_dirs), source_files=list(SOURCE_FILES),
                missing=missing)


def output_file(path):
    """常驻产物只能写在固定清单内的路径上；清单之外一律拒绝。

    不要求文件已存在——新单元第一次运行就是靠这里把产物建出来的。
    """
    path = Path(path).resolve()
    if not path.is_relative_to(UNIT_ROOT) or path.relative_to(UNIT_ROOT).as_posix() not in fixed_files():
        raise ValueError(f'禁止增加或移动独立单元文件: {path}')
    if path.name in SOURCE_FILES:
        raise ValueError(f'源码文件只能由开发者修改，不能当产物覆写: {path}')
    return path


def checked_name(value):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', value):
        raise ValueError(f'非法运行名称: {value}')
    return value


def run_name():
    name = os.environ.get('MODEL_RUN_NAME')
    if not name:
        for filename, key in [('launch.json', 'run_name'), ('final_audit.json', 'run')]:
            path = info_root() / filename
            if path.is_file():
                name = json.loads(path.read_text()).get(key)
                if name:
                    break
    name = checked_name(name or 'current')
    if name in ('legacy', 'before_restructure'):
        raise ValueError('请选择新的运行名称')
    return name


def run_root():
    """固定工作单元；运行名称只标记报告，不再创建额外目录层。"""
    return UNIT_ROOT


def info_root():
    return UNIT_ROOT / 'model_info'


def data_root():
    """trainingdata 根目录（含 `meta.json`）；传 `factors` / `fac_sample` 子目录也接受。

    ★ V2 的特征块默认 `factors`（全部 479 列）。V1 是 `fac_sample`（20 列样例）。
    """
    root = Path(os.environ.get('MX_DATA', PROJECT_ROOT / 'trainingdata' / 'factors')).expanduser().resolve()
    if root.name in ('fac_sample', 'factors') and (root.parent / 'meta.json').is_file():
        root = root.parent
    if not (root / 'meta.json').is_file():
        raise FileNotFoundError(f'{root}/meta.json 不存在；传入trainingdata根目录或其 factors 子目录')
    return root


def feature_block():
    """特征块名。`factors` = 全部 479 列，`fac_sample` = 20 列抽样副本。

    ★★ 2026-09-25 修掉一个**静默陷阱**：原实现的回退字面量 `'fac_sample'` 本身就在允许
      名单里，于是 `Path(...).name` 永远落在 `explicit` 分支、`else` 分支**永远走不到**
      —— 设 `MODEL_FEATURE_BLOCK=factors` 完全不生效（在 V1 上实跑验证过：仍是 20 列）。
      后果是"以为设了环境变量、其实用 20 列跑满 16 折"且产物自洽、不报任何错。
      改法：回退值改成 V2 自己的默认块 `DEFAULT_FEATURE_BLOCK`，**并且在本函数末尾加了硬闸**：
      解析出来的块不是 `DEFAULT_FEATURE_BLOCK` 就直接报错。闸放在**这里**而不是只放
      `model.py` —— 因为 `run.py check` / `split` / `doctor` 根本不 import `model.py`，
      只放那边的话这几条入口照样会拿 20 列跑完（实测：闸放 model.py 时
      `run.py check --data .../fac_sample` 一路通过、不报错）。
      `model.py` 里也有一道同名闸，覆盖"直接跑 model.py"的路径。
    """
    explicit = Path(os.environ.get('MX_DATA', DEFAULT_FEATURE_BLOCK)).name
    block = explicit if explicit in ('fac_sample', 'factors') else os.environ.get(
        'MODEL_FEATURE_BLOCK', DEFAULT_FEATURE_BLOCK)
    if block not in ('fac_sample', 'factors'):
        raise ValueError('特征块仅支持 fac_sample 或 factors')
    if block != DEFAULT_FEATURE_BLOCK:
        raise ValueError(
            f'{UNIT} 是**全因子正式训练单元**，特征块必须是 {DEFAULT_FEATURE_BLOCK}（当前 ={block}）。\n'
            f'  · MX_DATA={os.environ.get("MX_DATA", "(未设)")}'
            f'  MODEL_FEATURE_BLOCK={os.environ.get("MODEL_FEATURE_BLOCK", "(未设)")}\n'
            f'  · 想跑 20 列的样例：用 experiments/V1，不要在 {UNIT} 里降级。\n'
            f'  · 确实要改本单元的特征块：先改 run.py 的 DEFAULT_FEATURE_BLOCK，'
            f'并同步 docs/iterations.md 的配置摘要（属科研决定，不是运行参数）。')
    return block


def price_root():
    """回测价格的地址 = 快照块 `trainingdata/prices`（只读）。

    ★ 用户 2026-09-26 定：价格由上游 `preparingdata.py` **复制进快照**，本单元只读
      `trainingdata/`、不再自己连模块①（`datadownload/`）。这一块与 `factors` 同一条
      日期轴，日常增量一起刷新 —— 反转的是 2026-09-25"价格不属于训练数据层"那条边界，
      理由写在 `preparingdata.py` 的模块头部。

    与 `stock_list_path()`/`calendar_path()` 不同，它**不是外部读取**了：路径从
    `data_root()` 推出来（于是 `MX_DATA` 换快照时它跟着换），`MX_PRICES` 覆盖已去掉。
    这份数据**参与数值计算**：可交易性判定（停牌/涨跌停/复权）与含费净值都靠它。
    读到的值由 `Prices.verify_labels()` 与 `target` 的标签逐格对拍兜底 —— 口径不一致
    会直接报错，不会静默跑完。
    """
    return data_root() / 'prices'


def stock_list_path():
    """股票中文名对照表的固定地址（用户 2026-09-25 授权，只读）。

    与`calendar_path()`一样只用于 picks.md 的显示，不参与任何数值计算。
    从 PROJECT_ROOT 推出来而不是写死绝对路径，共享盘换挂载点时不用改代码。
    """
    return Path(os.environ.get(
        'MX_STOCK_LIST', PROJECT_ROOT.parent / 'datadownload' / 'data' / 'stock_list' / 'data.parquet'))


def calendar_path():
    """交易日历的固定地址（用户 2026-09-25 授权，只读）。

    用途只有一个：picks.md 里把「T+1 买入 / T+6 卖出」写成**具体日期**。快照内的日期
    取 `panel.days`（与标签口径完全一致），快照之后的日期只能来自这份日历——它含未来
    交易日（实测到 2026-10-25），所以最新一天的买卖日也写得出来。
    """
    return Path(os.environ.get(
        'MX_CALENDAR', PROJECT_ROOT.parent / 'datadownload' / 'data' / 'basic_calendar' / 'data.parquet'))


def feature_columns(meta):
    """Frozen chip, money-flow, margin and intraday view; no annual financial fields."""
    prefixes = ('chip_', 'cyqp_', 'cost_', 'mf_', 'margin_', 'short_', 'idt_', 'id2_', 'close30_')
    names = {'big_vs_small_divergence_5d', 'fundflow_retail_inst_divergence',
             'order_size_concentration', 'small_order_crowding',
             'super_large_order_intensity', 'winner_rate_acceleration',
             'winner_rate_reversal_signal', 'dragon_tiger_org_net_20',
             'elg_net_60d_to_mv', 'top_list_net_rate_20'}
    columns = [c for c in meta['columns']['features'] if c.startswith(prefixes) or c in names]
    if len(columns) < 70:
        raise ValueError('flow view unexpectedly small')
    return columns


def market_columns(meta):
    """市场门控的输入列：61 个市场因子的 `_z252`（过去252个交易日含当日的 z-score）。

    只取 `_z252` 而不取原值：原值列量纲不可比（指数点位、比率、金额混在一起），
    z-score 已是尺度无关口径，与股票因子块同一套约定。预热期（2018 全年等）为 NaN，
    载入时按「无信息」填 0，门控退化为恒等。
    """
    block = meta['market_factors']
    columns = [c for c in block['columns'] if c.endswith(f"_z{block['z_window']}")]
    if len(columns) != block['n_factors']:
        raise ValueError(f'市场因子 z 窗口列数 {len(columns)} 与 n_factors {block["n_factors"]} 不一致')
    return columns


def analysis_input_root():
    requested = os.environ.get('MODEL_INPUT_RUN')
    if requested and requested != run_name():
        raise ValueError('本单元使用固定目录，不再支持另一个 --input-run；直接读取 model_train')
    return UNIT_ROOT


#: 固定工作单元的两个常用常量；必须排在上面两个函数定义之后。
RUN_ROOT = run_root()
INPUT_ROOT = analysis_input_root()

# =============================================================================
# ③ 配方（RECIPE）
#
# 一轮训练的超参与口径集中配置，并记录在每折训练信息中。
RECIPE=dict(
    name='V61',
    architecture='flow_temporal',
    label='label_ret_5d',
    label_horizons=[3, 5, 10],
    score_blend='head:label_ret_5d',
    score_label='label_ret_5d',
    label_weights={3: 0.2, 5: 0.5, 10: 0.3},
    valid_ic_weights={3: 0.2, 5: 0.5, 10: 0.3},
    valid_ic_scale=1.0,
    market_gate=False,
    market_block='market_factors',
    market_clip=10.0,
    picks_topn=10,
    backtest_money=float(os.environ.get('MX_MONEY', '100000')),
    picks_holding=1,
    valid_criterion='rankic',
    loss_rank_decay=float(os.environ.get('MX_DECAY', '0.5')),
    train_pool_frac=1.0,
    hot_pool=dict(frac=1.0, amount_window=20, amp_window=5, limit_window=20, min_names=20),
    entry_rule='open',
    hold_band=40,
    budget_rule=os.environ.get('MX_BUDGET', 'divide'),
    bag_topk=5,
    bag_recalibrate=True,
    features='all',
    feature_block=feature_block(),
    folds=4,
    seed=6253,
    batch_days=4,
    optimizer='adamw',
    lr=0.001,
    weight_decay=0.02,
    dropout=tuple((float(x) for x in os.environ.get('MX_DROP', '0.2,0.1').split(','))),
    hidden=tuple((int(x) for x in os.environ.get('MX_HIDDEN', '256,128,32').split(','))),
    max_epochs=30,
    early_stop_patience=6,
    lr_patience=3,
    lr_factor=0.5,
    lr_cooldown=2,
    min_lr=5e-06,
    min_feature_coverage=0.2,
    linear_weight=0.0,
    device='auto',
    threads=int(os.environ.get('MX_THREADS', DEFAULT_THREADS)),
    valid_topn=5,
    valid_metric='weighted raw-label RankIC of delivered score',
    preprocessing='daily_zscore_win1_99_fixed_missing_0',
    test_start='2025-07-01',
    test_end='2026-06-30',
    purge_horizon=max([1, 3, 5, 10, 20]),
    training_window='2018起扩展窗，每季度初更新；季度边界purge max(h)+1=21交易日',
    sequence_days=8,
    family='flow_temporal',
    training_target='daily standardized return; WPCC',
    weights_origin='strategy-only full copy of V53; all32 checkpoint files byte-identical; no new model training',
    research_protocol='four seeds 3253/4253/5253/6253; 4 quarters x 4 folds; all train-before-validation blocks; frozen configurations before test; test display only; no releases promotion',
    strategy_candidates={'topn': [5, 10, 20], 'period': [1, 5, 10], 'band': [0, 20, 40], 'phase': 'all phases for each period'},
    snapshot_cutoff='2026-09-29',
    prediction_horizon=5,
    strategy_parent='V53',
    portfolio_diversity_strength=0.25,
    portfolio_history=63,
    portfolio_min_observations=40,
    portfolio_pool=100,
    portfolio_policy='carry-aware positive-correlation penalty on fresh entries; Top1 baseline and existing40-name hold buffer unchanged',
)

# =============================================================================
# ③b 硬闸：V4 是**全因子正式版**，特征块不是 `factors` 就直接停
#
# 为什么要有这道闸：`run.py` 的特征块默认值、`MODEL_FEATURE_BLOCK` 环境变量与
# `MX_DATA` 三者有互相遮蔽的历史（见 `run.py:feature_block()` 的 ★★ 注释），
# 配错时会**静默**用 20 列跑满 16 折并把产物写成 V4 的结果 —— 产物自洽、不报错。
# 这道闸让"V28 = 479 列全因子"由代码保证，不靠操作纪律。
# =============================================================================
if feature_block() != 'factors':
    raise ValueError(
        f'V4 是全因子正式训练单元，特征块必须是 factors（当前 ={feature_block()}）。\n'
        f'  · 若想跑 20 列的样例：用 experiments/V1，不要在 V4 里降级。\n'
        f'  · 若确实要改本单元的特征块：先改 run.py 的 DEFAULT_FEATURE_BLOCK 与本闸，'
        f'并同步更新 docs/iterations.md 的配置摘要（这属于科研决定，不是运行参数）。')

# =============================================================================
# ④ 网络与损失（来自用户 MLP.py 参考脚本）
# =============================================================================


def head_of(label):
    """标签名 → 输出头列号；列序固定为 RECIPE['label_horizons']。"""
    horizon = int(label.split('_ret_')[1].rstrip('d'))
    return RECIPE['label_horizons'].index(horizon)


class MarketGate(nn.Module):
    """市场门控：由当日市场状态生成**逐通道**乘性门，调制股票分支的隐藏表示。

    门取 `1 + tanh(·)` ∈ (0,2)，两个头都零初始化 ⇒ 起点 gate 恒等于 1，也就是
    「训练开始那一刻与无门控模型逐位等价」，门控只学相对中性的偏移。换成 sigmoid
    门则在初始状态就砍掉一半信号，是另一种（更弱的）模型，不是同一个起点。

    ★ 门必须逐通道（宽度 = 被乘隐层的宽度）。若简化成每天一个标量，`wpcc` 对同日
    截面的正仿射变换精确不变 ⇒ 梯度恒为 0、市场输入永远不起作用，且**不会报错**。
    """

    def __init__(self, market_dim, widths):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(market_dim, 64), nn.GELU())
        self.heads = nn.ModuleList(nn.Linear(64, width) for width in widths)

    def zero_heads(self):
        """把门控头清零：gate 恒等于 1。必须在通用权重初始化之后调用。"""
        for head in self.heads:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(self, market):
        z = self.encoder(market.float())
        return [1. + torch.tanh(head(z)) for head in self.heads]


class PredictModel(nn.Module):
    """Eight causal sessions of flow inputs; shared GRU, no common MLP/ridge/market gate."""
    def __init__(self, input_dim, market_dim, horizons=None):
        super().__init__()
        if market_dim:
            raise ValueError('flow temporal family has no shared market gate')
        self.horizons = list(horizons or RECIPE['label_horizons'])
        self.encoder = nn.Sequential(nn.Linear(input_dim, 24), nn.LayerNorm(24), nn.GELU())
        self.recurrent = nn.GRU(24, 32, batch_first=True)
        self.output_layer = nn.Sequential(nn.LayerNorm(32), nn.Linear(32, len(self.horizons)))

    def forward(self, tsdata, market=None):
        if tsdata.ndim != 3 or tsdata.shape[1] != RECIPE['sequence_days']:
            raise ValueError('expected stock x causal-session x feature input')
        hidden, _ = self.recurrent(self.encoder(tsdata.float().clamp(-5.0, 5.0)))
        # The final hidden state only consumes dates <= the signal date.
        return self.output_layer(hidden[:, -1])


def wpcc(preds, y):
    """按预测名次加权的 Pearson 相关（WPCC），保留参考脚本公式。

    权重依预测名次衰减（`RECIPE['loss_rank_decay'] ** (名次/(n-1))`）；方差围绕普通均值
    而不是加权均值——不能偷偷换成另一种加权 Pearson，只增加了常数截面的数值保护。
    每次输入必须是单个交易日的完整有效截面。

    ★ V19：衰减底数从写死的 0.5 改成旋钮。0.5 时顶部权重 1、底部 0.5 —— **几乎是均权**；
    而本项目测出「模型的钱在最顶端」，交付又只看前 5 名 ⇒ 更陡的加权是对这个观测的
    直接推论。**改成旋钮而不是改常数**，是为了能按协议做多种子判决（改损失必须 ≥3 种子）。
    """
    p = preds.reshape(-1)
    y = y.reshape(-1)
    n = p.numel()
    if n < 2:
        return p.sum() * 0
    order = torch.argsort(p, descending=True, stable=True)
    w = torch.empty_like(p)
    w[order] = torch.pow(RECIPE['loss_rank_decay'],
                         torch.arange(n, device=p.device, dtype=p.dtype) / (n - 1))
    w_sum = w.sum()
    cov = (p * y * w).sum() / w_sum - (p * w).sum() * (y * w).sum() / w_sum.square()
    vp = ((p - p.mean()).square() * w).sum() / w_sum
    vy = ((y - y.mean()).square() * w).sum() / w_sum
    return -cov / (vp * vy).clamp_min(1e-16).sqrt()

# =============================================================================
# =============================================================================


def metadata():
    return json.loads((data_root() / 'meta.json').read_text())


def training_info(recipe):
    """记录训练参数和实际输入列，供报告读取。"""
    return dict(recipe=recipe, feature_source=str(data_root()/recipe['feature_block']),
                feature_columns=feature_columns(metadata()),
                market_columns=market_columns(metadata()) if recipe['market_gate'] else [])


def frame(root, kind, year, columns):
    """读一个数据块的某一年，统一主键类型并断言无重复。"""
    p = root / kind / f'year={year}' / 'data.parquet'
    df = pq.ParquetFile(p).read(columns=['trade_date', 'stock_code'] + list(columns)).to_pandas()
    df['trade_date'] = df['trade_date'].astype(str).str[:10]
    df['stock_code'] = df['stock_code'].astype(str)
    if df.duplicated(['trade_date', 'stock_code']).any():
        raise ValueError(f'{p}: 重复主键')
    return df.set_index(['trade_date', 'stock_code']).sort_index()


#: 价格块的 8 列，与上游 `preparingdata.py:PRICE_COLUMNS` 逐字一致（前 7 列来自模块①
#: 的 `stock_daily`、`adj_factor` 来自 `stock_adj_factor`，由快照块 `prices/` 落好）。
PRICE_COLUMNS = ('open', 'high', 'low', 'close', 'pre_close', 'pct_chg', 'vol', 'adj_factor')


def price_frame(root, year, index):
    """读快照的价格块 `prices/year=YYYY/data.parquet`，按 `index`（当年的 (trade_date, stock_code) 乘积）对齐。

    ★ 2026-09-26 起价格由上游 `preparingdata.py` 从模块① 复制进快照（见该脚本的 `prices` 块），
      本单元只读 `trainingdata/`、不再直连 `datadownload/`。块里的列就是本文件
      `PRICE_COLUMNS` 那 8 列，逐字来自原始表（float64，未复权、未填充）。

    两条校验，对不上就直接报错、不静默错位：
      ① 快照里那一年的**交易日集合**必须与面板当年完全一致（块与 factors 同轴，这是快照契约）；
      ② 主键不得重复。
    缺格（停牌 / 池内票当年没有日线）是 NaN —— 消费端正是靠这个 NaN 判"买不进"。
    再往下的数值对拍由 `Prices.verify_labels()` 兜底：它用价格重算 5 个 horizon 的收益
    与 `target` 逐格比，口径不一致会直接 raise。
    """
    p = Path(root) / f'year={year}' / 'data.parquet'
    part = pq.ParquetFile(p).read(columns=['trade_date', 'stock_code', *PRICE_COLUMNS]).to_pandas()
    part['trade_date'] = part['trade_date'].astype(str).str[:10]
    part['stock_code'] = part['stock_code'].astype(str)
    if part.duplicated(['trade_date', 'stock_code']).any():
        raise ValueError(f'{p}: 重复主键')
    part = part.set_index(['trade_date', 'stock_code'])
    want = set(index.get_level_values(0))
    days_seen = set(part.index.get_level_values(0))
    if days_seen != want:
        raise ValueError(
            f'{year} 价格块日线轴与 factors 不一致（缺 {sorted(want - days_seen)[:3]} / '
            f'多 {sorted(days_seen - want)[:3]}）—— 回测会错位；跑 preparingdata.py --prices-only 重建')
    return part.reindex(index)


def market_frame(root, year, columns):
    """读市场块的某一年：主键只有 trade_date，每日恰好一行。"""
    p = root / RECIPE['market_block'] / f'year={year}' / 'data.parquet'
    df = pq.ParquetFile(p).read(columns=['trade_date'] + list(columns)).to_pandas()
    df['trade_date'] = df['trade_date'].astype(str).str[:10]
    if df['trade_date'].duplicated().any():
        raise ValueError(f'{p}: 市场块 trade_date 重复')
    return df.set_index('trade_date').sort_index()


def atomic_json(path, value):
    """先写 .tmp 再 os.replace，保证读到的永远是完整 JSON。"""
    path = output_file(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(temp, path)


def save_torch(path, value):
    """同上，torch 检查点的原子写。"""
    path = output_file(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    torch.save(value, tmp)
    os.replace(tmp, path)


def save_npy(path, value):
    """同上，npy 打分的原子写。"""
    path = output_file(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    with open(tmp, 'wb') as f:
        np.save(f, value)
    os.replace(tmp, path)

# =============================================================================
# ⑥ 交易日轴与季度切分
# =============================================================================


def axis():
    """交易日轴与股票轴：年份从 meta 的 built_years 来，顺序与 factors 块一致。"""
    root = data_root()
    meta = metadata()
    days = []
    for y in sorted(meta['built_years']):
        d = pq.ParquetFile(root / feature_block() / f'year={y}' / 'data.parquet').read(columns=['trade_date'])
        days.extend(sorted(set(d.column(0).to_pylist())))
    return np.asarray(days), np.asarray(meta['axis']['codes'])


def quarters(days):
    """评价窗口固定为 2025-07-01 ~ 2026-06-30（用户指定，V16 及以后版本都遵守）。

    入参 days 只为保持调用签名一致，不影响返回值——不能自行扩展或缩成部分季度。
    """
    return ['2025Q3', '2025Q4', '2026Q1', '2026Q2']


def splits(days, purge_horizon=RECIPE['purge_horizon'], folds=4, quarter=None):
    """一个季度的四折切分：训练/验证块在季度前 purge_horizon+1 日隔离，测试窗严格样本外。

    `purge_horizon` 必须取**最长**标签期（V2 是 20d）：标签 T 使用 T+1+h 开盘，按 6 日
    隔离时训练集里 20d 目标的窗口会伸进验证窗（验证指标偏乐观）。取 20 时实际隔离
    21 个交易日，训练目标不再窥视验证窗。
    """
    # 标签 T 使用 T+1+purge_horizon 开盘；严格早于验证/测试的边界，不共享端点。
    quarter = quarter or quarters(days)[0]
    q = pd.Period(quarter, freq='Q')
    start = str(q.start_time.date())
    end = str(q.end_time.date())
    ix = np.arange(len(days))
    first = int(np.searchsorted(days, start))
    cutoff = first - purge_horizon - 1
    eligible = ix[:cutoff]
    # 测试窗与 RECIPE 的 test_start / test_end 保持一致。
    test = ix[(days >= max(start, '2025-07-01')) & (days <= min(end, '2026-06-30'))]
    # 推演窗：从本季度第一个交易日一直到**数据末日**（不只是这个季度）。
    # 一次推演同时支撑两件事：① 评价窗内那一段（= test，是推演窗的前缀）
    # ② 之后的增量最新打分 —— 该日由"最新一个没见过它的季度模型组"负责，见 analysis.py。
    score = ix[first:]
    out = []
    for k, block in enumerate(np.array_split(eligible, folds), 1):
        a, b = int(block[0]), int(block[-1]) + 1
        # 验证块两端各留 h+1 日的 purge/embargo，训练集再把验证块整段挖掉。
        valid = block[block + purge_horizon + 1 < b]
        train = eligible[(eligible + purge_horizon + 1 < a) | (eligible >= b + purge_horizon + 1)]
        if not len(train) or not len(valid) or not len(test) or not len(score):
            raise ValueError('训练/验证/测试/推演窗为空')
        assert max(train.max(), valid.max()) + purge_horizon + 1 < first
        assert not np.intersect1d(train, valid).size
        assert test[0] == score[0] and test[-1] == score[len(test) - 1]   # test 必须是 score 的前缀
        out.append(dict(quarter=quarter, fold=k, train=train, valid=valid, test=test, score=score))
    return out


def split_summary(days, split):
    """把切分的索引数组压成可读的连续日期区间，供落盘与打印。"""
    def spans(ix):
        runs = np.split(ix, np.flatnonzero(np.diff(ix) > 1) + 1)
        return [{'start': str(days[r[0]]), 'end': str(days[r[-1]]), 'days': len(r)} for r in runs if len(r)]
    return {'quarter': split['quarter'], 'fold': split['fold'],
            **{k: spans(split[k]) for k in ['train', 'valid', 'test', 'score']}}

# =============================================================================
# ⑦ 面板：特征 / 标签 / 成交额
# =============================================================================


class Panel:
    """全历史面板，按 (交易日 × 股票) 的二维网格存放；市场块按 (交易日) 一行。

    load_x=False 时只读标签与成交额（评价阶段用），省掉最大的一块内存；市场块只有
    T×61 个数（实测 0.5 MB），无论 load_x 与否都读，供门控使用。
    """

    def __init__(self, features='all', load_x=True):
        self.root = data_root()
        self.meta = metadata()
        self.days, self.codes = axis()
        if str(self.days[-1]) != RECIPE['snapshot_cutoff']:
            raise ValueError('snapshot changed: create a new independent version')
        self.features = (feature_columns(self.meta) if features in ('all', RECIPE['features'])
                         else self.meta['feature_sets'][features])
        if not set(self.features).issubset(feature_columns(self.meta)):
            raise ValueError('请求的特征不在选定数据块中')
        self.labels = self.meta['columns']['labels']
        self.market_features = market_columns(self.meta) if RECIPE['market_gate'] else []
        self.market_dim = len(self.market_features)

        T, C = len(self.days), len(self.codes)
        self.X = np.empty((T, C, len(self.features)), np.float32) if load_x else None
        self.coverage = np.empty((T, C), np.float32) if load_x else None
        self.Y = {name: np.full((T, C), np.nan, np.float32) for name in self.labels}
        self.amount = np.full((T, C), np.nan, np.float64)
        self.M = np.zeros((T, self.market_dim), np.float32)      # 门控输入：逐日市场状态

        # 逐年读取并写进预分配好的网格；每年都断言轴一致，避免静默错位。
        t_load = time.time()
        years = list(self.meta['built_years'])
        filled = 0                       # 被逐年循环真正填过的交易日数（下面要断言 = T）
        for yi, y in enumerate(years, 1):
            rows = np.flatnonzero(np.char.startswith(self.days, str(y)))
            filled += len(rows)
            index = pd.MultiIndex.from_product([self.days[rows], self.codes], names=['trade_date', 'stock_code'])
            if load_x:
                df = frame(self.root, feature_block(), y, self.features)
                if len(df) != len(index) or not df.index.equals(index):
                    raise ValueError(f'{y} {feature_block()} 轴不一致')
                x = df.to_numpy(np.float32).reshape(len(rows), C, -1)
                self.coverage[rows] = np.isfinite(x).mean(axis=2)
                # 快照口径 meta['semantics']='zscore_win1_99_v1'：当日截面缩尾(1%/99%) →
                # 减均值除标准差 → 截断±10 → 按 direction 取负。已经是中心 0、尺度一致的
                # 当日截面量，**缺失的中性填充是 0**，不再做仿射变换（旧的 (x-.5)*2 是
                # [0,1] rank 口径的消费方式，会在新快照上静默把输入放大一倍）。
                x = np.nan_to_num(x, nan=0.)
                # ★ 无穷值检查放在**逐年**做（对 nan_to_num 之后的结果，与原先整块检查同义），
                #   不再在整块 X 上做：479 列下 `np.isfinite(self.X).all()` 会临时分配一份
                #   (T,C,F) 的 bool = 2.14 GB，而逐年只 245 MB 且随迭代释放。
                #   注意必须查**填充之后**的数组：`np.isfinite` 对 NaN 也是 False，
                #   而缺失值本来就是 NaN（快照口径就是"缺失填 0"）—— 查之前会全军覆没。
                if not np.isfinite(x).all():
                    raise ValueError(f'{y} 年特征含无穷值')
                self.X[rows] = x
                del df, x
            if self.market_dim:
                mkt = market_frame(self.root, y, self.market_features)
                if not mkt.index.equals(pd.Index(self.days[rows])):
                    raise ValueError(f'{y} 市场块日期轴与 {feature_block()} 不一致')
                # 预热期（2018 全年等）的 z252 是 NaN：填 0 在 z 语义下就是「无信息」，
                # 门控自然退化为恒等；截断口径与股票因子块一致。
                self.M[rows] = np.clip(np.nan_to_num(mkt.to_numpy(np.float32), nan=0.),
                                       -RECIPE['market_clip'], RECIPE['market_clip'])
            df = frame(self.root, 'target', y, self.labels).reindex(index)
            for name in self.labels:
                self.Y[name][rows] = df[name].to_numpy(np.float32).reshape(len(rows), C)
            self.amount[rows] = frame(self.root, 'amount', y, ['amount']).reindex(index)['amount'].to_numpy().reshape(len(rows), C)
            print(f'已读取 {y} 年，{len(rows)} 个交易日'
                  f'（{yi}/{len(years)}，已用 {time.time() - t_load:.0f}s，'
                  f'预计还需 {(time.time() - t_load) / yi * (len(years) - yi):.0f}s）', flush=True)

        if load_x and filled != T:
            # 逐年循环必须恰好覆盖整条日期轴；漏了就会留下 `np.empty` 的未初始化内存，
            # 而"特征含无穷值"那道检查已经挪进循环里、不会再兜住它。
            raise ValueError(f'特征面板只填了 {filled}/{T} 个交易日 —— 年份分区没覆盖整条日期轴')
        if not np.isfinite(self.M).all():
            raise ValueError('市场块含无穷值')
        # 读盘是整条流水线里最长的静默段（约 1 分钟），逐行给进度与 ETA。
        print(f'面板读取完成：{len(self.meta["built_years"])} 年，市场列 {self.market_dim}，'
              f'已用 {time.time() - t_load:.0f}s', flush=True)

    def inputs(self, day, mask):
        width = RECIPE['sequence_days']
        if day < width - 1:
            raise ValueError('insufficient causal history')
        return np.ascontiguousarray(self.X[day - width + 1:day + 1, mask].transpose(1, 0, 2))

    def mask(self, day, label=None):
        """当日可用的股票掩码：覆盖率达标，且（给定标签时）标签已知。

        ★ 阈值改读 `RECIPE['min_feature_coverage']`：原先这里硬编码 `.2`，而 RECIPE 里
          同名键**从未被读**（改了不生效）。实测两种特征块下该掩码都不触发
          （479 列时 per-(day,stock) 有效占比最小 0.41，20 列时最小 0.30），
          所以这次接线在数值上是 no-op，纯粹让配置成为唯一真源。
        """
        width = RECIPE['sequence_days']
        if day < width - 1:
            return np.zeros(len(self.codes), dtype=bool)
        out = (self.coverage[day-width+1:day+1] >= RECIPE['min_feature_coverage']).all(axis=0)
        if label is not None:
            out = out & np.isfinite(self.Y[label][day])
        return out

# =============================================================================
# ⑧ 价格与可交易性
# =============================================================================


class Prices:
    """原始价 + 停牌/限价的可交易性判定 + ★V11 的热门池。

    只前向填充状态量（复权因子、收盘价）；停牌无开盘价仍不可交易，不前填成交量。
    """

    def __init__(self, panel):
        names = list(PRICE_COLUMNS)
        T, C = len(panel.days), len(panel.codes)
        self.raw = {n: np.full((T, C), np.nan, np.float64) for n in names}

        # 逐年读快照的价格块（`trainingdata/prices`，见 price_frame）；日期轴必须与 factors 一致。
        for year in panel.meta['built_years']:
            ix = np.flatnonzero(np.char.startswith(panel.days, str(year)))
            index = pd.MultiIndex.from_product([panel.days[ix], panel.codes], names=['trade_date', 'stock_code'])
            df = price_frame(price_root(), year, index)
            for n in names:
                self.raw[n][ix] = df[n].to_numpy().reshape(len(ix), C)

        self.adj = pd.DataFrame(self.raw['adj_factor']).ffill().to_numpy()
        self.close = pd.DataFrame(self.raw['close']).ffill().to_numpy()
        self.open = self.raw['open']

        # 冻结池均为非ST主板；按分四舍五入。开盘触及限价即保守拒单，
        # 不读取当天 high/low 决定开盘订单，避免用未来全天行情挑选成交。
        pre = self.raw['pre_close']
        upper = np.floor(pre * 1.10 * 100 + .5) / 100
        lower = np.floor(pre * .90 * 100 + .5) / 100
        self.limit_up = upper
        self.limit_down = lower
        traded = (np.isfinite(self.open) & (self.open > 0) & np.isfinite(pre) & (pre > 0)
                  & np.isfinite(self.adj) & (self.adj > 0) & (self.raw['vol'] > 0))
        self.entry = traded & (self.open < upper - 1e-8)
        self.exit = traded & (self.open > lower + 1e-8)

        # ★★ V11 打板口径：**整日最低价低于涨停价** ⇒ 板被打开过 ⇒ 挂在涨停价的买单能成交。
        #    `low` 只用来判「能不能成交」，不参与选股（票是信号日就选定的），所以不是
        #    "用未来全天行情挑票"。一字板（整日封死，low ≥ 涨停价）拿不到货，仍然拒单 ——
        #    这是打板最核心的一档风险，放过它回测会系统性偏乐观。
        #    成交价就是开盘价：挂涨停价买入时，开盘低于涨停按开盘价成交；开盘即涨停而
        #    盘中开板则按涨停价成交 —— 而开盘价此时恰好等于涨停价。⇒ 两种情形都取开盘价。
        self.board_entry = traded & (self.raw['low'] < upper - 1e-8)

        # next_entry[d] = 第 d 日开盘能不能买进「T-1 日选出的票」。
        self.next_entry = np.zeros_like(self.entry)
        self.next_entry[:-1] = self.entry[1:]
        self.next_board_entry = np.zeros_like(self.board_entry)
        self.next_board_entry[:-1] = self.board_entry[1:]

        # 收盘卖的可执行性：判据与开盘卖同法，但用当日**收盘价**判跌停 —— 收盘单就在那个
        # 时点成交，不含任何超过当日收盘的信息（不是"用当天行情挑成交"）。停牌/无量同样拒单。
        raw_close = self.raw['close']
        self.exit_close = (np.isfinite(raw_close) & (raw_close > 0) & np.isfinite(pre) & (pre > 0)
                           & np.isfinite(self.adj) & (self.adj > 0) & (self.raw['vol'] > 0)
                           & (raw_close > lower + 1e-8))

        self._hot_pool(panel)

    # -------------------------------------------------------------------------
    # ★ V11：热门池（人气股 / 妖股）
    # -------------------------------------------------------------------------
    def _hot_pool(self, panel):
        """当日截面「热门度」排名前 `hot_pool['frac']` 的票 —— 只会用到快照自己的块。

        三项各做**当日截面**百分位名次再等权平均（名次在当日截面上取 ⇒ PIT 安全）：

        | 项 | 算法 | 为什么是「热门」 |
        | --- | --- | --- |
        | 金额异动 | `amount[d] ÷ mean(amount[d-w:d])` | 钱突然涌进来 = **人气**。用**前 w 日**均额做分母（不含当日），当日巨量才不会被自己稀释 |
        | 妖气 | 近 `amp_window` 日 `(high−low)/pre_close` 的均值 | 振幅大 = **妖股** |
        | 板性 | 近 `limit_window` 日「收盘触涨停」的天数 | **涨停生态 / 打板候选** |

        ★ 为什么不用 `factors` 块里的涨停/换手因子：那一块已经按 `columns.direction`
        做过符号翻转（`raw = 存储值 × direction`，479 列里 121 列是 −1），拿它组「越大越热门」
        的合成量要把 121 个符号全对一遍 —— **错了不会报错**，只会静默把一个反向的池子交出去。
        价格块没有这层歧义：涨没涨停、振幅多大，都是直接算出来的。

        `frac >= 1` 时池子退化为全截面（`hot` 全 True），用来做同权重的干净对照。
        """
        spec = RECIPE['hot_pool']
        frac, aw, lw = spec['frac'], spec['amount_window'], spec['limit_window']
        T, C = len(panel.days), len(panel.codes)
        # ★ 记下"这份掩码是按哪个 frac 算的"：`hot` 只在构造时算一次，之后改 RECIPE 里的
        #   frac **不会**让它重算 —— 有对照台就是靠改 frac 来做 A/B 的，静默沿用旧掩码
        #   会让整个扫描得到一排一模一样的数字（实测踩过）。`hot_filter()` 靠这个字段把它
        #   变成一声响的错误，而不是一个安静的假结果。
        self.hot_frac = frac
        self.hot = np.ones((T, C), bool)                 # 默认全池（frac=1 或算不出来时）
        self.hot_score = np.full((T, C), np.nan, np.float32)
        if frac >= 1.:
            return

        amount = panel.amount
        # 分母不含当日（`.shift(1)`）；前 aw 日的均额用 min_periods 放宽到 5 日，
        # 让 2018 年初的预热期短一点，但不足 5 日的仍然算不出来（留 NaN）。
        base = pd.DataFrame(amount).rolling(aw, min_periods=5).mean().shift(1).to_numpy()
        with np.errstate(invalid='ignore', divide='ignore'):
            activity = np.where((base > 0) & np.isfinite(amount), amount / base, np.nan)

        pre = self.raw['pre_close']
        with np.errstate(invalid='ignore', divide='ignore'):
            amp = np.where((pre > 0) & np.isfinite(self.raw['high']) & np.isfinite(self.raw['low']),
                           (self.raw['high'] - self.raw['low']) / pre, np.nan)
        amplitude = pd.DataFrame(amp).rolling(spec['amp_window'], min_periods=2).mean().to_numpy()

        # 「收盘触板」只在真的有成交（vol>0、价格齐全）的日子算，停牌日不能算成板。
        # 涨停价就是上面按分四舍五入算好的 `self.limit_up`，不另算一遍。
        raw_close = self.raw['close']
        touch = (np.isfinite(raw_close) & (raw_close > 0) & (pre > 0) & (self.raw['vol'] > 0)
                 & (raw_close >= self.limit_up - 1e-8))
        boards = pd.DataFrame(touch.astype(np.float64)).rolling(lw, min_periods=1).sum().to_numpy()

        # 池子算不出来的日子（预热期）。**不静默**：跑完打印，且这些日子按"不设池"处理。
        bare = []
        for d in range(T):
            ranked = []
            for values in (activity[d], amplitude[d], boards[d]):
                ok = np.isfinite(values)
                if ok.sum() < spec['min_names']:
                    continue
                r = np.full(C, np.nan)
                r[ok] = (rankdata(values[ok]) - 1) / (ok.sum() - 1)   # 当日截面百分位 ∈ [0,1]
                ranked.append(r)
            # ★ 至少要**两项**可用。只靠一项的池子在预热期真的出现过：2018-01-02 只有
            #   「板性」能算（前 20 日均额与近 5 日振幅都还没有窗口），而它是 0/1 两值、
            #   `rankdata` 把并列排成同一个名次 ⇒ 合成分成了常数 ⇒ `>= 分位数` 对**全体**
            #   成立，池子静默退化成"全池 2115 只"。要求两项就把这条堵死了。
            if len(ranked) < 2:
                bare.append(d)
                continue
            score = np.nanmean(np.stack(ranked), axis=0)
            self.hot_score[d] = score.astype(np.float32)
            ok = np.isfinite(score)
            # ★ 第二道保险：合成分没有区分度（并列成一坨）时同样不设池 —— 分位数阈值
            #   在常数向量上会把所有票一次放进来，这是上面那条的通用形式。
            if ok.sum() < spec['min_names'] or float(np.ptp(score[ok])) < 1e-12:
                bare.append(d)
                continue
            self.hot[d] = ok & (score >= np.quantile(score[ok], 1. - frac))

        if bare:
            print(f'热门池：{len(bare)} 天算不出来（预热期，按"不设池"处理，池内=全截面）：'
                  f'{[str(panel.days[d]) for d in bare[:5]]}'
                  f'{" …" if len(bare) > 5 else ""}', flush=True)

    def entry_ok(self):
        """执行日的「能不能买进」判定 —— 由 `RECIPE['entry_rule']` 决定，不在调用处写死。"""
        rule = RECIPE['entry_rule']
        if rule == 'open':
            return self.entry
        if rule == 'board':
            return self.board_entry
        raise ValueError(f"未知 entry_rule={rule!r}；见 RECIPE['entry_rule']")

    def next_entry_ok(self):
        """信号日口径的「T+1 能不能买进」—— `validation()` 的收益代理用它，与回测同源。"""
        rule = RECIPE['entry_rule']
        if rule == 'open':
            return self.next_entry
        if rule == 'board':
            return self.next_board_entry
        raise ValueError(f"未知 entry_rule={rule!r}；见 RECIPE['entry_rule']")

    def verify_labels(self, panel, days):
        """价格轴/复权口径对拍：若标签与价格不是同一口径，回测不能继续。"""
        px = pd.DataFrame(self.open).ffill().to_numpy() * self.adj
        errors = []
        for h in [1, 3, 5, 10, 20]:
            ix = np.asarray(days)
            ix = ix[ix + h + 1 < len(px)]
            calc = px[ix + h + 1] / px[ix + 1] - 1
            target = panel.Y[f'label_ret_{h}d'][ix]
            good = np.isfinite(calc) & np.isfinite(target)
            diff = np.abs(calc[good] - target[good])
            errors.append({'horizon': h, 'checked': int(good.sum()),
                           'max_abs_error': float(diff.max()) if len(diff) else None})
            if not len(diff) or np.mean(diff > 1e-5) > .001:
                raise ValueError(f'价格与 {h}d 标签口径不一致: {errors[-1]}')
        return errors

# =============================================================================
# ⑨ 预测、验证与单折训练
# =============================================================================


def corr(a, b):
    if len(a) < 3 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.
    return float(np.corrcoef(a, b)[0, 1])


def device_for(request):
    if request == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if request == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('指定 CUDA 但不可用')
    return torch.device(request)


@torch.no_grad()
def predict_heads(model, panel, days, device):
    """逐日对当日有效截面推**全部**输出头；返回 (天数, 股票数, 头数)，无效位置留 NaN。"""
    model.eval()
    out = np.full((len(days), len(panel.codes), len(model.horizons)), np.nan, np.float32)
    for i, d in enumerate(days):
        mask = panel.mask(d)
        if mask.sum():
            x = torch.from_numpy(panel.inputs(d, mask)).to(device)
            m = torch.from_numpy(panel.M[d]).to(device) if panel.market_dim else None
            out[i, mask] = model(x, m).cpu().numpy()
    return out


def score_of(heads):
    """★ V3 的打分：把 (天数, 股票数, 头数) 的头输出**逐日截面秩平均**成二维打分。

    `RECIPE['score_blend']` 是唯一真源：`'rank_mean_all_heads'`（V3）或
    `'head:label_ret_5d'`（V2 的旧口径，保留实现以便回退对照）。

    为什么在**秩**上平均而不是在原值上平均：五个头是同一个主干分出来的，尺度与分布
    会各自漂移（训练中 `wpcc` 对正仿射不变 ⇒ 原值尺度不受约束），原值平均会被某个
    头的大方差带跑；秩平均对每个头是单调不变，天然免疫。秩在当日截面上取，只用当日
    信息 ⇒ PIT 安全，与 `analysis.py` 的 RankIC 同一套口径。

    缺失位置（`predict_heads` 留的 NaN，= 当日覆盖率不达标的票）保持 NaN，不参与排名，
    也不被排成中间名次——它们既不该进 IC 也不该进组合。
    """
    spec = RECIPE['score_blend']
    if spec.startswith('head:'):
        return heads[:, :, head_of(spec.split(':', 1)[1])]
    if spec != 'rank_mean_all_heads':
        raise ValueError(f"未知的 score_blend={spec!r}；见 RECIPE['score_blend'] 的注释")
    out = np.full(heads.shape[:2], np.nan, np.float32)
    for i in range(heads.shape[0]):
        row = heads[i]
        # 五个头同一次前向、同一个掩码，所以"全头有限"与"头 0 有限"等价；取严的那个，
        # 缺口的票一律不参与排名（**不填 0 当中性**：那会在一堆真值里插进假值插队）。
        known = np.isfinite(row).all(axis=1)
        n = int(known.sum())
        if n < 2:                               # 少于两只票没有"秩"可言
            continue
        acc = np.zeros(n)
        for j in range(heads.shape[2]):
            v = row[known, j].astype(np.float64)
            # 名次归一到以 0 为中心：rank∈[1,n] → (rank − (n+1)/2)/n ∈ 约 (−.5, .5]，均值 0
            acc += (rankdata(v) - (n + 1) / 2) / n
        out[i, known] = acc / heads.shape[2]
    return out


def fit_ridge(panel, train_days, lam=1e3, pool=None):
    """在训练窗上闭式拟合岭回归（逐日截面已 z-score 的特征 → 当日标准化的 5 日标签）。

    返回 `(n_features + 1,)` 的权重（末位是截距）。逐日累积 Gram 矩阵，不整块驻留 X。

    ★ 为什么要在神经网络之外再挂一个线性分量：两者**信息互补**——1893 天样本外上，
    线性分量在全截面 RankIC 上全面更好（四块 +0.004~0.008），而网络在极值桶（top5）上更好。
    掺 20% 后 RankIC / Pearson IC / top5 **三个同时变好**（见 `blend_linear`）。
    """
    if not RECIPE['linear_weight']:
        return None
    D = len(panel.features)
    G = np.zeros((D + 1, D + 1))
    b = np.zeros(D + 1)
    for d in train_days:
        m = panel.mask(d, 'label_ret_5d')
        if pool is not None:                    # ★ V17：岭回归分量与网络用同一套样本
            m = m & pool[d]
        if m.sum() < 50:
            continue
        X = panel.X[d, m].astype(np.float64)
        y = panel.Y['label_ret_5d'][d, m].astype(np.float64)
        sd = y.std()
        if sd < 1e-8:
            continue
        y = (y - y.mean()) / sd
        A = np.c_[X, np.ones(len(X))]
        G += A.T @ A
        b += A.T @ y
    G[np.arange(D), np.arange(D)] += lam
    return np.linalg.solve(G, b)


def blend_linear(score, w, panel, days):
    """★ V9：把岭回归分量**仿射映射**到打分的均值/标准差后按 `linear_weight` 凸组合。

    ★ 必须仿射、不能秩混合：秩混合会把打分变成秩尺度、把 Pearson IC 从 0.0863 压到
    0.0708（V3 那次在秩上聚合的同一個坑）；仿射保持量纲，实测三指标同时改善。
    `w=None`（未拟合）或权重为 0 时原样返回，便于回退。
    """
    lw = RECIPE['linear_weight']
    if not lw or w is None:
        return score
    out = score.copy()
    for i, d in enumerate(days):
        s = score[i]
        known = np.isfinite(s)
        if known.sum() < 20:
            continue
        r = panel.X[d, known].astype(np.float64) @ w[:-1] + w[-1]
        sd = r.std()
        if sd < 1e-12:
            continue
        r = (r - r.mean()) * (s[known].std() / sd) + s[known].mean()
        out[i, known] = (1. - lw) * s[known] + lw * r
    return out


def average_state_dicts(states):
    """★ V12：把若干份 `state_dict` 逐键取均值（权重与 BN 的 running 统计一起平均）。

    ★ 合法性前提：这些状态来自**同一条训练轨迹上的不同 epoch**（同一次运行、同一个种子、
    同一批数据）。它们落在同一个吸引盆里，逐键平均才近似"盆心"。
    **跨种子平均不成立**（不同盆），不要拿这个函数去做多种子集成。

    ★ 非浮点键（`num_batches_tracked` 这类计数）不参与平均，原样取第一份 —— 对它们求均值
    既没有意义，`.mean()` 在整型上还会直接报错。
    """
    averaged = {}
    for key, first in states[0].items():
        if not first.is_floating_point():
            averaged[key] = first.clone()
            continue
        averaged[key] = torch.stack([state[key].float() for state in states]).mean(dim=0).to(first.dtype)
    return averaged


def recalibrate_bn(model, panel, train_days, device):
    """★ V12：按**平均后的权重**把 BatchNorm 的 running 统计重估一遍（SWA 的标准收尾）。

    为什么必须做：每份权重的 `running_mean/var` 是按"它自己"的权重统计出来的，对平均后的
    权重并不成立。沿用会让推理时的归一化按错误的尺度走 —— 而且**不会报错**，只会静默地
    把打分弄偏（本项目最忌讳的那种失败）。

    做法是一次**纯前向**：`train()` 模式让 BN 自己累计统计（`momentum=None` ⇒ 累计平均，
    正好等于整窗统计量），同时把 Dropout 关掉 —— Dropout 不是状态量，开着只会把统计量弄脏。
    """
    model.train()
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.eval()
    for module in model.modules():
        if isinstance(module, nn.BatchNorm1d):
            module.reset_running_stats()
            module.momentum = None          # 累计平均：这一步跑完正好是整窗的统计量
    with torch.no_grad():
        for d in train_days:
            mask = panel.mask(d)
            if mask.sum() < 3:
                continue
            x = torch.from_numpy(panel.inputs(d, mask)).to(device)
            m = torch.from_numpy(panel.M[d]).to(device) if panel.market_dim else None
            model(x, m)
    model.eval()


def hot_filter(score, prices, days):
    """★ V11：把打分限制在当日的热门池内 —— 池外一律置 NaN。

    为什么置 NaN 而不是排到最后：下游（`score_of` 的秩、`top_portfolio_daily` 的 `known`、
    `cash_backtest` 的 `isfinite`、`write_scores` 的落盘）**一致地把 NaN 当成"这只票今天
    没有分数"**，于是池外的票自然既不进 IC、也不进组合、也不进 picks.md。换成"打个低分"
    就会在一堆真分数里插进假分数，把 RankIC 和 top5 一起带偏。

    ★ 这里**不做**逐日重排：池内的相对顺序就是原打分的相对顺序（单调变换不改变池内名次），
    池内 RankIC 与池外无关。
    """
    frac = RECIPE['hot_pool']['frac']
    if frac != prices.hot_frac:
        raise ValueError(
            f'热门池掩码是按 frac={prices.hot_frac} 算的，当前 RECIPE 写的是 {frac} —— '
            f'掩码不会自动重算，直接跑会静默沿用旧掩码。\n'
            f'  · 要换 frac：重建 Prices（`Prices(panel)` 会在构造时按新 frac 重算掩码）。\n'
            f'  · 要做 A/B：改完 frac 必须重建，否则两边拿到的是同一份掩码。')
    if frac >= 1.:
        return score
    out = score.copy()
    for i, d in enumerate(days):
        out[i][~prices.hot[d]] = np.nan
    return out


def predict(model, panel, prices, days, device):
    """打分：`predict_heads` 的全部头 → `score_of` → `blend_linear` → 热门池，二维。

    下游（集成相加、落盘 npy、write_scores、picks、两策略）全部假设打分是二维，
    这里**不能**返回三维。

    ★ 池子放在 `blend_linear` **之后**：线性分量的仿射映射是在"当前有效的票"上重新定标
    的（均值/标准差），先筛池再定标会让线性分量按池内的尺度落位 —— 这与"打分只在池内
    排序"这个语义一致。反过来（先定标再筛池）会让池内打分的尺度取决于池外的分布。
    """
    return hot_filter(blend_linear(score_of(predict_heads(model, panel, days, device)),
                                   getattr(model, 'ridge', None), panel, days), prices, days)


def validation(model, panel, prices, days, device):
    """验证窗上的 val_wei = Σ_h v_h · RankIC_h（**打分**对 h 日标签的逐日截面秩相关）。

    ★ V3 换掉了 V2 的判据（`100 × 可执行 top5 的 5d 标签代理收益 + 1.0 × 加权 Pearson IC`）：
    那个收益项占判据约 87%，是全流程最吵的量，且实测五个头在评价窗上全部是"最后一轮"
    的秩相关优于它选出的 best 轮。下游策略消费**排序**，判据就用排序口径。

    关键：这里算的是**打分**（`score_of()`，与测试窗同一个函数、同一套逐日截面公式），
    不再是"第 h 个头对自己标签"。V2 的两张表因此同名不同物（valid 的 IC_1d 是 1d 头，
    test 的 IC_1d 是 5d 头），V3 的 valid 表与 test 表**逐列同源**。

    缺失处理也照抄测试窗（`analysis.py:top_portfolio_daily`）：RankIC 只用成对有效值，
    Pearson 按缺失补 0。只用于选 epoch / 早停，测试窗不参与任何选择。
    """
    heads = predict_heads(model, panel, days, device)
    score = hot_filter(blend_linear(score_of(heads), getattr(model, 'ridge', None), panel, days), prices, days)
    ic, rank = ({}, {})
    for h in (1, 3, 5, 10, 20):
        label = f'label_ret_{h}d'
        pe, sp = ([], [])
        for i, d in enumerate(days):
            y = panel.Y[label][d]
            p = score[i]
            known = np.isfinite(p)
            ok = known & np.isfinite(y)
            sp.append(corr(rankdata(p[ok]), rankdata(y[ok])))
            pe.append(corr(p[known], np.where(np.isfinite(y[known]), y[known], 0.0)))
        ic[label] = float(np.mean(pe)) if pe else 0.0
        rank[label] = float(np.mean(sp)) if sp else 0.0
    ic_term = sum((RECIPE['valid_ic_weights'][h] * rank[f'label_ret_{h}d'] for h in RECIPE['label_horizons']))
    criterion = RECIPE['valid_criterion']
    if criterion not in ('rankic', 'tradable_topn'):
        raise ValueError(f"未知 valid_criterion={criterion!r}；见 RECIPE['valid_criterion']")
    next_entry = prices.next_entry_ok()
    preds = score
    rets = []
    unknown_picks = 0
    for p, d in zip(preds, days):
        y = panel.Y[RECIPE['score_label']][d]
        can = np.flatnonzero(np.isfinite(p) & next_entry[d])
        picks = can[np.argsort(-p[can], kind='stable')[:RECIPE['valid_topn']]]
        miss = ~np.isfinite(y[picks])
        unknown_picks += int(miss.sum())
        rets.append(float(np.where(miss, 0.0, y[picks]).sum() / RECIPE['valid_topn']) if len(picks) else 0.0)
    ret = float(np.mean(rets))
    label5 = RECIPE['score_label']
    if label5 not in ic:
        label5 = f"label_ret_{max(RECIPE['label_horizons'])}d"
    primary = ret if criterion == 'tradable_topn' else RECIPE['valid_ic_scale'] * ic_term
    return dict(val_wei=primary, val_criterion=criterion, val_return_proxy=ret, val_ic_weighted=ic_term, val_unknown_label_picks=unknown_picks, Pearson_IC=ic[label5], RankIC=rank[label5], **{f'IC_{h}d': ic[f'label_ret_{h}d'] for h in (1, 3, 5, 10, 20)}, **{f'RankIC_{h}d': rank[f'label_ret_{h}d'] for h in (1, 3, 5, 10, 20)})


def train_fold(fold, device='auto', panel=None, prices=None, quarter=None):
    """训练一个「季度 × 折」模型，打分落盘；可断点续跑。

    顺序：已完成则跳过 → （必要时从 last.pt 恢复）逐轮训练，每轮用验证集
    算 val_wei 并据此选最佳权重 / 调学习率 / 早停 → 用最佳权重对测试窗打分落盘。
    """
    recipe = copy.deepcopy(RECIPE)
    recipe['device'] = device
    quarter = quarter or quarters(axis()[0])[0]
    recipe['quarter'] = quarter
    info = training_info(recipe)

    out = RUN_ROOT / 'model_train' / quarter / f'fold{fold}'
    output_file(out / 'complete.json')
    out.mkdir(parents=True, exist_ok=True)
    done = out / 'complete.json'
    # --- 已完成且所需产物存在时直接复用 ---
    # 判据："`complete.json` 在、且三个产物文件齐"。
    #   代价（唯一要记住的一条运维事实）：**改了配置想重训，得自己删掉对应折目录**
    #   （`rm -rf model_train/<季度>/fold<折>`），否则会被当成"已完成"跳过、静默沿用旧配置的产物。
    #   配置本身仍有留痕：`training_info.json` 与 `complete.json` 里都记了整份 recipe。
    if done.exists():
        old = json.loads(done.read_text())
        if not all((out / name).is_file() for name in ('best.pt', 'test_predictions.npy', 'score_predictions.npy')):
            raise RuntimeError('完成记录存在但产物缺失')
        print(f'fold{fold} 已完成，跳过', flush=True)
        return old

    # --- 复现性：线程数、设备、种子、确定性算法 ---
    torch.set_num_threads(recipe['threads'])
    dev = device_for(device)
    seed = recipe['seed'] + fold - 1
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if dev.type == 'cuda':
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)     # 依赖 ① 的 CUBLAS_WORKSPACE_CONFIG

    # --- 数据与切分 ---
    panel = panel or Panel(recipe['features'])
    prices = prices or Prices(panel)
    split = splits(panel.days, recipe['purge_horizon'], recipe['folds'], quarter)[fold - 1]
    atomic_json(out / 'training_info.json', info)
    atomic_json(out / 'split.json', split_summary(panel.days, split))

    # --- 模型、优化器、学习率调度 ---
    model = PredictModel(len(panel.features), panel.market_dim).to(dev)
    # ★★ V9：岭回归分量在**选轮之前**拟合（它是训练窗上的闭式解，与网络无关），
    #   挂到 model 上 ⇒ `validation()` 每轮用的就是交付打分（网络 ⊕ 线性），
    #   选轮判据与实际交易的对象一致。
    model.ridge = fit_ridge(panel, split['train'], pool=prices.hot if recipe['train_pool_frac'] < 1. else None)
    print("fold%d linear component %s (training %d days)" % (fold, "disabled by recipe" if model.ridge is None else "fitted", len(split["train"])), flush=True)
    # ★ 优化器与调度器都从 RECIPE 取（原先这里是字面量，RECIPE 里的
    #   `optimizer` / `lr_patience` / `lr_factor` / `lr_cooldown` / `min_lr` **从未被读**，
    #   于是"改了 RECIPE 不生效"、而记录里显示的又是 RECIPE 那份值 —— 口径与记录不一致）。
    if recipe['optimizer'] != 'adamw':
        raise ValueError(f"暂只实现 adamw，收到 {recipe['optimizer']!r}")
    opt = torch.optim.AdamW(model.parameters(), lr=recipe['lr'], weight_decay=recipe['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode='max', factor=recipe['lr_factor'], patience=recipe['lr_patience'],
        cooldown=recipe['lr_cooldown'], min_lr=recipe['min_lr'])

    # --- 断点恢复：模型 / 优化器 / 调度器 / 随机数状态整套回放 ---
    start = 0
    best = -float('inf')
    stale = 0
    history = []
    # ★ V13：val_wei 前 `bag_topk` 名的 (分数, 轮号, 权重快照)。权重放 CPU，模型只有 ~17 万参数
    #   ⇒ 五份不到 4 MB。随 `last.pt` 一起落盘，断点续跑时能接着攒（否则续跑会把前半程的
    #   前 k 名丢掉、bagging 退化成"后半程的 top-k"，而且不报错）。
    bank = []
    last = out / 'last.pt'
    if last.exists():
        ck = torch.load(last, map_location='cpu', weights_only=False)
        model.load_state_dict(ck['model'])
        opt.load_state_dict(ck['optimizer'])
        scheduler.load_state_dict(ck['scheduler'])
        for state in opt.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(dev)
        start = ck['epoch'] + 1
        best = ck['best']
        stale = ck['stale']
        history = ck['history']
        bank = ck.get('bank', [])       # ★ V13：旧的 last.pt 没有这个键，缺了就当空（不影响续跑）
        np.random.set_state(ck['numpy_rng'])
        random.setstate(ck['python_rng'])
        torch.set_rng_state(ck['torch_rng'])
        if dev.type == 'cuda':
            torch.cuda.set_rng_state_all(ck['cuda_rng'])
        print(f'fold{fold} 恢复 epoch {start}', flush=True)
    print(f'fold{fold} 设备 {dev}，训练 {len(split["train"])} / 验证 {len(split["valid"])} / 测试 {len(split["test"])} 天', flush=True)

    # --- 逐轮训练 ---
    t0 = time.time()
    for epoch in range(start, recipe['max_epochs']):
        if stale >= recipe['early_stop_patience']:
            break
        model.train()
        order = np.random.permutation(split['train'])
        losses = []
        t = time.time()

        # 一个 batch = batch_days 个完整日截面；损失逐日算完再平均。
        for pos in range(0, len(order), recipe['batch_days']):
            opt.zero_grad(set_to_none=True)
            batch = []
            for d in order[pos:pos + recipe['batch_days']]:
                # 掩码只看覆盖率：各 horizon 的标签缺失率不同（20d 在数据尾部缺得更多），
                # 标签的取舍放到每个头各自的子集里做。
                mask = panel.mask(d)
                # ★ V17：池内训练 —— 样本掩码再与当日热门池取交。
                if recipe['train_pool_frac'] < 1.:
                    mask = mask & prices.hot[d]
                if mask.sum() < 3:      # BatchNorm 训练态要求每通道多于 1 个样本，先挡住
                    continue
                # 先确认当日至少有一个头有效，再前向 —— 否则这次前向白做，且归一化分母为 0。
                targets = {}
                for h in recipe['label_horizons']:
                    y = panel.Y[f'label_ret_{h}d'][d, mask]
                    ok = np.isfinite(y)
                    if ok.sum() < 3 or np.std(y[ok]) < 1e-8:
                        continue
                    targets[h] = (ok, y[ok])
                if not targets:
                    continue
                x = torch.from_numpy(panel.inputs(d, mask)).to(dev)
                m = torch.from_numpy(panel.M[d]).to(dev) if panel.market_dim else None
                pred = model(x, m)
                terms, weights = [], []
                for h, (ok, y) in targets.items():
                    j = recipe['label_horizons'].index(h)      # 列序的唯一出处，不靠 dict 顺序
                    yt = torch.from_numpy(y).to(dev)
                    # 仅当天标准化标签；常数截面跳过，缺失标签从不填零。
                    yt = (yt - yt.mean()) / yt.std()
                    rows = torch.from_numpy(np.flatnonzero(ok)).to(dev)
                    terms.append(recipe['label_weights'][h] * wpcc(pred[rows, j], yt))
                    weights.append(recipe['label_weights'][h])
                batch.append(torch.stack(terms).sum() / sum(weights))
            if not batch:
                continue
            loss = torch.stack(batch).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError('WPCC 非有限')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.)
            opt.step()
            losses.append(float(loss.detach()))

        # --- 每轮收尾：验证选轮 → 调学习率 → 落盘 best / last / history ---
        metrics = validation(model, panel, prices, split['valid'], dev)
        score = metrics['val_wei']
        scheduler.step(score)
        improved = score > best + 1e-10
        if improved:
            best = score
            stale = 0
            save_torch(out / 'best.pt', dict(model=model.state_dict(), input_dim=len(panel.features),
                                            market_dim=panel.market_dim, horizons=list(model.horizons),
                                            features=panel.features, epoch=epoch, metrics=metrics,
                                            ridge=getattr(model, 'ridge', None)))
        else:
            stale += 1
        # ★ V13：无论这一轮有没有刷新 best 都要进候选池 —— bagging 取的是**前 k 名**，
        #   不是"每一段的最好"，漏掉并列的第二、三名会让平均只在单点上做，失去降噪的意义。
        if recipe['bag_topk']:
            bank.append((score, epoch,
                         {k: v.detach().to('cpu', copy=True) for k, v in model.state_dict().items()}))
            bank.sort(key=lambda item: -item[0])
            del bank[recipe['bag_topk']:]
        record = dict(epoch=epoch, train_loss=float(np.mean(losses)), lr=opt.param_groups[0]['lr'],
                      seconds=round(time.time() - t, 2), **metrics)
        history.append(record)
        atomic_json(out / 'history.json', history)
        save_torch(last, dict(model=model.state_dict(), optimizer=opt.state_dict(), scheduler=scheduler.state_dict(),
                              epoch=epoch, best=best, stale=stale, history=history, bank=bank,
                              numpy_rng=np.random.get_state(), python_rng=random.getstate(),
                              torch_rng=torch.get_rng_state(),
                              cuda_rng=torch.cuda.get_rng_state_all() if dev.type == 'cuda' else []))
        # 进度与 ETA：按本进程已完成轮次的平均耗时估算；早停会提前结束，所以是上界。
        elapsed = time.time() - t0
        done_n = epoch - start + 1
        eta = elapsed / done_n * (recipe['max_epochs'] - done_n)
        print(f'fold{fold} epoch {epoch+1}/{recipe["max_epochs"]}: loss={record["train_loss"]:.5f} '
              f'val_wei={score:.5f} (ret={metrics["val_return_proxy"]:.5f} '
              f'ICw={metrics["val_ic_weighted"]:.5f}) '
              f'IC in recipe horizon order=' + '/'.join(f'{metrics[f"IC_{h}d"]:.4f}' for h in recipe['label_horizons'])
              + f' lr={record["lr"]:.6g} {record["seconds"]:.1f}s best={improved} | '
              f'{done_n}/{recipe["max_epochs"]} 轮，已用 {elapsed / 60:.1f}m，'
              f'ETA≤{eta / 60:.1f}m（早停提前）', flush=True)

    # --- ★ V13 收尾：交付权重 = val_wei 前 k 名的平均（epoch-bagging），不再是 argmax ---
    ck = torch.load(out / 'best.pt', map_location=dev, weights_only=False)
    argmax_epoch, argmax_metrics = ck['epoch'], ck['metrics']
    if recipe['bag_topk'] and len(bank) > 1:
        model.load_state_dict(average_state_dicts([state for _, _, state in bank]))
        if recipe['bag_recalibrate']:
            recalibrate_bn(model, panel, split['train'], dev)
        # 平均后的权重必须**重新过一遍选轮判据**：交付的是它，valid 表的收益列也取自它。
        bagged_metrics = validation(model, panel, prices, split['valid'], dev)
        save_torch(out / 'best.pt', dict(model=model.state_dict(), input_dim=len(panel.features),
                                        market_dim=panel.market_dim, horizons=list(model.horizons),
                                        features=panel.features, epoch=argmax_epoch,
                                        metrics=bagged_metrics, ridge=ck.get('ridge'),
                                        # argmax 那一份原样留着 —— 「bagged − argmax」的配对
                                        # 比较要在同一批训练上做，丢了就只剩跨版本抽奖噪声。
                                        model_argmax=ck['model'], metrics_argmax=argmax_metrics))
    else:
        # bag_topk=0 或只有一轮进池：退回 argmax，与 V11 口径逐位相同。
        model.load_state_dict(ck['model'])
        bagged_metrics = argmax_metrics
    model.eval()

    pred = predict(model, panel, prices, split['score'], dev)
    save_npy(out / 'score_predictions.npy', pred)
    # 评价用的季度切片 = 推演窗的前缀（同一份结果切片，不额外前向）
    save_npy(out / 'test_predictions.npy', pred[:len(split['test'])])
    result = dict(quarter=quarter, fold=fold, best_epoch=argmax_epoch + 1, epochs=len(history),
                  test_dates=panel.days[split['test']].tolist(),
                  score_dates=panel.days[split['score']].tolist(),
                  best_validation=bagged_metrics,           # valid 表读它 ⇒ 报的是交付口径
                  argmax_validation=argmax_metrics,         # 同批训练的对照，配对比较用
                  bag=dict(topk=recipe['bag_topk'], used=len(bank),
                           epochs=[int(e) + 1 for _, e, _ in bank],
                           scores=[float(s) for s, _, _ in bank]),
                  peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2,
                  seconds=round(time.time() - t0, 2), device=str(dev))
    atomic_json(done, result)
    print(f'fold{fold} EXIT:0（bag {len(bank)}/{recipe["bag_topk"]} 份，'
          f'轮号 {result["bag"]["epochs"]}；valid val_wei argmax {argmax_metrics["val_wei"]:.5f} '
          f'→ bagged {bagged_metrics["val_wei"]:.5f}）', flush=True)
    return result

# =============================================================================
# ⑩ 命令行入口
#
# 分季度×折的调度在 run.py（`./run.py train`）；这里只做单折/自检/看切分。
# =============================================================================


def rescore(quarter_list, fold_list, device='auto', panel=None, prices=None):
    """只重推演、不重训：用已完成的 best.pt 按当前口径重算打分并落盘。

    口径改动（推演窗、切分）之后不必重训就能刷新产物 —— 模型本身没变。
    只按当前口径重推演：口径变了而权重没重训时不会报错，需要自己删折重训。

    ★ **`best_validation` 也一并按当前口径重算**（V3 补）：valid 表的来源就是它，
    而它里面装的是**打分**的指标。改 `RECIPE['score_blend']` 之后只刷新打分、不刷新它，
    valid 表就会继续印旧口径的数字，与新打分对不上——`analysis.py` 不会发现，
    产物自洽、不报错。重算一次验证窗约 1 分钟/折。
    """
    dev = device_for(device)
    torch.set_num_threads(RECIPE['threads'])
    panel = panel or Panel(RECIPE['features'])
    prices = prices or Prices(panel)
    for quarter in quarter_list:
        for fold in fold_list:
            out = RUN_ROOT / 'model_train' / quarter / f'fold{fold}'
            done_path = out / 'complete.json'
            if not done_path.exists():
                print(f'{quarter} fold{fold} 未完成，跳过', flush=True)
                continue
            done = json.loads(done_path.read_text())
            if not (out / 'best.pt').exists():
                raise RuntimeError(f'{quarter} fold{fold} 缺 best.pt')
            split = splits(panel.days, RECIPE['purge_horizon'], RECIPE['folds'], quarter)[fold - 1]
            ck = torch.load(out / 'best.pt', map_location=dev, weights_only=False)
            model = PredictModel(len(panel.features), panel.market_dim).to(dev)
            model.load_state_dict(ck['model'])
            model.ridge = ck.get('ridge')          # ★ V9：打分含线性分量，必须一并恢复
            pred = predict(model, panel, prices, split['score'], dev)
            save_npy(out / 'score_predictions.npy', pred)
            save_npy(out / 'test_predictions.npy', pred[:len(split['test'])])
            atomic_json(out / 'split.json', split_summary(panel.days, split))
            done['score_dates'] = panel.days[split['score']].tolist()
            done['test_dates'] = panel.days[split['test']].tolist()
            done['best_validation'] = validation(model, panel, prices, split['valid'], dev)
            atomic_json(done_path, done)
            print(f'{quarter} fold{fold} 重推演完成：推演窗 {pred.shape[0]} 天'
                  f'（其中评价 {len(split["test"])} 天），best_validation 已按当前口径重算',
                  flush=True)


def main():
    parser = ArgumentParser(description='V2：全因子 + 市场门控 + 多目标（1/3/5/10/20d）四折模型，单折训练（断点续跑）')
    parser.add_argument('--fold', type=int, choices=[1, 2, 3, 4], default=None, help='只跑指定折')
    parser.add_argument('--quarter', default=None, help='例如 2025Q3；默认全部评价季度')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--split', action='store_true', help='只打印各季度切分，不训练')
    parser.add_argument('--rescore', action='store_true', help='只重推演、不重训：用已有 best.pt 按当前口径刷新打分')
    parser.add_argument('--doctor', action='store_true', help='环境与数据自检')
    args = parser.parse_args()
    # 新单元的常驻产物由运行生成，所以起点允许缺失；多余路径仍然一律拒绝。
    validate_layout(allow_temporary=True, allow_missing=True)

    # --- 自检：只打印本单元自己的运行环境，不探测宿主（独立单元约束）---
    if args.doctor:
        print('Python:', sys.executable, 'PyTorch:', torch.__version__)
        print('Data:', data_root())
        print('CUDA:', torch.cuda.is_available(),
              torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')
        print('Device:', device_for(RECIPE['device']))
        return

    days, _ = axis()
    qs = [args.quarter] if args.quarter else quarters(days)
    if any(q not in quarters(days) for q in qs):
        raise ValueError('季度不在评价窗口内')

    # --- 看切分：不读特征面板 ---
    if args.split:
        for q in qs:
            for s in splits(days, quarter=q):
                print(json.dumps(split_summary(days, s), ensure_ascii=False))
        return

    # --- 只重推演：不训练，用已有 best.pt 按当前口径刷新打分与切分记录 ---
    if args.rescore:
        panel = Panel(RECIPE['features'])
        return rescore(qs, [args.fold] if args.fold else [1, 2, 3, 4], args.device, panel)

    # --- 训练：读一次面板，按季度 × 折依次训练 ---
    panel = Panel(RECIPE['features'])
    prices = Prices(panel)
    check_days = np.concatenate([splits(panel.days, quarter=q)[0]['test'] for q in qs])
    prices.verify_labels(panel, check_days)
    for q in qs:
        for fold in ([args.fold] if args.fold else range(1, 5)):
            train_fold(fold, args.device, panel, prices, q)


if __name__ == '__main__':
    main()
