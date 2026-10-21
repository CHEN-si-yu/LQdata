"""V1 · model。市场因子门控 + 多目标（1/3/5/10/20d）回归。约定见 docs/architecture.md。"""

# =============================================================================
# ① 环境前置 —— 必须排在 import torch 之前
#
# BLAS / OpenMP 的线程数只在库首次加载时读取一次，import torch 之后再设就无效，
# 所以这段必须放在文件最前面。线程数直接决定内存峰值（实测 306 特征 + 20 线程
# = 38 GiB/折），是内存纪律而不是性能偏好。
# =============================================================================
import os

# 启用确定性算法时 cuBLAS 必须用固定大小的工作区，不设这个变量 torch 会直接报错。
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
for _key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[_key] = os.environ.get('MX_THREADS', '6')

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
from argparse import ArgumentParser
from pathlib import Path

# 单元内自带路径帮助文件；代码和配置无需依赖其他实验或公共模块。
from run import (PROJECT_ROOT, run_root, analysis_input_root, feature_block, feature_columns,
                 market_columns, data_root as project_data_root, price_root, output_file, validate_layout)
RUN_ROOT = run_root()
INPUT_ROOT = analysis_input_root()


import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from scipy.stats import rankdata
from torch import nn

ROOT = Path(__file__).resolve().parent

# =============================================================================
# ③ 配方（RECIPE）
#
# 一轮训练的超参与口径集中配置，并记录在每折训练信息中。
RECIPE = dict(
    name='V1',
    architecture='MLP-4linear + market-gate + multi-head',
    label='label_ret_5d',
    # 多目标：五个 horizon 共用一个主干、各出一个头；打分只取 label_ret_5d 那一列。
    label_horizons=[1, 3, 5, 10, 20],
    score_label='label_ret_5d',
    # 损失权重（5d 为主）：5d=1.0，近端次之，远端最弱。
    label_weights={1: .3, 3: .5, 5: 1., 10: .4, 20: .2},
    # valid 判据里五个 horizon 的 IC 权重（归一）。
    valid_ic_weights={1: .1, 3: .15, 5: .4, 10: .2, 20: .15},
    valid_ic_scale=1.,
    # 市场门控：61 个市场因子的 _z252 列，逐日一行、广播到当日全截面。
    market_gate=True,
    market_block='market_factors',
    market_clip=10.,
    picks_topn=10,              # picks.md 每天列前几名（固定策略仍是 top5/top1，不受影响）
    features='sample_20' if feature_block() == 'fac_sample' else 'all',
    feature_block=feature_block(),
    folds=4,
    seed=3253,
    batch_days=4,
    optimizer='adamw',
    lr=.001,
    weight_decay=.02,
    max_epochs=int(os.environ.get('MX_EPOCHS', '2')),
    early_stop_patience=6,
    lr_patience=3,
    lr_factor=.5,
    lr_cooldown=2,
    min_lr=5e-6,
    min_feature_coverage=.2,
    device='auto',
    threads=int(os.environ.get('MX_THREADS', '2')),
    valid_money=100000,
    valid_topn=5,
    valid_metric=('100 * executable_top5_5d_mean_return + 1.0 * weighted_Pearson_IC_5horizons '
                  '(v={1:.1,3:.15,5:.4,10:.2,20:.15})'),
    preprocessing='daily_zscore_win1_99_fixed_missing_0',
    test_start='2025-07-01',
    test_end='2026-06-30',
    # 隔离带按最长 horizon 取：20d 标签伸到 T+21，6 日隔离会让训练目标覆盖验证窗。
    purge_horizon=max([1, 3, 5, 10, 20]),
    training_window='2018起扩展窗，每季度初更新；季度边界purge max(h)+1=21交易日',
)

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
    """MLP-4linear + 市场门控 + 多目标头：input_dim → 256 → 128 → 32 → len(horizons)。

    市场门控挂在 128 与 32 两处隐层之后（乘性、逐通道）；market_dim=0 时不建门控。
    主干被拆成 trunk1/trunk2 只是为了在中间插门控，层序与无门控版的 MLP-4linear
    完全一致（Dropout 的位置不变），所以关掉门控时输出与无门控模型逐位相同。
    """

    # market_dim 是必填位置参数：漏传要变成 TypeError，不能默认 0 把门控静默关掉
    # （nn.Linear(0,64) 是合法的，只会发一条 zero-element 警告）。
    def __init__(self, input_dim, market_dim, horizons=None):
        super().__init__()
        self.horizons = list(horizons or RECIPE['label_horizons'])   # 列序的唯一出处
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(.2),
        )
        self.trunk1 = nn.Sequential(nn.Linear(256, 128), nn.BatchNorm1d(128), nn.GELU())
        self.drop = nn.Dropout(.1)
        self.trunk2 = nn.Sequential(nn.Linear(128, 32), nn.BatchNorm1d(32), nn.GELU())
        self.output_layer = nn.Linear(32, len(self.horizons))
        self.gate = MarketGate(market_dim, (128, 32)) if market_dim else None
        self._initialize_weights()
        if self.gate is not None:
            self.gate.zero_heads()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv1d)):
                nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, tsdata, market=None):
        """market 是**单个交易日**的市场向量 (market_dim,)；同一交易日内全截面共用一个门。"""
        hidden = self.trunk1(self.input_layer(tsdata.float()))
        if self.gate is not None:
            if market is None:
                raise ValueError('建了市场门控就必须传入当日市场向量')
            g1, g2 = self.gate(market)
            hidden = hidden * g1
        hidden = self.trunk2(self.drop(hidden))
        if self.gate is not None:
            hidden = hidden * g2
        return self.output_layer(hidden)


def wpcc(preds, y):
    """按预测名次加权的 Pearson 相关（WPCC），保留参考脚本公式。

    权重依预测名次衰减；方差围绕普通均值而不是加权均值——不能偷偷换成另一种加权
    Pearson，只增加了常数截面的数值保护。每次输入必须是单个交易日的完整有效截面。
    """
    p = preds.reshape(-1)
    y = y.reshape(-1)
    n = p.numel()
    if n < 2:
        return p.sum() * 0
    order = torch.argsort(p, descending=True, stable=True)
    w = torch.empty_like(p)
    w[order] = torch.pow(.5, torch.arange(n, device=p.device, dtype=p.dtype) / (n - 1))
    w_sum = w.sum()
    cov = (p * y * w).sum() / w_sum - (p * w).sum() * (y * w).sum() / w_sum.square()
    vp = ((p - p.mean()).square() * w).sum() / w_sum
    vy = ((y - y.mean()).square() * w).sum() / w_sum
    return -cov / (vp * vy).clamp_min(1e-16).sqrt()

# =============================================================================
# =============================================================================


def data_root():
    root = project_data_root()
    if not (root / 'meta.json').is_file():
        raise FileNotFoundError(f'{root}/meta.json 不存在；传入trainingdata根目录或fac_sample子目录')
    return root


def metadata():
    return json.loads((data_root() / 'meta.json').read_text())


def training_info(recipe):
    """记录训练参数和实际输入列，供报告读取。"""
    return dict(recipe=recipe, feature_source=str(data_root()/recipe['feature_block']),
                feature_columns=feature_columns(metadata()),
                market_columns=market_columns(metadata()) if recipe['market_gate'] else [])


def fingerprint(recipe):
    """产物指纹：模型架构与训练口径。复用已产出的折之前必须比对。

    只看 recipe —— 特征块 + 特征集名字已经唯一决定输入维数，所以 model.py 与
    analysis.py 的复用判断可以用同一个函数，不必各自持有 Panel。缺这个字段
    （旧版产物）或对不上一律报错要求清理，不能静默复用另一版模型的权重和打分。
    """
    return dict(name=recipe['name'], architecture=recipe['architecture'],
                feature_block=recipe['feature_block'], feature_set=recipe['features'],
                label_horizons=list(recipe['label_horizons']), score_label=recipe['score_label'],
                label_weights={str(k): v for k, v in recipe['label_weights'].items()},
                market_gate=recipe['market_gate'], purge_horizon=recipe['purge_horizon'],
                preprocessing=recipe['preprocessing'])


def frame(root, kind, year, columns):
    """读一个数据块的某一年，统一主键类型并断言无重复。"""
    p = root / kind / f'year={year}' / 'data.parquet'
    df = pq.ParquetFile(p).read(columns=['trade_date', 'stock_code'] + list(columns)).to_pandas()
    df['trade_date'] = df['trade_date'].astype(str).str[:10]
    df['stock_code'] = df['stock_code'].astype(str)
    if df.duplicated(['trade_date', 'stock_code']).any():
        raise ValueError(f'{p}: 重复主键')
    return df.set_index(['trade_date', 'stock_code']).sort_index()


#: 价格的 7 列在模块① `stock_daily` 里，复权因子在 `stock_adj_factor` 里。
PRICE_FIELDS = ('open', 'high', 'low', 'close', 'pre_close', 'pct_chg', 'vol')
PRICE_COLUMNS = PRICE_FIELDS + ('adj_factor',)


def price_frame(root, year, index):
    """读模块① 原始价，按 `index`（当年的 (trade_date, stock_code) 乘积）对齐成 8 列帧。

    ★ 价格**不是因子侧产物**（因子侧只出股票因子 / 市场因子 / 标签），所以它不在
      `trainingdata/` 里，由本单元直接读模块① 的 `datadownload/data`（见 `run.price_root()`，
      2026-09-25 用户定：`prices` 块从训练数据层移出）。注意读取路径由 `PROJECT_ROOT`
      推导，换挂载点不用改代码。

    口径与原先训练数据层里的 `prices` 块**逐字相同**：`stock_daily` 出 7 列、
      `stock_adj_factor` 出 `adj_factor`，按 (trade_date, stock_code) 主键对齐后 `reindex`
      —— 缺格是 NaN，而消费端正是靠这个 NaN 判"停牌买不进"（不是靠它填数）。

    两条校验，对不上就直接报错、不静默错位：
      ① 原始日线的**交易日集合**必须与面板当年完全一致（实测 9 年逐日相同）；
      ② 主键不得重复。
    池内个别票在原始表里整年没有日线是**上游缺口**（实测 2018 有 1 只），它只会让该票
    整列 NaN ⇒ 判为不可交易，属于保守方向，不构错位。
    再往下的数值对拍由 `Prices.verify_labels()` 兜底：它用价格重算 5 个 horizon 的收益
    与 `trainingdata/target` 逐格比，口径不一致会直接 raise —— 这是这条外部依赖的安全网。
    """
    frames, days_seen = [], None
    for dataset, names in (('stock_daily', PRICE_FIELDS), ('stock_adj_factor', ('adj_factor',))):
        p = root / dataset / f'year={year}' / 'data.parquet'
        part = pq.ParquetFile(p).read(columns=['trade_date', 'stock_code', *names]).to_pandas()
        part['trade_date'] = part['trade_date'].astype(str).str[:10]
        part['stock_code'] = part['stock_code'].astype(str)
        if part.duplicated(['trade_date', 'stock_code']).any():
            raise ValueError(f'{p}: 重复主键')
        part = part.set_index(['trade_date', 'stock_code'])
        if dataset == 'stock_daily':
            days_seen = set(part.index.get_level_values(0))
        frames.append(part[list(names)])
    want = set(index.get_level_values(0))
    if days_seen != want:
        raise ValueError(
            f'{year} 原始价日线轴与 factors 不一致（缺 {sorted(want - days_seen)[:3]} / '
            f'多 {sorted(days_seen - want)[:3]}）—— 回测会错位')
    return frames[0].join(frames[1], how='outer').reindex(index)


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

    `purge_horizon` 必须取**最长**标签期（V1 是 20d）：标签 T 使用 T+1+h 开盘，按 6 日
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
        for yi, y in enumerate(years, 1):
            rows = np.flatnonzero(np.char.startswith(self.days, str(y)))
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
                self.X[rows] = np.nan_to_num(x, nan=0.)
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

        if load_x and not np.isfinite(self.X).all():
            raise ValueError('特征含无穷值')
        if not np.isfinite(self.M).all():
            raise ValueError('市场块含无穷值')
        # 读盘是整条流水线里最长的静默段（约 1 分钟），逐行给进度与 ETA。
        print(f'面板读取完成：{len(self.meta["built_years"])} 年，市场列 {self.market_dim}，'
              f'已用 {time.time() - t_load:.0f}s', flush=True)

    def mask(self, day, label=None):
        """当日可用的股票掩码：覆盖率达标，且（给定标签时）标签已知。"""
        out = self.coverage[day] >= .2
        if label is not None:
            out = out & np.isfinite(self.Y[label][day])
        return out

# =============================================================================
# ⑧ 价格与可交易性
# =============================================================================


class Prices:
    """原始价 + 停牌/限价的可交易性判定。

    只前向填充状态量（复权因子、收盘价）；停牌无开盘价仍不可交易，不前填成交量。
    """

    def __init__(self, panel):
        names = list(PRICE_COLUMNS)
        T, C = len(panel.days), len(panel.codes)
        self.raw = {n: np.full((T, C), np.nan, np.float64) for n in names}

        # 逐年读模块① 原始价（不在 trainingdata 里，见 price_frame）；日期轴必须与 factors 一致。
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
        traded = (np.isfinite(self.open) & (self.open > 0) & np.isfinite(pre) & (pre > 0)
                  & np.isfinite(self.adj) & (self.adj > 0) & (self.raw['vol'] > 0))
        self.entry = traded & (self.open < upper - 1e-8)
        self.exit = traded & (self.open > lower + 1e-8)

        # next_entry[d] = 第 d 日开盘能不能买进「T-1 日选出的票」。
        self.next_entry = np.zeros_like(self.entry)
        self.next_entry[:-1] = self.entry[1:]

        # 收盘卖的可执行性：判据与开盘卖同法，但用当日**收盘价**判跌停 —— 收盘单就在那个
        # 时点成交，不含任何超过当日收盘的信息（不是"用当天行情挑成交"）。停牌/无量同样拒单。
        raw_close = self.raw['close']
        self.exit_close = (np.isfinite(raw_close) & (raw_close > 0) & np.isfinite(pre) & (pre > 0)
                           & np.isfinite(self.adj) & (self.adj > 0) & (self.raw['vol'] > 0)
                           & (raw_close > lower + 1e-8))

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
            x = torch.from_numpy(panel.X[d, mask]).to(device)
            m = torch.from_numpy(panel.M[d]).to(device) if panel.market_dim else None
            out[i, mask] = model(x, m).cpu().numpy()
    return out


def predict(model, panel, days, device):
    """打分：只取 RECIPE['score_label'] 那一列，返回二维 (天数, 股票数)。

    下游（集成相加、落盘 npy、write_scores、picks、两策略）全部假设打分是二维，
    这里**不能**返回三维。
    """
    return predict_heads(model, panel, days, device)[:, :, head_of(RECIPE['score_label'])]


def validation(model, panel, prices, days, device):
    """验证窗上的 val_wei = 100 × 可执行 top5 的 5d 标签代理收益 + 1.0 × Σ_h v_h·Pearson_IC_h。

    IC_h 是**第 h 个输出头**对自己那个 horizon 标签的逐日 Pearson IC 日平均；五项按
    RECIPE['valid_ic_weights'] 加权（5d 为主）。只用于选 epoch / 早停，测试窗不参与任何选择。
    """
    heads = predict_heads(model, panel, days, device)
    ics = {}
    ranks = {}
    for j, h in enumerate(RECIPE['label_horizons']):
        label = f'label_ret_{h}d'
        pair = []
        for i, d in enumerate(days):
            y = panel.Y[label][d]
            p = heads[i, :, j]
            ok = np.isfinite(p) & np.isfinite(y)
            pair.append((p[ok], y[ok]))
        ic = float(np.mean([corr(p, y) for p, y in pair])) if pair else 0.
        rank = float(np.mean([corr(rankdata(p), rankdata(y)) for p, y in pair])) if pair else 0.
        ics[label] = ic
        ranks[label] = rank
    label = RECIPE['score_label']
    ic_term = sum(RECIPE['valid_ic_weights'][h] * ics[f'label_ret_{h}d'] for h in RECIPE['label_horizons'])
    preds = heads[:, :, head_of(label)]
    rets = []
    unknown_picks = 0
    for p, d in zip(preds, days):
        y = panel.Y[label][d]
        # 验证收益仅是选 epoch 的 5d 标签代理；不是现金策略净值。
        can = np.flatnonzero(np.isfinite(p) & prices.next_entry[d])
        picks = can[np.argsort(-p[can], kind='stable')[:RECIPE['valid_topn']]]
        # 选中的票若 5d 标签未知（窗口内停牌等），该票按 0 收益计 —— 与 analysis.py 四指标
        # 「标签缺失按 0 计」同一口径。**不换成下一名**：用标签可得性换票才是事后挑选。
        # （当前快照 2018 年约 3% 的票 5d 标签缺失，验证窗内平均约 1.3% 的天数会命中。）
        miss = ~np.isfinite(y[picks])
        unknown_picks += int(miss.sum())
        rets.append(float(np.where(miss, 0., y[picks]).sum() / RECIPE['valid_topn']) if len(picks) else 0.)
    ret = float(np.mean(rets))
    label5 = RECIPE['score_label']
    return dict(val_wei=100 * ret + RECIPE['valid_ic_scale'] * ic_term,
                val_return_proxy=ret, val_ic_weighted=ic_term,
                val_unknown_label_picks=unknown_picks,
                Pearson_IC=ics[label5], RankIC=ranks[label5],
                **{f'IC_{h}d': ics[f'label_ret_{h}d'] for h in RECIPE['label_horizons']},
                **{f'RankIC_{h}d': ranks[f'label_ret_{h}d'] for h in RECIPE['label_horizons']})


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
    fp = fingerprint(recipe)
    # --- 已完成且所需产物存在时直接复用（口径必须一致）---
    if done.exists():
        old = json.loads(done.read_text())
        if not all((out / name).is_file() for name in ('best.pt', 'test_predictions.npy', 'score_predictions.npy')):
            raise RuntimeError('完成记录存在但产物缺失')
        if old.get('fingerprint') != fp:
            raise RuntimeError(f'{quarter} fold{fold} 的产物属于另一个口径（指纹不符），'
                               f'不能静默复用：已有 {old.get("fingerprint")}；本次 {fp}')
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
    opt = torch.optim.AdamW(model.parameters(), lr=recipe['lr'], weight_decay=recipe['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', factor=.5, patience=3, cooldown=2, min_lr=5e-6)

    # --- 断点恢复：模型 / 优化器 / 调度器 / 随机数状态整套回放 ---
    start = 0
    best = -float('inf')
    stale = 0
    history = []
    last = out / 'last.pt'
    if last.exists():
        ck = torch.load(last, map_location='cpu', weights_only=False)
        if ck.get('fingerprint') != fp:
            raise RuntimeError(f'{quarter} fold{fold} 的断点属于另一个口径（指纹不符），'
                               '删除 last.pt 后重训，不能接着旧模型的优化器状态跑')
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
                x = torch.from_numpy(panel.X[d, mask]).to(dev)
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
                                            fingerprint=fp))
        else:
            stale += 1
        record = dict(epoch=epoch, train_loss=float(np.mean(losses)), lr=opt.param_groups[0]['lr'],
                      seconds=round(time.time() - t, 2), **metrics)
        history.append(record)
        atomic_json(out / 'history.json', history)
        save_torch(last, dict(model=model.state_dict(), optimizer=opt.state_dict(), scheduler=scheduler.state_dict(),
                              epoch=epoch, best=best, stale=stale, history=history, fingerprint=fp,
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
              f'IC1/3/5/10/20=' + '/'.join(f'{metrics[f"IC_{h}d"]:.4f}' for h in recipe['label_horizons'])
              + f' lr={record["lr"]:.6g} {record["seconds"]:.1f}s best={improved} | '
              f'{done_n}/{recipe["max_epochs"]} 轮，已用 {elapsed / 60:.1f}m，'
              f'ETA≤{eta / 60:.1f}m（早停提前）', flush=True)

    # --- 收尾：最佳权重重载 → 推演整段（季度首日 → 数据末日）→ 落盘 ---
    ck = torch.load(out / 'best.pt', map_location=dev, weights_only=False)
    model.load_state_dict(ck['model'])
    pred = predict(model, panel, split['score'], dev)
    save_npy(out / 'score_predictions.npy', pred)
    # 评价用的季度切片 = 推演窗的前缀（同一份结果切片，不额外前向）
    save_npy(out / 'test_predictions.npy', pred[:len(split['test'])])
    result = dict(quarter=quarter, fold=fold, best_epoch=ck['epoch'] + 1, epochs=len(history),
                  test_dates=panel.days[split['test']].tolist(),
                  score_dates=panel.days[split['score']].tolist(), best_validation=ck['metrics'],
                  fingerprint=fp,
                  peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2,
                  seconds=round(time.time() - t0, 2), device=str(dev))
    atomic_json(done, result)
    print(f'fold{fold} EXIT:0', flush=True)
    return result

# =============================================================================
# ⑩ 命令行入口
#
# 跨折并发调度在 analysis.py（train.sh 走的就是那条）；这里只做单折/自检/看切分。
# =============================================================================


def rescore(quarter_list, fold_list, device='auto', panel=None):
    """只重推演、不重训：用已完成的 best.pt 按当前口径重算打分并落盘。

    口径改动（推演窗、切分）之后不必重训就能刷新产物 —— 模型本身没变。
    只是「模型本身没变」由指纹兜底：口径变了而权重没重训，直接报错而不是刷新。
    """
    dev = device_for(device)
    torch.set_num_threads(RECIPE['threads'])
    panel = panel or Panel(RECIPE['features'])
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
            if done.get('fingerprint') != fingerprint(RECIPE):
                raise RuntimeError(f'{quarter} fold{fold} 的产物属于另一个口径（指纹不符），不能按当前口径重推演')
            split = splits(panel.days, RECIPE['purge_horizon'], RECIPE['folds'], quarter)[fold - 1]
            ck = torch.load(out / 'best.pt', map_location=dev, weights_only=False)
            model = PredictModel(len(panel.features), panel.market_dim).to(dev)
            model.load_state_dict(ck['model'])
            pred = predict(model, panel, split['score'], dev)
            save_npy(out / 'score_predictions.npy', pred)
            save_npy(out / 'test_predictions.npy', pred[:len(split['test'])])
            atomic_json(out / 'split.json', split_summary(panel.days, split))
            done['score_dates'] = panel.days[split['score']].tolist()
            done['test_dates'] = panel.days[split['test']].tolist()
            atomic_json(done_path, done)
            print(f'{quarter} fold{fold} 重推演完成：推演窗 {pred.shape[0]} 天'
                  f'（其中评价 {len(split["test"])} 天）', flush=True)


def main():
    parser = ArgumentParser(description='V1：市场门控 + 多目标（1/3/5/10/20d）四折模型，单折训练（断点续跑）')
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
