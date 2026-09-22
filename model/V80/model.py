"""V77：新快照(trainingdata, 599 列)上的「全量」臂。

配方与 V62/V32 逐字相同，唯一差别是特征清单（features_armB_all599.json，599 列）。
数据快照由 SNAPSHOT 常量锁定，不随 MX_DATA 漂移。"""

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
import hashlib
import json
import random
import resource
import sys
import time
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from scipy.stats import rankdata
from torch import nn

SNAPSHOT_DIR = 'trainingdata'

ROOT = Path(__file__).resolve().parent

# ★ 本版锁定的数据快照：新因子快照（653 列）。写成 setdefault 而不是读取 ——
#   同一个目录会被 V62/V63 两个臂共用，靠"记得设环境变量"太脆（忘设就会静默换数据）。
os.environ.setdefault('MX_DATA', str(ROOT.parent / SNAPSHOT_DIR))

# =============================================================================
# ③ 配方（RECIPE）
#
# 一轮训练的全部超参与口径，集中写死在这里。provenance() 会把它算进 run_id，
# 所以改动这里等于换了模型来源，已有产物会被判为指纹不一致。
# =============================================================================
VARIANT = '新快照(trainingdata)上的「全量」臂：全部 599 列。其余配方与 V62/V32 逐字相同，只换输入列 ⇒ 与 V76 构成配对对照。'

# ★ 特征清单从文件读（= V32 快照的 columns.features，顺序也照它）⇒ 与 V32 的面板逐列同序。
#   刻意不用 meta['columns']['features']：那是新快照的 653 列，会把对照臂变成处置臂。
FEATURES_OLD = json.loads((Path(__file__).resolve().parent / 'features_armB_all599.json').read_text())

RECIPE = dict(
    name='V80',
    time_half_life=0,
    runtime_policy='torch-cu128-explicit-device-v1',
    epoch_agg='mean_all',
    architecture='MLP-4linear',
    label='label_ret_5d',
    features=FEATURES_OLD,
    folds=4,
    seed=3254,
    batch_days=4,
    optimizer='adamw',
    lr=.001,
    weight_decay=.02,
    max_epochs=30,
    early_stop_patience=6,
    lr_patience=3,
    lr_factor=.5,
    lr_cooldown=2,
    min_lr=5e-6,
    min_feature_coverage=.2,
    device='auto',
    threads=3,
    valid_money=100000,
    snapshot=SNAPSHOT_DIR,
    valid_topn=5,
    valid_metric='不选轮：打分取全部轮次预测的平均（epoch_agg=mean_all）；val_wei 只用来决定何时早停',
    preprocessing='daily_rank_centered_fixed_missing_0.5',
    test_start='2025-07-01',
    test_end='2026-06-30',
    training_window='2018起扩展窗，每季度初更新；季度边界purge h+1交易日',
)

# =============================================================================
# ④ 网络与损失（来自用户 MLP.py 参考脚本）
# =============================================================================


class PredictModel(nn.Module):
    """MLP-4linear：337 → 256 → 128 → 32 → 1；维数由快照特征列表决定，避免依赖旧机器的全局变量。"""

    def __init__(self, input_dim):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(.2),
        )
        self.net = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(.1),
            nn.Linear(128, 32),
            nn.BatchNorm1d(32),
            nn.GELU(),
        )
        self.output_layer = nn.Linear(32, 1)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv1d)):
                nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, tsdata):
        return self.output_layer(self.net(self.input_layer(tsdata.float())))


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
# ⑤ 数据路径、来源指纹与小工具
# =============================================================================


def data_root():
    """按 MX_DATA > V16/trainingdata > ../trainingdata 的顺序找数据根，以 meta.json 是否存在判断。"""
    explicit = os.environ.get('MX_DATA')
    candidates = [Path(explicit)] if explicit else [ROOT / 'trainingdata', ROOT.parent / 'trainingdata']
    for p in candidates:
        if (p / 'meta.json').is_file():
            return p.resolve()
    raise FileNotFoundError('找不到 trainingdata/meta.json，请设置 MX_DATA')


def metadata():
    return json.loads((data_root() / 'meta.json').read_text())


def provenance(recipe):
    """模型来源指纹：配方 + 全部数据块的 SHA。

    只对「配方 + 数据」取指纹，不哈希源码字节，因此整理/重排代码不会让已有产物失效；
    改了配方或重建了数据则会算出不同的 run_id，指纹不符即拒绝恢复。
    """
    meta = metadata()
    # panel_digest 没包含 target 内容；额外收录每个块的 SHA，防止标签变更沿用旧模型。
    files = {str(y): meta['years'][str(y)]['files'] for y in meta['built_years']}
    value = dict(recipe=recipe, panel_digest=meta['panel_digest'], files=files,
                 model_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                 runtime=dict(torch=torch.__version__, cuda=torch.version.cuda, numpy=np.__version__, python=sys.version.split()[0]))
    value['run_id'] = hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()[:20]
    return value


def frame(root, kind, year, columns):
    """读一个数据块的某一年，统一主键类型并断言无重复。"""
    p = root / kind / f'year={year}' / 'data.parquet'
    df = pq.ParquetFile(p).read(columns=['trade_date', 'stock_code'] + list(columns)).to_pandas()
    df['trade_date'] = df['trade_date'].astype(str).str[:10]
    df['stock_code'] = df['stock_code'].astype(str)
    if df.duplicated(['trade_date', 'stock_code']).any():
        raise ValueError(f'{p}: 重复主键')
    return df.set_index(['trade_date', 'stock_code']).sort_index()


def atomic_json(path, value):
    """先写 .tmp 再 os.replace，保证读到的永远是完整 JSON。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(temp, path)


def save_torch(path, value):
    """同上，torch 检查点的原子写。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    torch.save(value, tmp)
    os.replace(tmp, path)


def save_npy(path, value):
    """同上，npy 打分的原子写。"""
    path = Path(path)
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
        d = pq.ParquetFile(root / 'factors' / f'year={y}' / 'data.parquet').read(columns=['trade_date'])
        days.extend(sorted(set(d.column(0).to_pylist())))
    return np.asarray(days), np.asarray(meta['axis']['codes'])


def quarters(days):
    """评价窗口固定为 2025-07-01 ~ 2026-06-30（用户指定，V16 及以后版本都遵守）。

    入参 days 只为保持调用签名一致，不影响返回值——不能自行扩展或缩成部分季度。
    """
    return ['2025Q3', '2025Q4', '2026Q1', '2026Q2']


def label_horizon(label):
    """标签名 → 期限（`label_ret_5d` → 5）。

    purge/embargo 必须按**实际训的那个标签**算：换 20d 标签时若仍退 5 日，
    训练样本的标签就会伸进测试窗 —— 那是硬约束（SPEC.md §2 铁律一）里唯一不能破的一条。
    """
    return int(label.rsplit('_', 1)[-1].rstrip('d'))


def splits(days, horizon=5, folds=4, quarter=None):
    """一个季度的四折切分：训练/验证块在季度前 h+1 日隔离，测试窗严格样本外。"""
    # 标签 T 使用 T+1+h 开盘；严格早于验证/测试的边界，不共享端点。
    quarter = quarter or quarters(days)[0]
    q = pd.Period(quarter, freq='Q')
    start = str(q.start_time.date())
    end = str(q.end_time.date())
    ix = np.arange(len(days))
    first = int(np.searchsorted(days, start))
    cutoff = first - horizon - 1
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
        valid = block[block + horizon + 1 < b]
        train = eligible[(eligible + horizon + 1 < a) | (eligible >= b + horizon + 1)]
        if not len(train) or not len(valid) or not len(test) or not len(score):
            raise ValueError('训练/验证/测试/推演窗为空')
        assert max(train.max(), valid.max()) + horizon + 1 < first
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
    """全历史面板，按 (交易日 × 股票) 的二维网格存放。

    load_x=False 时只读标签与成交额（评价阶段用），省掉最大的一块内存。
    """

    def __init__(self, features='all', load_x=True):
        self.root = data_root()
        self.meta = metadata()
        self.days, self.codes = axis()
        self.features = (list(features) if isinstance(features, (list, tuple)) else
                         self.meta['columns']['features'] if features == 'all' else self.meta['feature_sets'][features])
        self.labels = self.meta['columns']['labels']

        T, C = len(self.days), len(self.codes)
        self.X = np.empty((T, C, len(self.features)), np.float32) if load_x else None
        self.coverage = np.empty((T, C), np.float32) if load_x else None
        self.Y = {name: np.full((T, C), np.nan, np.float32) for name in self.labels}
        self.amount = np.full((T, C), np.nan, np.float64)

        # 逐年读取并写进预分配好的网格；每年都断言轴一致，避免静默错位。
        t_load = time.time()
        years = list(self.meta['built_years'])
        for yi, y in enumerate(years, 1):
            rows = np.flatnonzero(np.char.startswith(self.days, str(y)))
            index = pd.MultiIndex.from_product([self.days[rows], self.codes], names=['trade_date', 'stock_code'])
            if load_x:
                df = frame(self.root, 'factors', y, self.features)
                if len(df) != len(index) or not df.index.equals(index):
                    raise ValueError(f'{y} factors 轴不一致')
                x = df.to_numpy(np.float32).reshape(len(rows), C, -1)
                self.coverage[rows] = np.isfinite(x).mean(axis=2)
                # 输入已经是当日截面 rank；固定中心 0.5，缺失中性填充，无全样本拟合。
                self.X[rows] = (np.nan_to_num(x, nan=.5) - .5) * 2
                del df, x
            df = frame(self.root, 'target', y, self.labels).reindex(index)
            for name in self.labels:
                self.Y[name][rows] = df[name].to_numpy(np.float32).reshape(len(rows), C)
            self.amount[rows] = frame(self.root, 'amount', y, ['amount']).reindex(index)['amount'].to_numpy().reshape(len(rows), C)
            print(f'已读取 {y} 年，{len(rows)} 个交易日'
                  f'（{yi}/{len(years)}，已用 {time.time() - t_load:.0f}s，'
                  f'预计还需 {(time.time() - t_load) / yi * (len(years) - yi):.0f}s）', flush=True)

        if load_x and not np.isfinite(self.X).all():
            raise ValueError('特征含无穷值')
        # 读盘是整条流水线里最长的静默段（约 1 分钟），逐行给进度与 ETA。
        print(f'面板读取完成：{len(self.meta["built_years"])} 年，'
              f'已用 {time.time() - t_load:.0f}s', flush=True)

    def mask(self, day, label=None):
        """当日可用的股票掩码：覆盖率达标，且（给定标签时）标签已知。"""
        out = self.coverage[day] >= RECIPE['min_feature_coverage']
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
        names = ['open', 'high', 'low', 'close', 'pre_close', 'pct_chg', 'vol', 'adj_factor']
        T, C = len(panel.days), len(panel.codes)
        self.raw = {n: np.full((T, C), np.nan, np.float64) for n in names}

        # 逐年读原始价；轴必须与 factors 完全一致，否则回测会错位。
        for year in panel.meta['built_years']:
            ix = np.flatnonzero(np.char.startswith(panel.days, str(year)))
            index = pd.MultiIndex.from_product([panel.days[ix], panel.codes], names=['trade_date', 'stock_code'])
            df = frame(panel.root, 'prices', year, names)
            if not df.index.equals(index):
                raise ValueError(f'{year} prices 轴与 factors 不一致')
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
def predict(model, panel, days, device):
    """给定日期，逐日对当日有效截面打分；无效位置留 NaN。"""
    model.eval()
    out = np.full((len(days), len(panel.codes)), np.nan, np.float32)
    for i, d in enumerate(days):
        mask = panel.mask(d)
        if mask.sum():
            out[i, mask] = model(torch.from_numpy(panel.X[d, mask]).to(device)).squeeze(-1).cpu().numpy()
    return out


def validation(model, panel, prices, days, device, label):
    """验证窗上的 val_wei = 100 × 可执行 top5 的 5d 标签代理收益 + 0.7 × Pearson IC。

    只用于选 epoch / 早停，测试窗不参与任何选择。返回 (指标字典, 验证窗原始打分)——
    打分要留下来供离线换选轮规则（V17 新增）。
    """
    preds = predict(model, panel, days, device)
    ics = []
    ranks = []
    rets = []
    for p, d in zip(preds, days):
        y = panel.Y[label][d]
        ok = np.isfinite(p) & np.isfinite(y)
        ics.append(corr(p[ok], y[ok]))
        ranks.append(corr(rankdata(p[ok]), rankdata(y[ok])))
        # 验证收益仅是选 epoch 的 5d 标签代理；不是现金策略净值。
        can = np.flatnonzero(np.isfinite(p) & prices.next_entry[d])
        picks = can[np.argsort(-p[can], kind='stable')[:RECIPE['valid_topn']]]
        if len(picks) and not np.isfinite(y[picks]).all():
            raise ValueError('验证选股存在未知标签，不能事后换票')
        rets.append(float(y[picks].sum() / RECIPE['valid_topn']) if len(picks) else 0.)
    ret = float(np.mean(rets))
    ic = float(np.mean(ics))
    return dict(val_wei=100 * ret + .7 * ic, val_return_proxy=ret, Pearson_IC=ic,
                RankIC=float(np.mean(ranks))), preds


def train_fold(fold, device='auto', panel=None, prices=None, quarter=None):
    """训练一个「季度 × 折」模型，打分落盘；可断点续跑。

    顺序：算指纹 → 已完成则跳过 → （必要时从 last.pt 恢复）逐轮训练，每轮用验证集
    算 val_wei 并据此选最佳权重 / 调学习率 / 早停 → 按本版全轮平均规则对测试窗打分落盘。
    """
    recipe = copy.deepcopy(RECIPE)
    device = str(device_for(device))
    recipe['device'] = device
    quarter = quarter or quarters(axis()[0])[0]
    recipe['quarter'] = quarter
    stamp = provenance(recipe)
    rid = stamp['run_id']

    out = ROOT / 'model_train' / quarter / f'fold{fold}'
    out.mkdir(parents=True, exist_ok=True)
    done = out / 'complete.json'

    # --- 已完成即跳过：指纹一致才允许跳过，产物缺失则视为未完成 ---
    if done.exists():
        old = json.loads(done.read_text())
        if old['run_id'] != rid:
            raise RuntimeError('已有模型与配方/数据不一致；请归档产物后重跑')
        if not (out / 'best.pt').exists() or not (out / 'test_predictions.npy').exists():
            raise RuntimeError('完成记录存在但产物缺失')
        print(f'fold{fold} 已完成且指纹匹配，跳过', flush=True)
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
    split = splits(panel.days, label_horizon(recipe['label']), recipe['folds'], quarter)[fold - 1]
    atomic_json(out / 'recipe.lock.json', stamp)
    atomic_json(out / 'split.json', split_summary(panel.days, split))

    # --- 模型、优化器、学习率调度 ---
    model = PredictModel(len(panel.features)).to(dev)
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
        if ck['run_id'] != rid:
            raise RuntimeError('断点指纹不一致，禁止混用')
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

    # --- 逐轮留档：权重快照 + 验证窗原始打分 ---
    # 断点续跑时按已完成的轮次重建，超出的行丢掉（宁可少几行，也不让两份产物对不上）。
    epoch_folder = out / 'epochs'
    epoch_folder.mkdir(exist_ok=True)
    valid_file = out / 'valid_predictions_by_epoch.npy'
    epoch_preds = list(np.load(valid_file)) if valid_file.exists() else []
    epoch_preds = epoch_preds[:start]

    # --- 逐轮训练 ---
    t0 = time.time()
    time_weight_mean = (float(np.mean(2. ** (-(split['train'].max() - split['train']) / recipe['time_half_life'])))
                        if recipe['time_half_life'] else 1.)
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
                mask = panel.mask(d, recipe['label'])
                if mask.sum() < 3:
                    continue
                x = torch.from_numpy(panel.X[d, mask]).to(dev)
                y = torch.from_numpy(panel.Y[recipe['label']][d, mask]).to(dev)
                # 仅当天标准化标签；常数截面跳过，缺失标签从不填零。
                if y.std() < 1e-8:
                    continue
                y = (y - y.mean()) / y.std()
                day_loss = wpcc(model(x), y)
                if recipe['time_half_life']:
                    # 权重仅依赖本折训练截止日；测试收益不参与权重计算。
                    age = int(split['train'].max()) - int(d)
                    weight = 2. ** (-age / recipe['time_half_life'])
                    day_loss = day_loss * weight / time_weight_mean
                batch.append(day_loss)
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
        metrics, valid_pred = validation(model, panel, prices, split['valid'], dev, recipe['label'])
        score = metrics['val_wei']
        # 逐轮权重与验证打分：让"换选轮规则 / 选前 k 轮平均预测"能离线复算（V17 新增）
        save_torch(epoch_folder / f'epoch{epoch:02d}.pt',
                   dict(model=model.state_dict(), epoch=epoch, metrics=metrics, run_id=rid))
        epoch_preds.append(valid_pred)
        save_npy(valid_file, np.stack(epoch_preds))
        scheduler.step(score)
        improved = score > best + 1e-10
        if improved:
            best = score
            stale = 0
            save_torch(out / 'best.pt', dict(model=model.state_dict(), input_dim=len(panel.features),
                                            features=panel.features, run_id=rid, epoch=epoch, metrics=metrics))
        else:
            stale += 1
        record = dict(epoch=epoch, train_loss=float(np.mean(losses)), lr=opt.param_groups[0]['lr'],
                      seconds=round(time.time() - t, 2), **metrics)
        history.append(record)
        atomic_json(out / 'history.json', history)
        save_torch(last, dict(model=model.state_dict(), optimizer=opt.state_dict(), scheduler=scheduler.state_dict(),
                              epoch=epoch, best=best, stale=stale, history=history, run_id=rid,
                              numpy_rng=np.random.get_state(), python_rng=random.getstate(),
                              torch_rng=torch.get_rng_state(),
                              cuda_rng=torch.cuda.get_rng_state_all() if dev.type == 'cuda' else []))
        # 进度与 ETA：按本进程已完成轮次的平均耗时估算；早停会提前结束，所以是上界。
        elapsed = time.time() - t0
        done_n = epoch - start + 1
        eta = elapsed / done_n * (recipe['max_epochs'] - done_n)
        print(f'fold{fold} epoch {epoch+1}/{recipe["max_epochs"]}: loss={record["train_loss"]:.5f} '
              f'val_wei={score:.5f} RankIC={metrics["RankIC"]:.5f} lr={record["lr"]:.6g} '
              f'{record["seconds"]:.1f}s best={improved} | {done_n}/{recipe["max_epochs"]} 轮，'
              f'已用 {elapsed / 60:.1f}m，ETA≤{eta / 60:.1f}m（早停提前）', flush=True)

    # --- 收尾：推演整段（季度首日 → 数据末日）→ 落盘 ---
    #
    # ★ V22 的改动在这里：**不再"选一轮"，而是把全部轮次的预测取平均**（recipe['epoch_agg']）。
    # 理由是实测的：同一份训练里按 val_wei 选一轮 → Σtop1 −0.16，而全轮平均 → +1.19，
    # 四折配对 4/4 更好。机制上说得通 —— 打分是跨模型的共识时，头部那几只才稳（跨折集成
    # 的 top-1 也远好于任一单折，见 rank_profile.py）。`best.pt` 仍然保留，只为留档"哪一轮
    # 被 val_wei 选中"，不再参与打分。
    ck = torch.load(out / 'best.pt', map_location=dev, weights_only=False)
    chosen = ck['epoch'] + 1
    files = sorted(epoch_folder.glob('epoch*.pt'))
    if recipe['epoch_agg'] == 'mean_all' and files:
        acc = None
        for path in files:
            snapshot = torch.load(path, map_location=dev, weights_only=False)
            model.load_state_dict(snapshot['model'])
            p = predict(model, panel, split['score'], dev)
            # NaN 用求和传播：某票某轮没有分数时，平均结果仍应是"没有分数"，不能当成 0 分。
            acc = p if acc is None else acc + p
        pred = (acc / len(files)).astype(np.float32)
        agg_note = f'mean_all({len(files)}轮)'
    else:
        model.load_state_dict(ck['model'])
        pred = predict(model, panel, split['score'], dev)
        agg_note = f'best_epoch1({chosen})'
    save_npy(out / 'score_predictions.npy', pred)
    # 评价用的季度切片 = 推演窗的前缀（同一份结果切片，不额外前向）
    save_npy(out / 'test_predictions.npy', pred[:len(split['test'])])
    result = dict(run_id=rid, quarter=quarter, fold=fold, best_epoch=chosen,
                  epochs=len(history), epoch_agg=agg_note,
                  test_dates=panel.days[split['test']].tolist(),
                  score_dates=panel.days[split['score']].tolist(), best_validation=ck['metrics'],
                  peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2,
                  seconds=round(time.time() - t0, 2), device=str(dev))
    atomic_json(done, result)
    print(f'fold{fold} EXIT:0（打分口径 {agg_note}）', flush=True)
    return result

# =============================================================================
# ⑩ 命令行入口
#
# 跨折并发调度在 analysis.py（train.sh 走的就是那条）；这里只做单折/自检/看切分。
# =============================================================================


def rescore(quarter_list, fold_list, device='auto', panel=None):
    """只重推演、不重训：按当前口径重算打分并落盘。

    口径改动（推演窗、切分、**打分聚合方式**）之后不必重训就能刷新产物 —— 模型本身没变。
    推演前校验指纹，避免把不同配方/数据的预测混在一起。
    ★ 打分要按 `recipe['epoch_agg']` 重算：全轮平均与"取 best.pt"是两份不同的产物，
    漏掉这一处会静默地拿旧口径的打分冒充新口径（见 LESSONS.md §10 死胡同详录的同族教训）。
    """
    dev = device_for(device)
    torch.set_num_threads(RECIPE['threads'])
    panel = panel or Panel(RECIPE['features'])
    for quarter in quarter_list:
        for fold in fold_list:
            out = ROOT / 'model_train' / quarter / f'fold{fold}'
            done_path = out / 'complete.json'
            if not done_path.exists():
                print(f'{quarter} fold{fold} 未完成，跳过', flush=True)
                continue
            done = json.loads(done_path.read_text())
            lock = json.loads((out / 'recipe.lock.json').read_text())
            if provenance(lock['recipe'])['run_id'] != done['run_id']:
                raise RuntimeError(f'{quarter} fold{fold} 指纹不一致，禁止重推演')
            if not (out / 'best.pt').exists():
                raise RuntimeError(f'{quarter} fold{fold} 缺 best.pt')
            split = splits(panel.days, label_horizon(RECIPE['label']), RECIPE['folds'], quarter)[fold - 1]
            ck = torch.load(out / 'best.pt', map_location=dev, weights_only=False)
            model = PredictModel(len(panel.features)).to(dev)
            files = sorted((out / 'epochs').glob('epoch*.pt'))
            if RECIPE['epoch_agg'] == 'mean_all' and files:
                acc = None
                for path in files:
                    snapshot = torch.load(path, map_location=dev, weights_only=False)
                    model.load_state_dict(snapshot['model'])
                    p = predict(model, panel, split['score'], dev)
                    acc = p if acc is None else acc + p
                pred = (acc / len(files)).astype(np.float32)
            else:
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
    parser = ArgumentParser(description='V32：季度滚动四折 MLP 基线，单折训练（断点续跑）')
    parser.add_argument('--fold', type=int, choices=[1, 2, 3, 4], default=None, help='只跑指定折')
    parser.add_argument('--quarter', default=None, help='例如 2025Q3；默认全部评价季度')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--split', action='store_true', help='只打印各季度切分，不训练')
    parser.add_argument('--rescore', action='store_true', help='只重推演、不重训：用已有 best.pt 按当前口径刷新打分')
    parser.add_argument('--doctor', action='store_true', help='环境与数据自检')
    args = parser.parse_args()

    # --- 自检：只打印环境信息 ---
    if args.doctor:
        print('Python:', sys.executable, 'PyTorch:', torch.__version__)
        print('Data:', data_root(), 'Panel:', metadata()['panel_digest'])
        print('CUDA:', torch.cuda.is_available(),
              torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')
        for f in ['/sys/fs/cgroup/memory.max', '/sys/fs/cgroup/cpu.max']:
            if Path(f).exists():
                print(f, Path(f).read_text().strip())
        return

    days, _ = axis()
    qs = [args.quarter] if args.quarter else quarters(days)
    if any(q not in quarters(days) for q in qs):
        raise ValueError('季度不在评价窗口内')

    # --- 看切分：不读特征面板 ---
    if args.split:
        for q in qs:
            for s in splits(days, label_horizon(RECIPE['label']), RECIPE['folds'], quarter=q):
                print(json.dumps(split_summary(days, s), ensure_ascii=False))
        return

    # --- 只重推演：不训练，用已有 best.pt 按当前口径刷新打分与切分记录 ---
    if args.rescore:
        panel = Panel(RECIPE['features'])
        return rescore(qs, [args.fold] if args.fold else [1, 2, 3, 4], args.device, panel)

    # --- 训练：读一次面板，按季度 × 折依次训练 ---
    panel = Panel(RECIPE['features'])
    prices = Prices(panel)
    check_days = np.concatenate([splits(panel.days, label_horizon(RECIPE['label']), RECIPE['folds'], quarter=q)[0]['test'] for q in qs])
    prices.verify_labels(panel, check_days)
    for q in qs:
        for fold in ([args.fold] if args.fold else range(1, 5)):
            train_fold(fold, args.device, panel, prices, q)


if __name__ == '__main__':
    main()
