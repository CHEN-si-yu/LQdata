"""V51: train the V34 multi-head model on a frozen SW2021 L2 stock pool."""

# V51 freezes one SW2021 second-level sector pool before performance evaluation.
# Stock tables are Arrow-filtered to RECIPE["universe"] before pandas or tensor allocation.
# Market factors remain date keyed; the runner schedules quarter×fold jobs and each fold
# trains its four independent seeds sequentially.
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
UNIT_ROOT = ROOT
PROJECT_ROOT = UNIT_ROOT.parents[1]
UNIT = UNIT_ROOT.name

DEFAULT_THREADS = "2"
DEFAULT_FEATURE_BLOCK = "factors"
LAYOUT_QUARTERS = ("2025Q3", "2025Q4", "2026Q1", "2026Q2")
LAYOUT_FOLDS = (1, 2, 3, 4)
LAYOUT_YEARS = (2025, 2026)
QUARTERS = LAYOUT_QUARTERS
FOLDS = LAYOUT_FOLDS
SEEDS = (17, 29, 43, 71)
PURGE_HORIZON = 20
SOURCE_FILES = ("run.py", "model.py", "analysis.py")
FOLD_FILES = (
    "best.pt", "last.pt", "complete.json", "history.json",
    "score_predictions.npy", "test_predictions.npy", "split.json",
    "training_info.json",
)
TOOL_DIRS = {".ipynb_checkpoints", "__pycache__"}

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = os.environ.get("MX_THREADS", DEFAULT_THREADS)


def fixed_files():
    """V1's exact fixed 159-file / 28-directory contract."""
    out = set(SOURCE_FILES)
    for quarter in LAYOUT_QUARTERS:
        for fold in LAYOUT_FOLDS:
            base = f"model_train/{quarter}/fold{fold}"
            out.update(f"{base}/{name}" for name in FOLD_FILES)
            out.add(f"model_logs/{quarter}_{fold}.log")
    out.update({
        "model_pred/actions.md", "model_pred/equity_curves.png",
        "model_pred/ensemble/score_meta.json",
        "model_pred/ensemble/year=2025/data.parquet",
        "model_pred/ensemble/year=2026/data.parquet",
        "model_pred/ensemble/CSV/cash_action_3state.csv",
        "model_pred/ensemble/CSV/trades_action_3state.csv",
        "model_pred/ensemble/CSV/cash_buy_hold.csv",
        "model_pred/ensemble/CSV/trades_buy_hold.csv",
        "model_logs/analysis_0.log",
        "model_info/final_audit.json",
        "model_info/action_validation.json",
    })
    assert len(out) == 159, len(out)
    return out


def validate_layout(*, allow_missing=True, allow_temporary=False):
    expected = fixed_files()
    actual_files, actual_dirs = set(), set()
    for path in ROOT.rglob("*"):
        rel = path.relative_to(ROOT).as_posix()
        if any(part in TOOL_DIRS for part in Path(rel).parts):
            continue
        if path.is_dir():
            actual_dirs.add(rel)
        elif path.is_file():
            actual_files.add(rel)
    expected_dirs = set()
    for name in expected:
        parent = Path(name).parent
        while str(parent) not in ("", "."):
            expected_dirs.add(parent.as_posix())
            parent = parent.parent
    if allow_temporary:
        temporary = set()
        for name in expected:
            path = Path(name)
            if path.suffix == ".json":
                temporary.add(path.with_suffix(".json.tmp").as_posix())
            elif path.suffix in (".pt", ".npy"):
                temporary.add(path.with_suffix(".tmp").as_posix())
            elif path.suffix == ".parquet":
                temporary.add(path.with_name("data.tmp.parquet").as_posix())
        actual_files -= temporary
    extra_files = sorted(actual_files - expected)
    extra_dirs = sorted(actual_dirs - expected_dirs)
    missing = sorted(expected - actual_files)
    if extra_files or extra_dirs or (missing and not allow_missing):
        parts = []
        if extra_files:
            parts.append("清单外文件: " + ", ".join(extra_files[:20]))
        if extra_dirs:
            parts.append("清单外目录: " + ", ".join(extra_dirs[:20]))
        if missing and not allow_missing:
            parts.append("缺少文件: " + ", ".join(missing[:20]))
        raise RuntimeError("validate_layout 失败；" + "；".join(parts))
    return {"expected_files": len(expected), "present_files": len(actual_files),
            "expected_directories": len(expected_dirs), "present_directories": len(actual_dirs),
            "missing_files": len(missing), "extra_files": extra_files, "extra_dirs": extra_dirs}


def output_file(path):
    path = Path(path).resolve()
    if not path.is_relative_to(UNIT_ROOT) or path.relative_to(UNIT_ROOT).as_posix() not in fixed_files():
        raise ValueError(f"禁止增加或移动独立单元文件: {path}")
    if path.name in SOURCE_FILES:
        raise ValueError(f"源码文件只能由开发者修改，不能当产物覆写: {path}")
    return path


def data_root():
    root = Path(os.environ.get("MX_DATA", PROJECT_ROOT / "trainingdata" / "factors")).expanduser().resolve()
    if root.name in ("factors", "fac_sample") and (root.parent / "meta.json").is_file():
        root = root.parent
    if not root.is_relative_to(PROJECT_ROOT.resolve()):
        raise ValueError(f"V51 仅允许从 model/trainingdata 读取数据，当前路径为 {root}")
    if not (root / "meta.json").is_file():
        raise FileNotFoundError(f"{root}/meta.json 不存在；需传 model/trainingdata 根目录或 factors 子目录")
    return root


def feature_block():
    requested = Path(os.environ.get("MX_DATA", DEFAULT_FEATURE_BLOCK)).name
    block = requested if requested in ("factors", "fac_sample") else os.environ.get(
        "MODEL_FEATURE_BLOCK", DEFAULT_FEATURE_BLOCK)
    if block != DEFAULT_FEATURE_BLOCK:
        raise ValueError(f"V51 固定使用 {DEFAULT_FEATURE_BLOCK} 特征块，当前为 {block}")
    return block


def price_root():
    return data_root() / "prices"


def feature_columns(meta):
    columns = list(meta["columns"]["features"])
    excluded = {
        "chip_near_overhang_5pct", "close30_amt_share", "close30_giveback",
        "open5_capacity_p20_log", "open5_capacity_floor_ratio_20", "close30_ret",
    }
    return [column for column in columns if column not in excluded]


def market_columns(meta):
    block = meta["market_factors"]
    columns = [c for c in block["columns"] if c.endswith(f"_z{block['z_window']}")]
    if len(columns) != block["n_factors"]:
        raise ValueError(f"市场因子 z 窗口列数 {len(columns)} 与 n_factors {block['n_factors']} 不一致")
    return columns


RUN_ROOT = UNIT_ROOT
INPUT_ROOT = UNIT_ROOT

RECIPE = dict(
    name="V51",
    purpose="冻结SW2021申万二级270500消费电子池内训练排序模型并供策略优化",
    universe=[
        "000021.SZ", "002045.SZ", "002139.SZ", "002351.SZ", "002369.SZ",
        "002475.SZ", "002600.SZ", "002635.SZ", "002655.SZ", "002660.SZ",
        "002855.SZ", "002861.SZ", "002866.SZ", "002881.SZ", "002888.SZ",
        "600203.SH", "601231.SH", "603380.SH", "603626.SH", "603633.SH",
    ],
    pool={
        "taxonomy": "申万2021",
        "level1_code": "270000",
        "level1_name": "电子",
        "level2_code": "270500",
        "level2_name": "消费电子",
        "snapshot_date": "2025-06-30",
        "freeze_rule": "trainingdata轴内各申万2021二级行业点时成分数最接近20；并列按行业代码升序；在查看策略收益前冻结",
        "selection_used_backtest_returns": False,
        "member_count": 20,
        "members": None,
        "mean_pairwise_correlation": 0.3858239,
        "first_pc_variance_share": 0.4190298,
        "health_common_dates": 1328,
        "health_period": ["2020-01-02", "2025-06-30"],
        "health_return_basis": "复权收盘日收益",
    },
    architecture="V34 multi-head MLP + market gate + ridge blend + early stopping + epoch bagging; only sample universe is restricted",
    feature_block="factors",
    features="all",
    preprocessing="daily_zscore_win1_99_fixed_missing_0",
    label_horizons=[1, 3, 5, 10, 20],
    labels=[f"label_ret_{h}d" for h in (1, 3, 5, 10, 20)],
    label_formula="adj_open[T+1+h] / adj_open[T+1] - 1",
    score_blend="head:label_ret_1d",
    score_label="label_ret_1d",
    label_weights={1: 0.3, 3: 0.5, 5: 1.0, 10: 0.4, 20: 0.2},
    valid_ic_weights={1: 0.1, 3: 0.15, 5: 0.4, 10: 0.2, 20: 0.15},
    valid_ic_scale=1.0,
    valid_criterion="rankic",
    market_gate=True,
    market_block="market_factors",
    market_clip=10.0,
    hidden=(256, 128, 32),
    dropout=(0.2, 0.1),
    loss_rank_decay=0.5,
    optimizer="adamw",
    lr=0.001,
    weight_decay=0.02,
    max_epochs=30,
    early_stop_patience=6,
    lr_patience=3,
    lr_factor=0.5,
    lr_cooldown=2,
    min_lr=5e-6,
    batch_days=4,
    bag_topk=5,
    bag_recalibrate=True,
    ridge_lambda=1e3,
    linear_weight=0.2,
    min_feature_coverage=0.2,
    valid_topn=5,
    train_pool_frac=1.0,
    hot_pool=dict(frac=1.0, amount_window=20, amp_window=5, limit_window=20, min_names=3),
    entry_rule="open",
    folds=4,
    seeds=list(SEEDS),
    purge_horizon=PURGE_HORIZON,
    purge_embargo_trading_days=PURGE_HORIZON + 1,
    threads=int(os.environ.get("MX_THREADS", DEFAULT_THREADS)),
    device="auto",
    account_money=100000.0,
    test_start="2025-07-01",
    test_end="2026-06-30",
    quarters=list(QUARTERS),
    strategy_search=dict(
        topk=[1, 3, 5],
        target_exposure=[0.25, 0.50, 1.0],
        threshold_modes=["rank_only", "rank_and_score_gt_0"],
        buffer_slots=[0, 1, 2],
        rebalance_days=[1, 5, 20],
    ),
    clean_selection=dict(
        quarter="2025Q3",
        fold=4,
        split="valid",
        contiguous_segments=4,
        minimum_positive_segments=3,
        minimum_positive_seeds=3,
        alpha_formula="strategy_net_return - average_exposure * same_pool_equal_weight_net_return",
    ),
    execution=dict(
        account_money=100000.0,
        lot_size=100,
        max_participation=0.01,
        commission_rate=0.00025,
        min_commission=5.0,
        transfer_rate=0.00001,
        stamp_sell_rate=0.0005,
        slippage_rate=0.0003,
        signal_time="T_close",
        execution_time="T_plus_1_open",
        mark_time="daily_close",
        gross_fee_scenarios=["gross_zero_cost", "explicit_fees_no_slippage", "net_with_slippage"],
    ),
    reference_baselines=dict(
        four_bank_codes=["601288.SH", "601398.SH", "601939.SH", "601988.SH"],
        expected_net_return_242d=dict(momentum_top1=0.1835, momentum_top2=0.1158, equal_weight=0.0162),
        long_window_net_return_2024_01_to_2026_06=dict(
            momentum_60d=0.8169, v34_single_factor=0.9221, equal_weight=0.7044,
        ),
    ),
    data_built_at=None,
)
RECIPE["pool"]["members"] = RECIPE["universe"]
CODES = tuple(RECIPE["universe"])

if feature_block() != "factors":
    raise ValueError("V51 只允许读取 trainingdata/factors 全因子特征块")


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
        # ★ V23：三个宽度与两个 dropout 全部取自 RECIPE（原先写死，做不了强度对照）
        w1, w2, w3 = RECIPE['hidden']
        d1, d2 = RECIPE['dropout']
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, w1),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(d1),
        )
        self.trunk1 = nn.Sequential(nn.Linear(w1, w2), nn.BatchNorm1d(w2), nn.GELU())
        self.drop = nn.Dropout(d2)
        self.trunk2 = nn.Sequential(nn.Linear(w2, w3), nn.BatchNorm1d(w3), nn.GELU())
        self.output_layer = nn.Linear(w3, len(self.horizons))
        self.gate = MarketGate(market_dim, (w2, w3)) if market_dim else None
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
    """按冻结池过滤读取一年股票块，再转换成池内索引表。"""
    path = root / kind / f"year={year}" / "data.parquet"
    table = pq.read_table(
        path,
        columns=["trade_date", "stock_code", *columns],
        filters=[("stock_code", "in", list(RECIPE["universe"]))],
    )
    df = table.to_pandas()
    df["trade_date"] = df["trade_date"].astype(str).str[:10]
    df["stock_code"] = df["stock_code"].astype(str)
    unexpected = sorted(set(df["stock_code"]) - set(RECIPE["universe"]))
    if unexpected:
        raise ValueError(f"{path}: Arrow 过滤后仍出现池外代码 {unexpected[:5]}")
    if df.duplicated(["trade_date", "stock_code"]).any():
        raise ValueError(f"{path}: 重复主键")
    return df.set_index(["trade_date", "stock_code"]).sort_index()


PRICE_COLUMNS = ("open", "high", "low", "close", "pre_close", "pct_chg", "vol", "adj_factor")


def price_frame(root, year, index):
    """仅读取RECIPE池内价格，并核对交易日轴。"""
    path = Path(root) / f"year={year}" / "data.parquet"
    part = pq.read_table(
        path,
        columns=["trade_date", "stock_code", *PRICE_COLUMNS],
        filters=[("stock_code", "in", list(RECIPE["universe"]))],
    ).to_pandas()
    part["trade_date"] = part["trade_date"].astype(str).str[:10]
    part["stock_code"] = part["stock_code"].astype(str)
    unexpected = sorted(set(part["stock_code"]) - set(RECIPE["universe"]))
    if unexpected:
        raise ValueError(f"{path}: Arrow 过滤后仍出现池外代码 {unexpected[:5]}")
    if part.duplicated(["trade_date", "stock_code"]).any():
        raise ValueError(f"{path}: 重复主键")
    part = part.set_index(["trade_date", "stock_code"])
    want = set(index.get_level_values(0))
    days_seen = set(part.index.get_level_values(0))
    if days_seen != want:
        raise ValueError(
            f"{year} 池内价格日线轴与因子轴不一致（缺 {sorted(want - days_seen)[:3]} / "
            f"多 {sorted(days_seen - want)[:3]}）")
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
    """读取全日历日期轴，但股票轴严格来自已冻结的20只池内代码。"""
    root = data_root()
    meta = metadata()
    available = set(str(code) for code in meta["axis"]["codes"])
    missing = sorted(set(RECIPE["universe"]) - available)
    if missing:
        raise ValueError(f"RECIPE 冻结池代码不在 trainingdata/meta.json 股票轴内: {missing}")
    days = []
    for year in sorted(int(y) for y in meta["built_years"]):
        path = root / RECIPE["market_block"] / f"year={year}" / "data.parquet"
        table = pq.read_table(path, columns=["trade_date"])
        values = table.column(0).to_pylist()
        days.extend(sorted(set(str(value)[:10] for value in values)))
    days = np.asarray(sorted(set(days)), dtype=str)
    codes = np.asarray(RECIPE["universe"], dtype=str)
    if len(codes) != 20 or len(set(codes)) != 20:
        raise ValueError("V51 RECIPE 必须冻结20只互异股票")
    return days, codes


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
                #   不再在整块 X 上做：478 列下 `np.isfinite(self.X).all()` 会临时分配一份
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

    def mask(self, day, label=None):
        """当日可用的股票掩码：覆盖率达标，且（给定标签时）标签已知。

        ★ 阈值改读 `RECIPE['min_feature_coverage']`：原先这里硬编码 `.2`，而 RECIPE 里
          同名键**从未被读**（改了不生效）。实测两种特征块下该掩码都不触发
          （478 列时 per-(day,stock) 有效占比最小 0.41，20 列时最小 0.30），
          所以这次接线在数值上是 no-op，纯粹让配置成为唯一真源。
        """
        out = self.coverage[day] >= RECIPE['min_feature_coverage']
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
        做过符号翻转（`raw = 存储值 × direction`，478 列里 121 列是 −1），拿它组「越大越热门」
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
            x = torch.from_numpy(panel.X[d, mask]).to(device)
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
    """在训练窗上闭式拟合岭回归（逐日截面已 z-score 的特征 → 当日标准化的打分目标）。

    返回 `(n_features + 1,)` 的权重（末位是截距）。逐日累积 Gram 矩阵，不整块驻留 X。

    ★ 为什么要在神经网络之外再挂一个线性分量：两者**信息互补**——1893 天样本外上，
    线性分量在全截面 RankIC 上全面更好（四块 +0.004~0.008），而网络在极值桶（top5）上更好。
    掺 20% 后 RankIC / Pearson IC / top5 **三个同时变好**（见 `blend_linear`）。
    """
    D = len(panel.features)
    G = np.zeros((D + 1, D + 1))
    b = np.zeros(D + 1)
    for d in train_days:
        m = panel.mask(d, RECIPE['score_label'])
        if pool is not None:                    # ★ V17：岭回归分量与网络用同一套样本
            m = m & pool[d]
        if m.sum() < 3:
            continue
        X = panel.X[d, m].astype(np.float64)
        y = panel.Y[RECIPE['score_label']][d, m].astype(np.float64)
        sd = y.std()
        if sd < 1e-8:
            continue
        y = (y - y.mean()) / sd
        A = np.c_[X, np.ones(len(X))]
        G += A.T @ A
        b += A.T @ y
    G[np.arange(D), np.arange(D)] += lam
    G[-1, -1] += 1e-8
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
        if known.sum() < 3:
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
            x = torch.from_numpy(panel.X[d, mask]).to(device)
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
    # ★ V9：选轮与落盘的 valid 指标都算在**交付打分**（网络 ⊕ 线性分量）上；
    # ★ V11：交付打分还过一道热门池，选轮判据必须跟着过 —— 否则选出来的 epoch 是在
    #   全截面上排序最好的那一轮，而不是**在池内**排序最好的那一轮，判据与交付物不一致。
    score = hot_filter(blend_linear(score_of(heads), getattr(model, 'ridge', None), panel, days),
                       prices, days)
    ic, rank = {}, {}
    for h in RECIPE['label_horizons']:
        label = f'label_ret_{h}d'
        pe, sp = [], []
        for i, d in enumerate(days):
            y = panel.Y[label][d]
            p = score[i]
            known = np.isfinite(p)                       # 打分缺失的票排不了序、也不进 IC
            ok = known & np.isfinite(y)
            # 与 analysis.py:top_portfolio_daily 逐字同式，**不跳过无效日**：
            # 无效日由 corr() 返回 0 并计入日平均（两边同日同式才可比）。
            sp.append(corr(rankdata(p[ok]), rankdata(y[ok])))
            pe.append(corr(p[known], np.where(np.isfinite(y[known]), y[known], 0.)))
        ic[label] = float(np.mean(pe)) if pe else 0.
        rank[label] = float(np.mean(sp)) if sp else 0.
    ic_term = sum(RECIPE['valid_ic_weights'][h] * rank[f'label_ret_{h}d']
                  for h in RECIPE['label_horizons'])
    # ★ V21：判据可以选择"全截面加权 RankIC"或"可交易 top5 的收益"。
    #   后者在下面算（`ret`），所以这里先取个名字，最后再合。
    criterion = RECIPE['valid_criterion']
    if criterion not in ('rankic', 'tradable_topn'):
        raise ValueError(f"未知 valid_criterion={criterion!r}；见 RECIPE['valid_criterion']")
    # 收益代理：**只进 valid 表的「收益(%)」列，不进 val_wei**（表格版式固定，不能少列）。
    # 口径与 V2 逐字不变：可执行 top5、5d 标签、缺失按 0、不换下一名。
    # ★ V11：可执行性走 `RECIPE['entry_rule']`（打板口径下「开盘涨停但盘中开板」的票也算能买），
    #   与回测同源，否则 valid 表的收益列与策略回测是两个口径。
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
        rets.append(float(np.where(miss, 0., y[picks]).sum() / RECIPE['valid_topn']) if len(picks) else 0.)
    ret = float(np.mean(rets))
    # ★ V22：`score_label` 不一定在头集合里（比如只训 1d 头时）。
    #   这两个字段只是报表用的便利量，缺了就**退到最长的那个头**，不要让它 KeyError
    #   —— 头集合是可配的，代码不能假设某一路径永远只有一种形状。
    label5 = RECIPE['score_label']
    if label5 not in ic:
        label5 = f'label_ret_{max(RECIPE["label_horizons"])}d'
    # ★ V21：`val_wei` —— 判据为 `rankic` 时是加权 RankIC；为 `tradable_topn` 时是
    #    可执行 top5 的 5 日标签均值（**不乘 100**：它本身已经是收益量纲，乘了只是放大噪声）。
    primary = (ret if criterion == 'tradable_topn' else RECIPE['valid_ic_scale'] * ic_term)
    return dict(val_wei=primary,
                val_criterion=criterion,
                val_return_proxy=ret, val_ic_weighted=ic_term,
                val_unknown_label_picks=unknown_picks,
                Pearson_IC=ic[label5], RankIC=rank[label5],
                **{f'IC_{h}d': ic[f'label_ret_{h}d'] for h in RECIPE['label_horizons']},
                **{f'RankIC_{h}d': rank[f'label_ret_{h}d'] for h in RECIPE['label_horizons']})


def _cpu_state_dict(state):
    return {key: value.detach().to("cpu", copy=True) for key, value in state.items()}


def _fold_signature(recipe, panel):
    payload = {
        "recipe": recipe,
        "data_built_at": panel.meta.get("built_at"),
        "features": list(panel.features),
        "market_features": list(panel.market_features),
        "codes": list(panel.codes),
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _completed_payload(completed, panel, split, quarter, fold, signature):
    ordered = sorted(completed, key=lambda item: RECIPE["seeds"].index(item["seed"]))
    seed_ids = [int(item["seed"]) for item in ordered]
    score_by_seed = np.stack([item["score_prediction"] for item in ordered]).astype(np.float32)
    valid_by_seed = np.stack([item["valid_prediction"] for item in ordered]).astype(np.float32)
    with np.errstate(invalid="ignore"):
        score_mean = np.nanmean(score_by_seed, axis=0).astype(np.float32)
    return {
        "schema": "v51.seed-bundle.v1",
        "recipe_name": RECIPE["name"],
        "recipe_signature": signature,
        "data_built_at": panel.meta.get("built_at"),
        "quarter": quarter,
        "fold": int(fold),
        "codes": panel.codes.tolist(),
        "seed_ids": seed_ids,
        "seed_bundles": [item["bundle"] for item in ordered],
        "seed_predictions": score_by_seed,
        "valid_seed_predictions": valid_by_seed,
        "score_predictions_mean": score_mean,
        "score_dates": panel.days[split["score"]].tolist(),
        "valid_dates": panel.days[split["valid"]].tolist(),
        "test_dates": panel.days[split["test"]].tolist(),
        "completed": ordered,
    }


def _history_document(completed, active_seed=None, active_history=None):
    rows = [{
        "seed": int(item["seed"]),
        "epochs": item["history"],
        "bag": item["bundle"]["bag"],
        "argmax_validation": item["bundle"]["argmax_validation"],
        "bagged_validation": item["bundle"]["validation"],
    } for item in completed]
    if active_seed is not None:
        rows.append({"seed": int(active_seed), "epochs": active_history or [], "in_progress": True})
    return rows


def train_fold(fold, device="auto", panel=None, prices=None, quarter=None, force=False):
    """训练一个季度×折；四种子顺序运行，种子间不复用权重，支持从已完成种子续跑。"""
    quarter = quarter or QUARTERS[0]
    if quarter not in QUARTERS or fold not in FOLDS:
        raise ValueError(f"无效季度/折: {quarter} fold{fold}")
    recipe = copy.deepcopy(RECIPE)
    recipe["device"] = device
    recipe["quarter"] = quarter
    recipe["fold"] = int(fold)
    panel = panel or Panel(recipe["features"])
    prices = prices or Prices(panel)
    recipe["data_built_at"] = panel.meta.get("built_at")
    signature = _fold_signature(recipe, panel)

    out = ROOT / "model_train" / quarter / f"fold{fold}"
    out.mkdir(parents=True, exist_ok=True)
    done_path = output_file(out / "complete.json")
    best_path = output_file(out / "best.pt")
    last_path = output_file(out / "last.pt")
    if done_path.is_file() and not force:
        old = json.loads(done_path.read_text(encoding="utf-8"))
        required = ("best.pt", "last.pt", "score_predictions.npy", "test_predictions.npy",
                    "history.json", "split.json", "training_info.json")
        if old.get("recipe_signature") != signature:
            raise RuntimeError(f"{done_path} 与当前 RECIPE/数据快照不匹配；请显式 --force")
        if not all((out / name).is_file() for name in required):
            raise RuntimeError(f"{done_path} 存在但固定产物缺失")
        print(f"{quarter} fold{fold} 已完成且配方/快照匹配，跳过", flush=True)
        return old
    if force and done_path.exists():
        done_path.unlink()

    split = splits(panel.days, RECIPE["purge_horizon"], RECIPE["folds"], quarter)[fold - 1]
    atomic_json(out / "training_info.json", training_info(recipe))
    atomic_json(out / "split.json", split_summary(panel.days, split))

    # 小池仍沿用V34岭回归混合；岭项对特征系数正则，截距不受惩罚。
    ridge = fit_ridge(panel, split["train"])
    torch.set_num_threads(recipe["threads"])
    dev = device_for(device)
    torch.use_deterministic_algorithms(True)
    started = time.time()

    completed = []
    active_seed = None
    if not force:
        for candidate in (last_path, best_path):
            if not candidate.is_file():
                continue
            old = torch.load(candidate, map_location="cpu", weights_only=False)
            if old.get("schema") != "v51.seed-bundle.v1" or old.get("recipe_signature") != signature:
                if candidate == last_path and old.get("schema") == "v51.progress.v1":
                    if old.get("recipe_signature") != signature:
                        raise RuntimeError(f"{candidate} 与当前 RECIPE/数据快照不匹配；请显式 --force")
                    candidate_completed = old.get("completed", [])
                    if len(candidate_completed) > len(completed):
                        completed = candidate_completed
                    active_seed = old.get("active_seed")
                    continue
                raise RuntimeError(f"{candidate} 与当前 RECIPE/数据快照不匹配；请显式 --force")
            candidate_completed = old.get("completed", [])
            if len(candidate_completed) > len(completed):
                completed = candidate_completed
            if candidate == last_path:
                active_seed = old.get("active_seed")

    completed_ids = {int(item["seed"]) for item in completed}
    if not completed_ids.issubset(set(RECIPE["seeds"])):
        raise RuntimeError("last.pt 含 RECIPE 以外的种子")
    if not completed:
        empty_payload = {
            "schema": "v51.seed-bundle.v1", "recipe_name": RECIPE["name"],
            "recipe_signature": signature, "data_built_at": panel.meta.get("built_at"),
            "quarter": quarter, "fold": int(fold), "codes": panel.codes.tolist(),
            "seed_ids": [], "seed_bundles": [],
            "seed_predictions": np.empty((0, len(split["score"]), len(panel.codes)), np.float32),
            "valid_seed_predictions": np.empty((0, len(split["valid"]), len(panel.codes)), np.float32),
            "score_predictions_mean": np.empty((len(split["score"]), len(panel.codes)), np.float32),
            "score_dates": panel.days[split["score"]].tolist(),
            "valid_dates": panel.days[split["valid"]].tolist(),
            "test_dates": panel.days[split["test"]].tolist(),
            "completed": [],
        }
        save_torch(best_path, empty_payload)
        save_torch(last_path, {
            "schema": "v51.progress.v1", "recipe_signature": signature,
            "data_built_at": panel.meta.get("built_at"), "quarter": quarter,
            "fold": int(fold), "completed": [], "active_seed": None,
            "seed_ids_completed": [],
        })

    for seed in RECIPE["seeds"]:
        if seed in completed_ids:
            continue
        # 中断时只恢复完整种子；当前未完成种子从相同 seed 重新初始化。
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if dev.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        model = PredictModel(len(panel.features), panel.market_dim).to(dev)
        model.ridge = ridge
        if recipe["optimizer"] != "adamw":
            raise ValueError(f"暂只实现 adamw，收到 {recipe['optimizer']!r}")
        opt = torch.optim.AdamW(model.parameters(), lr=recipe["lr"], weight_decay=recipe["weight_decay"])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="max", factor=recipe["lr_factor"], patience=recipe["lr_patience"],
            cooldown=recipe["lr_cooldown"], min_lr=recipe["min_lr"])
        best = -float("inf")
        best_epoch = -1
        best_metrics = None
        best_state = None
        stale = 0
        history = []
        bank = []
        seed_started = time.time()
        for epoch in range(recipe["max_epochs"]):
            if stale >= recipe["early_stop_patience"]:
                break
            model.train()
            order = np.random.permutation(split["train"])
            losses = []
            epoch_started = time.time()
            for pos in range(0, len(order), recipe["batch_days"]):
                opt.zero_grad(set_to_none=True)
                batch_losses = []
                for day in order[pos:pos + recipe["batch_days"]]:
                    mask = panel.mask(day)
                    if mask.sum() < 3:
                        continue
                    targets = {}
                    for horizon in recipe["label_horizons"]:
                        y = panel.Y[f"label_ret_{horizon}d"][day, mask]
                        valid = np.isfinite(y)
                        if valid.sum() < 3 or np.std(y[valid]) < 1e-8:
                            continue
                        targets[horizon] = (valid, y[valid])
                    if not targets:
                        continue
                    x = torch.from_numpy(panel.X[day, mask]).to(dev)
                    market = torch.from_numpy(panel.M[day]).to(dev) if panel.market_dim else None
                    pred = model(x, market)
                    terms, weights = [], []
                    for horizon, (valid, y) in targets.items():
                        j = recipe["label_horizons"].index(horizon)
                        target = torch.from_numpy(y).to(dev)
                        target = (target - target.mean()) / target.std()
                        rows = torch.from_numpy(np.flatnonzero(valid)).to(dev)
                        terms.append(recipe["label_weights"][horizon] * wpcc(pred[rows, j], target))
                        weights.append(recipe["label_weights"][horizon])
                    batch_losses.append(torch.stack(terms).sum() / sum(weights))
                if not batch_losses:
                    continue
                loss = torch.stack(batch_losses).mean()
                if not torch.isfinite(loss):
                    raise FloatingPointError("WPCC 非有限")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                opt.step()
                losses.append(float(loss.detach()))

            metrics = validation(model, panel, prices, split["valid"], dev)
            score = float(metrics["val_wei"])
            scheduler.step(score)
            improved = score > best + 1e-10
            if improved:
                best, best_epoch, best_metrics, stale = score, epoch, metrics, 0
                best_state = _cpu_state_dict(model.state_dict())
            else:
                stale += 1
            if recipe["bag_topk"]:
                bank.append((score, epoch, _cpu_state_dict(model.state_dict())))
                bank.sort(key=lambda item: -item[0])
                del bank[recipe["bag_topk"]:]
            record = {
                "seed": int(seed), "epoch": int(epoch),
                "train_loss": float(np.mean(losses)) if losses else None,
                "lr": float(opt.param_groups[0]["lr"]),
                "seconds": round(time.time() - epoch_started, 2),
                "best": bool(improved), **metrics,
            }
            history.append(record)
            atomic_json(out / "history.json", _history_document(completed, seed, history))
            save_torch(last_path, {
                "schema": "v51.progress.v1", "recipe_signature": signature,
                "data_built_at": panel.meta.get("built_at"), "quarter": quarter,
                "fold": int(fold), "completed": completed, "active_seed": int(seed),
                "active_epoch": int(epoch), "seed_ids_completed": sorted(completed_ids),
            })
            elapsed = time.time() - seed_started
            print(f"{quarter} fold{fold} seed={seed} epoch={epoch + 1}/{recipe['max_epochs']} "
                  f"val_RankIC={score:.5f} lr={record['lr']:.6g} "
                  f"elapsed={elapsed / 60:.1f}m", flush=True)

        if best_state is None:
            raise RuntimeError(f"{quarter} fold{fold} seed={seed} 没有可用验证轮次")
        model.load_state_dict(best_state)
        argmax_state = _cpu_state_dict(best_state)
        argmax_metrics = best_metrics
        if recipe["bag_topk"] and len(bank) > 1:
            model.load_state_dict(average_state_dicts([state for _, _, state in bank]))
            if recipe["bag_recalibrate"]:
                recalibrate_bn(model, panel, split["train"], dev)
            bagged_metrics = validation(model, panel, prices, split["valid"], dev)
        else:
            model.load_state_dict(argmax_state)
            bagged_metrics = argmax_metrics
        model.ridge = ridge
        model.eval()
        valid_prediction = predict(model, panel, prices, split["valid"], dev).astype(np.float32)
        score_prediction = predict(model, panel, prices, split["score"], dev).astype(np.float32)
        bundle = {
            "seed": int(seed),
            "model": _cpu_state_dict(model.state_dict()),
            "model_argmax": argmax_state,
            "input_dim": len(panel.features),
            "market_dim": panel.market_dim,
            "horizons": list(model.horizons),
            "features": list(panel.features),
            "ridge": np.asarray(ridge, dtype=np.float64),
            "best_epoch": int(best_epoch),
            "argmax_validation": argmax_metrics,
            "validation": bagged_metrics,
            "bag": {
                "topk": int(recipe["bag_topk"]),
                "used": len(bank),
                "epochs": [int(epoch) + 1 for _, epoch, _ in bank],
                "scores": [float(value) for value, _, _ in bank],
            },
        }
        completed.append({
            "seed": int(seed), "bundle": bundle,
            "score_prediction": score_prediction,
            "valid_prediction": valid_prediction,
            "history": history,
        })
        completed_ids.add(int(seed))
        payload = _completed_payload(completed, panel, split, quarter, fold, signature)
        save_torch(best_path, payload)
        save_torch(last_path, {
            "schema": "v51.progress.v1", "recipe_signature": signature,
            "data_built_at": panel.meta.get("built_at"), "quarter": quarter,
            "fold": int(fold), "completed": completed, "active_seed": None,
            "seed_ids_completed": sorted(completed_ids),
        })
        atomic_json(out / "history.json", _history_document(completed))
        print(f"{quarter} fold{fold} seed={seed} complete: "
              f"argmax={argmax_metrics['val_wei']:.5f}, bagged={bagged_metrics['val_wei']:.5f}",
              flush=True)
        del model, opt, scheduler
        if dev.type == "cuda":
            torch.cuda.empty_cache()

    if completed_ids != set(RECIPE["seeds"]):
        raise RuntimeError(f"种子未全部完成: {sorted(completed_ids)}")
    payload = _completed_payload(completed, panel, split, quarter, fold, signature)
    save_torch(best_path, payload)
    score_mean = payload["score_predictions_mean"]
    test_mean = score_mean[:len(split["test"])]
    save_npy(out / "score_predictions.npy", score_mean)
    save_npy(out / "test_predictions.npy", test_mean)
    atomic_json(out / "history.json", _history_document(completed))
    rss_gib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2)
    result = {
        "schema": "v51.complete.v1",
        "recipe_name": RECIPE["name"],
        "recipe_signature": signature,
        "data_built_at": panel.meta.get("built_at"),
        "quarter": quarter, "fold": int(fold),
        "seed_ids": [int(seed) for seed in RECIPE["seeds"]],
        "valid_by_seed": [
            {"seed": int(item["seed"]),
             "argmax_validation": item["bundle"]["argmax_validation"],
             "bagged_validation": item["bundle"]["validation"],
             "bag": item["bundle"]["bag"]}
            for item in sorted(completed, key=lambda row: RECIPE["seeds"].index(row["seed"]))
        ],
        "train_days": int(len(split["train"])),
        "valid_days": int(len(split["valid"])),
        "test_days": int(len(split["test"])),
        "score_days": int(len(split["score"])),
        "train_start": str(panel.days[split["train"][0]]),
        "train_end": str(panel.days[split["train"][-1]]),
        "valid_start": str(panel.days[split["valid"][0]]),
        "valid_end": str(panel.days[split["valid"][-1]]),
        "test_start": str(panel.days[split["test"][0]]),
        "test_end": str(panel.days[split["test"][-1]]),
        "score_start": str(panel.days[split["score"][0]]),
        "score_end": str(panel.days[split["score"][-1]]),
        "clean_forward_test": bool(panel.days[split["train"][-1]] < panel.days[split["test"][0]]),
        "score_prediction_shape": list(score_mean.shape),
        "seed_prediction_shape": list(payload["seed_predictions"].shape),
        "valid_seed_prediction_shape": list(payload["valid_seed_predictions"].shape),
        "peak_rss_gib": round(float(rss_gib), 3),
        "seconds": round(time.time() - started, 2),
        "device": str(dev),
    }
    atomic_json(done_path, result)
    return result


def rescore(quarter_list, fold_list, device="auto", panel=None, prices=None):
    """用现存四种子 checkpoints 重推有效、推演窗分数；不重新训练。"""
    dev = device_for(device)
    torch.set_num_threads(RECIPE["threads"])
    panel = panel or Panel(RECIPE["features"])
    prices = prices or Prices(panel)
    for quarter in quarter_list:
        for fold in fold_list:
            out = ROOT / "model_train" / quarter / f"fold{fold}"
            done_path = out / "complete.json"
            best_path = out / "best.pt"
            if not done_path.is_file() or not best_path.is_file():
                print(f"{quarter} fold{fold} 尚未完成，跳过重推演", flush=True)
                continue
            done = json.loads(done_path.read_text(encoding="utf-8"))
            payload = torch.load(best_path, map_location=dev, weights_only=False)
            if payload.get("schema") != "v51.seed-bundle.v1":
                raise RuntimeError(f"{best_path} 不是 V51 多种子 checkpoint")
            split = splits(panel.days, RECIPE["purge_horizon"], RECIPE["folds"], quarter)[fold - 1]
            completed = payload["completed"]
            if {int(row["seed"]) for row in completed} != set(RECIPE["seeds"]):
                raise RuntimeError(f"{quarter} fold{fold} checkpoint 未含齐四个种子")
            for row in completed:
                bundle = row["bundle"]
                model = PredictModel(len(panel.features), panel.market_dim).to(dev)
                model.load_state_dict(bundle["model"])
                model.ridge = bundle["ridge"]
                bundle["validation"] = validation(model, panel, prices, split["valid"], dev)
                row["valid_prediction"] = predict(
                    model, panel, prices, split["valid"], dev).astype(np.float32)
                row["score_prediction"] = predict(
                    model, panel, prices, split["score"], dev).astype(np.float32)
                del model
            refreshed = _completed_payload(completed, panel, split, quarter, fold,
                                           payload["recipe_signature"])
            refreshed["rescore_only"] = True
            save_torch(best_path, refreshed)
            save_torch(out / "last.pt", {
                "schema": "v51.progress.v1",
                "recipe_signature": payload["recipe_signature"],
                "data_built_at": panel.meta.get("built_at"),
                "quarter": quarter, "fold": int(fold),
                "completed": completed, "active_seed": None,
                "seed_ids_completed": [int(x) for x in RECIPE["seeds"]],
            })
            save_npy(out / "score_predictions.npy", refreshed["score_predictions_mean"])
            save_npy(out / "test_predictions.npy",
                     refreshed["score_predictions_mean"][:len(split["test"])])
            atomic_json(out / "split.json", split_summary(panel.days, split))
            done.update({
                "score_dates": refreshed["score_dates"],
                "valid_dates": refreshed["valid_dates"],
                "test_dates": refreshed["test_dates"],
                "seed_prediction_shape": list(refreshed["seed_predictions"].shape),
                "valid_seed_prediction_shape": list(refreshed["valid_seed_predictions"].shape),
                "score_prediction_shape": list(refreshed["score_predictions_mean"].shape),
                "valid_by_seed": [
                    {"seed": int(row["seed"]),
                     "argmax_validation": row["bundle"]["argmax_validation"],
                     "bagged_validation": row["bundle"]["validation"],
                     "bag": row["bundle"]["bag"]}
                    for row in sorted(completed, key=lambda item: RECIPE["seeds"].index(item["seed"]))
                ],
            })
            atomic_json(done_path, done)


def main(argv=None):
    parser = ArgumentParser(description="V51 池内多头排序模型；四季度×四折×四种子")
    parser.add_argument("--quarter", choices=QUARTERS)
    parser.add_argument("--fold", type=int, choices=FOLDS)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--layout-only", action="store_true")
    parser.add_argument("--split", action="store_true")
    parser.add_argument("--rescore", action="store_true")
    parser.add_argument("--doctor", action="store_true")
    args = parser.parse_args(argv)
    if args.layout_only:
        print(json.dumps(validate_layout(), ensure_ascii=False, indent=2))
        return
    validate_layout(allow_missing=True, allow_temporary=True)
    if (args.quarter is None) != (args.fold is None):
        parser.error("--quarter 与 --fold 要么同时指定，要么都不指定")
    days, _ = axis()
    quarters_to_run = [args.quarter] if args.quarter else list(QUARTERS)
    folds_to_run = [args.fold] if args.fold else list(FOLDS)

    if args.doctor:
        print("Python:", sys.executable, "PyTorch:", torch.__version__)
        print("Data:", data_root())
        print("Pool codes:", len(CODES), "Device:", device_for(args.device))
        return
    if args.split:
        for quarter in quarters_to_run:
            for split in splits(days, RECIPE["purge_horizon"], RECIPE["folds"], quarter):
                print(json.dumps(split_summary(days, split), ensure_ascii=False))
        return

    panel = Panel(RECIPE["features"])
    prices = Prices(panel)
    check_days = np.concatenate([
        splits(panel.days, RECIPE["purge_horizon"], RECIPE["folds"], quarter)[0]["test"]
        for quarter in quarters_to_run
    ])
    prices.verify_labels(panel, check_days)
    if args.rescore:
        return rescore(quarters_to_run, folds_to_run, args.device, panel, prices)
    for quarter in quarters_to_run:
        for fold in folds_to_run:
            train_fold(fold, args.device, panel, prices, quarter, force=args.force)


if __name__ == '__main__':
    main()
