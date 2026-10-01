#!/usr/bin/env python3
"""experiments2/V2 candidate: V1 models with fixed 2.0x validation-spread hysteresis."""
from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import resource
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parents[1]
SOURCE_ROOT = PROJECT_ROOT / "experiments2" / "V1"
DATA_ROOT = PROJECT_ROOT / "trainingdata"
CODES = ("601288.SH", "601398.SH", "601939.SH", "601988.SH")
LABELS = ("label_ret_1d", "label_ret_5d")
ALL_LABELS = ("label_ret_1d", "label_ret_3d", "label_ret_5d", "label_ret_10d", "label_ret_20d")
PRICE_COLUMNS = ("open", "high", "low", "close", "pre_close", "pct_chg", "vol", "adj_factor")
QUARTERS = ("2025Q3", "2025Q4", "2026Q1", "2026Q2")
FOLDS = (1, 2, 3, 4)
PURGE_HORIZON = 5
SEEDS = (17, 29, 43, 71)

RECIPE = {
    "name": "V2_candidate",
    "purpose": "复用V1模型，仅将因果q30/q70滞回带固定为2.0倍",
    "universe": list(CODES),
    "feature_block": "factors + market_factors",
    "feature_semantics": "trainingdata/meta.json semantics; cross-section winsorized z-score, missing filled with 0",
    "labels": list(LABELS),
    "score_blend": {"label_ret_1d": 0.30, "label_ret_5d": 0.70},
    "model": {
        "library": "lightgbm",
        "objective": "huber",
        "n_estimators": 180,
        "learning_rate": 0.03,
        "num_leaves": 7,
        "max_depth": 3,
        "min_child_samples": 80,
        "feature_fraction": 0.80,
        "bagging_fraction": 0.80,
        "bagging_freq": 1,
        "reg_alpha": 2.0,
        "reg_lambda": 20.0,
        "label_clip_quantiles": [0.005, 0.995],
        "time_decay_half_life_days": 756,
        "threads": 1,
    },
    "seeds": list(SEEDS),
    "purge_horizon": PURGE_HORIZON,
    "folds": 4,
    "account_money": 100000.0,
    "action": {
        "name": "action_3state",
        "levels": [0.0, 0.5, 1.0],
        "initial_state": 0.0,
        "threshold_quantiles": [0.30, 0.70],
        "hysteresis_fraction": 2.0,
        "max_participation": 0.01,
        "commission_rate": 0.00025,
        "min_commission": 5.0,
        "transfer_rate": 0.00001,
        "stamp_sell_rate": 0.0005,
        "slippage_rate": 0.0003,
        "lot_size": 100,
    },
    "test_start": "2025-07-01",
    "test_end": "2026-06-30",
    "data_built_at": None,
}


def fixed_files():
    """The unit's complete, fixed 159-file contract."""
    # V2 references immutable V1 model artifacts; only its own code and outputs live here.
    out = {"run.py", "model.py", "analysis.py"}
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
    assert len(out) == 15, len(out)
    return out


TOOL_DIRS = {".ipynb_checkpoints", "__pycache__"}


def _tool_path(rel):
    return any(part in TOOL_DIRS for part in Path(rel).parts)


def validate_layout(*, allow_missing=True):
    expected = fixed_files()
    actual_files, actual_dirs = set(), set()
    for path in ROOT.rglob("*"):
        rel = path.relative_to(ROOT).as_posix()
        if _tool_path(rel):
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
            "missing_files": len(missing), "extra_files": extra_files, "extra_dirs": extra_dirs}


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")
    os.replace(tmp, path)


def atomic_npy(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        np.save(f, value)
    os.replace(tmp, path)


def atomic_pickle(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(value, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def metadata():
    return json.loads((DATA_ROOT / "meta.json").read_text(encoding="utf-8"))


def _read_table(kind, year, columns):
    path = DATA_ROOT / kind / f"year={year}" / "data.parquet"
    if not path.is_file():
        raise FileNotFoundError(path)
    return pq.read_table(path, columns=list(columns),
                         filters=[("stock_code", "in", list(CODES))]).to_pandas()


class Prices:
    def __init__(self, frame, days, codes):
        shape = (len(days), len(codes))
        self.raw = {name: frame[name].to_numpy(np.float64).reshape(shape) for name in PRICE_COLUMNS}
        self.open = self.raw["open"]
        self.close = self.raw["close"]
        self.adj = self.raw["adj_factor"].copy()
        self.adj = self._forward_fill(self.adj)
        self.raw["adj_factor"] = self.adj
        base = np.isfinite(self.open) & (self.open > 0)
        vol = np.nan_to_num(self.raw["vol"], nan=0.0)
        self.entry = base & (vol > 0)
        self.exit = base & (vol > 0)
        prev = self.raw["pre_close"]
        high = self.raw["high"]
        low = self.raw["low"]
        one_price = np.isfinite(high) & np.isfinite(low) & np.isclose(high, low, rtol=0, atol=1e-8)
        up_lock = one_price & np.isfinite(prev) & (self.open >= prev * 1.095)
        down_lock = one_price & np.isfinite(prev) & (self.open <= prev * 0.905)
        self.entry &= ~up_lock
        self.exit &= ~down_lock

    @staticmethod
    def _forward_fill(values):
        out = values.copy()
        for c in range(out.shape[1]):
            good = np.flatnonzero(np.isfinite(out[:, c]) & (out[:, c] > 0))
            if not len(good):
                continue
            first = good[0]
            out[:first, c] = out[first, c]
            for i in range(first + 1, out.shape[0]):
                if not np.isfinite(out[i, c]) or out[i, c] <= 0:
                    out[i, c] = out[i - 1, c]
        return out


class Panel:
    pass


def load_panel():
    meta = metadata()
    factor_years = sorted(int(y) for y in meta["built_years"])
    schema = pq.ParquetFile(DATA_ROOT / "factors" / f"year={factor_years[0]}" / "data.parquet").schema_arrow.names
    factor_cols = [c for c in schema if c not in ("trade_date", "stock_code")]
    price_schema = pq.ParquetFile(DATA_ROOT / "prices" / f"year={factor_years[0]}" / "data.parquet").schema_arrow.names
    amount_schema = pq.ParquetFile(DATA_ROOT / "amount" / f"year={factor_years[0]}" / "data.parquet").schema_arrow.names
    amount_col = next((c for c in amount_schema if c not in ("trade_date", "stock_code")), None)
    if amount_col is None:
        raise ValueError("amount 块没有成交额列")
    factor_set = set(factor_cols)
    market_schema = pq.ParquetFile(DATA_ROOT / "market_factors" / f"year={factor_years[0]}" / "data.parquet").schema_arrow.names
    market_cols = [c for c in market_schema if c != "trade_date"]
    # 保证市场列与股票因子同名时仍有显式、稳定的特征名。
    market_names = [c if c not in factor_set else "market__" + c for c in market_cols]
    pieces = []
    for yi, year in enumerate(factor_years, 1):
        f = _read_table("factors", year, ["trade_date", "stock_code", *factor_cols])
        y = _read_table("target", year, ["trade_date", "stock_code", *ALL_LABELS])
        a = _read_table("amount", year, ["trade_date", "stock_code", amount_col])
        p = _read_table("prices", year, ["trade_date", "stock_code", *PRICE_COLUMNS])
        mpath = DATA_ROOT / "market_factors" / f"year={year}" / "data.parquet"
        m = pq.read_table(mpath, columns=["trade_date", *market_cols]).to_pandas()
        m = m.rename(columns=dict(zip(market_cols, market_names)))
        df = f.merge(y, on=["trade_date", "stock_code"], validate="one_to_one")
        df = df.merge(a, on=["trade_date", "stock_code"], validate="one_to_one")
        df = df.merge(p, on=["trade_date", "stock_code"], validate="one_to_one")
        df = df.merge(m, on="trade_date", validate="many_to_one")
        pieces.append(df)
        print(f"载入 {year}：{len(df):,} 行（{yi}/{len(factor_years)}）", flush=True)
    frame = pd.concat(pieces, ignore_index=True)
    frame["trade_date"] = frame["trade_date"].astype(str)
    frame["stock_code"] = frame["stock_code"].astype(str)
    days = np.asarray(sorted(frame["trade_date"].unique()), dtype=str)
    codes = np.asarray(CODES, dtype=str)
    idx = pd.MultiIndex.from_product([days, codes], names=["trade_date", "stock_code"])
    frame = frame.set_index(["trade_date", "stock_code"]).reindex(idx)
    if frame.index.has_duplicates or len(frame) != len(days) * len(codes):
        raise ValueError("四大银行因子/标签/价格轴不完整或重复")
    panel = Panel()
    panel.days, panel.codes = days, codes
    panel.meta = meta
    panel.factor_features = tuple(factor_cols)
    panel.market_features = tuple(market_names)
    panel.features = tuple(factor_cols + market_names + [f"asset_{i}" for i in range(len(codes))])
    numeric = frame[factor_cols + market_names].to_numpy(np.float32)
    numeric = np.nan_to_num(numeric, nan=0.0, posinf=0.0, neginf=0.0)
    code_id = np.tile(np.arange(len(codes)), len(days))
    onehot = np.eye(len(codes), dtype=np.float32)[code_id]
    panel.X = np.column_stack((numeric, onehot)).reshape(len(days), len(codes), -1)
    panel.Y = {name: frame[name].to_numpy(np.float64).reshape(len(days), len(codes))
               for name in ALL_LABELS}
    panel.amount = frame[amount_col].to_numpy(np.float64).reshape(len(days), len(codes))
    panel.prices = Prices(frame.reset_index(), days, codes)
    panel.amount = np.nan_to_num(panel.amount, nan=0.0, posinf=0.0, neginf=0.0)
    panel.amount = np.maximum(panel.amount, 0.0)
    panel.data_built_at = meta.get("built_at")
    if not np.isfinite(panel.X).all():
        raise ValueError("特征矩阵在缺失填零后仍含非有限值")
    print(f"面板就绪：{len(days)} 日 × {len(codes)} 票 × {panel.X.shape[-1]} 特征；"
          f"快照 {panel.data_built_at}", flush=True)
    return panel


def axis():
    panel = load_panel()
    return panel.days, panel.codes


def quarters(days=None):
    return list(QUARTERS)


def _quarter_bounds(quarter):
    period = pd.Period(quarter, freq="Q")
    return period.start_time.strftime("%Y-%m-%d"), period.end_time.strftime("%Y-%m-%d")


def splits(days, purge_horizon=PURGE_HORIZON, folds=4, quarter=None):
    quarter = quarter or QUARTERS[0]
    start, end = _quarter_bounds(quarter)
    days = np.asarray(days, dtype=str)
    first = int(np.searchsorted(days, start))
    cutoff = first - int(purge_horizon) - 1
    eligible = np.arange(max(cutoff, 0), dtype=np.int64)
    test = np.flatnonzero((days >= max(start, RECIPE["test_start"])) &
                          (days <= min(end, RECIPE["test_end"])))
    score = np.arange(first, len(days), dtype=np.int64)
    if not len(eligible) or not len(test) or not len(score):
        raise ValueError(f"{quarter} 的 train/test/score 为空")
    out = []
    for fold, block in enumerate(np.array_split(eligible, folds), 1):
        if not len(block):
            raise ValueError(f"{quarter} fold{fold} valid 为空")
        a, b = int(block[0]), int(block[-1]) + 1
        valid = block[block + purge_horizon + 1 < b]
        train = eligible[(eligible + purge_horizon + 1 < a) |
                         (eligible >= b + purge_horizon + 1)]
        if not len(train) or not len(valid):
            raise ValueError(f"{quarter} fold{fold} train/valid 为空")
        assert train.max() + purge_horizon + 1 < first
        assert test[0] == score[0]
        out.append({"quarter": quarter, "fold": fold, "train": train,
                    "valid": valid, "test": test, "score": score})
    return out


def split_summary(days, split):
    def spans(ix):
        ix = np.asarray(ix, dtype=np.int64)
        if not len(ix):
            return []
        chunks = np.split(ix, np.flatnonzero(np.diff(ix) > 1) + 1)
        return [{"start": str(days[c[0]]), "end": str(days[c[-1]]), "days": len(c)} for c in chunks]
    return {key: spans(split[key]) for key in ("train", "valid", "test", "score")} | {
        "quarter": split["quarter"], "fold": split["fold"],
        "purge_horizon": PURGE_HORIZON, "train_ends_before_test": True}


def _sample_weight(day_ids, train_days):
    age = (int(np.max(train_days)) - day_ids).astype(np.float64)
    half_life = float(RECIPE["model"]["time_decay_half_life_days"])
    return np.power(0.5, age / half_life).astype(np.float32)


def _flatten_mask(day_ids, label):
    return np.repeat(np.isin(np.arange(len(day_ids)), label), len(CODES))


def _predict_days(model, X, days):
    values = X[np.asarray(days, dtype=np.int64)]
    return model.predict(values.reshape(-1, values.shape[-1]), num_threads=1).reshape(len(days), len(CODES))


def _corr(a, b):
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 4 or np.std(a[ok]) < 1e-12 or np.std(b[ok]) < 1e-12:
        return None
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def _thresholds(score_valid, codes, qlo, qhi, band_fraction):
    out = {}
    for j, code in enumerate(codes):
        v = score_valid[:, j]
        v = v[np.isfinite(v)]
        if len(v) < 10:
            lo, hi = -0.25, 0.25
        else:
            lo, hi = [float(x) for x in np.quantile(v, [qlo, qhi])]
            if hi - lo < 1e-6:
                mid = float(np.median(v))
                lo, hi = mid - 0.25, mid + 0.25
        band = max((hi - lo) * band_fraction, 0.0)
        out[str(code)] = {"lo": lo, "hi": hi, "band": band}
    return out


def _threshold_variants(score_valid, codes):
    variants = {}
    for qlo, qhi, label in ((0.20, 0.80, "q20_80"),
                             (0.30, 0.70, "q30_70"),
                             (0.40, 0.60, "q40_60")):
        for band in (0.0, 0.10, 0.25, 2.0):
            name = f"{label}_b{int(round(band * 100)):02d}"
            variants[name] = _thresholds(score_valid, codes, qlo, qhi, band)
    return variants


def train_fold(panel, quarter, fold, *, force=False):
    out = ROOT / "model_train" / quarter / f"fold{fold}"
    done = out / "complete.json"
    if done.is_file() and not force:
        result = json.loads(done.read_text(encoding="utf-8"))
        if result.get("recipe_name") == RECIPE["name"] and result.get("data_built_at") == panel.data_built_at:
            print(f"{quarter} fold{fold} 已完成并匹配当前配方/快照，跳过", flush=True)
            return result
        raise RuntimeError(f"{done} 已存在但版本或快照不匹配；请用 --force 重训")
    out.mkdir(parents=True, exist_ok=True)
    split = splits(panel.days, PURGE_HORIZON, 4, quarter)[fold - 1]
    atomic_json(out / "training_info.json", {
        "recipe": RECIPE, "data_built_at": panel.data_built_at,
        "features": {"factor_count": len(panel.factor_features),
                     "market_count": len(panel.market_features),
                     "total_count": len(panel.features)},
        "python": sys.version.split()[0], "lightgbm": lgb.__version__,
        "device": "cpu", "threads": 1,
    })
    atomic_json(out / "split.json", split_summary(panel.days, split))
    x = panel.X.reshape(-1, panel.X.shape[-1])
    train_mask_days = _flatten_mask(panel.days, split["train"])
    valid_mask_days = _flatten_mask(panel.days, split["valid"])
    score_days = split["score"]
    weights_by_day = _sample_weight(np.arange(len(panel.days), dtype=np.int64), split["train"])
    weights = np.repeat(weights_by_day, len(panel.codes))
    model_config = RECIPE["model"]
    bundles, seed_scores, seed_thresholds, history = [], [], [], []
    started = time.time()
    for seed in SEEDS:
        trained = {}
        valid_preds = {}
        score_preds = {}
        seed_record = {"seed": seed, "labels": {}}
        for label in LABELS:
            yall = panel.Y[label].reshape(-1)
            train_mask = train_mask_days & np.isfinite(yall)
            valid_mask = valid_mask_days & np.isfinite(yall)
            if int(train_mask.sum()) < 1000 or int(valid_mask.sum()) < 100:
                raise ValueError(f"{quarter} fold{fold} {label} 有效样本过少")
            ytrain = yall[train_mask]
            ql, qh = model_config["label_clip_quantiles"]
            clip_lo, clip_hi = np.quantile(ytrain, [ql, qh])
            yfit = np.clip(ytrain, clip_lo, clip_hi)
            reg = lgb.LGBMRegressor(
                objective="huber", alpha=0.9, n_estimators=model_config["n_estimators"],
                learning_rate=model_config["learning_rate"], num_leaves=model_config["num_leaves"],
                max_depth=model_config["max_depth"], min_child_samples=model_config["min_child_samples"],
                colsample_bytree=model_config["feature_fraction"], subsample=model_config["bagging_fraction"],
                subsample_freq=model_config["bagging_freq"], reg_alpha=model_config["reg_alpha"],
                reg_lambda=model_config["reg_lambda"], random_state=seed,
                n_jobs=1, verbosity=-1, deterministic=True, force_col_wise=True,
            )
            reg.fit(x[train_mask], yfit, sample_weight=weights[train_mask])
            trained[label] = reg
            pv = _predict_days(reg, panel.X, split["valid"])
            ps = _predict_days(reg, panel.X, score_days)
            valid_preds[label], score_preds[label] = pv, ps
            label_ix = LABELS.index(label)
            yv = panel.Y[label][split["valid"]]
            seed_record["labels"][label] = {
                "valid_pearson": _corr(pv.ravel(), yv.ravel()),
                "valid_days": int(np.isfinite(yv).any(axis=1).sum()),
                "n_train": int(train_mask.sum()),
                "label_clip": [float(clip_lo), float(clip_hi)],
                "feature_importance_top": [
                    {"feature": str(panel.features[k]), "gain": round(float(v), 4)}
                    for k, v in sorted(enumerate(reg.booster_.feature_importance(importance_type="gain")),
                                       key=lambda pair: -pair[1])[:12] if v > 0
                ],
            }
            del yv
        mean1 = float(np.nanmean(valid_preds[LABELS[0]]))
        mean5 = float(np.nanmean(valid_preds[LABELS[1]]))
        raw_sd1 = float(np.nanstd(valid_preds[LABELS[0]]))
        raw_sd5 = float(np.nanstd(valid_preds[LABELS[1]]))
        # A constant 1d regressor carries no ranking information; center it to neutral
        # instead of amplifying its intercept by dividing by a near-zero standard deviation.
        sd1 = raw_sd1 if raw_sd1 >= 1e-6 else 1.0
        sd5 = raw_sd5 if raw_sd5 >= 1e-6 else 1.0
        w1, w5 = RECIPE["score_blend"][LABELS[0]], RECIPE["score_blend"][LABELS[1]]
        valid_score = w1 * (valid_preds[LABELS[0]] - mean1) / sd1 + w5 * (valid_preds[LABELS[1]] - mean5) / sd5
        future_score = w1 * (score_preds[LABELS[0]] - mean1) / sd1 + w5 * (score_preds[LABELS[1]] - mean5) / sd5
        threshold_variants = _threshold_variants(valid_score, panel.codes)
        thresholds = threshold_variants["q30_70_b200"]
        seed_scores.append(future_score.astype(np.float32))
        seed_thresholds.append(threshold_variants)
        bundles.append({"seed": seed, "models": trained,
                        "mean": {LABELS[0]: mean1, LABELS[1]: mean5},
                        "sd": {LABELS[0]: sd1, LABELS[1]: sd5},
                        "thresholds": thresholds, "threshold_variants": threshold_variants,
                        "features": list(panel.features)})
        seed_record["score_scale"] = {"mean_1d": mean1, "mean_5d": mean5,
                                      "sd_1d": sd1, "sd_5d": sd5}
        seed_record["thresholds"] = thresholds
        history.append(seed_record)
        ic1, ic5 = (seed_record["labels"][label]["valid_pearson"] for label in LABELS)
        show1 = "—" if ic1 is None else f"{ic1:.4f}"
        show5 = "—" if ic5 is None else f"{ic5:.4f}"
        print(f"{quarter} fold{fold} seed={seed} validIC1={show1} validIC5={show5}", flush=True)
        del trained, valid_preds, score_preds, reg
    seed_predictions = np.stack(seed_scores, axis=0)
    ensemble = np.nanmean(seed_predictions, axis=0).astype(np.float32)
    atomic_npy(out / "score_predictions.npy", ensemble)
    atomic_npy(out / "test_predictions.npy", ensemble[:len(split["test"])])
    payload = {
        "recipe_name": RECIPE["name"], "data_built_at": panel.data_built_at,
        "quarter": quarter, "fold": fold, "features": list(panel.features),
        "bundles": bundles, "seed_predictions": seed_predictions,
        "score_dates": panel.days[score_days].tolist(),
        "test_dates": panel.days[split["test"]].tolist(),
        "valid_thresholds": seed_thresholds,
    }
    atomic_pickle(out / "best.pt", payload)
    atomic_pickle(out / "last.pt", payload)
    atomic_json(out / "history.json", history)
    rss_gib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2)
    result = {
        "recipe_name": RECIPE["name"], "data_built_at": panel.data_built_at,
        "quarter": quarter, "fold": fold, "seeds": list(SEEDS),
        "train_days": int(len(split["train"])), "valid_days": int(len(split["valid"])),
        "test_days": int(len(split["test"])), "score_days": int(len(score_days)),
        "train_start": str(panel.days[split["train"][0]]),
        "train_end": str(panel.days[split["train"][-1]]),
        "valid_start": str(panel.days[split["valid"][0]]),
        "valid_end": str(panel.days[split["valid"][-1]]),
        "test_start": str(panel.days[split["test"][0]]),
        "test_end": str(panel.days[split["test"][-1]]),
        "clean_forward_test": bool(panel.days[split["train"][-1]] < panel.days[split["test"][0]]),
        "valid_by_seed": history, "peak_rss_gib": round(float(rss_gib), 3),
        "seconds": round(time.time() - started, 2),
    }
    atomic_json(done, result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quarter", choices=QUARTERS)
    parser.add_argument("--fold", type=int, choices=FOLDS)
    parser.add_argument("--device", choices=("auto", "cpu"), default="cpu")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--layout-only", action="store_true")
    args = parser.parse_args(argv)
    if args.layout_only:
        print(json.dumps(validate_layout(), ensure_ascii=False, indent=2))
        return
    if (args.quarter is None) != (args.fold is None):
        parser.error("--quarter 与 --fold 要么同时指定，要么都不指定")
    panel = load_panel()
    tasks = [(args.quarter, args.fold)] if args.quarter else [
        (q, f) for q in QUARTERS for f in FOLDS]
    for quarter, fold in tasks:
        log = ROOT / "model_logs" / f"{quarter}_{fold}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        if args.quarter:
            train_fold(panel, quarter, fold, force=args.force)
        else:
            train_fold(panel, quarter, fold, force=args.force)


if __name__ == "__main__":
    main()

