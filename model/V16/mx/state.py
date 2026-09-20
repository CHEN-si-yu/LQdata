"""状态落盘 —— 原子写 JSON（与模块①② 同一套做法：tmp + os.replace + 同组可读写）。"""
from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime
from pathlib import Path


def save_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=path.name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=1, default=_default)
        try:
            os.chmod(tmp, 0o664)
        except OSError:
            pass
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _default(o):
    """把 numpy 标量/数组转成可序列化的东西（训练指标里到处是 np.float32）。"""
    try:
        import numpy as np
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
    except Exception:            # noqa: BLE001
        pass
    return str(o)


def load_json(path: Path, default=None):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ---------------------------------------------------------------- 单元（V{N}）路径
# ★ 单元顶层只有三类产物桶（2026-09-20 收敛：用户要求「逻辑划分清晰一点」）：
#
#   Model/<单元>/model_train/   训练过程产物（逐折权重 / 指标 / 逐折打分 / 折汇总）
#   Model/<单元>/model_pred/    回测相关结果（集成打分 / 榜单 / 回测 JSON / 净值图）
#   Model/<单元>/logs/          记录文件（逐折日志 / 冻结配方）
#
#   代码另有三份，**不是产物**、不占桶：`main.py`、`model.py`（本版配方）、`mx/`（引擎）。
#
# ★ 冒烟/小样本**嵌进桶内**（`model_train/smoke`、`model_pred/sample`、`logs/smoke`），
#   不再在顶层平铺 `smoke_pred/ sample_state/` 这类兄弟目录。
#   分树本身必须保留：`--sample` 是**快看链路**用的，500 只票训出来的模型绝不该覆盖
#   正式 `model_train/fold*/model.pkl`，更不该覆盖 `model_pred/leaderboard.json`
#   —— `promote` 读的正是那份榜单。只是换个位置摆，不平铺在顶层。
BUCKET_TRAIN = "model_train"
BUCKET_PRED = "model_pred"
BUCKET_LOGS = "logs"


def _uname(unit) -> str:
    """单元路径参数既能接名字（"V1"）也能接 `Unit` 对象 —— 少一类低级错误。"""
    return str(getattr(unit, "name", unit))


def _flavor(smoke: bool = False, sample: bool = False) -> str:
    """产物分树：正式（""）/ 冒烟（"smoke"）/ 小样本（"sample"）。样本优先于冒烟。"""
    return "sample" if sample else ("smoke" if smoke else "")


def _bucket(cfg, unit, bucket: str, smoke: bool = False, sample: bool = False) -> Path:
    base = cfg.unit_dir(_uname(unit)) / bucket
    f = _flavor(smoke, sample)
    return base / f if f else base


def unit_train_dir(cfg, unit: str, smoke: bool = False, sample: bool = False) -> Path:
    """训练产物桶 —— 逐折权重/指标/打分 + 折汇总。"""
    return _bucket(cfg, unit, BUCKET_TRAIN, smoke, sample)


def unit_pred_dir(cfg, unit: str, smoke: bool = False, sample: bool = False) -> Path:
    """回测相关结果桶 —— 集成打分 / 榜单 / 回测 JSON / 净值图。

    ★ 打分与「回测的结果」同桶：`score__*` 是回测的**输入**，`leaderboard.json`、
      净值图、`backtest__*.json` 是它的**输出**。分开摆只会让「跑完 analysis 该去哪看」
      多一次跳转 —— 用户要的是逻辑清晰，不是目录多。
    """
    return _bucket(cfg, unit, BUCKET_PRED, smoke, sample)


def unit_logs_dir(cfg, unit: str, smoke: bool = False, sample: bool = False) -> Path:
    """记录桶 —— 逐折日志 + 冻结配方。"""
    return _bucket(cfg, unit, BUCKET_LOGS, smoke, sample)


def unit_fold_dir(cfg, unit: str, fold: int, smoke: bool = False, sample: bool = False) -> Path:
    return unit_train_dir(cfg, unit, smoke, sample) / f"fold{fold}"


def unit_model_path(cfg, unit: str, fold: int, model: str, label: str,
                    smoke: bool = False, sample: bool = False) -> Path:
    return unit_fold_dir(cfg, unit, fold, smoke, sample) / f"{model}__{label}" / "model.pkl"


def unit_eval_path(cfg, unit: str, fold: int, model: str, label: str,
                   smoke: bool = False, sample: bool = False) -> Path:
    return unit_fold_dir(cfg, unit, fold, smoke, sample) / f"{model}__{label}" / "eval.json"


def unit_fold_score_path(cfg, unit: str, fold: int, model: str, label: str,
                         smoke: bool = False, sample: bool = False) -> Path:
    """某折某头的**整面板**打分（4 列契约，按年分区目录）—— analysis 的输入。

    ★ 返回的是**目录**（`score__<头>__<标签>/year=YYYY/data.parquet`），与模块② 的
      因子产物同构；不要给它加 `.parquet` 后缀（那是文件语义，两边会对不上）。
    """
    return unit_fold_dir(cfg, unit, fold, smoke, sample) / f"score__{model}__{label}"


def unit_pics_dir(cfg, unit: str, smoke: bool = False, sample: bool = False) -> Path:
    """净值图等 —— 直接落进 `model_pred/`，不另开桶（图是回测的输出）。"""
    return unit_pred_dir(cfg, unit, smoke, sample)


def unit_summary_path(cfg, unit: str, smoke: bool = False, sample: bool = False) -> Path:
    """单元总账 —— 与榜单同桶（都是分析层产物，`status` 会先后读它们）。"""
    return unit_pred_dir(cfg, unit, smoke, sample) / "unit_summary.json"


def unit_leaderboard_path(cfg, unit: str, smoke: bool = False, sample: bool = False) -> Path:
    return unit_pred_dir(cfg, unit, smoke, sample) / "leaderboard.json"


def unit_fold_summary_path(cfg, unit: str, fold: int, smoke: bool = False,
                           sample: bool = False) -> Path:
    """折汇总 —— 训练层产物，与 `fold<k>/` 同级摆在 `model_train/` 里。"""
    return unit_train_dir(cfg, unit, smoke, sample) / f"fold{fold}_summary.json"


def unit_recipe_path(cfg, unit: str) -> Path:
    """冻结配方 —— 本质是「一条记录」，所以进 `logs/`（读写见 mx/unit.py）。"""
    return unit_logs_dir(cfg, unit) / "recipe.yaml"


def unit_ensure_dirs(cfg, unit: str, smoke: bool = False, sample: bool = False) -> None:
    for p in (unit_train_dir(cfg, unit, smoke, sample), unit_pred_dir(cfg, unit, smoke, sample),
              unit_logs_dir(cfg, unit, smoke, sample)):
        p.mkdir(parents=True, exist_ok=True)
