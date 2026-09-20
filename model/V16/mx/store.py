"""产物读写 —— **与模块② 完全同一个契约**。

为什么坚持同一个 4 列契约（`trade_date/stock_code/value/rank`，string/string/float32/float32）：
  ① 模块② 的 MD5 台账、可视化、审计工具能直接读模型产物（"预测的历史不变性"同样可证）；
  ② 下游/回测/组合都只需要认一种格式；
  ③ 出问题时可以用模块② 的读层交叉验证。

单元的**打分**产物把 `value` 解释为 score、`rank` 仍是当日截面百分位 [0,1]，
落点由单元目录决定（`state.unit_fold_score_path` / `state.unit_pred_dir`），
**不再有全局的 `data/predictions` 落点**（2026-09-18 重构移除）。
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

COLUMNS = ("trade_date", "stock_code", "value", "rank")
DTYPES = {"trade_date": "string", "stock_code": "string",
          "value": "float32", "rank": "float32"}


def _assert_schema(df: pd.DataFrame) -> None:
    if tuple(df.columns) != COLUMNS:
        raise ValueError(f"列不是 {COLUMNS}，而是 {tuple(df.columns)}")
    for c, want in DTYPES.items():
        if str(df[c].dtype) != want:
            raise ValueError(f"列 {c} 的 dtype 是 {df[c].dtype}，契约要求 {want}")


def _atomic_write(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp, engine="pyarrow", compression="zstd", index=False)
    try:
        os.chmod(tmp, 0o664)          # 共享盘：claude 与 root 两个用户都要能读写
    except OSError:
        pass
    os.replace(tmp, path)


def cs_rank_from_score(dates: pd.Series, score: pd.Series) -> pd.Series:
    """打分 → 当日截面百分位 [0,1]（与模块② 的 `rank` 语义一致：并列取平均名次）。

    ★ 只按**当日**分组 —— 这是"只用当日截面统计量"红线的具体体现。
    """
    s = pd.DataFrame({"d": dates.astype(str).str[:10], "v": score})
    return s.groupby("d")["v"].rank(pct=True, na_option="keep").astype("float32")


# ---------------------------------------------------------------- 任意路径的读写（单元产物用）
def write_frame(path: Path, df: pd.DataFrame) -> Path:
    """把 4 列契约的 DataFrame 原子写到一个**显式路径**（按年分区）。

    单元（`Model/V{N}`）的打分产物用它 —— 路径由单元决定，不再走 `data/predictions/`。
    """
    _assert_schema(df)
    path = Path(path)
    if path.suffix == ".parquet":
        path = path.with_suffix("")            # 传了文件名就当目录名用
    for y, sub in df.groupby(df["trade_date"].astype(str).str[:4], sort=True):
        _atomic_write(sub.reset_index(drop=True), path / f"year={y}" / "data.parquet")
    return path


def read_frame(path: Path) -> pd.DataFrame:
    """读回 `write_frame` 写出的产物（4 列契约）。"""
    d = Path(path)
    if d.is_file():
        parts = [pd.read_parquet(d)]
    else:
        parts = [pd.read_parquet(p) for p in sorted(d.glob("year=*/data.parquet"))]
    if not parts:
        return pd.DataFrame(columns=list(COLUMNS))
    df = pd.concat(parts, ignore_index=True)
    df["trade_date"] = df["trade_date"].astype(str).str[:10]
    return df.sort_values(["trade_date", "stock_code"], kind="stable").reset_index(drop=True)
