import pandas as pd
from typing import List, Dict
from tqdm import tqdm
import numpy as np
import pandas as pd
from typing import List, Dict, Optional

# ============= 1) 假设节假日模板（示例 & 填写指引） =============
# 你只需要维护这个 HOLIDAYS 列表：每一段放假区间是一条记录。
# name: 统一的英文蛇形命名（列名会基于它生成），start/end: 使用 "YYYY-MM-DD" 字符串。
# 注意：这里的 start/end 是实际放假的连续自然日区间（调休后形成的连休也当作连续区间写进去）。
HOLIDAYS: List[Dict[str, str]] = [
{"name": "new_year",            "start": "2015-01-01", "end": "2015-01-03"},
{"name": "spring_festival",     "start": "2015-02-18", "end": "2015-02-24"},
{"name": "qingming",            "start": "2015-04-04", "end": "2015-04-06"},
{"name": "labor_day",           "start": "2015-05-01", "end": "2015-05-03"},
{"name": "dragon_boat",         "start": "2015-06-20", "end": "2015-06-22"},
{"name": "mid_autumn",          "start": "2015-09-26", "end": "2015-09-27"},
{"name": "national_day",        "start": "2015-10-01", "end": "2015-10-07"},
{"name": "new_year",            "start": "2016-01-01", "end": "2016-01-03"},
{"name": "spring_festival",     "start": "2016-02-07", "end": "2016-02-13"},
{"name": "qingming",            "start": "2016-04-02", "end": "2016-04-04"},
{"name": "labor_day",           "start": "2016-04-30", "end": "2016-05-02"},
{"name": "dragon_boat",         "start": "2016-06-09", "end": "2016-06-11"},
{"name": "mid_autumn",          "start": "2016-09-15", "end": "2016-09-17"},
{"name": "national_day",        "start": "2016-10-01", "end": "2016-10-07"},
{"name": "new_year",            "start": "2017-01-01", "end": "2017-01-02"},
{"name": "spring_festival",     "start": "2017-01-27", "end": "2017-02-02"},
{"name": "qingming",            "start": "2017-04-02", "end": "2017-04-04"},
{"name": "labor_day",           "start": "2017-04-29", "end": "2017-05-01"},
{"name": "dragon_boat",         "start": "2017-05-28", "end": "2017-05-30"},
{"name": "mid_autumn",          "start": "2017-10-04", "end": "2017-10-04"},
{"name": "national_day",        "start": "2017-10-01", "end": "2017-10-08"},
{"name": "new_year",            "start": "2018-01-01", "end": "2018-01-01"},
{"name": "spring_festival",     "start": "2018-02-15", "end": "2018-02-21"},
{"name": "qingming",            "start": "2018-04-05", "end": "2018-04-07"},
{"name": "labor_day",           "start": "2018-04-29", "end": "2018-05-01"},
{"name": "dragon_boat",         "start": "2018-06-16", "end": "2018-06-18"},
{"name": "mid_autumn",          "start": "2018-09-22", "end": "2018-09-24"},
{"name": "national_day",        "start": "2018-10-01", "end": "2018-10-07"},
{"name": "new_year",            "start": "2018-12-30", "end": "2019-01-01"},
{"name": "spring_festival",     "start": "2019-02-04", "end": "2019-02-10"},
{"name": "qingming",            "start": "2019-04-05", "end": "2019-04-07"},
{"name": "labor_day",           "start": "2019-05-01", "end": "2019-05-04"},
{"name": "dragon_boat",         "start": "2019-06-07", "end": "2019-06-09"},
{"name": "mid_autumn",          "start": "2019-09-13", "end": "2019-09-15"},
{"name": "national_day",        "start": "2019-10-01", "end": "2019-10-07"},
{"name": "new_year",            "start": "2020-01-01", "end": "2020-01-01"},
{"name": "spring_festival",     "start": "2020-01-24", "end": "2020-02-02"}, # 2020-01-27 通知将假期延长至 2/02
{"name": "qingming",            "start": "2020-04-04", "end": "2020-04-06"},
{"name": "labor_day",           "start": "2020-05-01", "end": "2020-05-05"},
{"name": "dragon_boat",         "start": "2020-06-25", "end": "2020-06-27"},
{"name": "mid_autumn",          "start": "2020-10-01", "end": "2020-10-01"}, # 与国庆叠加
{"name": "national_day",        "start": "2020-10-01", "end": "2020-10-08"},
{"name": "new_year",            "start": "2021-01-01", "end": "2021-01-03"},
{"name": "spring_festival",     "start": "2021-02-11", "end": "2021-02-17"},
{"name": "qingming",            "start": "2021-04-03", "end": "2021-04-05"},
{"name": "labor_day",           "start": "2021-05-01", "end": "2021-05-05"},
{"name": "dragon_boat",         "start": "2021-06-12", "end": "2021-06-14"},
{"name": "mid_autumn",          "start": "2021-09-19", "end": "2021-09-21"},
{"name": "national_day",        "start": "2021-10-01", "end": "2021-10-07"},
{"name": "new_year",            "start": "2022-01-01", "end": "2022-01-03"},
{"name": "spring_festival",     "start": "2022-01-31", "end": "2022-02-06"},
{"name": "qingming",            "start": "2022-04-03", "end": "2022-04-05"},
{"name": "labor_day",           "start": "2022-04-30", "end": "2022-05-04"},
{"name": "dragon_boat",         "start": "2022-06-03", "end": "2022-06-05"},
{"name": "mid_autumn",          "start": "2022-09-10", "end": "2022-09-12"},
{"name": "national_day",        "start": "2022-10-01", "end": "2022-10-07"},
{"name": "new_year",            "start": "2022-12-31", "end": "2023-01-02"},
{"name": "spring_festival",     "start": "2023-01-21", "end": "2023-01-27"},
{"name": "qingming",            "start": "2023-04-05", "end": "2023-04-05"},
{"name": "labor_day",           "start": "2023-04-29", "end": "2023-05-03"},
{"name": "dragon_boat",         "start": "2023-06-22", "end": "2023-06-24"},
{"name": "mid_autumn",          "start": "2023-09-29", "end": "2023-09-29"}, # 与国庆连休
{"name": "national_day",        "start": "2023-09-29", "end": "2023-10-06"},
{"name": "new_year",            "start": "2023-12-30", "end": "2024-01-01"},
{"name": "spring_festival",     "start": "2024-02-10", "end": "2024-02-17"}, # 除夕(2/09)鼓励休息但非统一放假
{"name": "qingming",            "start": "2024-04-04", "end": "2024-04-06"},
{"name": "labor_day",           "start": "2024-05-01", "end": "2024-05-05"},
{"name": "dragon_boat",         "start": "2024-06-08", "end": "2024-06-10"},
{"name": "mid_autumn",          "start": "2024-09-15", "end": "2024-09-17"},
{"name": "national_day",        "start": "2024-10-01", "end": "2024-10-07"},
{"name": "new_year",            "start": "2025-01-01", "end": "2025-01-01"},
{"name": "spring_festival",     "start": "2025-01-28", "end": "2025-02-04"}, # 新规自除夕起放 8 天
{"name": "qingming",            "start": "2025-04-04", "end": "2025-04-06"},
{"name": "labor_day",           "start": "2025-05-01", "end": "2025-05-05"},
{"name": "dragon_boat",         "start": "2025-05-31", "end": "2025-06-02"},
{"name": "mid_autumn",          "start": "2025-10-06", "end": "2025-10-06"}, # 与国庆叠加
{"name": "national_day",        "start": "2025-10-01", "end": "2025-10-08"},
]

# ============= 2) 工具函数 =============

def _to_dt_yyyymmdd(x) -> pd.Timestamp:
    """将 20150106 / '20150106' 转为 Timestamp；不能解析则返回 NaT。"""
    if pd.isna(x):
        return pd.NaT
    s = str(int(x)) if isinstance(x, (int, float)) and not pd.isna(x) else str(x)
    # 去掉可能的 '.0' 等
    s = s.split('.')[0]
    try:
        return pd.to_datetime(s, format="%Y%m%d", errors="coerce")
    except Exception:
        return pd.NaT

def _safe_to_timestamp(s: str) -> pd.Timestamp:
    return pd.to_datetime(s, format="%Y-%m-%d")

def _normalize_holiday_name(name: str) -> str:
    """将节名转为列名后缀，保持干净统一。"""
    return name.strip().lower().replace(" ", "_")


# 可选：并行
try:
    from joblib import Parallel, delayed
    _HAS_JOBLIB = True
except Exception:
    _HAS_JOBLIB = False

print(f"[INFO] joblib 可用：{_HAS_JOBLIB}")

def make_holiday_features_fast(
    df: pd.DataFrame,
    holidays: List[Dict[str, str]],
    pre_days: int = 5,
    post_days: int = 5,
    date_col: str = "date",
    n_jobs: Optional[int] = None,   # None 或 1 表示不并行；>1 表示并行
) -> pd.DataFrame:
    """
    生成每个节日的打标列（前窗 -1，后窗 +1，当天 0），
    使用矢量化区间比较，避免 per-holiday 的 set/isin。
    当 n_jobs>1 且已安装 joblib 时，对每个节日的列构建并行。
    """
    if date_col not in df.columns:
        raise KeyError(f"输入 DataFrame 缺少 `{date_col}` 列")

    out = df.copy()

    # 1) 解析日期 -> datetime64[D] -> int64（天序号）
    out["_date_dt"] = out[date_col].apply(_to_dt_yyyymmdd)
    if out["_date_dt"].isna().any():
        bad = out[out["_date_dt"].isna()][date_col].unique()[:5]
        print(f"[WARN] 有无法解析的日期样本（只展示前5个）：{list(bad)}")

    # 注意：astype('datetime64[D]') 再转 int64 是最快的序号化手段之一
    d_np = out["_date_dt"].values.astype('datetime64[D]').astype('int64')
    n = d_np.shape[0]
    print(f"[INFO] 共有 {n} 条样本，日期范围 {out['_date_dt'].min().date()} ~ {out['_date_dt'].max().date()}")

    # 2) 预创建所有需要的列名（规范化 + 去重）
    names = [_normalize_holiday_name(h["name"]) for h in holidays]
    uniq_names = sorted(set(names))
    col_names = [f"holiday_{nm}" for nm in uniq_names]
    print(f"[INFO] 共有 {len(holidays)} 个节日定义，涉及 {len(uniq_names)} 个不同节日，生成列：{col_names}")

    # 3) 定义单个节日列的构建函数（矢量化区间比较）
    def _build_one_col(h: Dict[str, str]) -> (str, np.ndarray):
        nm = _normalize_holiday_name(h["name"])
        col = f"holiday_{nm}"

        start_ts = _safe_to_timestamp(h["start"])
        end_ts   = _safe_to_timestamp(h["end"])
        if end_ts < start_ts:
            raise ValueError(f"节假日区间非法：{h}")

        # 转天序号
        start_i = np.datetime64(pd.to_datetime(start_ts).date(), 'D').astype('int64')
        end_i   = np.datetime64(pd.to_datetime(end_ts).date(), 'D').astype('int64')

        # 前窗: [start - pre_days, start)  -> -1
        pre_mask  = (d_np >= (start_i - pre_days)) & (d_np < start_i)

        # 后窗: (end, end + post_days]    -> +1
        post_mask = (d_np > end_i) & (d_np <= (end_i + post_days))

        col_vals = np.zeros(n, dtype=np.int8)
        # 注意顺序无所谓，因为 pre=-1, post=+1 不冲突（同一节日的前后窗不重叠）
        col_vals[pre_mask]  = -1
        col_vals[post_mask] =  1
        return col, col_vals


    # 4) 串行或并行构建各节日列
    if n_jobs and n_jobs != 1 and _HAS_JOBLIB:
        # 并行版本：每个节日产生一列（如同 map-reduce）
        results = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(_build_one_col)(h) for h in holidays
        )
    else:
        # 串行版本（已足够快，多为内存带宽受限）
        results = [_build_one_col(h) for h in holidays]
    print(f"[INFO] 节日列构建完毕，开始合并...")

    # 5) 将同名节日（如果重复出现）做“后写覆盖”或“聚合”
    # 常见做法：相同列名取逐元素最大绝对值（-1/0/1 中，优先 ±1）
    # 也可以选择“最后一个节日定义覆盖前者”——这里选择“最大绝对值”更稳妥：
    col_buffers = {c: np.zeros(n, dtype=np.int8) for c in col_names}
    for col, arr in tqdm(results, desc="合并同名节日", unit="列"):
        # 同名列聚合：abs 优先级（把非零写进去，不覆盖非零）
        buf = col_buffers.setdefault(col, np.zeros(n, dtype=np.int8))
        # 仅在 buf==0 的位置写入（避免覆盖已有的 -1/+1）
        write_pos = (buf == 0) & (arr != 0)
        buf[write_pos] = arr[write_pos]

    # 6) 赋值回 DataFrame（缺的列补 0）
    for col in tqdm(col_names, desc="写回DataFrame", unit="列"):
        out[col] = col_buffers.get(col, np.zeros(n, dtype=np.int8))

    out.drop(columns=["_date_dt"], inplace=True)
    return out


data = pd.read_feather("/home/user96/label_data1/hist_time.fea")

result = make_holiday_features_fast(data, HOLIDAYS, pre_days=5, post_days=5, date_col="date", n_jobs=64)
print(result)

result.to_feather("/home/user96/score_file/hist_time_holiday.fea")