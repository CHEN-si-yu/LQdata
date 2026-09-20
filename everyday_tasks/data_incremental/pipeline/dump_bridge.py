"""daily_dump → stock_history_5min 桥接。

为什么需要它：`stock_history_5min` 是 **685,917,526 行 / 8.4 GB** 的逐股 5min 表，
逐股爬一天要 **5,901 个请求**；而 `daily_dump` **1 个请求就能拿全市场当日 5min**
（264,384 行 = 5,508 只 × 48 根，约 5 秒）。**这是本工程最大的单点提效（约 5900×）。**

用户 2026-09-14 原话：「对于 history_5min，你就不应该去全量爬取，而是利用好
daily_dump 这个接口，在本地实现快速的更新。」

流程：
    ① `main.py calibrate-dump` —— 先标定单位（**一次性，必须先做**）
    ② 每天：找出 history_5min 缺的交易日 → **先看本地缓存** → 没有再逐日 dump
       → 原始数据按日存进缓存 → 过滤+换算后落进同一张表

★ 按日缓存（2026-09-15 按用户要求加，落点见 `paths.DUMP_CACHE_ROOT`）：
    「daily_dump 可以落盘到 datadownload/data，以单个日期为单位进行保存，
      以后优先读取本地是否有相关的数据」。
    缓存里存的是**原始件**（时间不做 48 时刻过滤、单位不做换算）——
    过滤与换算在读取时做，所以将来改判据/重标定都不用重抓（配额每日期每天 10 次）。
    缓存坏了/是盘中抓的半天的，一律不采信，照常问服务端（详见 `cache_is_final`）。

★ 三条不能违反的约束（都有实测依据）：
  1. **必须先标定 `vol` 单位**。history 的 vol 单位是**手**（文档明确 + 与 stock_daily
     实测 ×100），而 dump 的 5min Map **文档没写单位**。标定不通过就拒绝运行 ——
     一天 26 万行写错量纲，返工要按日重抓。
  2. **必须 drop `trade_date`**。history 是 8 列、没有 trade_date。直接 upsert 会让
     `store._unify_columns` 取列并集 → 给 **44,600,000 行**的分区补一列全 NA，
     并污染 `manifest.columns`。落盘前用 `df.reindex(columns=man.columns)` 锁死列序。
  3. **按标准 48 个时刻白名单过滤**。history 里有已知脏根：176 个 (股,日) 多一根
     `13:00:00`，其 `amount = vol × close`（**量纲差 100 倍**），集中在 2026-07-29~08-06。
     用白名单（而不是黑名单）更稳。
"""
from __future__ import annotations

import json
import statistics
from datetime import date as _date
from datetime import datetime

import pandas as pd

from .. import paths, registry as R
from ..core import state, store
from ..core.client import Client
from .strategies import Ctx, Result, _Buf

# ★★ 2026-09-17：`每天要不要补 dump` 与 `这份数据完不完整` **必须用同一套判据**。
#
#   旧实现是两条互相矛盾的线：
#     · "要补吗"用 `行数 < 正常一天 × 50%`（INCOMPLETE_RATIO）→ 很松
#     · "完整吗"用 `bar 覆盖 ≥98% 且 行数 ≥90%`（cache_is_final）→ 很严
#   于是 **50%~90% 之间的残缺日**被判成"够完整、不用补"，而它同时又过不了完整性
#   判据 —— 那天就永远停在残缺状态，报告还会写"都完整"。实测口径自洽的半天
#   （上午 09:35~11:30 那半场 = 正常行数的整整 50%）正好压在边界上、用严格 `<` 判为"不补"。
#
#   现在两条线共用下面两个常数（= `cache_is_final` 的判据）：
FINAL_BAR_RATIO = 0.98       # bar 覆盖率下限：行数 ÷ (股票数 × 48)
FINAL_ROWS_RATIO = 0.90      # 行数下限：相对"正常一天"的行数
MIN_ROWS = 1000              # 只用来判"这天到底有没有数据"（防 0 行参与中位数）

# 修复扫描窗口（日历天）。dump 接口只覆盖最近 ~90 天，所以：
#   · 太窄（= 尾部窗口 5 个交易日）：某天没在窗口内修好就**永久滑出**，再也补不回；
#   · 太宽：每次跑都要多扫本地数据（零请求，但网络盘读得慢）。
# 45 天是折中（覆盖连续停机一个多月），扫描成本约几秒。可用
# `conf/daily.yaml` 的 `dump.repair_lookback_days` 覆盖（最大 88）。
REPAIR_LOOKBACK_DAYS = 45


def standard_times() -> list[str]:
    """标准 48 个 5min bar 的**结束时刻**。

    上午 09:35 ~ 11:30（24 根），下午 13:05 ~ 15:00（24 根）。
    ⚠️ 没有 09:30（第一根 bar 是 09:35，覆盖 09:30~09:35）；
       也没有 13:00（下午从 13:01 开始计，第一根是 13:05）——
       13:00 那根是上游的脏数据，其 amount = vol × close（量纲差 100 倍）。
    """
    out: list[str] = []
    for start_h, start_m, n in ((9, 35, 24), (13, 5, 24)):
        h, m = start_h, start_m
        for _ in range(n):
            out.append(f"{h:02d}:{m:02d}")
            m += 5
            if m >= 60:
                h, m = h + 1, 0
    return out


STANDARD_5MIN = tuple(standard_times())
STANDARD_SET = frozenset(STANDARD_5MIN)

# 落进 history 的列（顺序即契约，见模块头★2）与缓存原始件的列
HISTORY_COLS = ["stock_code", "trade_time", "open", "high", "low", "close", "vol", "amount"]
RAW_COLS = ["stock_code", "time", "open", "high", "low", "close", "vol", "amount"]


def _raw_time(v) -> str | None:
    """dump 给的是 `"HH:MM"` 或 `"HH:MM:SS"` → 统一成 `"HH:MM:SS"`（**不过白名单**）。"""
    t = str(v).strip()
    if len(t) == 5:
        t = t + ":00"
    return t if len(t) == 8 else None


def _norm_time(v, day: str) -> str | None:
    """原始时刻 → `"YYYY-MM-DD HH:MM:SS"`；不在 48 时刻白名单里的返回 None。"""
    t = _raw_time(v)
    if t is None or t[:5] not in STANDARD_SET:   # ★ 白名单：挡掉 13:00 之类的脏根
        return None
    return f"{day} {t}"


def to_raw_df(data, day: str) -> pd.DataFrame:
    """dump 的 `Map<股票代码, [[时间,开,高,低,收,量,额],…]>` → **原始**表。

    ★ 两个"不过滤"是缓存层的立身之本（见 `prepare_for_history` 的反向操作）：
      · 时刻只规范化成 `HH:MM:SS`，**不**套 48 时刻白名单 → 上游的 13:00:00 脏根
        也一起留下来（想知道它分布在哪些天，不必重抓）；
      · `vol` / `amount` **不**乘标定系数 → 将来重新标定（系数变了）同一个缓存还能用。
    """
    rows: list[dict] = []
    if isinstance(data, dict):
        for code, bars in data.items():
            for b in bars or []:
                if not isinstance(b, (list, tuple)) or len(b) < 7:
                    continue
                t = _raw_time(b[0])
                if t is None:
                    continue
                rows.append({"stock_code": code, "time": t,
                             "open": b[1], "high": b[2], "low": b[3], "close": b[4],
                             "vol": b[5], "amount": b[6]})
    if not rows:
        return pd.DataFrame(columns=RAW_COLS)
    return pd.DataFrame(rows).drop_duplicates(subset=["stock_code", "time"], keep="last")


def prepare_for_history(raw: pd.DataFrame, day: str, cols: list[str] | None = None,
                        vol_factor: float = 1.0,
                        amount_factor: float = 1.0) -> pd.DataFrame:
    """原始件 → 可落进 `stock_history_5min` 的表（读取时做的加工都在这）。

    ① 48 时刻白名单过滤（脏根 13:00:00 的 `amount = vol × close`，量纲差 100 倍）
    ② 单位换算（标定出的 vol_factor / amount_factor）
    ③ 列序按 manifest.columns 锁死

    ★ 产出**不含 trade_date** —— history 没有这一列，带上会让整张历史表多出一列 NA。
    """
    want = list(cols or HISTORY_COLS)
    if raw is None or len(raw) == 0:
        return pd.DataFrame(columns=want)
    df = raw.copy()
    times = df["time"].astype(str)
    df = df[times.str[:5].isin(STANDARD_SET)].copy()
    if len(df) == 0:
        return pd.DataFrame(columns=want)
    df["trade_time"] = day + " " + df["time"].astype(str)
    df = df.drop(columns=["time"])
    if vol_factor != 1.0:
        df["vol"] = df["vol"].astype(float) * vol_factor
    if amount_factor != 1.0:
        df["amount"] = df["amount"].astype(float) * amount_factor
    df = df.drop_duplicates(subset=["stock_code", "trade_time"], keep="last")
    return df.reindex(columns=[c for c in want if c in df.columns])


def to_history_df(data, day: str, cols: list[str] | None = None) -> pd.DataFrame:
    """一次性把 API 响应转成 history 的表（等价于 raw + prepare，保留给标定/测试用）。"""
    return prepare_for_history(to_raw_df(data, day), day, cols)


# ================================================================ 按日缓存（本地优先）
def cache_is_final(raw: pd.DataFrame, base: float = 0.0) -> bool:
    """缓存里的这一天是不是一个**完整的交易日**（决定"本地优先"要不要采信它）。

    两条判据**同时**满足才算完整：
      ① bar 覆盖率 = 行数 ÷ (股票数 × 48) ≥ 0.98
         · 完整的一天：5,550 只 × 48 根 = 266,400 行 → 1.00（实测 266,256 = 5,547 × 48）
         · **盘中抓的半天**（例如上午 11:00 那一轮）：每只只有 24 根 → 0.50
      ② 行数 ≥ 0.9 × `base`（`base` = 本地该表近期的"正常一天"行数中位数）

    ★ ② 是实测补上的：只靠 ① 有一个洞 —— **"只有一半股票、但每只都齐 48 根"** 的
      半份数据，覆盖率算出来还是 1.00（实测：把完整的一天取前一半行，覆盖率 1.00、
      被判成"完整"）。服务端在极端情况下确实可能只回一部分股票。所以再拿
      "正常一天有多少行"卡一道，两道都过才认。
      `base` 拿不到（首次运行/新表）时只走 ①。

    ★ 为什么要这两条：没有它们，"本地优先"会把盘中/半份数据**永久固化**成
      那一天的最终版本（下次再也不会去问服务端）。不完整就照常问服务端，
      拿到更全的覆盖缓存 —— 缓存永远只是加速件，不是真相源。
    """
    if raw is None or len(raw) == 0 or "stock_code" not in raw.columns:
        return False
    stocks = int(raw["stock_code"].nunique())
    if stocks <= 0:
        return False
    if len(raw) < FINAL_BAR_RATIO * stocks * len(STANDARD_5MIN):
        return False
    if base and len(raw) < FINAL_ROWS_RATIO * float(base):
        return False
    return True


def _is_complete_stats(stat, base: float) -> bool:
    """(行数, 股票数) 是否构成一个完整交易日 —— 与 `cache_is_final` **同一条判据**。

    存在的意义：判断"本地这天要不要补 dump"时**不该把整天的数据读进来**
    （45 天 × 26 万行 = 1200 万行），只读两列数行数即可；但判据必须与
    `cache_is_final` 完全一致，否则又会出现"够完整不用补"与"不完整"两套标准。
    """
    if not stat:
        return False
    rows, stocks = int(stat[0]), int(stat[1])
    if rows < MIN_ROWS or stocks <= 0:
        return False
    if rows < FINAL_BAR_RATIO * stocks * len(STANDARD_5MIN):
        return False
    if base and rows < FINAL_ROWS_RATIO * float(base):
        return False
    return True


def _day_stats(ds_name: str, date_field: str, lo: str, hi: str) -> dict[str, tuple[int, int]]:
    """本地该表 `[lo, hi]` 内每天的 **(行数, 股票数)** —— 一次过滤读，零请求。

    过滤条件下推给 pyarrow，靠 row group 统计剪枝（本表按日期有序）。
    """
    import pyarrow.parquet as pq

    years = store.list_years(ds_name)
    files = ([store.partition_path(ds_name, y) for y in years] if years
             else [store.flat_path(ds_name)])
    hi_excl = _next_day(hi)
    out: dict[str, tuple[int, int]] = {}
    for f in files:
        if not f.exists():
            continue
        try:
            pf = pq.ParquetFile(f)
            names = pf.schema_arrow.names
            if date_field not in names or "stock_code" not in names:
                continue
            t = pq.read_table(f, columns=[date_field, "stock_code"],
                              filters=[(date_field, ">=", lo), (date_field, "<", hi_excl)])
            if t.num_rows == 0:
                continue
            df = t.to_pandas()
            df["_d"] = df[date_field].astype(str).str[:10]
            for day, sub in df.groupby("_d")["stock_code"]:
                prev = out.get(str(day), (0, 0))
                out[str(day)] = (prev[0] + int(len(sub)), prev[1] + int(sub.nunique()))
        except (OSError, ValueError, KeyError):
            continue
    return out


def save_dump(day: str, raw: pd.DataFrame, *, source: str = "api", note: str = "",
              base: float = 0.0) -> None:
    """把一天的原始 dump 原子写进本地缓存（parquet + meta.json）。

    `base` = 该表"正常一天"的行数中位数（判定完整性用，见 `cache_is_final`）。
    """
    if raw is None or len(raw) == 0:
        return
    try:
        store.write_atomic(paths.dump_cache_path(day), raw.reindex(columns=RAW_COLS))
        state._atomic_json(paths.dump_cache_meta(day), {
            "date": day, "level": "5min", "rows": int(len(raw)),
            "stocks": int(raw["stock_code"].nunique()),
            "final": bool(cache_is_final(raw, base)),
            "fetched_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "source": source, "note": note,
        })
    except (OSError, ValueError) as exc:  # noqa: BLE001 缓存写失败不能影响主流程
        print(f"   ⚠️ {day} 的 dump 缓存写失败（不影响本轮）：{exc}")


def load_dump(day: str) -> tuple[pd.DataFrame, dict] | None:
    """读本地缓存。没有 / 读不出来 → None（缓存坏了就当没有，绝不打断流程）。"""
    p = paths.dump_cache_path(day)
    if not p.exists():
        return None
    try:
        raw = pd.read_parquet(p)
    except Exception:  # noqa: BLE001 加速件坏了就重抓，不能让它把一轮跑挂掉
        return None
    if raw is None or len(raw) == 0 or "stock_code" not in raw.columns:
        return None
    try:
        meta = json.loads(paths.dump_cache_meta(day).read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        meta = {}
    return raw, (meta if isinstance(meta, dict) else {})


def cache_days() -> list[dict]:
    """列出缓存里有哪些天（**零请求**）—— 给 `main.py dump-cache` 用。"""
    root = paths.DUMP_CACHE_ROOT
    out: list[dict] = []
    if not root.exists():
        return out
    for d in sorted(root.iterdir()):
        if not d.is_dir() or not d.name.startswith("date="):
            continue
        day = d.name[5:]
        try:
            meta = json.loads(paths.dump_cache_meta(day).read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            meta = {}
        p = paths.dump_cache_path(day)
        out.append({"date": day,
                    "rows": meta.get("rows"), "stocks": meta.get("stocks"),
                    "final": meta.get("final"), "source": meta.get("source"),
                    "fetched_at": meta.get("fetched_at"),
                    "bytes": p.stat().st_size if p.exists() else 0})
    return out


def _load_or_fetch(client: Client, day: str, log=print,
                   base: float = 0.0) -> tuple[pd.DataFrame | None, str, str]:
    """取某一天的**原始** dump：本地缓存优先，没有再花 1 次配额问服务端。

    返回 `(raw | None, source, note)`：
      · source = `"local"` / `"api"` / `"none"`
      · note   = 给人看的一句话（为什么用了缓存 / 为什么失败），直接进报告

    ★ 优先级与回退（每一档都有实测依据）：
      ① 本地缓存**且是完整交易日** → 直接用，**0 请求 0 配额**（`cache_is_final`）
      ② 否则问服务端；拿回来的存进缓存（比缓存少就不覆盖 —— 防盘中抓的半份
         把一份完整的挤掉）
      ③ 服务端失败/返回 0 行时，**退用缓存里的那份（哪怕不完整）**：
         总比那天空着强，而且下轮的行数扫描会发现它不完整、自动重抓。
    """
    hit = load_dump(day)
    cached = hit[0] if hit is not None else None
    if cached is not None and cache_is_final(cached, base):
        return cached, "local", ""

    def _fallback_to_cache(why: str) -> tuple[pd.DataFrame | None, str, str]:
        if cached is not None and len(cached):
            return cached, "local", f"{why} → 改用本地缓存（不完整，下轮会重抓）"
        return None, "none", why

    quota = state.DumpQuota()
    if not quota.can_download(day, "5min"):
        return _fallback_to_cache(
            f"{day} 的 dump 配额**今天**已用 {quota.count_on(day)}/"
            f"{state.DumpQuota.LIMIT_PER_DATE} 次（服务端口径：每个数据日期每天 10 次，"
            f"超了封 3 天）")
    def _reserve_request() -> None:
        # Reserve before EVERY HTTP attempt, including retries/timeouts. Settling
        # after all retries is too late to prevent a 9 -> 15 request quota breach.
        from ..core.client import ApiError
        if not quota.can_download(day, "5min"):
            raise ApiError(f"{day} dump 配额已用满，本轮停止重试")
        quota.record(day, "5min")

    try:
        data = client.call("/stock/daily_dump", {"date": day, "level": "5min"},
                           method="POST", expect_rows=False, before_request=_reserve_request)
    except Exception as exc:  # noqa: BLE001
        return _fallback_to_cache(f"{day} dump 失败: {str(exc)[:70]}")

    raw = to_raw_df(data, day)
    if len(raw) == 0:
        # 空/失败**不标 done** → 下一轮尾部窗口自动重抓
        return _fallback_to_cache(f"{day} dump 返回 0 行（不标完成，下轮重试）")
    if cached is not None and len(cached) > len(raw):
        return cached, "local", (f"服务端这次只给 {len(raw):,} 行、少于本地缓存 "
                                 f"{len(cached):,} 行（盘中？）→ 保留缓存那份")
    save_dump(day, raw, source="api", base=base)
    return raw, "api", ""


# ================================================================ 单位标定
def calibrate(client: Client, cfg: dict, log=print) -> dict:
    """标定 dump 的 vol/amount 单位 —— **一次性，必须先做**。

    做法：取一个「本地 history_5min 已有 且 在 dump 的 90 天窗口内」的交易日，
    拉一次 dump（消耗 1/10 配额），逐根比对同一 (stock_code, trade_time)。
    """
    hist = R.get("stock_history_5min")
    man = state.Manifest.load("stock_history_5min")
    local_max = man.max_partition_date()
    if not local_max:
        return {"verdict": "no_data", "note": "本地 stock_history_5min 没有数据，无法标定"}
    # 用本地已有数据的最后一天（在 90 天窗口内，一定可拉）
    day = local_max
    log(f"标定基准日 = {day}（本地 history 的最后一天）")

    raw, source, note = _load_or_fetch(client, day, log=log)
    if raw is None:
        verdict = "quota" if "配额" in note else "empty"
        return {"verdict": verdict, "note": note, "day": day}
    log(f"基准日数据来源：{'本地缓存' if source == 'local' else '服务端'}"
        + (f"（{note}）" if note else ""))
    dump = prepare_for_history(raw, day)
    if len(dump) == 0:
        return {"verdict": "empty", "note": f"{day} 的 dump 过滤后 0 行（Map 解析失败？）", "day": day}

    # 读本地同一天的数据（只读那一天所在分区）
    y = int(day[:4])
    f = store.partition_path("stock_history_5min", y)
    if not f.exists():
        return {"verdict": "no_local", "note": f"本地没有 {y} 年的分区", "day": day}
    import pyarrow.parquet as pq
    parts = []
    try:
        pf = pq.ParquetFile(f)
        for b in pf.iter_batches(batch_size=2_000_000,
                                 columns=["stock_code", "trade_time", "open", "high",
                                          "low", "close", "vol", "amount"]):
            d = b.to_pandas()
            parts.append(d[d["trade_time"].astype(str).str[:10] == day])
    except (OSError, ValueError) as exc:
        return {"verdict": "err", "note": f"读本地失败: {exc}", "day": day}
    loc = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    if len(loc) == 0:
        return {"verdict": "no_local", "note": f"本地没有 {day} 的数据", "day": day}

    m = dump.merge(loc, on=["stock_code", "trade_time"], suffixes=("_d", "_h"))
    if len(m) == 0:
        return {"verdict": "no_overlap", "note": "dump 与本地没有可比的 (股票,时刻) 对",
                "day": day, "dump_rows": len(dump), "local_rows": len(loc)}
    out: dict = {"day": day, "dump_rows": len(dump), "local_rows": len(loc),
                 "matched": len(m), "n_bars_per_stock": int(dump.groupby("stock_code").size().median()),
                 "n_stocks": int(dump["stock_code"].nunique()),
                 "times": sorted(set(dump["trade_time"].str[11:16]))[:3]
                 + ["…"] + sorted(set(dump["trade_time"].str[11:16]))[-2:]}

    diffs = {}
    for col in ("open", "high", "low", "close"):
        a, b = m[f"{col}_d"].astype(float), m[f"{col}_h"].astype(float)
        diffs[col] = int((abs(a - b) > 1e-9).sum())
    out["price_mismatch"] = diffs
    for col in ("vol", "amount"):
        a, b = m[f"{col}_d"].astype(float), m[f"{col}_h"].astype(float)
        r = (b / a).replace([float("inf"), float("-inf")], pd.NA).dropna()
        out[f"{col}_ratio_median"] = float(r.median()) if len(r) else None
        out[f"{col}_ratio_p10"] = float(r.quantile(0.10)) if len(r) else None
        out[f"{col}_ratio_p90"] = float(r.quantile(0.90)) if len(r) else None

    price_ok = all(v == 0 for v in diffs.values())
    vr = out.get("vol_ratio_median")
    ar = out.get("amount_ratio_median")
    if not price_ok:
        out["verdict"] = "mismatch"
        out["note"] = f"价格对不上（{diffs}）—— dump 与 history 不是同源，**拒绝开启 dump 通道**"
    elif vr is None:
        out["verdict"] = "unknown"
        out["note"] = "算不出 vol 比值"
    else:
        # dump 的 vol 相对 history 的倍数：1.0 = 同单位（手）；100.0 = dump 是股
        factor = 1.0 / float(vr) if vr else 1.0
        out["vol_factor"] = factor
        out["amount_factor"] = (1.0 / float(ar)) if ar else 1.0
        out["verdict"] = "ok"
        out["note"] = (f"价格全对；dump 的 vol 是 history 的 {vr:.4g} 倍 → "
                       f"落盘前 vol × {factor:.6g}；每只股票 bar 数中位 {out['n_bars_per_stock']}")
    out["verified_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    state._atomic_json(paths.DUMP_CALIB, out)
    return out


def _load_calib() -> dict:
    try:
        v = json.loads(paths.DUMP_CALIB.read_text(encoding="utf-8"))
        return v if isinstance(v, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


# ================================================================ 每日增量
def run(ds: R.DS, man: state.Manifest, ctx: Ctx) -> Result:
    """把当日全市场 5min 并进 `stock_history_5min`。

    窗口 = 尾部窗口（T-redundancy ~ T）∩ dump 的 90 天可用窗口。
    """
    res = Result(ds.name, ds.mode)
    t0 = __import__("time").monotonic()

    calib = _load_calib()
    if calib.get("verdict") != "ok":
        res.status = "⊘"
        res.note = (f"dump 通道未标定（verdict={calib.get('verdict')}）—— "
                    f"先跑 `main.py calibrate-dump`。{calib.get('note','')}")
        res.seconds = __import__("time").monotonic() - t0
        return res

    # ★★ 2026-09-15 重写"哪天需要 dump"的判据。旧实现有两处会**永久丢数据**：
    #
    #   ① `local_max = man.max_partition_date()` 被当成"本地已到这天，且这天是完整的"。
    #      可 `max_date` 只是"这天**有**数据"，不是"这天**齐了**"。
    #      实测事故：2026-09-14 只有 **2,304 行**（48 只 × 48 根），正常一天约 266,400 行；
    #      水位却因此变成 09-14 → `lo = _next_day(09-14) = 09-15 > hi` → `days=[]`
    #      → 记一句"没有需要补的交易日"，status 还是 ✔、报告还是"无异常"。
    #      这张 9GB 的表就此卡在 09-14 永远不补（dump 台账里根本没有 09-14 的记账）。
    #
    #   ② `not man.is_done("dumps", ...)` 把"dump 成功过"当成"这天以后都不用再 dump"。
    #      而 `delete-day` **故意不动 done**（它测的就是"逻辑看不看数据"）→
    #      删掉的这天被 is_done 滤掉 → 同样永远补不回，且没有任何报错。
    #
    #   现在只认**真实数据**：扫描尾部窗口每天的本地行数，把"行数明显不足"的
    #   那一天拿出来 dump。既不漏（部分数据也认得出来），也不浪费配额
    #   （数据完整的天不会被重复下载 —— dump 配额是每日期每天 10 次，超了封 3 天）。
    from .calendar import trading_days_between

    # ★★ 2026-09-17 重写「哪天要补 dump」的两个维度：
    #   ① **窗口放宽**：旧实现只看尾部窗口（`redundancy=5` → 最多 6 个交易日），
    #      而 dump 接口其实能取最近 ~90 天 —— 一个残缺日只要没在这 5 天内修好，
    #      就**永久滑出扫描范围**（旧的 `>= -88` 过滤是死代码：窗口最长 6 天，
    #      那个判断永远为真）。停机一周、或者残缺日没被及时发现，洞就永远在了。
    #      现在按 `dump.repair_lookback_days`（默认 45 天）扫**纯本地行数**，零请求。
    #   ② **判据统一**：与 `cache_is_final` 同一条线（bar 覆盖 ≥98% 且 行数 ≥90%
    #      正常一天），不再是那个 50% 的松线（见 FINAL_BAR_RATIO 的说明）。
    hi = ctx.data_hi(ds)
    today = _date.today()
    lookback = int(ctx.cfg.get("dump", {}).get("repair_lookback_days",
                                               REPAIR_LOOKBACK_DAYS))
    lo = max(_minus_days(hi, lookback), ds.start)
    win = [d for d in trading_days_between(ctx.cal, lo, hi)
           if (_date.fromisoformat(d) - today).days >= -88]   # dump 只可取最近 90 天
    if not win:
        res.note = f"窗口 {lo}~{hi} 内没有可取 dump 的交易日"
        res.seconds = __import__("time").monotonic() - t0
        return res

    stats = _day_stats(ds.name, ds.date_field, win[0], win[-1])
    # "正常一天" = 窗口内行数**较大那一半**的中位数 —— 比"全体中位数"抗残缺日污染
    # （若一半以上的天是残缺的，全体中位数会被拉到残缺那一档，基准就失效了）
    rows_sorted = sorted(r for (r, _s) in stats.values() if r > 0)
    if rows_sorted:
        upper = rows_sorted[len(rows_sorted) // 2:]
        base = float(statistics.median(upper)) if upper else 0.0
    else:
        base = 0.0

    days = [d for d in win if not _is_complete_stats(stats.get(d), base)]
    if not days:
        res.note = (f"近 {len(win)} 个交易日（{win[0]}~{win[-1]}）每天的数据都完整"
                    f"（基准 {base:,.0f} 行/日），无需 dump")
        res.seconds = __import__("time").monotonic() - t0
        return res
    res.note = (f"窗口 {win[0]}~{win[-1]} 内 {len(days)} 天不完整（基准 {base:,.0f} 行/日）："
                + ", ".join(
                    f"{d}({stats.get(d, (0, 0))[0]:,}行/{stats.get(d, (0, 0))[1]}只)"
                    for d in days[:6])
                + (" …" if len(days) > 6 else ""))
    if ctx.dry_run:
        res.note = (f"将用 daily_dump 补 {len(days)} 天：{days}（每天 1 个请求，"
                    f"替代逐股 5901 个请求）；" + res.note)
        res.seconds = __import__("time").monotonic() - t0
        return res

    cols = man.columns or HISTORY_COLS
    buf = _Buf(ds, man, flush_rows=400_000, flush_batches=3)
    vf = float(calib.get("vol_factor") or 1.0)
    af = float(calib.get("amount_factor") or 1.0)
    n_local = n_api = 0
    for d in days:
        # ★ 本地优先：缓存里有一份完整的这天 → 0 请求、0 配额直接用（用户 2026-09-15 要求）
        raw, source, why = _load_or_fetch(ctx.client, d, base=base)
        if raw is None:
            if "配额" in why:
                res.status = "⚠️"       # 配额用尽不是"数据可疑"，不记 suspect
                res.note = why
                continue
            man.mark_suspect(f"dump|{d}|error")
            res.status = "⚠️"
            res.note = why
            continue
        df = prepare_for_history(raw, d, cols, vf, af)
        if len(df) == 0:
            # 空/失败**不标 done** → 下一轮尾部窗口自动重抓
            man.mark_suspect(f"dump|{d}|empty")
            res.status = "⚠️"
            res.note = f"{d} dump 过滤后 0 行（不标完成，下轮重试）"
            continue
        if source == "local":
            n_local += 1
        else:
            n_api += 1
        # ★★ 2026-09-17：写进来的这天如果**仍不完整**，必须记账 + 报警。
        #   旧实现是"服务端给什么就写什么、状态照样报成功" —— 实测 2026-09-14 那天
        #   只有 2,304 行（正常 266,400）也照落库，报告还写"正常补了 X 行"。
        #   现在：照写（比空着强），但记 suspect + 报告里标 ⚠️，且**下轮仍会被选中重试**
        #   （选日判据与这里用的是同一条线，见 FINAL_BAR_RATIO）。
        if not cache_is_final(raw, base):
            man.mark_suspect(f"dump|{d}|incomplete")
            res.status = "⚠️"
            res.note = (f"{d} 服务端给的仍是**残缺**数据（{len(raw):,} 行，"
                        f"基准 {base:,.0f} 行/日）—— 已写入并保留，下轮继续重试")
        buf.add(df)
        # ★ 顺序铁律：先落盘（buf.flush 内部做）→ 再标完成
        buf.flush()
        man.mark_done("dumps", f"5min|{d}")
        if ctx.runner:
            tag = "本地缓存" if source == "local" else "服务端"
            ctx.runner.note(f"   {d} dump → {len(df):,} 行（{df['stock_code'].nunique()} 只）"
                            f" · {tag}" + (f" · {why}" if why else "")
                            + f" · 累计 {buf.rows:,}")
    buf.flush()
    res.rows = buf.rows
    res.seconds = __import__("time").monotonic() - t0
    if not res.note:
        src = f"本地缓存 {n_local} / API {n_api} 天"
        res.note = (f"daily_dump 补 {len(days)} 天（{src}）→ 新增 {res.rows:,} 行"
                    if (n_local or n_api) else f"daily_dump 补 {len(days)} 天 → 新增 {res.rows:,} 行")
    return res


def _next_day(s: str) -> str:
    from datetime import timedelta
    return (_date.fromisoformat(str(s)[:10]) + timedelta(days=1)).isoformat()


def _minus_days(s: str, n: int) -> str:
    from datetime import timedelta
    return (_date.fromisoformat(str(s)[:10]) - timedelta(days=int(n))).isoformat()
