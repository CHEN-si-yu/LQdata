"""服务端探测 —— 闸门的判据来源。

★ 与旧工程闸门最本质的区别：**两边都问服务端**。
    旧工程：T = 本地 stock_daily 的 max_date，判据 = 本地各表 max_date 够不够新。
            → 两边都在自己家里，**上游没发布时全表一起退**，闸门反而"通过"。
    本工程：T = **服务端**已发布的最新交易日；
            ready = **服务端**在 `T - delay` 那天有没有数据。
            → 这才能实现用户要的「确认上游到齐了才开跑」。

成本：T 探测 1~3 请求，20 张日频表各 1 请求 —— 一轮约 23 请求，
      5 分钟一轮 = 每分钟不到 5 个请求，对 260/min 的额度毫无压力。
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass

import pyarrow.parquet as pq

from .. import paths, registry as R
from ..core import state, store
from ..core.client import Client


# ================================================================ 本地统计（零请求）
def daily_row_counts(ds_name: str, date_field: str, limit_days: int = 60) -> dict[str, int]:
    """读本地该数据集**最近若干天**的每日行数。用来判断"某天 0 行"是否正常。"""
    years = store.list_years(ds_name)
    out: dict[str, int] = {}
    if not years:
        f = store.flat_path(ds_name)
        files = [f] if f.exists() else []
    else:
        files = [store.partition_path(ds_name, y) for y in years[-2:]]   # 最近两年足够
    for f in files:
        if not f.exists():
            continue
        try:
            pf = pq.ParquetFile(f)
            if date_field not in pf.schema_arrow.names:
                continue
            for batch in pf.iter_batches(batch_size=2_000_000, columns=[date_field]):
                s = batch.column(0).to_pandas().astype(str).str[:10]
                vc = s.value_counts()
                for d, c in vc.items():
                    out[d] = out.get(d, 0) + int(c)
        except (OSError, ValueError):
            continue
    if not out:
        return {}
    days = sorted(out)[-limit_days:]
    return {d: out[d] for d in days}


def median_daily_rows(ds_name: str, date_field: str, n: int = 5) -> float:
    """近 n 个"有数据日"的行数中位数 —— 判定"是否完整发布"的基准。"""
    counts = daily_row_counts(ds_name, date_field)
    vals = [v for v in counts.values() if v > 0]
    if not vals:
        return 0.0
    return float(statistics.median(vals[-n:]))


def trailing_nonzero_ratio(ds_name: str, date_field: str, n: int = 60) -> float:
    """近 n 个交易日里"有数据的天数"占比。

    ≥0.8 → 这张表每天都在更新，某天 0 行 = 真缺（用严格判据）
    <0.8 → 事件型/稀疏表，0 行是常态（用宽松判据）

    ★★ 2026-09-15 修一个让这个函数**恒等于 1.0** 的分母错误：
       旧实现的分母是 `len(counts.values())`，而 `counts`（来自
       `daily_row_counts` 的 `value_counts()`）**键只包含数据里出现过的日期** ——
       也就是说"没有数据的那天"根本不在分母里，分子则按定义恒等于分母。
       实测：33 张表返回 1.000，6 张无数据返回 0.000，**没有任何中间值**；
       `index_weight` 在近两年只覆盖 2.2% 的交易日，函数照样返回 1.000。

       后果不是"判错某张表"，而是两条设计好的退路**结构性不可达**：
         · `probe_table_ready` 里"稀疏表退化为 10 天窗口判据"永远走不到 →
           稀疏表在合法空日被判 `ready=False` → 闸门白等满 4 小时 → 超时 partial
           → 再叠加 `Ctx.selected` 的过滤，整个非日频半场被静默跳过；
         · `report._judge` 里"稀疏表 0 行豁免"失效 → 告警疲劳。

       现在分母改成**日历里该窗口的交易日数**（数据里多出来的非交易日不参与）。
    """
    counts = daily_row_counts(ds_name, date_field, limit_days=n)
    if not counts:
        return 0.0
    days = sorted(counts)[-int(n):]
    lo, hi = days[0], days[-1]
    try:
        cal = state.effective_calendar()
    except Exception:  # noqa: BLE001
        cal = []
    expect = [d for d in cal if lo <= d <= hi] if cal else []
    # 日历覆盖不到（例如 date_field 不是交易日、或日历缺失）→ 退回旧口径
    vals = ([counts.get(d, 0) for d in expect] if len(expect) > len(days)
            else [counts[d] for d in days])
    if not vals:
        return 0.0
    return sum(1 for v in vals if v > 0) / len(vals)


# ================================================================ 代表实体
_ENTITY_CACHE: dict[str, str] = {}


def _count_col(f, col: str, k_groups: int = 6):
    """数某一列里各值出现的次数 —— 在**整份文件里均匀取样** `k_groups` 个 row group。

    ★ 为什么不是"只读最后几个 row group"：主键有序的表里，最后几个 row group
      = **主键最大的那批实体**（`stock_history_5min` 按 `stock_code` 排 → 最后一段
      全是 `9xxxxx.BJ`），取样会系统性偏向某一类实体。均匀取样 + 主板优先
      （见 `representative_entities`）才能选出"最普通、最不会缺席"的那一类。
      代价：对 6 亿行的表也只读 6 个 row group（约 1 秒内）。
    """
    from collections import Counter
    cnt: Counter = Counter()
    try:
        pf = pq.ParquetFile(f)
        if col not in pf.schema_arrow.names:
            return cnt
        n = pf.num_row_groups
        if n <= k_groups:
            idx = list(range(n))
        else:
            step = (n - 1) / (k_groups - 1)
            idx = sorted({round(i * step) for i in range(k_groups)})
        for gi in idx:
            t = pf.read_row_group(gi, columns=[col])
            cnt.update(x for x in t.column(0).to_pylist() if x)
    except (OSError, ValueError):
        return cnt
    return cnt


def _newest_partition(name: str):
    years = store.list_years(name)
    if years:
        return store.partition_path(name, years[-1])
    f = store.flat_path(name)
    return f if f.exists() else None


def representative_entities(ds: R.DS, cal: list[str], k: int = 3) -> list[str]:
    """per_entity 表探针用的代表实体（**按优先级**最多 k 个）。

    ★★ 2026-09-17 重写（旧实现有两个真问题，都已实测）：

      ① **它取的是源表，不是本表。** docstring 写"取本地数据里行数最多的实体"，
         实现的却是"在 `entity_from` 源表里数出现次数" —— 而源表（`stock_list`）
         里每个代码只出现 1~2 次，`most_common` 实际退化成**按文件行序取第一个**。
      ② **可能取到退市股。** 实测当前取到 `000003.SZ`（2002-06-14 退市，
         本地 `stock_cyq_chips` 里一行都没有）→ 探针恒返回"无"
         → 闸门为 `stock_cyq_chips` / `stock_history_5min` **每晚白等满 6 小时预算**
         （2026-09-16 23:40 那轮就是这个现象：日志 + status.json + delay_history 三处互证）。

      现在的顺序：
        ① 本表**自己的本地数据**里最活跃的实体（这才是原 docstring 的本意）；
        ② 找不到就退回源表，且**排除退市**（有 `list_status`/`delist_date` 就用），
           优先主板前缀 —— 至少不会挑一只 2002 年就退市的票。
    """
    if ds.name in _ENTITY_CACHE:
        return list(_ENTITY_CACHE[ds.name])
    out: list[str] = []

    # ---- ① 本表自己的数据里最活跃的实体
    if ds.date_field:                     # 无日期轴的纯快照表不参与闸门，用不上
        f = _newest_partition(ds.name)
        col = ds.code_param
        if f is not None and f.exists():
            cnt = _count_col(f, col)
            # ★ 并列时**主板优先**（`000/001/002/003/600/601/603/605` 之外的一律靠后）：
            #   行数几乎总是并列（每股每天都是 48 根），不排序就等于"按取样顺序随便挑"。
            pref = tuple(R.MAIN_BOARD_PREFIXES)
            ranked = sorted(cnt.items(),
                            key=lambda kv: (-kv[1],
                                            0 if str(kv[0]).startswith(pref) else 1,
                                            str(kv[0])))
            out += [c for c, _ in ranked[:k]]

    # ---- ② 退回源表（排除退市 + 优先主板）
    if len(out) < k and ds.entity_from:
        src, col = ds.entity_from
        f = _newest_partition(src)
        if f is not None and f.exists():
            try:
                pf = pq.ParquetFile(f)
                names = pf.schema_arrow.names
                cols = [col] + [c for c in ("list_status", "delist_date") if c in names]
                t = pq.read_table(f, columns=cols)
                codes = [str(x) for x in t.column(0).to_pylist() if x]
                status = ([str(x) for x in t.column("list_status").to_pylist()]
                          if "list_status" in cols else None)
                ddate = ([str(x) for x in t.column("delist_date").to_pylist()]
                         if "delist_date" in cols else None)
                alive: list[str] = []
                for i, c in enumerate(codes):
                    if status is not None and status[i].strip().upper() in ("D", "退市"):
                        continue
                    if ddate is not None and ddate[i].strip() not in ("", "None", "nan"):
                        continue          # 有退市日 = 已退市
                    alive.append(c)
                # 主板优先（与 registry.MAIN_BOARD_PREFIXES 同一套口径）
                pref = tuple(R.MAIN_BOARD_PREFIXES)
                alive.sort(key=lambda c: (0 if c.startswith(pref) else 1, c))
                for c in alive:
                    if c not in out:
                        out.append(c)
                    if len(out) >= k:
                        break
            except (OSError, ValueError, KeyError):
                pass
    _ENTITY_CACHE[ds.name] = list(out)
    return out


def representative_entity(ds: R.DS, cal: list[str]) -> str | None:
    """兼容旧调用：只取优先级最高的那一个。"""
    es = representative_entities(ds, cal)
    return es[0] if es else None


# ================================================================ 探测请求
def probe_payload(ds: R.DS, date: str, cal: list[str],
                  entity: str | None = None) -> dict | None:
    """构造"问服务端某天有没有数据"的最小请求。None = 这张表无法用单日探测。

    ★★ 2026-09-15 深夜修三处缺口（闸门要从"只日频"扩到全表，不补会让新纳入的表
       **永久阻塞**）：
      · `range` 分支**不透传 `variants[0]`** → `stock_kline` 的 `period` 是必填，
        实测 monitor 里它 `server_max=None`（探测异常被吞成"未知"）。
      · `per_entity` 分支不认 `entity_range=False` → 对无日期轴的纯快照表
        （`tdx_block_stocks` / `index_ths_constituent_stocks`）造出
        `{code_param, start_time, end_time}` 这种**语义不对**的请求。
      · `per_entity` 不认 `date_step="month"` → 月频表要的是 `date_param=YYYY-MM`，
        给 `start_time/end_time` 必然返回 0 行。
      三条与取数侧 `strategies._entity_payload`（:561-571）对齐，避免两边再次漂移。
    """
    if ds.mode == "snapshot":
        return None
    if ds.mode == "per_date":
        p = {**ds.params, ds.date_param: date}
        if ds.variants:
            p.update(ds.variants[0])
        return p
    if ds.mode == "per_entity":
        if not ds.entity_range:
            return None          # 无日期轴的纯快照：没有"某天"可问
        ent = entity or representative_entity(ds, cal)
        if not ent:
            return None
        p = {**ds.params, ds.code_param: ent}
        if ds.date_step == "month":
            p[ds.date_param] = date[:7]          # YYYY-MM
        else:
            p[ds.start_param] = date
            p[ds.end_param] = date
        return p
    if ds.mode == "dump":
        return {ds.start_param: date, "level": "5min"}
    if ds.mode == "per_stock" and ds.date_field == "end_date":
        # ★ 报告期表（stock_income / balancesheet / cashflow / financial_indicator）：
        #   取数侧用的是 `{end_date: 报告期}`，**不是** start_time/end_time
        #   （见 strategies.py 的 REPORT_PERIOD_TABLES 分支）。参数名错了就永远 0 行。
        return {**ds.params, "end_date": date}
    # range / per_stock（其余）
    p = {**ds.params, ds.start_param: date, ds.end_param: date}
    if ds.variants:
        p.update(ds.variants[0])     # ★ 缺了这行会让 stock_kline 探不动
    return p


def probe_rows(client: Client, ds: R.DS, date: str, cal: list[str],
               page_size: int = 3, entity: str | None = None) -> int | None:
    """问服务端：这张表在 `date` 那天有多少行？None = 探测失败/不适用。

    ⚠️ 必须返回**服务端信封里的 total**，不是"当前页取回了几行" ——
       为了省流量我们只请求 page_size=3，若拿 len(list) 当行数，
       永远只有 3，会被发布完整性判据（要 ≥ 中位数×0.7，约 3900）一直判为"未发布"。

    `entity`：per_entity 表指定代表实体（不传则取优先级最高的那个）。
    """
    payload = probe_payload(ds, date, cal, entity=entity)
    if payload is None:
        return None
    try:
        data = client.call(ds.path, {**payload, "page": 0, "page_size": page_size},
                           method=ds.method, expect_rows=False)
        from ..core.client import extract_list, extract_total
        total = extract_total(data)              # ★ 优先用服务端给的 total
        if total is not None:
            return int(total)
        return len(extract_list(data))
    except Exception:  # noqa: BLE001
        return None


# ================================================================ T 探测
@dataclass
class TEvidence:
    T: str | None
    checked: list[dict]                 # 每个候选日的探测结果
    cross_check: str | None = None
    cross_ok: bool | None = None
    note: str = ""


def resolve_T(client: Client, cal: list[str], cfg: dict,
              probe_ds: str = "stock_daily", candidates: int = 5) -> TEvidence:
    """确定 T = **服务端**已发布的最新交易日。

    做法：从日历里今天及以前的最后 N 个交易日，**新→旧**逐个问服务端；
    第一个"行数达到近 5 日中位数 × publish_ratio"的日期即 T。
       · 防"收盘前跑"：今天数据还没出 → 退回昨天
       · 防"半量发布"：今天只有 3000 行（正常 5500）→ 也退回昨天
    """
    ratio = float(cfg.get("gate", {}).get("publish_ratio", 0.7))
    ds = R.get(probe_ds)
    today = __import__("datetime").date.today().isoformat()
    cand = [d for d in cal if d <= today][-candidates:][::-1]
    base = median_daily_rows(probe_ds, ds.date_field, n=5)

    checked: list[dict] = []
    T: str | None = None
    for d in cand:
        n = probe_rows(client, ds, d, cal, page_size=3)
        if n is None:
            checked.append({"date": d, "rows": None, "ok": False, "why": "探测失败"})
            continue
        ok = (n > 0) and (base <= 0 or n >= base * ratio)
        checked.append({"date": d, "rows": n, "ok": ok,
                        "why": "" if ok else f"行数不足（基准 {base:.0f} × {ratio}）"})
        if ok:
            T = d
            break

    ev = TEvidence(T=T, checked=checked)
    if T is None:
        ev.note = "服务端没有任何候选日达到发布阈值 —— 可能还没出数据，或今天不是交易日"
        return ev

    # 交叉验证：用**另一张 date 轴表**独立确认 T（不同的接口 + 不同的行数量级）
    #   /stock/daily 是 5550 行/日，/stock/adj_factor 是 5562 行/日 —— 两条独立证据。
    if cfg.get("gate", {}).get("cross_check", True):
        try:
            alt = R.get("stock_adj_factor")
            n = probe_rows(client, alt, T, cal)
            if n:
                ev.cross_check = f"{alt.name}:{T}({n}行)"
                base_alt = median_daily_rows(alt.name, alt.date_field, n=5)
                ev.cross_ok = bool(base_alt <= 0 or n >= base_alt * 0.7)
        except Exception:  # noqa: BLE001
            pass
    return ev


# ================================================================ 单表就绪判定
@dataclass
class TableProbe:
    name: str
    expected: str          # T - delay（交易日）
    actual: str | None     # 服务端该表在 expected 那天的可用情况（"有"/"无"/"未知"）
    ready: bool
    rows: int | None = None
    note: str = ""
    # ★ 2026-09-17：`unknown=True` = **探测本身失败**（超时/异常），
    #   与"服务端确认没有数据"是两回事。前者不该阻塞闸门（见 probe_table_ready）。
    unknown: bool = False


def probe_table_ready(client: Client, ds: R.DS, expected: str, cal: list[str],
                      sparse_ratio: float | None = None) -> TableProbe:
    """判据：**服务端**在 `expected` 这天有没有这张表的数据。

    ⚠️ 稀疏表的 0 行是真常态（stock_suspension 多数交易日无停牌；
       stock_st_info 有 63 天服务端自己也是空的）。
       所以对 `trailing_nonzero_ratio < 0.8` 的表退化为"看最近 10 天的窗口，
       落后不超过 1 个交易日就算就绪"。
    """
    if sparse_ratio is None:
        sparse_ratio = trailing_nonzero_ratio(ds.name, ds.date_field)

    if sparse_ratio >= 0.8:
        n = probe_rows(client, ds, expected, cal)
        # ★ 2026-09-17 单实体探测的**假阴性兜底**：per_entity 表只问一个代表实体，
        #   那个实体当天恰好停牌/无数据就会误判"整表未发布" → 闸门白等满预算。
        #   第一个实体为 0 时再问 1~2 个别的实体（只在"疑似无数据"时多花请求）。
        if n == 0 and ds.mode == "per_entity":
            for ent in representative_entities(ds, cal)[1:3]:
                n2 = probe_rows(client, ds, expected, cal, entity=ent)
                if n2:
                    return TableProbe(ds.name, expected, "有", True, n2,
                                      f"代表实体 {ent} 有数据（首个代表实体当天为空，已兜底）")
                if n2 is None:
                    break
        if n is None:
            # ★★ 2026-09-17：**探测失败 ≠ 服务端没有**。判成 not-ready 会让闸门
            #   白等满 6 小时预算再 partial 跳过；而闸门假阳性的代价只是
            #   "拿回空、记 suspect、下轮重试"（尾部窗口兜底，见 gate.py 的设计原则）。
            #   所以"看不清"时**不阻塞**，但要在闸门表和报告里看得出来。
            return TableProbe(ds.name, expected, "未知", True, None,
                              "⚠️ 探测失败（按不阻塞处理：无法区分「没发布」与「探测出错」，"
                              "数据由尾部窗口兜底）", unknown=True)
        return TableProbe(ds.name, expected, "有" if n > 0 else "无", n > 0, n)

    # ---- 稀疏表：看一个 10 天窗口，取服务端返回的最新日期
    payload = probe_payload(ds, expected, cal)
    if payload is None:
        return TableProbe(ds.name, expected, "不适用", True, None, "无法单日探测，跳过")
    lo = cal[max(0, cal.index(expected) - 10)] if expected in cal else expected
    if ds.mode == "per_date":
        p = dict(payload)
    else:
        p = {**payload}
        p[ds.start_param] = lo
        p[ds.end_param] = expected
    try:
        data = client.call(ds.path, {**p, "page": 0, "page_size": 200},
                           method=ds.method, expect_rows=False, empty_retries=0)
        from ..core.client import extract_list
        rows = extract_list(data)
        got = sorted({str(r.get(ds.date_field))[:10] for r in rows if r.get(ds.date_field)})
        latest = got[-1] if got else None
        lag_ok_date = cal[cal.index(expected) - 1] if expected in cal and cal.index(expected) > 0 else expected
        ready = bool(latest and latest >= lag_ok_date)
        return TableProbe(ds.name, expected, latest or "无", ready, len(rows),
                          f"稀疏表（有数据日占比 {sparse_ratio:.0%}），按 10 天窗口判")
    except Exception as exc:  # noqa: BLE001
        return TableProbe(ds.name, expected, "未知", False, None, f"探测异常: {str(exc)[:60]}")


def server_max_date(client: Client, ds: R.DS, cal: list[str], back: int = 12) -> str | None:
    """问服务端：这张表最新有数据的日期是哪天（往前最多找 back 个交易日）。"""
    today = __import__("datetime").date.today().isoformat()
    cand = [d for d in cal if d <= today][-back:][::-1]
    for d in cand:
        n = probe_rows(client, ds, d, cal, page_size=2)
        if n:
            return d
    return None


# ================================================================ 闸门·观测档（T2）
def _watch_candidates(ds: R.DS, cal: list[str], T: str, n: int = 3) -> list[str]:
    """观测档的候选探测日期（最多 n 个，新的在前）。按表的**时间步长**给不同的粒度。

    ★ 候选日期只代表"我去问了这一格"，**不等于服务端数据的真实日期** ——
      所以判定"是否落后"必须用 `_period_key` 把两端都压到同一粒度再比，
      否则月频表会被误报（把候选的 07-31 当成服务端水位，而服务端其实只到 07-01）。
    """
    past = [d for d in cal if d <= T]
    if ds.freq == "quarterly":
        # 报告期表：要问的是季末，不是交易日（交易日永远不在报告期里 → 必然"无数据"）
        y = int(T[:4])
        ends = [f"{yy}-{md}" for yy in (y - 1, y, y + 1)
                for md in ("03-31", "06-30", "09-30", "12-31")]
        ends = sorted(e for e in ends if e <= T)
        return list(reversed(ends))[:n]
    if not past:
        return []
    if ds.date_step == "month":
        out, seen = [], set()
        for d in reversed(past):
            m = d[:7]
            if m not in seen:
                seen.add(m)
                out.append(d)
            if len(out) >= n:
                break
        return out
    return list(reversed(past))[:n]


def _period_key(ds: R.DS, d: str | None) -> str:
    """把日期压到该表的时间粒度上，用于"本地 vs 服务端"的同口径比较。"""
    if not d:
        return ""
    return d[:7] if ds.date_step == "month" else d


def probe_table_watch(client: Client, ds: R.DS, cal: list[str], T: str) -> TableProbe:
    """T2（观测档）：探服务端最新日期，与本地水位对比。**ready 恒为 True —— 永不阻塞闸门。**

    为什么 T2 不能做成阻塞判据（死锁论证）见 `registry.gate_tier` 的注释。
    这里的价值是：把"服务端有新的、而我们本地还没有"**显式展示出来**，
    而不是像改造前那样对 22 张非日频表**完全不看**。

    探测成本封顶 3 个请求（候选日期最多 3 个），且闸门里 T2 只在第 1 轮 + 每 5 轮探一次。
    """
    from ..core import state

    try:
        local_max = state.Manifest.load(ds.name).max_partition_date()
    except Exception:  # noqa: BLE001  manifest 读不出来不该让闸门挂掉
        local_max = None

    cands = _watch_candidates(ds, cal, T, n=3)
    hit = None
    for d in cands:
        if probe_rows(client, ds, d, cal, page_size=2):
            hit = d
            break

    if hit is None:
        why = "探测失败/不适用" if not cands else f"服务端近 {len(cands)} 个候选期均无数据"
        return TableProbe(ds.name, T, "无", True, None, f"观测档：{why}（本地 {local_max or '—'}）")

    # ★ 必须同粒度比较（月频压到 YYYY-MM），否则把"候选日期"误当服务端水位 → 误报落后
    behind = bool(local_max) and _period_key(ds, local_max) < _period_key(ds, hit)
    note = f"观测档：本地 {local_max or '—'}　服务端最近有数据的一格 {hit}"
    if behind:
        note += "　⚠️ 落后于服务端"
    return TableProbe(ds.name, T, hit, True, None, note)


# ================================================================ 完整性比例判据
def probe_ready_by_ratio(client: Client, ds: R.DS, expected: str, cal: list[str],
                         publish_ratio: float = 0.7, ref_back: int = 4) -> TableProbe:
    """稠密判据 + **完整性比例**：该日行数 ≥ 同口径历史日行数 × ratio 才算就绪。

    为什么单看"有没有行"不够：分钟级表**盘中也会返回半份数据**
    （实测 `stock_history_5min` 11:00 那轮每只只有 24 根），
    `probe_table_ready` 的稠密分支会把半份判成"就绪"。

    ★ 基准必须用**同口径的历史日**，不能用整表中位数 —— 对 `per_entity` 表，
      探针只问一个代表实体，返回的是"该实体当天的行数"（几十行），
      而整表中位数是 26 万行级，两者不可比（会把所有 per_entity 表判成永远不就绪）。
      所以这里往前找第一个**探测成功**的交易日当基准，自校准、无硬编码常数。
    """
    n = probe_rows(client, ds, expected, cal)
    if n is None:
        # 同 probe_table_ready：探测失败不阻塞（见那里的注释）
        return TableProbe(ds.name, expected, "未知", True, None,
                          "⚠️ 探测失败（按不阻塞处理，数据由尾部窗口兜底）", unknown=True)

    ref = None
    if expected in cal:
        idx = cal.index(expected)
        for d in reversed(cal[max(0, idx - ref_back):idx]):
            m = probe_rows(client, ds, d, cal)
            if m:
                ref = (d, m)
                break

    if not ref:
        return TableProbe(ds.name, expected, "有" if n > 0 else "无", n > 0, n,
                          "无同口径基准日，退化为「有行即就绪」")
    ref_d, ref_n = ref
    need = ref_n * publish_ratio
    ok = n >= need
    return TableProbe(ds.name, expected, "有" if n > 0 else "无", ok, n,
                      f"完整性 {n:,} vs {ref_d}({ref_n:,}) 的 {publish_ratio:.0%}"
                      + ("" if ok else "　⚠️ 疑似半份发布"))
