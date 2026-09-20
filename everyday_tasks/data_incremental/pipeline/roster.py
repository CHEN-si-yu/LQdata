"""固化名单的自动维护：**厂商名单同步** + **定期完整性 sweep**。

★★ 为什么这两件事必须挂在日更主流程里（而不是单独脚本）：
   用户 2026-09-19 的要求是「我以后只执行 `main.py`，就能更新数据」。
   凡是要手动跑第二条命令的事，实际上就等于没做 —— 会忘、会拖、会烂。
   所以名单同步和 sweep 都并进 `cmd_run`，用户无感。

两件事解决两类不同的洞：
  ① `sync`   —— 厂商**新上**的指数不会自动进固化名单（固化是拿"覆盖可复现"换来的取舍）。
                不同步 ⇒ 新指数永远不抓。**1 个请求**。
                新增的代码当轮就会被 `run_per_entity` 的 `todo_new` 路径**全量回填历史**
                （见 strategies.py：不在 `done.entities` 里的实体取 `ds.start`→now）。
  ② `sweep`  —— 已在 `done.entities` 里的实体，日更**只补尾部窗口**（约 6 个交易日），
                **更早的洞永远不自愈**。实测：2026-03 那 480 只的洞就是这么留了半年的。
                故每隔 `SWEEP_EVERY_DAYS` 天做一次全历史刷新兜底。**约 40 个请求**。

★ 只查不存：`index_ths_sector_categories` 作为**数据表**已于 2026-09-19 删除
  （快照无历史 ⇒ 回测前视）。这里只把它当**权威名单源**问一次，不落盘、不建表。
★ 只增不减：`sync` **只追加**新代码，从不删。厂商停供的代码留着 ——
  代价只是每次多一个空响应，好处是**厂商哪天恢复供应能自动发现**。
"""
from __future__ import annotations

import json
from datetime import date, datetime

from .. import paths
from ..core.client import ApiError, QueryLimitError, Client
from ..core import state
from .strategies import _Buf
from .. import registry as R

THS_DS = "index_ths_daily"
SRC_PATH = "/index/ths_sector_categories"      # 权威名单源（只查不存）
ROSTER_PATH = paths.LOCAL / "conf" / "ths_index_codes.txt"
SWEEP_STATE = paths.STATE / "roster_sweep.json"

SWEEP_EVERY_DAYS = 30      # 全历史刷新周期；太密是浪费配额，太疏则洞留得久
SWEEP_BATCH = 20           # 起始批量；超服务端行数上限时自动对半劈（见 _fetch_group）
ADD_WARN = 200             # 单次新增超过这个数就在报告里显著提示（防厂商名单异常）


def _load_roster() -> list[str]:
    return [ln.strip() for ln in ROSTER_PATH.read_text(encoding="utf-8").splitlines()
            if ln.strip() and not ln.startswith("#")]


# ------------------------------------------------------------------ ① 名单同步
def sync(cfg: dict, client: Client, write: bool = True) -> dict:
    """问厂商当前的 THS 名单，把新增的追加进固化名单。返回统计（供报告用）。

    `write=False`（dry-run 用）：照常查询与比对，但**不落盘**。
    """
    try:
        rows = client.fetch_all(SRC_PATH, {}, page_size=1000, expect_rows=False)
        if not getattr(rows, "complete", True):
            return {"ok": False, "note": "名单分页不完整，未修改名单，下轮重试"}
    except Exception as exc:                     # noqa: BLE001
        return {"ok": False, "note": f"名单源查询失败：{str(exc)[:80]}"}

    vendor = {str(r.get("index_code")) for r in rows if r.get("index_code")}
    if not vendor:
        # ★ 空返回**绝不当成"厂商清空了名单"** —— 那会导致把名单误删成空、静默停更。
        return {"ok": False, "note": "厂商返回 0 条名单，跳过（不改名单）"}

    known = set(_load_roster())
    added = sorted(vendor - known)
    if not added:
        return {"ok": True, "added": 0, "vendor": len(vendor), "roster": len(known)}

    # 只追加、不重排 —— 保持文件 diff 最小、可读
    if write:
        with ROSTER_PATH.open("a", encoding="utf-8") as f:
            f.write(f"# ---- {date.today().isoformat()} 日更自动同步新增 {len(added)} 只 ----\n")
            for c in added:
                f.write(c + "\n")
        # 进程内的注册表是模块加载时读的，追加后要就地刷新，本轮才能立刻抓到它们
        ds = R.get(THS_DS)
        ds.entity_codes = _load_roster()

    res = {"ok": True, "added": len(added), "vendor": len(vendor),
           "roster": len(R.get(THS_DS).entity_codes), "sample": added[:5],
           "written": write}
    if len(added) > ADD_WARN:
        res["warn"] = (f"单次新增 {len(added)} 只，异常地多 —— 请人工确认厂商名单是否正常"
                       f"（已照常追加；代码若无效只会返回空，不会污染数据）")
    return res


# ------------------------------------------------------------------ ② 定期 sweep
def _due() -> tuple[bool, str | None]:
    if not SWEEP_STATE.exists():
        return True, None
    try:
        last = json.loads(SWEEP_STATE.read_text(encoding="utf-8")).get("last_sweep")
    except (OSError, ValueError):
        return True, None
    if not last:
        return True, None
    age = (date.today() - date.fromisoformat(str(last)[:10])).days
    return age >= SWEEP_EVERY_DAYS, str(last)[:10]


def _fetch_group(cli: Client, ds, grp: list[str], start: str, end: str,
                 depth: int = 0) -> list[dict]:
    """取一组；命中服务端行数上限就对半劈、递归重试（不丢实体）。

    ★ 批量**不能写死**：线上批量 100 是配尾部窗口的（每只几行），全历史下
      老代码每只可达 7,000+ 行，20 只一批实测冲到 16 万行，超服务端
      `page*page_size ≤ 10万` 的硬上限 —— 写死批量的首版跑出 8 个失败批。
    """
    payload = {ds.code_param: grp, "start_time": start, "end_time": end, **ds.params}
    try:
        rows = cli.fetch_all(ds.path, payload, page_size=8000,
                             method=ds.method, expect_rows=False)
        if not getattr(rows, "complete", True):
            raise RuntimeError("名单历史分页不完整，未推进成功日期")
        return rows
    except QueryLimitError as exc:
        if len(grp) == 1 or depth >= 8:
            raise
        mid = len(grp) // 2
        return (_fetch_group(cli, ds, grp[:mid], start, end, depth + 1)
                + _fetch_group(cli, ds, grp[mid:], start, end, depth + 1))


def sweep(cfg: dict, client: Client, runner=None) -> dict:
    """对名单里每只代码取一次全历史并幂等落盘。保证"名单上的都齐全"是构造性的。"""
    import pandas as pd

    ds = R.get(THS_DS)
    roster = list(ds.entity_codes or [])
    if not roster:
        return {"ok": False, "note": "名单为空，拒绝 sweep（会造成静默停更）"}

    end = datetime.now().strftime("%Y-%m-%d")
    man = state.Manifest.load(THS_DS)
    buf = _Buf(ds, man, flush_rows=200_000, flush_batches=400)
    got = added = failed = 0
    n_batch = (len(roster) + SWEEP_BATCH - 1) // SWEEP_BATCH

    for i in range(0, len(roster), SWEEP_BATCH):
        grp = roster[i:i + SWEEP_BATCH]
        try:
            recs = _fetch_group(client, ds, grp, ds.start, end)
        except Exception as exc:                 # noqa: BLE001
            failed += 1
            if runner:
                runner.note(f"   ⚠️ sweep 第 {i//SWEEP_BATCH + 1}/{n_batch} 批失败："
                            f"{str(exc)[:70]}")
            continue
        if recs:
            buf.add(pd.DataFrame(recs))
            got += len(recs)
        if runner and (i // SWEEP_BATCH + 1) % 20 == 0:
            runner.note(f"   sweep {i//SWEEP_BATCH + 1}/{n_batch} 批，"
                        f"取回 {got:,} 行，新增 {buf.rows:,} 行")

    buf.flush()
    man.save()
    added = buf.rows
    previous = {}
    if SWEEP_STATE.exists():
        try:
            previous = json.loads(SWEEP_STATE.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            pass
    record = {**previous, "last_attempt": end, "added": added,
              "fetched": got, "failed_batches": failed}
    if not failed:
        record["last_sweep"] = end
    state._atomic_json(SWEEP_STATE, record)
    return {"ok": not bool(failed), "added": added, "fetched": got, "failed": failed,
            "codes": len(roster), "note": f"{failed} 批失败，未推进成功日期，下轮重试" if failed else ""}


def sweep_if_due(cfg: dict, client: Client, runner=None) -> dict | None:
    """到期才跑。未到期返回 None（报告里不出现，保持日报干净）。"""
    due, last = _due()
    if not due:
        return None
    res = sweep(cfg, client, runner)
    res["last_before"] = last
    return res
