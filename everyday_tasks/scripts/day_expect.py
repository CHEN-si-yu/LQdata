"""「跑前立预期 → 跑后逐条验收」工具（用户 2026-09-15 交办）。

用户原话：
    「你先预估一下数据更新的预期，看看是否真的准确的更新了新的一天，并且冗余爬取的
     数据与之前的数据计算出来 MD5 保持一致，做好预期规划，等待爬取完成后验证。」

所以这里做两件事，顺序不能反：

  ① `snapshot`（**跑之前**跑）：把"应该发生什么"存成基线
     · 每张表**现在**的本地水位 / 总行数
     · 每张表在**尾部窗口内每一天**的单日指纹（md5 + 行数）← 冗余重抓的比对基准
     · 服务端对"新的一天 / T-delay"各有多少行 ← 新一天应有的行数
     存到 `state/backtest/expect_<date>.json`

  ② `verify`（跑完之后跑）：逐条对，输出 `state/backtest/REPORT_day_<date>.md`
     A. **新的一天**：本地水位是否推进到 `T-delay`；新一天的行数是否 = 服务端行数
     B. **冗余重抓一致性（核心）**：尾部窗口里每个 (表, 日期) 的 md5 必须与跑前**逐字节一致**；
        不一致的逐条分类：
          · 行数变多   → 我们自己补齐（正常）
          · 行数不变、只有数值列变 → 厂商重算（记录，不算我们的错）
          · **行数变少 → 🚩 红旗**（要查，可能是我们写丢了）
     C. 这次没抓到的日期（跑前就没有、跑后仍然没有）单独列出来

只读：除写自己那份 json/md 报告外，不碰任何数据。
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data_incremental import paths, registry as R          # noqa: E402
from data_incremental.core import state                     # noqa: E402
from data_incremental.pipeline import dayhash as DH         # noqa: E402
from data_incremental.pipeline import probe as probe_mod    # noqa: E402

BACKTEST = paths.BACKTEST_DIR
CAL_FILE = paths.STATE / "calendar_ext.json"


def calendar() -> list[str]:
    try:
        return sorted(str(x)[:10] for x in json.loads(CAL_FILE.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError):
        return []


def _tail_days(cal: list[str], T: str, n: int = 7) -> list[str]:
    if T not in cal:
        return []
    i = cal.index(T)
    return [cal[max(0, i - k)] for k in range(n)]


def snapshot(date: str, out: Path | None = None) -> dict:
    from data_incremental import config as C
    from data_incremental.core.client import Client

    cal = calendar()
    tail = set(_tail_days(cal, date, 7))
    cli = Client(C.load())

    tables: dict[str, dict] = {}
    for ds in R.enabled():
        man = state.Manifest.load(ds.name)
        rec: dict = {
            "mode": ds.mode, "freq": ds.freq, "delay": int(ds.delay_days or 0),
            "watermark": man.max_partition_date(), "rows": man.partition_rows(),
            "ledger": {}, "server": {},
        }
        # 尾部窗口内的单日指纹（冗余重抓的比对基准）
        try:
            led = DH.load_ledger(ds.name)
            if len(led):
                for _, row in led.iterrows():
                    d = str(row["date"])[:10]
                    if d in tail:
                        rec["ledger"][d] = {"md5": str(row["md5"]), "rows": int(row["rows"])}
        except Exception:  # noqa: BLE001
            pass
        # 服务端：新的一天（以及 delay≥1 表的 T-1）各多少行
        if ds.mode != "snapshot":
            days = [date] if ds.delay_days == 0 else [date, _shift(cal, date, int(ds.delay_days))]
            for d in [x for x in days if x]:
                try:
                    rec["server"][d] = probe_mod.probe_rows(cli, ds, d, cal)
                except Exception:  # noqa: BLE001
                    rec["server"][d] = None
        tables[ds.name] = rec

    doc = {"date": date, "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
           "tail_days": sorted(tail, reverse=True), "tables": tables}
    p = out or BACKTEST / f"expect_{date}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"✔ 基线已存：{p}（{len(tables)} 张表）")
    _render_plan(doc)
    return doc


def _shift(cal: list[str], T: str, back: int) -> str | None:
    if T not in cal:
        return None
    i = cal.index(T)
    return cal[i - back] if i - back >= 0 else None


def _render_plan(doc: dict) -> None:
    """把"预期"渲染成人看的清单，一并落到 REPORT 里（跑前也能看）。

    ★ 两处口径（2026-09-15 晚自查修正）：
      · 「期望新增」必须取**期望那天**（delay=1 的表是 T-1）的服务端行数，
        取 T 那天的话数字是别的日子、还会让验收判据空转；
      · 「水位必须推进」只对**闸门管的日频表**（daily_full / daily_sparse）成立；
        `index_weight`（月频）/季度 / 不定期表的水位本来就可能不动，不该判 ⚠️。
    """
    date = doc["date"]
    cal = calendar()
    daily = {d.name for d in R.daily_tables()}
    lines = [f"# {date} 日更「预期 → 验收」清单", "",
             f"基线生成：{doc['created_at']}　（跑之前）", "",
             "## 预期：本轮应该发生什么", "",
             "| 数据集 | 模式 | 现在水位 | 期望水位 | 期望新增(=服务端行数) | 尾部窗口基准 |",
             "|:--|:--|:--|:--|--:|--:|"]
    for name, t in sorted(doc["tables"].items()):
        if t["freq"] == "snapshot":
            continue
        exp = _shift(cal, date, t["delay"]) if name in daily else None
        srv = t["server"].get(exp) if exp else t["server"].get(date)
        lines.append(f"| {name} | {t['mode']} | {t['watermark'] or '—'} | {exp or '—（非日频，看数据）'} | "
                     f"{'—' if srv is None else f'{srv:,}'} | {len(t['ledger'])} 天 |")
    lines += ["", "## 验收规则", "",
              "1. **新的一天**：水位要推进到「期望水位」；新一天本地行数要 = 服务端行数（非 0 时）；",
              "2. **冗余重抓必须幂等**：尾部窗口内每个 (表, 日期) 的 md5 要与上表基线**完全一致**；",
              "   不一致的必须能解释：行数变多=我们补齐；行数不变只有数值列变=厂商重算；",
              "   **行数变少 = 🚩 红旗**（要查是不是我们写丢了）；",
              "3. 跑前没有、跑后仍没有的日期单独列出（可能是 delay/厂商没发，不是错）。"]
    p = BACKTEST / f"PLAN_day_{date}.md"
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"✔ 预期清单：{p}")


def verify(date: str, reprobe: bool = False) -> int:
    src = BACKTEST / f"expect_{date}.json"
    if not src.exists():
        print(f"✘ 找不到基线 {src} —— 必须先在跑之前执行 snapshot")
        return 2
    base = json.loads(src.read_text(encoding="utf-8"))
    cal = calendar()
    # ★ 基线是"跑之前"存的：有些表（如 margin_detail）的服务端数据在**基线之后、
    #   开跑之前**才发布，基线里记的是 0 → 行数比对会被跳过。
    #   `--reprobe` 只对这些"基线为 0/None 而本地抓到了行"的格子定点复探一次，
    #   让"本地 == 服务端"这条硬判据对它们也成立（请求数 = 这类表的数量，很小）。
    cli = None
    if reprobe:
        from data_incremental import config as C
        from data_incremental.core.client import Client
        cli = Client(C.load())

    rows_new: list[str] = []          # 新一天的验收
    idem_ok: list[str] = []
    changed: list[dict] = []
    missing: list[str] = []

    daily = {d.name for d in R.daily_tables()}
    for name, t in sorted(base["tables"].items()):
        ds = R.get(name)
        man = state.Manifest.load(name)
        after_wm, after_rows = man.max_partition_date(), man.partition_rows()
        exp = _shift(cal, date, int(ds.delay_days or 0)) if name in daily else None
        srv = t["server"].get(exp) if exp else t["server"].get(date)
        # ★ 行数能不能跟服务端"1:1 对"取决于探测方式（2026-09-15 晚自查修）：
        #   `probe_rows` 对 per_entity / 带变体的表只查**一个代表实体 / 第一个变体**
        #   （tdx_daily 查 1 个板块、history_5min 查 1 只股票的 48 根、ths_hot 查"热股"），
        #   而本地是全表全变体 —— 拿它们相减必然"不等"，那是探测口径问题，不是数据问题。
        #   只有"一次请求就拿全市场"的 range 表 / 无变体的 per_date 表才可比。
        comparable = ds.mode == "range" or (ds.mode == "per_date" and not ds.variants)

        # A. 新的一天（只有闸门管的日频表才"必须"推进；其余表看数据）
        if ds.freq != "snapshot":
            led_now = DH.load_ledger(name)
            got = 0
            if len(led_now):
                hit = led_now[led_now["date"].astype(str).str[:10] == (exp or date)]
                got = int(hit.iloc[-1]["rows"]) if len(hit) else 0
            ok_wm = (after_wm or "") >= (exp or "") if exp else True
            if (srv in (None, 0)) and got > 0 and cli is not None:
                # 基线之后才发布的数据：定点复探，让行数比对也能成立
                try:
                    srv = probe_mod.probe_rows(cli, ds, exp or date, cal)
                except Exception:  # noqa: BLE001
                    srv = None
            if not comparable:
                mark = "—（抽样探测，不可比）"
                ok_rows = True
            else:
                # ★ 只把"本地**少于**服务端"当缺口：`probe_rows` 对不返回 total 的接口
                #   只能拿回一页（page_size=3），那会让本地行数"显得"多于服务端
                #   （实测 stock_dragon_tiger 本地 557 vs 探测 3）—— 那是探测口径，不是缺数据。
                ok_rows = (srv in (None, 0)) or (got >= srv)
                mark = "✔" if (ok_wm and ok_rows) else "⚠️"
            rows_new.append(
                f"| {name} | {t['watermark'] or '—'} | {after_wm or '—'} | "
                f"{exp or '—（非日频）'} | {got:,} | "
                f"{'—' if srv is None else f'{srv:,}'} | {mark} |")
            if not ok_wm and exp:
                missing.append(f"{name}：水位 {after_wm} 未到 {exp}")

        # B. 冗余重抓一致性
        led_now = DH.load_ledger(name)
        now = {}
        if len(led_now):
            for _, row in led_now.iterrows():
                now[str(row["date"])[:10]] = {"md5": str(row["md5"]), "rows": int(row["rows"])}
        for d, b in t["ledger"].items():
            n = now.get(d)
            if n is None:
                changed.append({"name": name, "date": d, "kind": "消失",
                                "old_rows": b["rows"], "new_rows": 0, "note": "台账里没有这天了"})
                continue
            if n["md5"] == b["md5"]:
                idem_ok.append(f"{name}@{d}")
            else:
                if n["rows"] > b["rows"]:
                    kind, note = "补齐", "行数变多 = 我们自己补的（正常）"
                elif n["rows"] == b["rows"]:
                    kind, note = "厂商重算", "行数不变、只有值变 = 上游重算（记录即可）"
                else:
                    kind, note = "🚩变少", "行数变少 —— 必须查"
                changed.append({"name": name, "date": d, "kind": kind,
                                "old_rows": b["rows"], "new_rows": n["rows"], "note": note})

    red = [c for c in changed if c["kind"].startswith("🚩")]
    other = [c for c in changed if not c["kind"].startswith("🚩")]

    out = [f"# {date} 日更验收报告", "",
           f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}　（跑之后，零写入）", "",
           "## 结论", "",
           f"- 冗余重抓**逐字节一致**：{len(idem_ok)} 个 (表, 日期) 指纹",
           f"- 有不一致：{len(other)} 个（分类见下）",
           f"- 🚩 行数变少：{len(red)} 个",
           f"- 水位没跟上的表：{len(missing)} 张" + (f"　{missing}" if missing else ""),
           "", "## A. 新的一天（本地 vs 服务端）", "",
           "| 数据集 | 跑前水位 | 跑后水位 | 期望 | 本地新一天行数 | 服务端行数 | 判定 |",
           "|:--|:--|:--|:--|--:|--:|:--|", *rows_new, ""]
    if changed:
        out += ["## B. 冗余重抓出现差异的 (表, 日期)", "",
                "| 数据集 | 日期 | 类型 | 跑前行数 | 跑后行数 | 说明 |",
                "|:--|:--|:--|--:|--:|:--|"]
        for c in sorted(changed, key=lambda x: x["kind"]):
            out.append(f"| {c['name']} | {c['date']} | {c['kind']} | {c['old_rows']:,} | "
                       f"{c['new_rows']:,} | {c['note']} |")
        out.append("")
    out += ["## C. 冗余重抓一致明细（抽样 40 条）", "", "```",
            *idem_ok[:40], "```", ""]
    text = "\n".join(out) + "\n"
    p = BACKTEST / f"REPORT_day_{date}.md"
    p.write_text(text, encoding="utf-8")
    print(text)
    print(f"\n（报告已写入 {p}）")
    return 1 if (red or missing) else 0


def _ledger_now(name: str) -> dict:
    """当前台账里这张表**全部**日期的指纹（读本地 parquet，零请求）。"""
    led = DH.load_ledger(name)
    if not len(led):
        return {}
    return {str(r["date"])[:10]: {"md5": str(r["md5"]), "rows": int(r["rows"])}
            for _, r in led.iterrows()}


def compare(old_file: str) -> int:
    """**跨天/跨时刻比对同一批日期的指纹**（用户 2026-09-15 交办的核心诉求）。

    用户原话：「我们的 MD 回测时间是 7 天，等于今天有 6 个和昨天算出来的应该是重合的，
    对比一下今天算出来的和昨天算出来的结果是否一致（确认未来数据更新不会导致历史数据
    发生变化，服务器运营商没有偷偷修改历史数据）」。

    做法：拿一份**早先时刻的快照**（`snapshot` 存下来的 `expect_<date>.json` 里的 ledger 段，
    每天存一份就有"昨天"）与**当前台账**逐 (表, 日期) 比 md5，差异分四类：
      · 行数变多   → 我们自己补齐（正常，例如回刷窗口/主键修正回填）
      · 行数不变、md5 变 → 值被重算（上游重算 or 我们重抓，需看当天发生了什么）
      · 行数变少   → 🚩 必须查
      · 只在旧快照有 → 该日期已滚出 7 天窗口（正常）
    """
    src = Path(old_file)
    if not src.is_absolute():
        src = BACKTEST / old_file
    if not src.exists():
        print(f"✘ 找不到快照 {src}")
        return 2
    old = json.loads(src.read_text(encoding="utf-8"))
    print(f"旧快照：{src.name}（{old.get('created_at')}）")

    same = 0
    added: list[tuple] = []
    changed: list[dict] = []
    dropped: list[tuple] = []
    for name, t in sorted(old["tables"].items()):
        now = _ledger_now(name)
        for d, b in (t.get("ledger") or {}).items():
            n = now.get(d)
            if n is None:
                dropped.append((name, d))
                continue
            if n["md5"] == b["md5"]:
                same += 1
                continue
            if n["rows"] > b["rows"]:
                kind = "补齐（我们自己加的）"
            elif n["rows"] == b["rows"]:
                kind = "值被重算（上游重算 / 我们重抓）"
            else:
                kind = "🚩 行数变少"
            changed.append({"name": name, "date": d, "kind": kind,
                            "old_rows": b["rows"], "new_rows": n["rows"]})
        for d, n in now.items():
            if d not in (t.get("ledger") or {}) and d >= old["date"]:
                added.append((name, d, n["rows"]))

    print(f"\n逐 (表, 日期) 比对结果：")
    print(f"  ✔ 完全一致            {same} 个")
    print(f"  ⚠️ 有差异              {len(changed)} 个")
    print(f"  ＋ 新日期（旧快照没有）{len(added)} 个")
    print(f"  － 已滚出窗口/消失      {len(dropped)} 个")
    if changed:
        print("\n  差异明细：")
        for c in sorted(changed, key=lambda x: (x["name"], x["date"])):
            print(f"    {c['name']:32} {c['date']}  {c['kind']}  行数 {c['old_rows']:,} → {c['new_rows']:,}")
    red = [c for c in changed if c["kind"].startswith("🚩")]
    print("\n结论：" + ("✅ 没有任何一天的历史被偷偷改动（差异仅来自我们自己补数据）"
                     if not red and all("补齐" in c["kind"] for c in changed)
                     else "见上表逐条判断"))
    return 1 if red else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["snapshot", "plan", "verify", "compare"])
    ap.add_argument("--old", default="", help="compare 用：早先时刻的 expect_*.json")
    ap.add_argument("--date", default="2026-09-15")
    ap.add_argument("--reprobe", action="store_true",
                    help="对「基线时服务端还没有、但本地抓到了行」的表定点复探服务端行数")
    a = ap.parse_args()
    if a.cmd == "snapshot":
        snapshot(a.date)
        return 0
    if a.cmd == "compare":
        return compare(a.old or f"expect_{a.date}.json")
    if a.cmd == "plan":
        # 只按已存的基线重渲染清单（**零请求、不覆盖基线**）——
        # 跑起来之后再想改口径看预期，用这个，别重跑 snapshot。
        src = BACKTEST / f"expect_{a.date}.json"
        if not src.exists():
            print(f"✘ 找不到基线 {src}")
            return 2
        _render_plan(json.loads(src.read_text(encoding="utf-8")))
        return 0
    return verify(a.date, reprobe=a.reprobe)


if __name__ == "__main__":
    raise SystemExit(main())
