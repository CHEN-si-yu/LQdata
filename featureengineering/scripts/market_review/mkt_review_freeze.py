"""构建 / 校验市场因子复审 freeze（R0 封存件）。

freeze 同时满足两个角色：
1. 复审自身的「规则+身份」封存：门定义文本、逐因子通道/方向预注册表、探索性预览披露、
   输入指纹（61 个市场 parquet + 标签 + 上游水位）、复审脚本 sha256；
2. `research batch` 的 source freeze（validate_freeze 要求覆盖全部 fea/factors/research
   *.py、main.py、configs、universe、protocol 哈希，且 release_id/cutoff 与当前发布一致）。

写入用 research.catalog.immutable_json（exclusive create + fsync）；
任何源文件/输入在 freeze 之后变化都会被 `verify` 或 batch 拒绝。

用法：
    python scripts/market_review/mkt_review_freeze.py build   [--out freeze/review_freeze_v1.json]
    python scripts/market_review/mkt_review_freeze.py verify  [--out ...]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

sys.path.insert(0, str(ROOT / "scripts" / "market_review"))
from mkt_review_lib import (  # noqa: E402
    M6_CONTROLS, M6_LAG, M6_T, M4_FAIL, M4_EXPLAIN, MIN_M4_OVERLAP,
    MIN_SIGNAL_DAYS, MIN_SIGNAL_YEARS, REVIEW_ID, sha256_file,
)

REVIEW_DIR = ROOT / "artifacts" / "research" / REVIEW_ID
DEFAULT_OUT = REVIEW_DIR / "freeze" / "review_freeze_v6.json"
TABLE_PATH = REVIEW_DIR / "registration" / "channel_direction_table.json"
DOC_PATH = ROOT / "doc" / "03_准入_市场因子审核体系_20261004.md"
BATCH_DEFS = ROOT / "conf" / "research" / "market_review_20261003"
AUDIT_BATCH = REVIEW_DIR / "audit_batch"

GATE_TEXT = (
    "M0 结构合同=main.py check 市场分支全过（含 align_fill 前缀豁免）；"
    "M1 工程因果=research verify 截断复算+两阶段增量重放 float32 精确一致；"
    "M2 覆盖与变化=声明起点后有效日≥98% ∧ 全历史去重值≥10 ∧ 每个有效年(≥60有限日)去重值≥2 ∧ "
    "声明对齐前缀外无前缀有限值；"
    "M3 口径与可用性=逐因子元数据成文 + audit-pit 静态检查；"
    "M4 去重=61×61 时序 Spearman(共同有效日≥250)，|ρ|≤0.90 过 / 0.90<|ρ|≤0.95 需论证 / |ρ|>0.95 判不过；"
    "M5 预测检验=ρ=mean(zrank(x)·zrank(y))，HAC-Bartlett(19) 95% 区间由 fea/eval.py::mean_uncertainty 给出，"
    "主期限 h∈{1,5}：declared 区间下界>0 ∧ 同号年数≥ceil(7×有效年数/9) ∧ 有效年数≥6；"
    "M6 增量检验=zrank(目标)~zrank(候选)+冻结控制集 zrank 的 HAC-OLS(Bartlett lag 19)，declared t≥2.0；"
    "M7 机制审议=追溯登记先于统计 + raw/declared 双列 + 冻结裁决规则树。"
    f"控制集={list(M6_CONTROLS)}；M6 lag={M6_LAG}、t≥{M6_T}；M4 阈值 {M4_EXPLAIN}/{M4_FAIL}、最短重叠 {MIN_M4_OVERLAP}；"
    f"单年最少共同有效日 {MIN_SIGNAL_DAYS}、有效年数下限 {MIN_SIGNAL_YEARS}。"
)

PREVIEW_DISCLOSURE = {
    "occurred": "2026-10-03 方案核验阶段（设计代理的可行性探针，未写任何文件）",
    "seen_aggregates": [
        "M5 探针：全部 61 个因子在 h=1 与 h=5 上做过一次探索性统计（聚合层面：h=1 有 15 个 |t|≥1.96，"
        "h=5 有 17 个 |t|≥1.96；在『正=顺周期』读法下约 10 个同时通过区间判定与分年规则；未用于调整任何阈值或方向）",
        "M4 探针：发现 2 对 |ρ|>0.95（mkt_am_ew_return↔mkt_am_up_ratio=0.973；"
        "mkt_am_dispersion↔mkt_dispersion=0.956）与 2 对 0.90–0.95",
        "M6 探针：一例风格投影示例（mkt_am_range_mean h=1 raw t=+3.18 → 控制后 declared t=−2.29）",
    ],
    "controls_applied": [
        "门定义与阈值来自股票 G1–G7 先例与本体系规则（0.95/0.90 沿自 G5；HAC 内核沿自 G4/G7），不来自探针数字",
        "M2 的细节修正（去重值下限、按年去重、对齐前缀豁免）仅关乎不误伤合法结构，与结果无关",
        "方向表按公式语义独立推导并在任何复审统计之前成文封存",
        "全部结论标注 retrospective_screening;historical_selection_is_not_untouched_holdout",
    ],
}


def _collect_sources():
    out = {}
    for sub in ("fea", "factors", "research"):
        for p in sorted((ROOT / sub).rglob("*.py")):
            out[p.relative_to(ROOT).as_posix()] = sha256_file(p)
    required = ["main.py", "conf/config.yaml", "conf/frequency.yaml",
                "conf/research/protocol.yaml", "conf/universe_frozen.tsv"]
    for rel in required:
        out[rel] = sha256_file(ROOT / rel)
    # 自愿收录：复审脚本 / 别名家族 / 登记定义 / 制度文档 / 方向表 / 候选清单
    for p in sorted((ROOT / "scripts" / "market_review").glob("*.py")):
        out[p.relative_to(ROOT).as_posix()] = sha256_file(p)
    for p in sorted(BATCH_DEFS.rglob("*.yaml")):
        out[p.relative_to(ROOT).as_posix()] = sha256_file(p)
    out["doc/03_准入_市场因子审核体系_20261004.md"] = sha256_file(DOC_PATH)
    out[TABLE_PATH.relative_to(ROOT).as_posix()] = sha256_file(TABLE_PATH)
    cl = AUDIT_BATCH / "candidate_list.json"
    out[cl.relative_to(ROOT).as_posix()] = sha256_file(cl)
    return out


def _collect_inputs(cfg):
    market = {}
    for p in sorted((ROOT / "data" / "market_factors").glob("*/year=*/data.parquet")):
        st = p.stat()
        market[p.relative_to(ROOT).as_posix()] = {
            "size": st.st_size, "mtime_ns": st.st_mtime_ns, "sha256": sha256_file(p)}
    labels = {}
    for h in (1, 5):
        for p in sorted((ROOT / "data" / "factors" / f"label_ret_{h}d").glob("year=*/data.parquet")):
            st = p.stat()
            labels[p.relative_to(ROOT).as_posix()] = {
                "size": st.st_size, "mtime_ns": st.st_mtime_ns, "sha256": sha256_file(p)}
    from fea.engine import Engine
    eng = Engine(cfg)
    deps = ["stock_daily", "stock_adj_factor", "index_daily", "stock_history_5min",
            "stock_limit_list", "stock_margin_detail", "tdx_daily", "tdx_blocks",
            "dc_blocks", "stock_main_fund_flow", "stock_market_distribution_history"]
    watermarks = {name: eng.watermark(name) for name in deps}
    return {"market_parquet": market, "label_parquet": labels, "upstream_watermarks": watermarks}


def build(cfg, out_path):
    import factors  # noqa: F401
    from research.catalog import immutable_json
    from fea.release import production_specs, definition

    production = json.loads((ROOT / "conf" / "production.json").read_text(encoding="utf-8"))
    prod_hash = {o["name"]: o["definition_sha256"] for o in production["objects"]}
    table = json.loads(TABLE_PATH.read_text(encoding="utf-8"))
    table_names = [f["name"] for f in table["factors"]]
    assert len(table_names) == 61 and len(set(table_names)) == 61

    definition_hashes = {}
    for spec in sorted(production_specs(cfg), key=lambda s: s.name):
        if not spec.is_market:
            continue
        live = definition(spec, cfg)
        if live != prod_hash[spec.name]:
            raise ValueError(f"{spec.name}: 现算定义哈希与生产清单不一致")
        definition_hashes[spec.name] = live
    assert sorted(definition_hashes) == sorted(table_names)

    release = json.loads((ROOT / "state" / "current_release.json").read_text(encoding="utf-8"))
    payload = {
        "schema_version": 1,
        "kind": "market_review_freeze",
        "review_id": REVIEW_ID,
        "stock_count": 2115,
        "sealed_at": __import__("datetime").datetime.now(
            __import__("datetime").timezone.utc).isoformat(),
        "reason": "市场因子复审封存 v6（取代 v5：M5 declared 置信区间的方向翻端实现修正——对 declared=-1 的因子"
                  "须对区间两端取负重排（[min(low·d,high·d), max(low·d,high·d)]），v5 前的实现把 low×d 直接当低端，"
                  "使 15 个负方向因子的 M5 门退化为『原始区间低端<0』而过宽；修正后 M5 门恢复为『declared 区间"
                  "排除 0』的原文判据。该修正使 M5 判定更严（无因子获得有利偏向），并补充负方向自测用例。"
                  "M6 的 declared=t×d 为标量符号翻转、无此问题。修正后重算 M4–M6 与裁决；v5 的门序澄清继续有效）"
                  "：门定义+方向表+源/输入身份+脚本哈希；兼作 research batch source freeze",
        "supersedes": "review_freeze_v5.json",
        "release_id": release["release_id"],
        "cutoff": release["cutoff"],
        "protocol_sha256": sha256_file(ROOT / "conf" / "research" / "protocol.yaml"),
        "gate_definitions": GATE_TEXT,
        "gate_definitions_sha256": sha256_file(DOC_PATH),
        "direction_table": table,
        "definition_hashes": definition_hashes,
        "source_sha256": _collect_sources(),
        "inputs_frozen": _collect_inputs(cfg),
        "preview_disclosure": PREVIEW_DISCLOSURE,
        "retains_seals": [],
    }
    immutable_json(out_path, payload)
    print(f"freeze 已写入：{out_path}")
    print(f"  source 文件 {len(payload['source_sha256'])} 个；市场 parquet "
          f"{len(payload['inputs_frozen']['market_parquet'])} 个；标签分区 "
          f"{len(payload['inputs_frozen']['label_parquet'])} 个")
    return payload


def verify(cfg, out_path):
    import factors  # noqa: F401
    frozen = json.loads(Path(out_path).read_text(encoding="utf-8"))
    assert frozen["schema_version"] == 1 and frozen["stock_count"] == 2115
    # 源哈希
    current = _collect_sources()
    for rel, expected in frozen["source_sha256"].items():
        if rel not in current:
            raise ValueError(f"freeze 源文件缺失：{rel}")
        if current[rel] != expected:
            raise ValueError(f"freeze 之后源文件发生变化：{rel}")
    missing = set(current) - set(frozen["source_sha256"])
    if missing:
        raise ValueError(f"freeze 之后新增源文件（需重新封存）：{sorted(missing)[:5]}")
    # protocol / release
    if sha256_file(ROOT / "conf" / "research" / "protocol.yaml") != frozen["protocol_sha256"]:
        raise ValueError("protocol 已变化")
    release = json.loads((ROOT / "state" / "current_release.json").read_text(encoding="utf-8"))
    if release["release_id"] != frozen["release_id"] or release["cutoff"] != frozen["cutoff"]:
        raise ValueError("生产发布指针已变化（需重新 freeze）")
    # 方向表
    if sha256_file(TABLE_PATH) != frozen["source_sha256"][TABLE_PATH.relative_to(ROOT).as_posix()]:
        raise ValueError("方向表已变化")
    # 输入
    inputs = _collect_inputs(cfg)
    for group in ("market_parquet", "label_parquet"):
        for rel, meta in frozen["inputs_frozen"][group].items():
            st = (ROOT / rel).stat()
            if st.st_size != meta["size"] or st.st_mtime_ns != meta["mtime_ns"]:
                raise ValueError(f"输入文件身份变化：{rel}")
            if sha256_file(ROOT / rel) != meta["sha256"]:
                raise ValueError(f"输入文件内容变化：{rel}")
        if set(inputs[group]) != set(frozen["inputs_frozen"][group]):
            raise ValueError(f"{group} 分区集合变化")
    print("freeze verify OK：源文件、协议、发布指针、方向表、输入指纹全部一致")
    return frozen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["build", "verify"])
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    from fea.config import load
    cfg = load()
    if args.mode == "build":
        build(cfg, args.out)
    else:
        verify(cfg, args.out)


if __name__ == "__main__":
    main()
