"""只读生产筛选：年度画像、统一协议评价和可复核建议，不改变生产清单。

入口也可为 ``python -m research.selection --source-freeze ... --jobs 4``。
每个年度/对象保存完整单元，资源护栏中止后用 --resume 继续。
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import datetime as dt
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import re
import time
import uuid

if __name__ == "__main__":
    for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[_key] = "1"

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from fea.eval import evaluate_panel, model_value_panel, summarize_daily, _json_safe
from fea.manifest import _atomic_json
from fea.resources import memory_limits, safe_jobs
from .protocol import load_protocol

SELECTION_CONTRACT = "production_readonly_selection_v1"
_YEAR_CONTEXT = None


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 ** 2), b""):
            h.update(block)
    return h.hexdigest()


def project_file(cfg, value):
    path = Path(value)
    path = (path if path.is_absolute() else cfg.root / path).resolve()
    if not path.is_relative_to(cfg.root.resolve()):
        raise ValueError("Selection paths must stay inside project: " + str(path))
    return path


def selection_output_root(cfg, value, source_freeze):
    path = project_file(cfg, value)
    artifacts = cfg.project_path("artifacts", "artifacts")
    if path == artifacts or not path.is_relative_to(artifacts):
        raise ValueError("Keep selection in its own project artifacts namespace")
    freeze = project_file(cfg, source_freeze)
    batches = artifacts / "research" / "batches"
    if freeze.is_relative_to(batches):
        parts = freeze.relative_to(batches).parts
        if len(parts) < 2 or not path.is_relative_to(batches / parts[0]):
            raise ValueError("Selection output must stay in the source-freeze batch namespace")
    return path


def verify_source_freeze(cfg, path, release_id, protocol_sha256):
    path = project_file(cfg, path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1 or payload.get("release_id") != release_id:
        raise ValueError("Source freeze does not match the selected production release")
    if payload.get("protocol_sha256") != protocol_sha256:
        raise ValueError("Source freeze protocol identity differs")
    sources = payload.get("source_sha256")
    if not isinstance(sources, dict) or not sources:
        raise ValueError("Source freeze needs project source/config SHA256 identities")
    for relative, digest in sorted(sources.items()):
        if Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise ValueError("Frozen source path must be project-relative")
        source = project_file(cfg, relative)
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest) or file_hash(source) != digest:
            raise ValueError("Frozen source/config changed: " + relative)
    return {"path": str(path.relative_to(cfg.root.resolve())), "sha256": file_hash(path), "payload": payload}


def profile_panel(values, dates, name, kind="stock", valid_start="2018-01-01", min_pairs=30, rank=None):
    """纯画像；真实起点前的对齐区间与有效区间分别记录。"""
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    dates = np.asarray(dates, dtype=str)
    if values.ndim != 2 or len(dates) != len(values) or kind not in {"stock", "market", "label"}:
        raise ValueError("Profile needs aligned dates and a known object kind")
    if kind == "market" and values.shape[1] != 1:
        raise ValueError("Market object must be one native scalar per date")
    ranks = np.asarray(rank, dtype=float) if rank is not None else None
    if ranks is not None and ranks.shape != values.shape:
        raise ValueError("Rank panel shape differs")
    rows = []
    for t, (date, row) in enumerate(zip(dates, values)):
        finite = np.isfinite(row)
        x = row[finite]
        active = date >= valid_start
        unique = len(np.unique(x))
        rows.append({"name": str(name), "kind": kind, "trade_date": date,
                     "year": int(date[:4]), "active": bool(active), "n_pool": values.shape[1],
                     "finite": int(finite.sum()), "coverage": float(finite.mean()),
                     "inf_count": int(np.isinf(row).sum()), "unique_values": int(unique),
                     "cross_std": float(x.std()) if len(x) and kind != "market" else np.nan,
                     "constant_cross_section": bool(len(x) and unique == 1 and kind != "market"),
                     "zero_fraction_finite": float(np.mean(x == 0)) if len(x) else np.nan,
                     "value_mean": float(x.mean()) if len(x) else np.nan,
                     "value_min": float(x.min()) if len(x) else np.nan,
                     "value_max": float(x.max()) if len(x) else np.nan,
                     "sufficient_pairs": bool(len(x) >= (1 if kind == "market" else min_pairs)),
                     "finite_rank": int(np.isfinite(ranks[t]).sum()) if ranks is not None else 0,
                     "inf_rank": int(np.isinf(ranks[t]).sum()) if ranks is not None else 0})
    return pd.DataFrame(rows)


def summarize_profile(daily):
    if daily.empty:
        return {}
    active = daily[daily.active]
    observed = daily[daily.finite > 0]
    good = active[active.finite > 0]
    kind = str(daily.kind.iloc[0])
    annual = []
    for year, part in daily.groupby("year", sort=True):
        use = part[part.active]
        annual.append({"year": int(year), "dates": len(part), "active_dates": len(use),
                       "coverage_mean": float(use.coverage.mean()) if len(use) else None,
                       "coverage_min": float(use.coverage.min()) if len(use) else None,
                       "finite_dates": int((use.finite > 0).sum()),
                       "variable_dates": int(((use.unique_values > 1) & (use.finite > 0)).sum()) if kind != "market" else None,
                       "temporal_std": float(use.value_mean.std(ddof=0)) if kind == "market" and use.value_mean.notna().any() else None})
    return _json_safe({"name": str(daily.name.iloc[0]), "kind": kind, "dates": len(daily),
            "active_dates": len(active), "finite_active_dates": len(good),
            "first_observed_value_date": str(observed.trade_date.iloc[0]) if len(observed) else None,
            "first_active_value_date": str(good.trade_date.iloc[0]) if len(good) else None,
            "last_observed_value_date": str(observed.trade_date.iloc[-1]) if len(observed) else None,
            "active_coverage_mean": float(active.coverage.mean()) if len(active) else None,
            "active_coverage_median": float(active.coverage.median()) if len(active) else None,
            "active_coverage_min": float(active.coverage.min()) if len(active) else None,
            "sufficient_pair_dates": int(active.sufficient_pairs.sum()),
            "constant_cross_section_dates": int(active.constant_cross_section.sum()),
            "variable_active_dates": int((active.unique_values > 1).sum()) if kind != "market" else None,
            "sparse_active_dates": int(((active.unique_values > 0) & (active.unique_values <= 20)).sum()) if kind != "market" else None,
            "zero_fraction_finite_median": float(good.zero_fraction_finite.median()) if len(good) else None,
            "prefix_finite_cells": int(daily.loc[~daily.active, "finite"].sum()),
            "inf_count": int(daily.inf_count.sum()), "inf_rank": int(daily.inf_rank.sum()),
            "annual": annual,
            "interpretation": "Observed output start is not proof of the historical vendor field availability date"})


def definition_duplicate_groups(objects):
    """声明公式相同仅触发复查，不宣称实现或数值相同。"""
    groups = {}
    for obj in objects:
        formula = str(obj.get("formula", "")).strip()
        if not formula or obj["kind"] == "label":
            continue
        key = json.dumps({"kind": obj["kind"], "formula": re.sub(r"\s+", " ", formula),
                          "deps": sorted(obj.get("deps", [])), "start": obj.get("valid_start"),
                          "higher_is_better": obj.get("higher_is_better"),
                          "warmup_days": obj.get("warmup_days")}, sort_keys=True)
        groups.setdefault(key, []).append(obj["name"])
    return sorted([sorted(names) for names in groups.values() if len(names) > 1])


def recommendations(objects, profiles, evaluations, duplicate_groups=None, redundancy=None, field_evidence=None):
    """工程分层与研究证据并列；任何结果均不是自动退出/盈利准入。"""
    profile_map = {p["name"]: p for p in profiles}
    eval_map = {(e["name"], int(e["horizon"])): e for e in evaluations}
    duplicate_map = {name: group for group in (duplicate_groups or []) for name in group}
    pair_map = {}
    for pair in redundancy or []:
        rank = pair.get("rank_corr", {})
        value = pair.get("model_value_corr", {})
        # Recall/one shared rank is insufficient evidence of redundant values.
        # Thresholds only prioritize joint review, never apply an exit.
        rank_mean, value_mean = rank.get("mean"), value.get("mean")
        supported = (rank.get("days", 0) >= 60 and value.get("days", 0) >= 60
                     and rank_mean is not None and value_mean is not None
                     and abs(rank_mean) >= .995 and abs(value_mean) >= .995
                     and pair.get("common_fraction_pool", 0) >= .3
                     and pair.get("coverage_jaccard", 0) >= .8)
        if not supported:
            continue
        for name in (pair.get("a"), pair.get("b")):
            if name:
                pair_map.setdefault(name, []).append(pair)
    rows = []
    for obj in objects:
        name, kind = obj["name"], obj["kind"]
        p = profile_map.get(name, {})
        primary = [eval_map.get((name, h), {}) for h in (1, 5)]
        fields = (field_evidence or {}).get(name, {"status": "not_provided"})
        reasons, diagnostics = [], []
        action = "retain_engineering_review"
        if kind == "label":
            action, reasons = "retain_formal_benchmark", ["Official future-return benchmark; not an input factor"]
        elif p.get("errors") or p.get("inf_count", 0) or p.get("inf_rank", 0):
            action, reasons = "repair_review", ["Actual structural/read/infinite-value evidence requires review"]
        elif fields.get("status") == "verified_missing":
            action, reasons = "repair_review", ["Explicit field evidence reports a required raw field missing"]
        elif not p or p.get("finite_active_dates", 0) == 0:
            action, reasons = "engineering_review", ["No observed finite output in declared active dates"]
        elif kind == "stock" and p.get("variable_active_dates", 0) == 0:
            action, reasons = "engineering_review", ["No observed varying stock cross-section; inspect field/event semantics"]
        elif name in duplicate_map or name in pair_map:
            action, reasons = "redundancy_review", ["Declared duplicate or pair evidence needs semantic and full-history review"]
        else:
            reasons = ["Output is present; preserve it pending explicit evidence review"]
        if fields.get("status") == "not_provided":
            diagnostics.append("Required-field truth/effective-start evidence has not been supplied")
        if kind == "stock":
            if any(e.get("days", 0) < 60 for e in primary):
                research_tier = "insufficient_primary_observations"
            elif all(e.get("rankic") is not None and e["rankic"] > 0 for e in primary):
                research_tier = "positive_primary_historical_observations"
            elif all(e.get("rankic") is not None and e["rankic"] < 0 for e in primary):
                research_tier = "negative_after_declared_direction_review"
            else:
                research_tier = "mixed_primary_historical_observations"
        else:
            research_tier = "formal_benchmark" if kind == "label" else "native_market_context_no_cross_section_ic"
        rows.append({"name": name, "kind": kind, "recommendation": action, "reasons": reasons,
                     "diagnostics": diagnostics, "research_evidence_tier": research_tier,
                     "profile": p, "primary_evaluation": primary if kind == "stock" else [],
                     "definition_duplicate_group": duplicate_map.get(name, []),
                     "redundancy_pairs": pair_map.get(name, []), "field_evidence": fields,
                     "validated_alpha": False, "model_contribution": None, "automatic_action": False})
    return rows


def canonical_keys(dates, codes, market=False):
    dates = np.asarray(dates, dtype=str)
    if market:
        return pa.table({"trade_date": pa.array(dates, type=pa.string())})
    return pa.table({"trade_date": pa.array(np.repeat(dates, len(codes)), type=pa.string()),
                     "stock_code": pa.array(np.tile(np.asarray(codes, dtype=str), len(dates)), type=pa.string())})


def advise_clean(path, enabled):
    if not enabled:
        return False
    if not hasattr(os, "posix_fadvise") or not hasattr(os, "POSIX_FADV_DONTNEED"):
        return False
    try:
        fd = os.open(path, os.O_RDONLY)
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)
    except OSError:
        return False  # Best-effort advice cannot change numerical conclusions.
    return True


def read_panel(path, expected, market=False, label=False, fadvise_clean=False, n_codes=None):
    """Arrow 完整键快路径；保持冻结股票轴，不以零值补缺失。"""
    columns = [*expected.column_names, "value"] + ([] if market else ["rank"])
    try:
        if pq.ParquetFile(path).schema_arrow.names != columns:
            raise ValueError("Production columns differ from the formal schema")
        table = pq.read_table(path, columns=columns, use_threads=False)
        if table.column_names != columns or not pa.types.is_float32(table.schema.field("value").type):
            raise ValueError("Production columns/value type differ from the formal schema")
        if not market and not pa.types.is_float32(table.schema.field("rank").type):
            raise ValueError("Production rank type differs from the formal schema")
        for key in expected.column_names:
            if not (pa.types.is_string(table.schema.field(key).type) or pa.types.is_large_string(table.schema.field(key).type)):
                raise ValueError("Production key type is not string")
        if not table.select(expected.column_names).cast(expected.schema).combine_chunks().equals(expected):
            raise ValueError("Production grid/order/unique keys differ from the complete task axis")
        c = 1 if market else int(n_codes or len(pc.unique(expected["stock_code"])))
        if c <= 0 or expected.num_rows % c:
            raise ValueError("Invalid frozen stock panel width")
        values = table["value"].to_numpy().astype(np.float64).reshape(-1, c)
        ranks = None if market else table["rank"].to_numpy().astype(np.float64).reshape(-1, c)
        if label and ranks is not None and np.isfinite(ranks).any():
            raise ValueError("Formal label rank must remain missing")
        return values, ranks
    finally:
        advise_clean(path, fadvise_clean)


class MemoryGuard:
    """本作业同时计入页缓存，父进程约每秒检查；不操作全机缓存。"""
    def __init__(self, policy):
        v2 = Path("/sys/fs/cgroup/memory.current")
        if v2.exists():
            self.usage = v2
            self.limit = Path("/sys/fs/cgroup/memory.max")
            self.stat = Path("/sys/fs/cgroup/memory.stat")
            self.anon_key = "anon"
        else:
            self.usage = Path("/sys/fs/cgroup/memory/memory.usage_in_bytes")
            self.limit = Path("/sys/fs/cgroup/memory/memory.limit_in_bytes")
            self.stat = Path("/sys/fs/cgroup/memory/memory.stat")
            self.anon_key = "total_rss"
        if not self.usage.exists():
            raise RuntimeError("Cgroup charged-memory counters are required for production selection")
        text = self.limit.read_text().strip()
        container = int(text) if text != "max" else None
        if container and container > 1 << 62:
            container = None
        self.limits = memory_limits(container, policy)
        self.peak_charged = self.peak_anon = self.samples = 0

    def check(self):
        charged = int(self.usage.read_text())
        stat = dict(line.split() for line in self.stat.read_text().splitlines())
        anon = int(stat.get(self.anon_key, 0))
        self.samples += 1
        self.peak_charged = max(self.peak_charged, charged)
        self.peak_anon = max(self.peak_anon, anon)
        if charged > self.limits["charged"] or anon > self.limits["anon"]:
            raise MemoryError("Selection cgroup memory guard exceeded; completed annual units can be resumed")

    def report(self):
        return {"limits_bytes": self.limits, "peak_charged_bytes": self.peak_charged,
                "peak_anonymous_bytes": self.peak_anon, "samples": self.samples,
                "source": str(self.usage), "sample_scope": "selection only; charged usage includes clean page cache"}


def _unit_complete(folder, identity):
    path = folder / "complete.json"
    if not path.exists():
        return None
    record = json.loads(path.read_text())
    if record.get("identity") != identity:
        raise ValueError("Completed unit identity differs from the frozen request")
    for relative, digest in record["files"].items():
        target = (folder / relative).resolve()
        if not target.is_relative_to(folder.resolve()) or file_hash(target) != digest:
            raise ValueError("Completed selection unit changed")
    return record


def _annual_unit(obj):
    context = _YEAR_CONTEXT
    folder = Path(context["out"]) / "units" / str(context["year"]) / obj["name"]
    if not folder.resolve().is_relative_to(Path(context["out"]).resolve()):
        raise ValueError("Selection unit folder escapes the run namespace")
    folder.mkdir(parents=True, exist_ok=True)
    identity = {"request_sha256": context["request_sha256"], "year": context["year"], "name": obj["name"]}
    completed = _unit_complete(folder, identity)
    if completed:
        return completed
    files, errors, summary = {}, [], None
    market = obj["kind"] == "market"
    try:
        expected = context["market_keys"] if market else context["stock_keys"]
        path = Path(obj["root"]) / obj["name"] / ("year=" + str(context["year"])) / "data.parquet"
        if not path.resolve().is_relative_to(Path(obj["root"]).resolve()):
            raise ValueError("Production partition escapes the declared factor root")
        values, ranks = read_panel(path, expected, market, obj["kind"] == "label", context["fadvise_clean"], context["n_codes"])
        daily = profile_panel(values, context["dates"], obj["name"], obj["kind"], obj["valid_start"],
                              context["min_pairs"], ranks)
        target = folder / "profile.parquet"
        daily.to_parquet(target, index=False, compression="zstd")
        files[target.name] = file_hash(target)
        summary = summarize_profile(daily)
        if summary["inf_count"] or summary["inf_rank"]:
            errors.append("Infinite production values/ranks observed")
        if obj["kind"] == "stock" and context["evaluate"] and not errors:
            evaluation_values = values.copy()
            before_start = np.asarray(context["dates"], dtype="datetime64[D]") < np.datetime64(obj["valid_start"], "D")
            evaluation_values[before_start] = np.nan
            model_values = model_value_panel(evaluation_values, obj["higher_is_better"], (.01, .99), 10.)
            metrics = evaluate_panel(model_values, context["labels"], context["dates"],
                                     context["min_pairs"], context["bins"], obj["name"])
            metrics["value_transform"] = context["value_transform"]
            metrics["score_head"] = "pure_factor"
            target = folder / "evaluation.parquet"
            pq.write_table(_evaluation_table(metrics), target, compression="zstd")
            files[target.name] = file_hash(target)
    except Exception as exc:
        errors.append(type(exc).__name__ + ": " + str(exc))
    record = {"identity": identity, "files": files, "profile": summary, "errors": errors}
    _atomic_json(_json_safe(record), folder / "complete.json")
    return record


def _run_year(objects, context, jobs, guard, completed):
    global _YEAR_CONTEXT
    _YEAR_CONTEXT = context
    if "fork" not in mp.get_all_start_methods() and jobs == 1:
        for obj in objects:
            guard.check()
            completed(_annual_unit(obj))
            guard.check()
        return
    # Fork inherits the small canonical keys/formal-label arrays as readonly
    # pages. It never shares a whole multi-year/multi-factor matrix.
    if "fork" not in mp.get_all_start_methods():
        raise RuntimeError("Parallel production selection requires Linux fork; use --jobs 1 elsewhere")
    pool = ProcessPoolExecutor(max_workers=jobs, mp_context=mp.get_context("fork"))
    pending = {pool.submit(_annual_unit, obj) for obj in objects}
    try:
        while pending:
            guard.check()
            done, pending = wait(pending, timeout=1, return_when=FIRST_COMPLETED)
            for result in done:
                completed(result.result())
    except BaseException:
        # Stop only children created by this executor, preserving production
        # readers/writers and every complete selection checkpoint.
        for child in list((getattr(pool, "_processes", None) or {}).values()):
            if child.is_alive():
                child.terminate()
        pool.shutdown(wait=True, cancel_futures=True)
        raise
    else:
        pool.shutdown(wait=True)


def _evaluation_table(frame):
    table = pa.Table.from_pandas(frame, preserve_index=False)
    schema = pa.schema([pa.field(f.name, pa.list_(pa.int64()) if f.name == "bin_counts"
                                else pa.list_(pa.float64()) if f.name == "bin_returns" else f.type)
                        for f in table.schema])
    return table.cast(schema)


def _export_tables(folder, objects, guard=None):
    profile_writer = evaluation_writer = None
    profiles, evaluations = [], []
    try:
        for obj in objects:
            if guard:
                guard.check()
            parts, metrics, errors = [], [], []
            for unit in sorted((folder / "units").glob("*/" + obj["name"] + "/complete.json")):
                record = json.loads(unit.read_text())
                errors.extend(record["errors"])
                if "profile.parquet" in record["files"]:
                    table = pq.read_table(unit.parent / "profile.parquet")
                    if profile_writer is None:
                        profile_writer = pq.ParquetWriter(folder / "profile_daily.parquet", table.schema, compression="zstd")
                    profile_writer.write_table(table)
                    parts.append(table.to_pandas())
                if "evaluation.parquet" in record["files"]:
                    table = pq.read_table(unit.parent / "evaluation.parquet")
                    if evaluation_writer is None:
                        evaluation_writer = pq.ParquetWriter(folder / "evaluation_daily.parquet", table.schema, compression="zstd")
                    evaluation_writer.write_table(table)
                    metrics.append(table.to_pandas())
            profile = summarize_profile(pd.concat(parts, ignore_index=True)) if parts else {"name": obj["name"], "kind": obj["kind"]}
            profiles.append({**profile, "errors": errors})
            if metrics:
                evaluations.extend(summarize_daily(pd.concat(metrics, ignore_index=True)))
    finally:
        if profile_writer is not None:
            profile_writer.close()
        if evaluation_writer is not None:
            evaluation_writer.close()
    return profiles, evaluations


def _csv_records(path, records):
    frame = pd.DataFrame([{k: json.dumps(_json_safe(v), ensure_ascii=False) if isinstance(v, (dict, list)) else v
                           for k, v in row.items()} for row in records])
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def run_selection(args, cfg):
    import factors  # noqa: F401 — direct Python/module entry also registers approved definitions.
    from fea.dates import Calendar, int_to_str
    from fea.release import read_release, production_specs
    from fea.universe import frozen_codes
    from fea.upstream import Upstream
    if cfg.raw.get("_sandbox"):
        raise ValueError("Production selection must reference the project production release")
    protocol = load_protocol(cfg)
    horizons = tuple(sorted(set(int(h) for h in args.horizons.split(","))))
    if not {1, 5}.issubset(horizons) or not set(horizons).issubset(protocol.horizons):
        raise ValueError("Selection needs primary 1/5 horizons, with auxiliaries only 3/10/20")
    jobs = safe_jobs(args.jobs, cfg.raw.get("resources"))
    guard = MemoryGuard(cfg.raw.get("resources"))
    guard.check()
    started = time.monotonic()
    outroot = selection_output_root(cfg, args.out, args.source_freeze)
    run_id = args.resume or ("s_" + dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S") + "_" + uuid.uuid4().hex[:10])
    if not re.fullmatch(r"[A-Za-z0-9_-]+", run_id):
        raise ValueError("Invalid selection run ID")
    folder = outroot / "runs" / run_id
    if not folder.resolve().is_relative_to(cfg.root.resolve()):
        raise ValueError("Selection run folder escapes project")
    if args.resume:
        if not (folder / "request.json").exists():
            raise ValueError("Selection resume request not found")
    else:
        folder.mkdir(parents=True, exist_ok=False)
    previous_final = None
    if args.resume and (folder / "run.json").exists():
        saved_report = json.loads((folder / "run.json").read_text())
        if saved_report.get("status") in {"complete", "complete_with_engineering_errors"}:
            previous_final = saved_report
    complete_count = 0
    try:
        with read_release(cfg) as release:
            codes = frozen_codes(cfg)
            if codes is None or len(codes) != protocol.expected_stocks or release["stock_count"] != len(codes):
                raise ValueError("Selection requires the frozen 2115-stock task axis")
            protocol_sha = file_hash(cfg.contract_root / "conf/research/protocol.yaml")
            freeze = verify_source_freeze(cfg, args.source_freeze, release["release_id"], protocol_sha)
            if freeze["payload"].get("cutoff") != release["cutoff"] or freeze["payload"].get("stock_count") != len(codes):
                raise ValueError("Source freeze date/pool differs from the selected release")
            evidence_inputs, evidence_payloads = {}, {}
            for key, value in (("field_evidence", args.field_evidence), ("redundancy_report", args.redundancy_report)):
                if value:
                    path = project_file(cfg, value)
                    payload = json.loads(path.read_text(encoding="utf-8"))
                    if isinstance(payload, dict) and payload.get("release_id") not in (None, release["release_id"]):
                        raise ValueError("Selection evidence references a different production release")
                    evidence_inputs[key] = {"path": str(path.relative_to(cfg.root.resolve())), "sha256": file_hash(path)}
                    evidence_payloads[key] = payload
            if not isinstance(evidence_payloads.get("field_evidence", {}), dict):
                raise ValueError("Field evidence must map object names to explicit evidence records")
            specs = production_specs(cfg)
            if {s.name for s in specs} != {o["name"] for o in release["objects"]}:
                raise ValueError("Current production set differs from published objects")
            want = set(args.factors or [s.name for s in specs])
            if want - {s.name for s in specs}:
                raise ValueError("Only approved published names can be selected")
            objects = []
            released = {o["name"]: o for o in release["objects"]}
            for spec in specs:
                if spec.name not in want:
                    continue
                if not re.fullmatch(r"[A-Za-z0-9_.-]+", spec.name):
                    raise ValueError("Invalid production object name")
                objects.append({"name": spec.name, "kind": "label" if spec.is_label else "market" if spec.is_market else "stock",
                                "valid_start": spec.resolved_start(cfg), "formula": spec.formula,
                                "deps": list(spec.deps), "warmup_days": spec.warmup_days,
                                "higher_is_better": spec.higher_is_better, "root": str(cfg.factor_root(spec)),
                                "recipe": released[spec.name]["recipe"], "definition_sha256": released[spec.name]["definition_sha256"]})
            if not objects:
                raise ValueError("No selected production objects")
            if args.profile_years[0] != 2018 or args.profile_years[1] < 2018:
                raise ValueError("Production profiles must retain the 2018 output floor")
            if args.evaluation_years[0] < 2018 or args.evaluation_years[1] < args.evaluation_years[0]:
                raise ValueError("Invalid evaluation year interval")
            calendar = Calendar.load(Upstream(cfg.upstream, cfg.root))
            cutoff = int(release["cutoff"].replace("-", ""))
            years = list(range(2018, min(args.profile_years[1], cutoff // 10000) + 1))
            request = {"contract": SELECTION_CONTRACT, "release_id": release["release_id"], "cutoff": release["cutoff"],
                       "universe_sha256": release["universe_sha256"], "stock_count": len(codes),
                       "source_freeze": freeze, "protocol": protocol.to_dict(), "protocol_identity": protocol.identity,
                       "profile_years": years, "evaluation_years": args.evaluation_years, "horizons": list(horizons),
                       "evidence_inputs": evidence_inputs,
                       "fadvise_clean": bool(args.fadvise_clean), "objects": objects,
                       "common_sha256": release.get("common_sha256"), "kernel_version": release.get("kernel_version"),
                       "raw_input_semantics": "Published stored production values; historical outputs are not freshly recomputed vendor vintages",
                       "current_source_identity_semantics": "Frozen project source/config SHA256 plus published factor recipes and partition identities"}
            request = json.loads(json.dumps(request, ensure_ascii=False, allow_nan=False))
            request_digest = hashlib.sha256(json.dumps(request, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()
            if args.resume:
                saved = json.loads((folder / "request.json").read_text())
                if saved != request:
                    raise ValueError("Resume requires exactly the same source/release/protocol/object request")
                if previous_final is not None:
                    if previous_final.get("request_sha256") != request_digest or not previous_final.get("outputs_sha256"):
                        raise ValueError("Completed selection evidence identity is invalid")
                    for relative, digest in previous_final["outputs_sha256"].items():
                        target = (folder / relative).resolve()
                        if not target.is_relative_to(folder.resolve()) or file_hash(target) != digest:
                            raise ValueError("Completed selection output changed")
                    if verify_source_freeze(cfg, args.source_freeze, release["release_id"], protocol_sha) != freeze:
                        raise ValueError("Frozen source record changed during completed-run validation")
                    for entry in evidence_inputs.values():
                        if file_hash(project_file(cfg, entry["path"])) != entry["sha256"]:
                            raise ValueError("Selection input evidence changed during completed-run validation")
                    guard.check()
                    print(json.dumps({"run_id": run_id, "status": previous_final["status"], "reused_complete_run": True}), flush=True)
                    return previous_final
            else:
                _atomic_json(request, folder / "request.json")
            def completed(record):
                nonlocal complete_count
                complete_count += 1
                if complete_count % 50 == 0:
                    print(f"selection: {complete_count} annual units complete", flush=True)
                _atomic_json({"status": "running", "run_id": run_id, "completed_units": complete_count,
                              "resources": guard.report()}, folder / "run.json")
            for year in years:
                guard.check()
                days = calendar.between(year * 10000 + 101, min(year * 10000 + 1231, cutoff))
                if not len(days):
                    continue
                dates = np.asarray([int_to_str(int(day)) for day in days])
                stock_keys, market_keys = canonical_keys(dates, codes), canonical_keys(dates, codes, True)
                evaluate = args.evaluation_years[0] <= year <= args.evaluation_years[1]
                labels = {}
                if evaluate and any(o["kind"] == "stock" for o in objects):
                    for horizon in horizons:
                        path = cfg.factors_dir / f"label_ret_{horizon}d" / f"year={year}" / "data.parquet"
                        if not path.resolve().is_relative_to(cfg.root.resolve()):
                            raise ValueError("Formal label partition escapes project")
                        label, _ = read_panel(path, stock_keys, label=True, fadvise_clean=args.fadvise_clean, n_codes=len(codes))
                        if np.isinf(label).any():
                            raise ValueError("Infinite formal stored label")
                        label.setflags(write=False)
                        labels[horizon] = label
                        guard.check()
                context = {"year": year, "out": str(folder), "request_sha256": request_digest,
                           "dates": dates, "stock_keys": stock_keys, "market_keys": market_keys,
                           "n_codes": len(codes),
                           "labels": labels, "evaluate": evaluate, "min_pairs": protocol.min_pairs,
                           "bins": protocol.bins, "value_transform": protocol.value_transform,
                           "fadvise_clean": args.fadvise_clean}
                _run_year(objects, context, jobs, guard, completed)
            guard.check()
            profiles, evaluations = _export_tables(folder, objects, guard)
            field_evidence = evidence_payloads.get("field_evidence", {})
            redundancy = evidence_payloads.get("redundancy_report", {})
            pairs = redundancy.get("refined_pairs", []) if isinstance(redundancy, dict) else redundancy
            duplicates = definition_duplicate_groups(objects)
            decisions = recommendations(objects, profiles, evaluations, duplicates, pairs, field_evidence)
            end_freeze = verify_source_freeze(cfg, args.source_freeze, release["release_id"], protocol_sha)
            if end_freeze != freeze:
                raise ValueError("Frozen source record changed during selection")
            for entry in evidence_inputs.values():
                if file_hash(project_file(cfg, entry["path"])) != entry["sha256"]:
                    raise ValueError("Selection input evidence changed during the run")
            pointer = json.loads((cfg.state_dir / "current_release.json").read_text())
            if pointer.get("release_id") != release["release_id"]:
                raise ValueError("Production release changed during selection")
            for filename, records in (("profiles", profiles), ("evaluation_summary", evaluations), ("selection", decisions)):
                _atomic_json(_json_safe({"records": records}), folder / (filename + ".json"))
                _csv_records(folder / (filename + ".csv"), records)
            report = {"status": "complete_with_engineering_errors" if any(p["errors"] for p in profiles) else "complete",
                      "run_id": run_id, "output": str(folder), "request_sha256": request_digest,
                      "release_id": release["release_id"], "source_freeze_sha256": freeze["sha256"],
                      "objects": len(objects), "completed_units": complete_count, "horizons": list(horizons),
                      "resources": guard.report(), "elapsed_seconds": time.monotonic() - started,
                      "definition_duplicate_groups": duplicates, "recommendations": decisions,
                      "evidence_policy": "No automatic apply, rankic threshold deletion, direction flip or model-contribution claim",
                      "label_source": "formal_store_only", "fadvise_clean_requested": bool(args.fadvise_clean),
                      "fadvise_clean_available": hasattr(os, "posix_fadvise") and hasattr(os, "POSIX_FADV_DONTNEED")}
            report["outputs_sha256"] = {path.name: file_hash(path) for path in folder.iterdir()
                                        if path.is_file() and path.name != "run.json"}
            report = _json_safe(report)
            _atomic_json(report, folder / "run.json")
            _atomic_json({"run_id": run_id, "run": str(folder / "run.json")}, outroot / "latest.json")
            print(json.dumps({"run_id": run_id, "status": report["status"], "output": str(folder),
                              "elapsed_seconds": report["elapsed_seconds"]}, ensure_ascii=False), flush=True)
            return report
    except BaseException as exc:
        failure_target = folder / "run.json"
        if previous_final is not None:
            failure_target = folder / "resume_failures" / (uuid.uuid4().hex + ".json")
        _atomic_json({"status": "interrupted_resumable" if isinstance(exc, MemoryError) else "failed",
                      "run_id": run_id, "error": repr(exc), "completed_units": complete_count,
                      "resources": guard.report(), "resume": "Reuse the same request and --resume " + run_id}, failure_target)
        raise


def add_arguments(parser):
    parser.add_argument("--source-freeze", required=True, help="项目内统一源码/配置与发布冻结 JSON")
    parser.add_argument("--jobs", type=int, default=4, help="年度内按对象并行，默认4，受项目/CPU上限约束")
    parser.add_argument("--factors", nargs="*", help="只画像/评价指定正式对象；默认全部批准对象")
    parser.add_argument("--profile-years", nargs=2, type=int, default=[2018, 2026], metavar=("Y0", "Y1"))
    parser.add_argument("--evaluation-years", nargs=2, type=int, default=[2022, 2026], metavar=("Y0", "Y1"))
    parser.add_argument("--horizons", default="1,3,5,10,20", help="至少1,5；默认所有正式期限")
    parser.add_argument("--out", default="artifacts/research/batches/reselection_20261001_v1/selection", help="项目内筛选报告根目录")
    parser.add_argument("--resume", help="续作相同冻结请求的既有选择运行ID")
    parser.add_argument("--redundancy-report", help="可选项目内共同覆盖rank/value精判证据JSON")
    parser.add_argument("--field-evidence", help="可选项目内按对象名索引的原字段核验JSON")
    parser.add_argument("--fadvise-clean", action="store_true", help="只对已读项目Parquet提示释放干净页缓存")


def main():
    from fea.config import load
    from fea.resources import configure
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    args = parser.parse_args()
    cfg = load()
    configure(cfg.raw.get("resources"))
    report = run_selection(args, cfg)
    return int(report["status"] != "complete")


if __name__ == "__main__":
    raise SystemExit(main())
