"""Project-owned production library and complete-batch handoff contract.

Candidate code never becomes production merely because it was imported. Data
remain at the established annual paths. Readers take a shared daily lock and
refuse an unfinished batch; no historical data copies/backups are created.
"""
from __future__ import annotations

from contextlib import contextmanager
import datetime as dt
import hashlib
import inspect
import json
from pathlib import Path
from functools import wraps

from .manifest import Manifest, _atomic_json
from .spec import REGISTRY
_SOURCE_HASH_CACHE = {}

# Shared numerical code also changes a factor's meaning. Updating it requires
# an explicit reviewed kernel version, rather than mixing two historical rules.
_COMMON_KERNELS = ("context.py", "mathx.py", "prices.py", "deriv.py", "panel.py",
                   "dates.py", "universe.py", "store.py", "spec.py", "targets.py",
                   "chips.py", "chipwall.py", "closing30.py", "field_expansion.py",
                   "history.py", "intraday.py", "market_morning.py", "open5.py")


def common_definition(cfg):
    import ast
    files = {}
    for name in _COMMON_KERNELS:
        path = cfg.contract_root / "fea" / name
        if path.exists():
            tree = ast.parse(path.read_text(encoding="utf-8"))
            files[name] = hashlib.sha256(ast.dump(tree, include_attributes=False).encode()).hexdigest()
    return hashlib.sha256(_canonical(files).encode()).hexdigest()


def kernel_version(cfg):
    value = cfg.raw.get("compute", {}).get("kernel_version", 1)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("compute.kernel_version must be a positive integer")
    return value


def _canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def definition(spec, cfg):
    """Identity of a definition; descriptive comments do not change identity."""
    import ast
    source = inspect.getsourcefile(spec.fn) if spec.fn else None
    if source and Path(source).exists():
        info = Path(source).stat()
        key = (source,info.st_size,info.st_mtime_ns)
        if key not in _SOURCE_HASH_CACHE:
            tree = ast.parse(Path(source).read_text(encoding="utf-8"))
            _SOURCE_HASH_CACHE[key] = hashlib.sha256(ast.dump(tree, include_attributes=False).encode()).hexdigest()
        source_hash = _SOURCE_HASH_CACHE[key]
    else:
        source_hash = None
    align = spec.align_fill
    if isinstance(align, float) and align != align:
        align = "nan"
    fields = dict(name=spec.name, version=spec.version, formula=spec.formula,
                  deps=list(spec.deps), start=spec.resolved_start(cfg),
                  warmup_days=spec.warmup_days, higher_is_better=spec.higher_is_better,
                  is_market=spec.is_market, is_label=spec.is_label,
                  forward_days=spec.forward_days, align_fill=align,
                  fin_fields=list(spec.fin_fields), lagged_ok=list(spec.lagged_ok),
                  source_sha256=source_hash)
    return hashlib.sha256(_canonical(fields).encode()).hexdigest()


def registry_path(cfg):
    return cfg.input_contract_path("production_registry", "conf/production.json")


def _library_write(fn):
    @wraps(fn)
    def wrapped(cfg, *args, **kwargs):
        if cfg.raw.get("_sandbox"):
            raise ValueError("Sandbox cannot mutate production library")
        import fcntl
        cfg.state_dir.mkdir(parents=True, exist_ok=True)
        with (cfg.state_dir / "daily.lock").open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                return fn(cfg, *args, **kwargs)
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)
    return wrapped


def load_library(cfg):
    path = registry_path(cfg)
    if not path.exists():
        raise FileNotFoundError("Production library missing; run main.py library bootstrap once")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("schema_version") != 1 or not isinstance(raw.get("objects"), list):
        raise ValueError("Invalid production library")
    names = [x["name"] for x in raw["objects"]]
    if len(set(names)) != len(names):
        raise ValueError("Duplicate production library names")
    return raw


@_library_write
def bootstrap_library(cfg):
    """Explicit one-time adoption of existing objects, without efficacy claims."""
    path = registry_path(cfg)
    if path.exists():
        return load_library(cfg)
    entries = []
    for name, spec in sorted(REGISTRY.items()):
        if not spec.enabled:
            continue
        entries.append(dict(name=name, definition_sha256=definition(spec, cfg),
                            definition_version=spec.version,deps=list(spec.deps),origin="legacy", research_review="pending", status="production"))
    result = dict(schema_version=1, created_at=dt.datetime.now(dt.timezone.utc).isoformat(),
                  universe_sha256=_file_hash(cfg.frozen_universe) if cfg.frozen_universe else None,
                  default_start=cfg.default_start, common_sha256=common_definition(cfg),
                  kernel_version=kernel_version(cfg), objects=entries)
    _atomic_json(result, path)
    return result


def production_specs(cfg, *, check_definition=True):
    raw = load_library(cfg)
    if raw.get("default_start") != cfg.default_start:
        raise ValueError("Production library start differs from task contract")
    universe = _file_hash(cfg.frozen_universe) if cfg.frozen_universe else None
    if raw.get("universe_sha256") != universe:
        raise ValueError("Production library universe differs from frozen task contract")
    if check_definition and (raw.get("common_sha256") != common_definition(cfg)
                             or raw.get("kernel_version") != kernel_version(cfg)):
        raise ValueError("Shared numerical kernel changed; review with library accept-common before rebuilding")
    out = []
    for item in raw["objects"]:
        if item.get("status") != "production":
            continue
        name = item["name"]
        if name not in REGISTRY and item.get("candidate_id"):
            from research.catalog import spec_for_candidate
            candidate_spec = spec_for_candidate(cfg, item["candidate_id"])
            # Use the normal registration guard where possible. The candidate
            # catalog already validates all input/lag/causal metadata.
            REGISTRY[name] = candidate_spec
        if name not in REGISTRY:
            raise ValueError(f"Approved production implementation missing: {name}")
        spec = REGISTRY[name]
        if not spec.enabled:
            continue
        if check_definition and definition(spec, cfg) != item["definition_sha256"]:
            raise ValueError(f"Approved definition changed: {name}; review and accept a new version")
        out.append(spec)
    # Parents may still be computation nodes after being retired as outputs.
    # They cannot be unreviewed candidate dependencies.
    allowed = {x["name"] for x in raw["objects"]}
    for spec in out:
        unapproved = [d for d in spec.deps if d in REGISTRY and d not in allowed]
        if unapproved:
            raise ValueError(f"Unapproved parent dependency of {spec.name}: {unapproved}")
    return out


def _record_event(cfg, event):
    mark_dirty(cfg, "production_library_changed")
    payload = dict(event, at=dt.datetime.now(dt.timezone.utc).isoformat())
    token = hashlib.sha256(_canonical(payload).encode()).hexdigest()
    path = cfg.contract_root / "artifacts/library/events" / (token + ".json")
    if path.exists():
        raise FileExistsError(path)
    _atomic_json(payload, path)
    return token


def mark_dirty(cfg, reason):
    if not cfg.raw.get("_sandbox"):
        _atomic_json({"reason":reason,"at":dt.datetime.now(dt.timezone.utc).isoformat()},
                     cfg.state_dir/"publication_dirty.json")


@_library_write
def approve_candidate(cfg, candidate_id, spec, evidence):
    """Adopt a candidate only after its project-local lifecycle gate succeeds."""
    if getattr(cfg, "raw", {}).get("_sandbox"):
        raise ValueError("Research sandbox cannot change production library")
    if not evidence or not isinstance(evidence, dict):
        raise ValueError("Approval requires immutable evaluation and engineering evidence")
    from research.catalog import spec_for_candidate
    from research.catalog import Catalog
    if Catalog(cfg).state(candidate_id) != "shadow" or not str(evidence.get("review_reason","")).strip():
        raise ValueError("Production requires shadow state and an explicit review reason")
    expected = spec_for_candidate(cfg, candidate_id)
    if spec.name != expected.name or definition(spec, cfg) != definition(expected, cfg):
        raise ValueError("Approval does not match immutable candidate definition")
    from research.lifecycle import _check_evidence
    verified = _check_evidence(cfg, candidate_id, evidence.get("evaluation_id"), evidence.get("verification_id"))
    for key in ("evaluation_id", "verification_id"):
        if verified[key] != evidence.get(key):
            raise ValueError("Candidate approval evidence identity mismatch")
    library = load_library(cfg)
    if any(x["name"] == spec.name for x in library["objects"]):
        raise ValueError("Production name already used; create a new candidate version")
    approved = {x["name"] for x in library["objects"] if x.get("status") == "production"}
    private_names = {x["candidate_id"] for x in Catalog(cfg).records("candidates")}
    unapproved = [d for d in spec.deps if (d in REGISTRY or d in private_names) and d not in approved]
    if unapproved:
        raise ValueError(f"Candidate parents must be independently approved for production first: {unapproved}")
    event_id = _record_event(cfg, dict(action="approve_candidate", candidate_id=candidate_id,
                                     name=spec.name, evidence=evidence))
    library["objects"].append(dict(name=spec.name, candidate_id=candidate_id,
                                   definition_version=spec.version,
                                   deps=list(spec.deps),
                                   definition_sha256=definition(spec, cfg), origin="research",
                                   research_review="approved", status="production", approval_event=event_id))
    _atomic_json(library, registry_path(cfg))
    return {"candidate_id": candidate_id, "name": spec.name, "approval_event": event_id}


@_library_write
def accept_existing(cfg, name, reason):
    if not reason.strip():
        raise ValueError("A review reason is required")
    library = load_library(cfg)
    item = next((x for x in library["objects"] if x["name"] == name), None)
    if not item or item.get("origin") != "legacy" or name not in REGISTRY:
        raise ValueError("Only an already adopted legacy definition can be accepted here")
    old = item["definition_sha256"]
    new = definition(REGISTRY[name], cfg)
    if new != old and REGISTRY[name].version <= item.get("definition_version",REGISTRY[name].version):
        raise ValueError("Changed legacy numerical implementation must increment FactorSpec.version before acceptance/rebuild")
    event = _record_event(cfg, dict(action="accept_existing", name=name, before=old, after=new, reason=reason))
    item.update(definition_sha256=new, definition_version=REGISTRY[name].version,
                deps=list(REGISTRY[name].deps), approval_event=event)
    _atomic_json(library, registry_path(cfg))
    return item


@_library_write
def accept_common(cfg, reason):
    if not reason.strip():
        raise ValueError("A review reason is required")
    library = load_library(cfg)
    new = common_definition(cfg)
    version = kernel_version(cfg)
    if version <= int(library.get("kernel_version", 1)):
        raise ValueError("Increment compute.kernel_version before accepting shared numerical changes")
    event = _record_event(cfg, dict(action="accept_common", before=library.get("common_sha256"),
                                   after=new, kernel_version=version, reason=reason))
    library.update(common_sha256=new, kernel_version=version, common_approval_event=event)
    _atomic_json(library, registry_path(cfg))
    return {"common_sha256":new, "kernel_version":version, "approval_event":event}


@_library_write
def set_status(cfg, name, status, reason):
    if status not in {"production", "paused", "retired"} or not reason.strip():
        raise ValueError("Status and review reason are required")
    library = load_library(cfg)
    item = next((x for x in library["objects"] if x["name"] == name), None)
    if not item:
        raise ValueError("Unknown production object")
    if status != "production":
        dependents = [x["name"] for x in library["objects"] if x.get("status")=="production"
                      and name in x.get("deps",getattr(REGISTRY.get(x["name"]),"deps",()))]
        if dependents:
            raise ValueError(f"Still required by production dependencies: {dependents}")
    event = _record_event(cfg, dict(action="status", name=name, before=item["status"], after=status, reason=reason))
    item.update(status=status, approval_event=event)
    _atomic_json(library, registry_path(cfg))
    return item


def _file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def publish(cfg, specs, cutoff, run_id, report, monitor):
    """Publish one completed object set after quality/input-stability checks.

    Caller owns daily exclusive lock. Files are annual, atomically overwritten;
    this manifest plus reader lock forms a consistent handoff, not versioned
    immutable data copies. Failed/inflight batches are unavailable to readers.
    """
    if report.get("upstream_changed_during_run") or report.get("unexplained_changes") or report.get("gone"):
        raise ValueError("Cannot publish unstable or unexplained batch")
    if monitor and (monitor.get("status") in {"failed", "error"} or monitor.get("errors")):
        raise ValueError("Cannot publish a batch failing quality checks")
    cache_path = cfg.state_dir / "publication_hashes.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    entries = []
    for spec in specs:
        man = Manifest.load(cfg.state_dir, spec.name)
        parts = []
        for year, meta in sorted(man.partitions.items()):
            path = cfg.factor_root(spec) / spec.name / ("year=" + str(year)) / "data.parquet"
            if not path.exists():
                raise FileNotFoundError(path)
            relative = str(path.resolve().relative_to(cfg.root.resolve()))
            stat = path.stat()
            identity = [stat.st_size, stat.st_mtime_ns]
            previous = cache.get(relative, {})
            digest = previous.get("sha256") if previous.get("identity") == identity else None
            # A newly closed file on this shared filesystem can settle its
            # mtime only after the first read. Capture a stable identity *after*
            # hashing; never publish a digest with stale pre-read metadata.
            for attempt in range(3):
                if not digest:
                    digest = _file_hash(path)
                observed = path.stat()
                observed_identity = [observed.st_size, observed.st_mtime_ns]
                if observed_identity == identity:
                    stat = observed
                    cache[relative] = dict(identity=identity, sha256=digest)
                    break
                stat, identity, digest = observed, observed_identity, None
            else:
                raise RuntimeError("Partition identity did not stabilize while publishing: " + relative)
            parts.append(dict(path=relative, year=int(year), rows=meta["rows"],
                              min_date=meta.get("min_date"), max_date=meta.get("max_date"),
                              sha256=digest, bytes=stat.st_size, mtime_ns=stat.st_mtime_ns))
        entries.append(dict(name=spec.name, kind="label" if spec.is_label else "market" if spec.is_market else "stock",
                            group=spec.group, version=spec.version, higher_is_better=spec.higher_is_better,
                            definition_sha256=definition(spec, cfg), recipe=man.recipe,
                            valid_start=spec.resolved_start(cfg), coverage=man.coverage, partitions=parts))
    from .universe import frozen_codes
    codes = frozen_codes(cfg)
    base = dict(schema_version=1, run_id=run_id, cutoff=cutoff,
                created_at=dt.datetime.now(dt.timezone.utc).isoformat(),
                universe_sha256=_file_hash(cfg.frozen_universe) if cfg.frozen_universe else None,
                default_start=cfg.default_start, stock_count=len(codes) if codes is not None else None,
                common_sha256=common_definition(cfg), kernel_version=kernel_version(cfg),
                schemas={"stock": {"trade_date":"string","stock_code":"string","value":"float32","rank":"float32"},
                         "label": {"trade_date":"string","stock_code":"string","value":"float32","rank":"float32"},
                         "market": {"trade_date":"string","value":"float32"}},
                value_semantics="raw_factor_value; rank_daily_percentile; direction_in_metadata",
                consumer_note="Use read_release(cfg); hold its shared lock throughout reading; unfinished/dirty/changed files are rejected",
                objects=entries, research_review="legacy objects pending; candidate approvals explicit")
    base["release_id"] = hashlib.sha256(_canonical(base).encode()).hexdigest()
    target = cfg.project_path("artifacts", "artifacts") / "releases" / (base["release_id"] + ".json")
    _atomic_json(base, target)
    _atomic_json(cache, cache_path)
    _atomic_json(dict(release_id=base["release_id"], path=str(target.relative_to(cfg.root)), cutoff=cutoff,
                      run_id=run_id), cfg.state_dir / "current_release.json")
    (cfg.state_dir/"publication_dirty.json").unlink(missing_ok=True)
    return base


@contextmanager
def read_release(cfg, *, verify_hashes=False):
    """Downstream handoff API: never expose a partially completed batch."""
    import fcntl
    lock_path = cfg.state_dir / "daily.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_SH)
        try:
            if (cfg.state_dir / "daily_inflight.json").exists():
                raise RuntimeError("Last batch unfinished; run main.py to recover before consuming")
            if (cfg.state_dir / "publication_dirty.json").exists():
                raise RuntimeError("Production changed since publication; run full main.py before consuming")
            pointer = json.loads((cfg.state_dir / "current_release.json").read_text())
            path = (cfg.root / pointer["path"]).resolve()
            if not path.is_relative_to(cfg.root.resolve()):
                raise ValueError("Release path escapes project")
            release = json.loads(path.read_text())
            if release["release_id"] != pointer["release_id"]:
                raise ValueError("Publication identity mismatch")
            payload = {k:v for k,v in release.items() if k != "release_id"}
            if hashlib.sha256(_canonical(payload).encode()).hexdigest() != release["release_id"]:
                raise ValueError("Publication manifest content changed")
            for obj in release["objects"]:
                for part in obj["partitions"]:
                    data_path = (cfg.root / part["path"]).resolve()
                    if not data_path.is_relative_to(cfg.root.resolve()):
                        raise ValueError("Partition path escapes project")
                    info = data_path.stat()
                    if info.st_size != part["bytes"] or info.st_mtime_ns != part["mtime_ns"]:
                        raise RuntimeError("Published partition changed; rerun main.py: " + part["path"])
                    if verify_hashes and _file_hash(data_path) != part["sha256"]:
                        raise RuntimeError("Published partition hash mismatch: " + part["path"])
            yield release
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def cli(args, cfg):
    action = args.library_action
    if action == "bootstrap":
        result = bootstrap_library(cfg)
    elif action == "accept":
        result = accept_existing(cfg, args.name, args.reason)
    elif action == "accept-common":
        result = accept_common(cfg, args.reason)
    elif action == "status":
        existing=next((x for x in load_library(cfg)["objects"] if x["name"]==args.name),None)
        if existing and existing.get("origin")=="research":
            raise ValueError("Use research promote for research-origin lifecycle decisions")
        result = set_status(cfg, args.name, args.status, args.reason)
    else:
        result = load_library(cfg)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def add_parser(subparsers):
    parser = subparsers.add_parser("library", help="Explicit production definitions and approvals")
    actions = parser.add_subparsers(dest="library_action", required=True)
    actions.add_parser("bootstrap", help="Adopt existing legacy definitions once")
    actions.add_parser("list")
    accept = actions.add_parser("accept", help="Review changed implementation of an adopted legacy object")
    accept.add_argument("name")
    accept.add_argument("--reason", required=True)
    common = actions.add_parser("accept-common", help="Review a versioned change of shared numerical kernels")
    common.add_argument("--reason", required=True)
    status = actions.add_parser("status")
    status.add_argument("name")
    status.add_argument("status", choices=["production", "paused", "retired"])
    status.add_argument("--reason", required=True)
