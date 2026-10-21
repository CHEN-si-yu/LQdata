"""Immutable research definitions and append-only trial/lifecycle evidence."""
from __future__ import annotations

import importlib
import inspect
import contextlib
import json
import math
import os
import re
import uuid
from dataclasses import asdict
from pathlib import Path

import yaml

from .protocol import canonical, digest, utc_now

STATES = ("candidate", "qualified", "shadow", "production", "paused", "retired")
TRANSITIONS = {
    "candidate": {"qualified", "retired"}, "qualified": {"shadow", "paused", "retired"},
    "shadow": {"production", "paused", "retired"}, "production": {"paused", "retired"},
    "paused": {"shadow", "production", "retired"}, "retired": set(),
}
_SLUG = re.compile(r"^[a-z][a-z0-9_]{0,100}$")


def slug(value):
    value = str(value)
    if not _SLUG.fullmatch(value):
        raise ValueError(f"Unsafe research identifier: {value!r}; use lowercase letters, digits and underscore")
    return value


def project_path(cfg, relative):
    root = Path(cfg.root).resolve()
    path = (root / relative).resolve()
    if path != root and root not in path.parents:
        raise ValueError(f"Research path escapes project: {relative}")
    return path


def immutable_json(path, payload):
    """Exclusive create prevents replacement; fsync before publication completes."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = clean_json(payload)
    content = canonical(payload) + "\n"
    try:
        with path.open("x", encoding="utf-8", newline="\n") as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
    except FileExistsError:
        old = json.loads(path.read_text(encoding="utf-8"))
        if old != payload:
            raise ValueError(f"Immutable research record already exists with different content: {path}")
    return path


def clean_json(value):
    """Metric NaNs are missing evidence, represented by JSON null."""
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean_json(v) for v in value]
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


class Catalog:
    def __init__(self, cfg):
        self.cfg = cfg
        self.root = project_path(cfg, "artifacts/research")

    def read(self, kind, identity):
        if kind not in ("ideas", "families", "candidates"):
            raise ValueError(kind)
        path = self.root / "catalog" / kind / (slug(identity) + ".json")
        if not path.exists():
            raise KeyError(f"Unknown {kind}: {identity}")
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("definition_hash") != digest(record.get("definition")):
            raise ValueError(f"Immutable {kind} definition identity changed: {identity}")
        if kind == "candidates":
            expected = slug(record["definition"]["name"])[:75] + "_" + record["definition_hash"][:16]
            if record.get("candidate_id") != identity or expected != identity:
                raise ValueError(f"Immutable candidate name/definition mismatch: {identity}")
        return record

    def records(self, kind):
        return [self.read(kind, p.stem) for p in sorted((self.root / "catalog" / kind).glob("*.json"))]

    def register_idea(self, data):
        data = dict(data)
        for key in ("idea_id", "hypothesis", "mechanism", "expected_direction", "target_horizons", "falsifiable", "fields"):
            if not data.get(key):
                raise ValueError(f"Idea requires {key}")
        identity = slug(data["idea_id"])
        return self._definition("ideas", identity, data)

    def register_family(self, data):
        data = dict(data)
        for key in ("family_id", "idea_id", "trial_budget", "allowed_variants", "baseline"):
            if not data.get(key):
                raise ValueError(f"Family requires {key}")
        self.read("ideas", data["idea_id"])
        if not 1 <= int(data["trial_budget"]) <= 100:
            raise ValueError("Family trial budget must be between 1 and 100")
        identity = slug(data["family_id"])
        return self._definition("families", identity, data)

    def _definition(self, kind, identity, data):
        path = self.root / "catalog" / kind / (identity + ".json")
        if path.exists():
            old = self.read(kind, identity)
            if old["definition"] != data:
                raise ValueError(f"{identity} is immutable; use a new versioned identifier")
            return old
        record = {"definition": data, "definition_hash": digest(data), "registered_at": utc_now()}
        immutable_json(path, record)
        return record

    def register_candidate(self, data):
        # Concurrent researchers cannot both claim the last family slot.
        lock_path = self.root / "catalog" / ".candidate_registration.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open("a+b") as lock:
            if os.name == "nt":
                import msvcrt
                lock.seek(0)
                if not lock.read(1):
                    lock.write(b"0")
                    lock.flush()
                lock.seek(0)
                msvcrt.locking(lock.fileno(), msvcrt.LK_LOCK, 1)
            else:
                import fcntl
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                return self._register_candidate(data)
            finally:
                if os.name == "nt":
                    lock.seek(0)
                    msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    def _register_candidate(self, data):
        data = dict(data)
        for key in ("name", "family_id", "implementation", "parameters", "spec", "units", "availability", "missing_policy"):
            if key not in data or (key != "parameters" and not data[key]):
                raise ValueError(f"Candidate requires {key}")
        name = slug(data["name"])
        family = self.read("families", data["family_id"])["definition"]
        allowed = family["allowed_variants"]
        if data["parameters"] not in allowed:
            raise ValueError("Parameters must be a preregistered family variant")
        builder, source_hash = implementation(self.cfg, data["implementation"])
        definition = {**data, "source_hash": source_hash, "family_hash": digest(family)}
        identity = name[:75] + "_" + digest(definition)[:16]
        try:
            existing = self.read("candidates", identity)
        except KeyError:
            existing = None
        if existing:
            return existing
        siblings = [r for r in self.records("candidates") if r["definition"]["family_id"] == data["family_id"]]
        if len(siblings) >= int(family["trial_budget"]):
            raise ValueError(f"Family trial budget exhausted ({family['trial_budget']}); record a new preregistered family to expand it")
        # Build now: invalid executable definitions do not silently enter the catalog.
        spec = build_spec(definition, builder, identity)
        record = {"candidate_id": identity, "definition": definition, "definition_hash": digest(definition),
                  "registered_at": utc_now(), "status_at_registration": "candidate", "validated_alpha": False}
        immutable_json(self.root / "catalog" / "candidates" / (identity + ".json"), record)
        self.event(identity, "registered", {"spec_name": spec.name, "family_trial_number": len(siblings) + 1})
        return record

    def event(self, identity, kind, payload):
        identity = slug(identity)
        event = {"candidate_id": identity, "event_id": uuid.uuid4().hex, "kind": kind,
                 "created_at": utc_now(), "payload": payload}
        stamp = event["created_at"].replace(":", "").replace(".", "")
        path = self.root / "ledger" / identity / f"{stamp}_{event['event_id']}.json"
        immutable_json(path, event)
        return event

    def events(self, identity):
        self.read("candidates", identity)
        return [json.loads(p.read_text(encoding="utf-8")) for p in sorted((self.root / "ledger" / slug(identity)).glob("*.json"))]

    def state(self, identity):
        state = "candidate"
        for event in self.events(identity):
            if event["kind"] == "lifecycle":
                state = event["payload"]["to"]
        return state

    def legacy_import(self, specs):
        """Catalog legacy contracts without claiming research validation or touching their status."""
        records = []
        for spec in specs:
            if spec.is_label:
                continue
            info = asdict(spec)
            info.pop("fn", None)
            # align_fill NaN is explicit text in evidence, never invalid JSON NaN.
            if isinstance(info.get("align_fill"), float) and info["align_fill"] != info["align_fill"]:
                info["align_fill"] = "nan"
            info = json.loads(json.dumps(info))
            payload = {"factor": info, "legacy_status": "production_grandfathered_review_pending", "validated_alpha": False}
            path = self.root / "legacy" / (slug(spec.name) + "_" + digest(payload)[:16] + ".json")
            immutable_json(path, payload)
            records.append(payload)
        return records


def implementation(cfg, reference):
    """Only project-contained Python builders; no dynamic eval or outside code path."""
    if not re.fullmatch(r"research\.[a-zA-Z0-9_.]+:[a-zA-Z_][a-zA-Z0-9_]*", str(reference)):
        raise ValueError("Implementation must be research.<module>:<builder>, stored inside this project")
    module_name, function_name = reference.split(":", 1)
    module = importlib.import_module(module_name)
    path = Path(inspect.getfile(module)).resolve()
    root = Path(cfg.root).resolve()
    if root not in path.parents:
        raise ValueError("Candidate implementation is outside the factor-engineering project")
    fn = getattr(module, function_name)
    if not callable(fn):
        raise ValueError("Candidate builder is not callable")
    import hashlib
    return fn, hashlib.sha256(path.read_bytes()).hexdigest()


def build_spec(definition, builder, identity):
    from fea.spec import FactorSpec, LAGGED_DATASETS, QFQ_DATASETS
    metadata = dict(definition["spec"])
    unknown = set(metadata) - set(FactorSpec.__dataclass_fields__)
    if unknown:
        raise ValueError(f"Unknown FactorSpec fields: {sorted(unknown)}")
    for forbidden in ("name", "fn", "is_label", "forward_days", "enabled", "allow_qfq"):
        if forbidden in metadata:
            raise ValueError(f"Candidate cannot set {forbidden}")
    if "align_fill" in metadata and metadata["align_fill"] is None:
        raise ValueError("Candidate cannot disable the required annual date grid from 2018; omit align_fill to preserve unavailable prefixes as NaN")
    for key in ("deps", "lagged_ok", "fin_fields"):
        if key in metadata:
            metadata[key] = tuple(metadata[key])
    if not metadata.get("deps") or not metadata.get("formula") or not metadata.get("desc"):
        raise ValueError("Candidate spec requires declared deps, formula and desc")
    if any(d in QFQ_DATASETS for d in metadata["deps"]):
        raise ValueError("Forward-adjusted data is forbidden in research candidates")
    missing_lag = [d for d in metadata["deps"] if d in LAGGED_DATASETS and d not in metadata.get("lagged_ok", ())]
    if missing_lag:
        raise ValueError(f"Candidate must declare handling of source availability lags: {missing_lag}")
    if int(metadata.get("warmup_days", 0)) <= 0:
        raise ValueError("Explicit positive calendar-day warmup is required")
    fn = builder(dict(definition["parameters"]))
    if not callable(fn):
        raise ValueError("Builder must return callable(ctx) -> factor array")
    return FactorSpec(name=identity, fn=fn, enabled=True, **metadata)


def spec_for_candidate(cfg, identity):
    record = Catalog(cfg).read("candidates", identity)
    builder, source_hash = implementation(cfg, record["definition"]["implementation"])
    if source_hash != record["definition"]["source_hash"]:
        raise ValueError("Candidate implementation changed; register a new immutable candidate version")
    return build_spec(record["definition"], builder, identity)


def load_definition(path):
    path = Path(path)
    return yaml.safe_load(path.read_text(encoding="utf-8")) if path.suffix.lower() in (".yaml", ".yml") else json.loads(path.read_text(encoding="utf-8"))
