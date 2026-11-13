"""从生产定义构建已注册因子，保持原规格和参数口径。"""
from .spec import FactorSpec, LAGGED_DATASETS, QFQ_DATASETS


def build_spec(definition, builder, identity):
    metadata = dict(definition["spec"])
    unknown = set(metadata) - set(FactorSpec.__dataclass_fields__)
    if unknown:
        raise ValueError(f"Unknown FactorSpec fields: {sorted(unknown)}")
    for forbidden in ("name", "fn", "is_label", "forward_days", "enabled", "allow_qfq"):
        if forbidden in metadata:
            raise ValueError(f"Registered definition cannot set {forbidden}")
    if "align_fill" in metadata and metadata["align_fill"] is None:
        raise ValueError("Registered definition must preserve the annual date grid")
    for key in ("deps", "lagged_ok", "fin_fields"):
        if key in metadata:
            metadata[key] = tuple(metadata[key])
    if not metadata.get("deps") or not metadata.get("formula") or not metadata.get("desc"):
        raise ValueError("Registered spec requires declared deps, formula and desc")
    if any(d in QFQ_DATASETS for d in metadata["deps"]):
        raise ValueError("Forward-adjusted data is forbidden in factor definitions")
    missing_lag = [d for d in metadata["deps"] if d in LAGGED_DATASETS and d not in metadata.get("lagged_ok", ())]
    if missing_lag:
        raise ValueError(f"Missing source availability lag handling: {missing_lag}")
    if int(metadata.get("warmup_days", 0)) <= 0:
        raise ValueError("Explicit positive calendar-day warmup is required")
    fn = builder(dict(definition["parameters"]))
    if not callable(fn):
        raise ValueError("Builder must return callable(ctx) -> factor array")
    return FactorSpec(name=identity, fn=fn, enabled=True, **metadata)
