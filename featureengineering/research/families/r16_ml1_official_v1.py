"""Seat-only R16ML1 Context adapter draft. NOT registered or executed.

This file is intentionally outside factors/, fea/, and research/. It contains
no module-level Torch import. The main purpose is to make the private 2016/17
parent-history bridge and the frozen monthly training contract reviewable.
"""
from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import os
import platform
from pathlib import Path
import sys

import numpy as np


FEATURES = (
    "momentum_120", "short_term_reversal_5", "vol_120",
    "idio_vol_60", "turnover_std_20",
)
RAW_UNION = ("stock_daily", "stock_adj_factor", "stock_finance", "index_daily")
DECLARED_DEPS = FEATURES + RAW_UNION
HISTORY_FLOOR = 20160101
PUBLIC_FLOOR = 20180101
SEED = 20261008
GPU_BUDGET_BYTES = 2**30
TRAIN_SLOT_COUNT = 252
MIN_SLOT_ROWS = 100
FROZEN_TRAINING_SOURCE_SHA256 = "e43394c5a0392a2a0c2d3927fae7532b493c3c5e8adbab56fbb51c972372985c"
FROZEN_DESIGN_SHA256 = "fdf1adb552fad680cc40ab57fffd14b8c8dd7f0117a0224a264b27a3619df46d"
_TORCH_OWNER = "R16ML1-official-context-singlefile-v1"
_BLAS_LIMITER = None
FROZEN_ENVIRONMENT = {
    "python": "3.14.7", "numpy": "2.5.3", "torch": "2.11.0+cu128",
    "cuda_runtime": "12.8", "gpu_name": "NVIDIA vGPU-32GB",
    "gpu_driver": "570.124.04", "cublas_workspace_config": ":4096:8",
    "python_executable": "/autodl-fs/data/miniconda3/bin/python",
    "blas_threads": 1, "torch_threads": 1,
}
FROZEN_PARAMETERS = {
    "candidate": "R16ML1", "horizon": 5,
    "features": list(FEATURES), "seed": SEED,
    "architecture": [5, 16, 8, 1], "activation": "ReLU",
    "dtype": "float64", "epochs": 10, "batch_size": 8192,
    "optimizer": "Adam", "learning_rate": 1e-3, "weight_decay": 1e-4,
    "training_slots": TRAIN_SLOT_COUNT, "minimum_rows_per_slot": MIN_SLOT_ROWS,
    "warmup_days": 600, "gpu_budget_bytes": GPU_BUDGET_BYTES,
}


def _file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _array_sha(array: np.ndarray) -> str:
    x = np.ascontiguousarray(array)
    h = hashlib.sha256()
    # Exact canonical encoding used by the frozen monthly training runner.
    h.update(str(x.dtype).encode())
    h.update(json.dumps(x.shape).encode())
    h.update(memoryview(x).cast("B"))
    return h.hexdigest()


def _initialize_torch():
    """Initialize Torch lazily in this compute worker and mark its ownership."""
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in (None, ":4096:8"):
        raise RuntimeError("CUBLAS_WORKSPACE_CONFIG conflicts with frozen R16ML1 environment")
    if "torch" in sys.modules:
        torch = sys.modules["torch"]
        if getattr(torch, "_r16ml1_owner", None) != _TORCH_OWNER:
            raise RuntimeError("Torch was imported externally before worker-only deterministic setup")
        if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
            raise RuntimeError("cached worker Torch environment differs from frozen CUBLAS contract")
        if (torch.get_num_threads() != 1 or torch.get_num_interop_threads() != 1
                or not torch.are_deterministic_algorithms_enabled()):
            raise RuntimeError("cached worker Torch numerical environment differs from frozen contract")
        _assert_torch_environment(torch)
        _assert_blas_threadpool()
        return torch
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS"):
        os.environ[key] = "1"
    import torch
    torch._r16ml1_owner = _TORCH_OWNER
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        if torch.get_num_interop_threads() != 1:
            raise
    torch.use_deterministic_algorithms(True)
    _limit_blas_threadpool()
    _assert_torch_environment(torch)
    return torch


def _limit_blas_threadpool():
    """Set and verify actual NumPy BLAS pools, since env vars may be late."""
    global _BLAS_LIMITER
    try:
        from threadpoolctl import threadpool_info, threadpool_limits
    except ImportError as exc:
        raise RuntimeError("threadpoolctl is required to enforce the frozen one-thread CPU contract") from exc
    _BLAS_LIMITER = threadpool_limits(limits=1)
    _assert_blas_threadpool(threadpool_info)


def _assert_blas_threadpool(info_fn=None):
    try:
        from threadpoolctl import threadpool_info
    except ImportError as exc:
        raise RuntimeError("threadpoolctl is required to verify the frozen one-thread CPU contract") from exc
    entries = (info_fn or threadpool_info)()
    if any(int(item.get("num_threads", 1)) != 1 for item in entries):
        raise RuntimeError(f"loaded BLAS threadpool is not single-threaded: {entries}")


def _assert_torch_environment(torch):
    if platform.python_version() != FROZEN_ENVIRONMENT["python"]:
        raise RuntimeError("Python version differs from the sealed R16ML1 training environment")
    if sys.executable != FROZEN_ENVIRONMENT["python_executable"]:
        raise RuntimeError("Python executable differs from the sealed R16ML1 training environment")
    if np.__version__ != FROZEN_ENVIRONMENT["numpy"]:
        raise RuntimeError("NumPy version differs from the sealed R16ML1 training environment")
    if str(torch.__version__) != FROZEN_ENVIRONMENT["torch"]:
        raise RuntimeError("Torch version differs from the sealed R16ML1 training environment")
    if str(torch.version.cuda) != FROZEN_ENVIRONMENT["cuda_runtime"]:
        raise RuntimeError("CUDA runtime differs from the sealed R16ML1 training environment")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != FROZEN_ENVIRONMENT["cublas_workspace_config"]:
        raise RuntimeError("CUBLAS workspace config differs from the sealed R16ML1 environment")
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        if os.environ.get(key) != "1":
            raise RuntimeError(f"{key} must be 1 for the sealed R16ML1 numerical environment")
    if torch.get_num_threads() != 1 or torch.get_num_interop_threads() != 1:
        raise RuntimeError("Torch thread count differs from the sealed R16ML1 environment")
    if not torch.are_deterministic_algorithms_enabled():
        raise RuntimeError("Torch deterministic algorithms are not enabled")


def rank_target(y: np.ndarray, good: np.ndarray) -> np.ndarray:
    """Frozen midrank target: stable ascending ranks, normalized to (-.5,.5)."""
    out = np.full(y.shape, np.nan, dtype=np.float64)
    idx = np.flatnonzero(good)
    order = idx[np.argsort(y[idx], kind="mergesort")]
    vals, n = y[order], len(order)
    i = 0
    while i < n:
        j = i + 1
        while j < n and vals[j] == vals[i]:
            j += 1
        out[order[i:j]] = ((i + 1 + j) / 2.0) / (n + 1.0) - 0.5
        i = j
    return out


def build_training(features, open_hfq, traded, universe, slots, cutoff):
    """Frozen fixed-slot h5 target and past-only training transforms."""
    exits = slots + 6
    assert len(slots) == TRAIN_SLOT_COUNT and np.all(exits <= cutoff)
    X3 = np.stack([np.asarray(f[slots], dtype=np.float64) for f in features], axis=2)
    buy = np.asarray(open_hfq[slots + 1], dtype=np.float64)
    sell = np.asarray(open_hfq[slots + 6], dtype=np.float64)
    traded_buy = np.asarray(traded[slots + 1], dtype=bool)
    traded_sell = np.asarray(traded[slots + 6], dtype=bool)
    universe_signal = np.asarray(universe[slots], dtype=bool)
    label = np.full(buy.shape, np.nan, dtype=np.float64)
    label_valid = (traded_buy & traded_sell & np.isfinite(buy) & np.isfinite(sell)
                   & (buy > 0) & (sell > 0) & universe_signal)
    label[label_valid] = sell[label_valid] / buy[label_valid] - 1.0
    target = np.full(label.shape, np.nan, dtype=np.float64)
    counts = []
    for r in range(TRAIN_SLOT_COUNT):
        good = np.isfinite(label[r]) & np.isfinite(X3[r]).all(axis=1)
        counts.append(int(good.sum()))
        if counts[-1] >= MIN_SLOT_ROWS:
            target[r] = rank_target(label[r], good)
    counts = np.asarray(counts, dtype=np.int32)
    if np.any(counts < MIN_SLOT_ROWS):
        return None, {"status": "fail_closed_slot_below_100", "counts": counts}
    row_valid = np.isfinite(target)
    raw_x, y = X3[row_valid], target[row_valid]
    med = np.median(raw_x, axis=0)
    q25, q75 = np.quantile(raw_x, [0.25, 0.75], axis=0)
    iqr = q75 - q25
    if not np.isfinite(iqr).all() or np.any(iqr <= 0):
        return None, {"status": "fail_closed_bad_iqr", "counts": counts, "median": med, "iqr": iqr}
    x = np.clip((raw_x - med) / iqr, -5.0, 5.0)
    assert not np.isinf(x).any() and not np.isinf(y).any()
    return {"x": x, "y": y, "median": med, "iqr": iqr, "counts": counts,
            "input_sha256": _array_sha(x), "target_sha256": _array_sha(y),
            "raw_feature_sha256": _array_sha(X3),
            "endpoint_sha256": _array_sha(np.stack([buy, sell, traded_buy, traded_sell,
                                                     universe_signal], axis=2))}, {"status": "ready", "counts": counts}


def state_sha(state: dict[str, np.ndarray]) -> str:
    h = hashlib.sha256()
    for key in sorted(state):
        x = np.ascontiguousarray(state[key])
        h.update(key.encode() + b"\0"); h.update(str(x.dtype).encode() + b"\0")
        h.update(json.dumps(x.shape).encode() + b"\0"); h.update(memoryview(x).cast("B")); h.update(b"\0")
    return h.hexdigest()


def fit_once(x, y):
    """The frozen 5-16-8-1 float64 MLP and fixed optimizer/training budget."""
    torch = _initialize_torch()
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    model = torch.nn.Sequential(
        torch.nn.Linear(5, 16, dtype=torch.float64), torch.nn.ReLU(),
        torch.nn.Linear(16, 8, dtype=torch.float64), torch.nn.ReLU(),
        torch.nn.Linear(8, 1, dtype=torch.float64),
    ).cuda()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    xt = torch.as_tensor(x, dtype=torch.float64, device="cuda")
    yt = torch.as_tensor(y[:, None], dtype=torch.float64, device="cuda")
    gen = torch.Generator(device="cpu"); gen.manual_seed(SEED)
    for _ in range(10):
        perm = torch.randperm(len(x), generator=gen, device="cpu")
        for lo in range(0, len(x), 8192):
            ix = perm[lo:lo+8192].to("cuda")
            opt.zero_grad(set_to_none=True)
            loss = (model(xt.index_select(0, ix)).squeeze(1) -
                    yt.index_select(0, ix).squeeze(1)).square().mean()
            loss.backward(); opt.step()
    model.eval(); torch.cuda.synchronize()
    state = {k.replace(".", "_"): v.detach().cpu().contiguous().numpy().copy()
             for k, v in model.state_dict().items()}
    return model, state


def prediction_month(model, features, traded, universe, indices, med, iqr):
    torch = _initialize_torch()
    out = np.full((len(indices), len(universe[0])), np.nan, dtype=np.float32)
    valid_out = np.zeros(out.shape, dtype=bool)
    for r, di in enumerate(indices):
        xall = np.column_stack([np.asarray(f[di], dtype=np.float64) for f in features])
        good = (np.asarray(traded[di], dtype=bool) & np.asarray(universe[di], dtype=bool)
                & np.isfinite(xall).all(axis=1))
        valid_out[r] = good
        x = np.clip((xall[good] - med) / iqr, -5.0, 5.0)
        if len(x):
            with torch.no_grad():
                yhat = model(torch.as_tensor(x, dtype=torch.float64, device="cuda")).squeeze(1).cpu().numpy()
            out[r, good] = yhat.astype(np.float32)
    assert not np.isinf(out).any()
    return out, valid_out


def _loader_float32(values, parent_name):
    """Restore the stored float32 contract after FactorIO's float64 placement."""
    loaded = np.asarray(values)
    if loaded.dtype != np.float64:
        raise RuntimeError(f"FactorIO loader dtype changed for {parent_name}: {loaded.dtype}")
    if np.isinf(loaded).any():
        raise RuntimeError(f"FactorIO parent contains Inf for {parent_name}")
    restored = loaded.astype(np.float32)
    roundtrip = restored.astype(np.float64)
    if not np.array_equal(loaded, roundtrip, equal_nan=True):
        raise RuntimeError(f"FactorIO float64 placement does not round-trip stored float32 for {parent_name}")
    return restored


@contextlib.contextmanager
def _single_gpu_lock(ctx):
    """Serialize all candidate GPU use across production worker processes."""
    lock_path = ctx.cfg.contract_root / "state" / "r16ml1_gpu.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("R16ML1 GPU is already owned by another process") from exc
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _parent_identity(aux_cfg):
    import factors  # import-time registration of existing factors only
    from fea.spec import REGISTRY

    identities = {}
    for name in FEATURES:
        spec = REGISTRY[name]
        path = Path(__import__("inspect").getfile(spec.fn)).resolve()
        identities[name] = {
            "module_path": str(path), "module_sha256": _file_sha(path),
            "recipe": spec.recipe(aux_cfg), "start": spec.resolved_start(aux_cfg),
            "warmup_days": int(spec.warmup_days), "deps": list(spec.deps),
        }
    raw_closure = sorted({dep for item in identities.values() for dep in item["deps"]
                          if dep not in identities})
    if raw_closure != sorted(RAW_UNION):
        raise RuntimeError(f"frozen five-parent raw dependency closure changed: {raw_closure}")
    return identities


def _aux_namespace(ctx, identities):
    derived = ctx.cfg.project_path("derived", "data/derived")
    cutoff = ctx.up.audit_cutoff
    cutoff_tag = "cutoff_none" if cutoff is None else f"cutoff_{int(cutoff):08d}"
    task_year = int(ctx.panel.dates[-1]) // 10000
    # The enclosing derived path in annual research runs already carries its
    # phase/year identity. Add a digest of actual parent code/spec identities.
    identity_sha = hashlib.sha256(json.dumps(identities, sort_keys=True).encode()).hexdigest()
    return derived / "r16ml1_aux_parent_history_v1" / f"taskyear_{task_year}" / cutoff_tag / identity_sha


def _build_private_parent_history(ctx):
    """Recompute exact registered parents into an isolated 2016-floor sandbox."""
    from fea.spec import REGISTRY
    from fea.factors_io import FactorIO
    from fea.engine import Engine
    from fea.panel import Panel
    from fea.universe import st_pool_fingerprint

    if int(ctx.panel.dates[0]) >= PUBLIC_FLOOR:
        raise ValueError("Aux parent history requested although this Context has no pre-2018 rows")
    outer_cfg = ctx.cfg
    # Hash the existing module/spec identity before choosing the write location.
    aux_root_for_identity = outer_cfg.project_path("derived", "data/derived") / "identity_probe"
    aux_cfg_identity = outer_cfg.sandbox(aux_root_for_identity)
    aux_cfg_identity.raw["default_start"] = "2016-01-01"
    aux_cfg_identity.raw["_audit_cutoff"] = ctx.up.audit_cutoff
    identities = _parent_identity(aux_cfg_identity)
    namespace = _aux_namespace(ctx, identities)
    aux_cfg = outer_cfg.sandbox(namespace)
    aux_cfg.raw["default_start"] = "2016-01-01"
    # Carry the actual raw-visibility identity into any diagnostics/config
    # consumers. Engine facade still reuses ctx.up by object identity.
    aux_cfg.raw["_audit_cutoff"] = ctx.up.audit_cutoff
    if ctx.up.audit_cutoff != aux_cfg.raw["_audit_cutoff"]:
        raise AssertionError("auxiliary raw cutoff differs from parent Context Upstream")

    old_env = {key: os.environ.get(key) for key in
               ("FEA_CACHE_DIR", "FEA_AUDIT_TRACE", "FEA_LOG_NAME")}
    os.environ["FEA_CACHE_DIR"] = str(namespace / "cache")
    os.environ["FEA_AUDIT_TRACE"] = "artifacts/access_trace/r16ml1_aux"
    os.environ["FEA_LOG_NAME"] = f"r16ml1_aux_{ctx.up.audit_cutoff or 'production'}"
    try:
        # Engine.__init__ sets Upstream.audit_cutoff before reading calendar or
        # stock-list contracts. Then hand all parent/raw reads to the exact
        # outer worker Upstream instance, preserving its phase cache and audit.
        engine = Engine(aux_cfg)
        if engine.up.audit_cutoff != ctx.up.audit_cutoff:
            raise AssertionError("auxiliary constructor did not receive the actual Context cutoff")
        if not np.array_equal(engine.cal.days, ctx.cal.days):
            raise AssertionError("auxiliary calendar differs from Context calendar")
        if not np.array_equal(engine.codes, ctx.panel.codes):
            raise AssertionError("auxiliary frozen code axis differs from Context axis")
        engine.up = ctx.up
        if engine.up is not ctx.up or engine.up.audit_cutoff != ctx.up.audit_cutoff:
            raise AssertionError("auxiliary Engine did not retain the exact Context Upstream/cutoff")
        # These five parent factors use PriceLayer/raw daily inputs only;
        # preventing Derivative's default all-fields build is part of the
        # bounded auxiliary contract.
        engine._deriv_fields_run = frozenset()
        engine._start_floor = HISTORY_FLOOR
        engine._run_end = int(ctx.panel.dates[-1])
        from fea.dates import int_to_str
        engine._end_str = int_to_str(engine._run_end)
        # Reproduce the ordinary visibility gates in the private state namespace;
        # universe_for below is then compared to the actual child Context mask.
        engine.st_gate()
        engine.delay_gate()
        if ctx.up.audit_cutoff is not None:
            # Research worker phases publish a visible-ST identity. Derive the
            # phase from the enclosing isolated derived path and verify it
            # rather than creating a second interpretation of that phase.
            import re
            derived_rel = str(ctx.cfg.raw.get("paths", {}).get("derived", ""))
            match = re.search(r"phase_(\d+)_cutoff_(\d{8})", derived_rel)
            if match is None or int(match.group(2)) != int(ctx.up.audit_cutoff):
                raise RuntimeError("Context derived namespace does not identify the actual raw-cutoff phase")
            visible_path = ctx.cfg.state_dir / f"st_phase_{match.group(1)}_{match.group(2)}" / "visible.json"
            visible = json.loads(visible_path.read_text(encoding="utf-8"))
            if visible.get("phase") != int(match.group(1)) or visible.get("raw_cutoff") != int(ctx.up.audit_cutoff):
                raise RuntimeError("outer phase ST baseline identity does not match Context cutoff")
            if st_pool_fingerprint(ctx.up, ctx.panel.codes) != visible.get("visible_st"):
                raise RuntimeError("outer phase ST pool fingerprint changed before auxiliary calculation")
        years = range(int(ctx.panel.dates[0]) // 10000,
                      int(ctx.panel.dates[-1]) // 10000 + 1)
        source_days = {}
        for name in FEATURES:
            spec = REGISTRY[name]
            for year in years:
                days = engine.cal.between(max(HISTORY_FLOOR, year * 10000 + 101),
                                          min(int(ctx.panel.dates[-1]), year * 10000 + 1231))
                if not days.size:
                    continue
                result = engine.run_year(spec, int(year), days)
                if result.get("error"):
                    raise RuntimeError(f"private parent recomputation failed: {name}/{year}: {result['error']}")
                source_days[f"{name}/{year}"] = [int(days[0]), int(days[-1]), int(len(days))]

        # Reconstruct the universe from the same cutoff-bound Upstream and prove
        # it is the identical fixed panel supplied to the outer factor function.
        aux_uni = engine.universe_for(ctx.panel)
        if not np.array_equal(aux_uni, np.asarray(ctx.universe, dtype=bool)):
            raise RuntimeError("auxiliary ST/frozen-universe mask differs from outer Context")

        aux_io = FactorIO(aux_cfg, ctx.cal)
        parent_panel = Panel(np.asarray(ctx.panel.dates, dtype=np.int32), ctx.panel.codes)
        post_dates = np.asarray(ctx.panel.dates[ctx.panel.dates >= PUBLIC_FLOOR], dtype=np.int32)
        if not post_dates.size:
            raise RuntimeError("pre-2018 auxiliary bridge has no public overlap/prediction axis")
        public_panel = Panel(post_dates, ctx.panel.codes)
        private_values, public_values = {}, {}
        pre = np.asarray(ctx.panel.dates < PUBLIC_FLOOR)
        overlap = ~pre
        for name in FEATURES:
            # Load each side only on the date rows it owns, so FactorIO's
            # required-date checks never demand private 2016/17 rows from the
            # 2018-floor public store or post-Context rows from the aux store.
            ctx.accessed.add(name)
            pv_post = _loader_float32(ctx.factor_io.load(name, public_panel), name)
            av = _loader_float32(aux_io.load(name, parent_panel), name)
            pv = np.full(ctx.panel.shape, np.nan, dtype=np.float32)
            pv[overlap] = pv_post
            if pv.dtype != np.float32 or av.dtype != np.float32 or pv.shape != av.shape:
                raise RuntimeError(f"parent value dtype/shape mismatch for {name}")
            if np.isinf(pv).any() or np.isinf(av).any():
                raise RuntimeError(f"parent contains Inf for {name}")
            if not np.array_equal(pv[overlap], av[overlap], equal_nan=True):
                raise RuntimeError(f"private/public parent float32 overlap mismatch: {name}")
            private_values[name] = av
            public_values[name] = pv

        universe_path = ctx.cfg.frozen_universe
        manifest = {
            "candidate": "R16ML1", "contract": "private_exact_parent_history_v1",
            "outer_context_dates_sha256": _array_sha(np.asarray(ctx.panel.dates)),
            "codes_sha256": _array_sha(np.asarray(ctx.panel.codes)),
            "cutoff": ctx.up.audit_cutoff, "aux_default_start": aux_cfg.default_start,
            "upstream_object_reused": engine.up is ctx.up,
            "outer_engine_default_start": outer_cfg.default_start,
            "frozen_universe_sha256": _file_sha(universe_path) if universe_path else None,
            "aux_builder_sha256": _file_sha(Path(__file__).resolve()),
            "parent_identities": identities, "parent_year_output_spans": source_days,
            "raw_dependency_union": list(RAW_UNION),
            "pre2018_cells": int(pre.sum() * ctx.panel.C),
            "overlap_cells": int(overlap.sum() * ctx.panel.C),
            "overlap_mismatch_count": 0,
            "parent_array_sha256": {n: _array_sha(v) for n, v in private_values.items()},
            "public_parent_array_sha256": {n: _array_sha(v) for n, v in public_values.items()},
            "combined_parent_array_sha256": {
                n: _array_sha(np.where(overlap[:, None], public_values[n], private_values[n]))
                for n in FEATURES
            },
        }
        namespace.mkdir(parents=True, exist_ok=True)
        (namespace / "aux_parent_history.manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        combined = {}
        for name in FEATURES:
            value = private_values[name].copy()
            value[overlap] = public_values[name][overlap]
            combined[name] = value
        return namespace, combined, manifest
    finally:
        for key, value in old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _feature_panels(ctx):
    if int(ctx.panel.dates[0]) < PUBLIC_FLOOR:
        _namespace, features, _manifest = _build_private_parent_history(ctx)
        return [features[name] for name in FEATURES]
    # From 2020 onward the 600-natural-day annual Context begins after the
    # public factor floor, so use only the official exact-aligned parent values.
    features = [_loader_float32(ctx.load_factor(name), name) for name in FEATURES]
    if any(f.dtype != np.float32 or f.shape != ctx.panel.shape for f in features):
        raise RuntimeError("official parent Context panel dtype/shape differs from frozen contract")
    if any(np.isinf(f).any() for f in features):
        raise RuntimeError("official parent Context panel contains Inf")
    return features


def _task_prediction_months(ctx):
    days = np.asarray(ctx.panel.dates, dtype=np.int64)
    task_year = int(days[-1]) // 10000
    keep = (days // 10000 == task_year) & (days >= PUBLIC_FLOOR)
    months = {}
    for i in np.flatnonzero(keep):
        months.setdefault(int(days[i]) // 100, []).append(int(i))
    return days, task_year, {month: np.asarray(rows, dtype=np.int64)
                             for month, rows in months.items()}


def compute(ctx):
    """Draft FactorSpec.fn-compatible implementation; importing it does not register it."""
    import factors  # existing production definitions; no new registration
    from fea.dates import int_to_str

    features = _feature_panels(ctx)
    days, task_year, months = _task_prediction_months(ctx)
    open_hfq = np.asarray(ctx.hfq("open"))
    traded = np.asarray(ctx.traded(), dtype=bool)
    universe = np.asarray(ctx.universe, dtype=bool)
    if open_hfq.dtype != np.float64 and open_hfq.dtype != np.float32:
        raise RuntimeError(f"unexpected hfq open dtype {open_hfq.dtype}")
    if any(a.shape != ctx.panel.shape for a in [*features, open_hfq, traded, universe]):
        raise RuntimeError("Context input panel shapes do not align")
    if np.isinf(open_hfq).any():
        raise RuntimeError("hfq open contains Inf")

    out = np.full(ctx.panel.shape, np.nan, dtype=np.float32)
    if not months:
        return out
    with _single_gpu_lock(ctx):
        torch = _initialize_torch()  # lazy import in the single compute worker
        torch.set_num_threads(1)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            if torch.get_num_interop_threads() != 1:
                raise
        torch.use_deterministic_algorithms(True)
        if not torch.cuda.is_available():
            raise RuntimeError("frozen R16ML1 official compute requires the validated CUDA device")
        if torch.cuda.get_device_name(0) != FROZEN_ENVIRONMENT["gpu_name"]:
            raise RuntimeError("CUDA device differs from the sealed R16ML1 training environment")
        total = int(torch.cuda.get_device_properties(0).total_memory)
        import subprocess
        driver = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, check=False)
        driver_version = driver.stdout.strip() if driver.returncode == 0 else "unavailable"
        if driver_version != FROZEN_ENVIRONMENT["gpu_driver"]:
            raise RuntimeError("GPU driver differs from the sealed R16ML1 training environment")
        torch.cuda.set_per_process_memory_fraction(min(1.0, GPU_BUDGET_BYTES / total), device=0)
        torch.cuda.reset_peak_memory_stats(0)
        model_dir = (ctx.cfg.project_path("derived", "data/derived") /
                     "r16ml1_month_models_v1" / f"year_{task_year}" /
                     ("cutoff_none" if ctx.up.audit_cutoff is None
                      else f"cutoff_{int(ctx.up.audit_cutoff):08d}"))
        model_dir.mkdir(parents=True, exist_ok=True)
        env_identity = {
            "python": sys.version.split()[0], "numpy": np.__version__,
            "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
            "gpu_name": torch.cuda.get_device_name(0), "gpu_total_bytes": total,
            "seed": SEED, "float64": True, "deterministic": torch.are_deterministic_algorithms_enabled(),
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "torch_threads": [torch.get_num_threads(), torch.get_num_interop_threads()],
            "gpu_budget_bytes": GPU_BUDGET_BYTES,
        }
        env_identity["gpu_driver"] = driver_version

        for month, pred_rows in sorted(months.items()):
            first = int(pred_rows[0])
            cutoff = first - 1
            slots = np.arange(cutoff - 257, cutoff - 5, dtype=np.int64)
            # The one fixed 252-slot window; no eligible-row backfill.
            if len(slots) != TRAIN_SLOT_COUNT or slots[0] < 0 or slots[-1] + 6 > cutoff:
                continue  # fail-closed NaN month (not enough fixed history)
            train, status = build_training(features, open_hfq, traded,
                                           universe, slots, cutoff)
            if train is None or status.get("status") != "ready":
                continue
            train_days = days[slots]
            exit_days = days[slots + 6]
            if np.any(exit_days > days[cutoff]):
                raise AssertionError("training endpoint exceeds monthly matured cutoff")
            month_name = f"{month:06d}"
            artifact = model_dir / f"month_{month_name}_cutoff_{int(days[cutoff])}.npz"
            lineage = {
                "candidate": "R16ML1", "month": month_name,
                "cutoff": int(days[cutoff]), "task_year": task_year,
                "features": list(FEATURES), "raw_dependencies": list(RAW_UNION),
                "training_days_sha256": _array_sha(train_days),
                "exit_days_sha256": _array_sha(exit_days),
                "training_input_sha256": train["input_sha256"],
                "training_target_sha256": train["target_sha256"],
                "raw_feature_sha256": train["raw_feature_sha256"],
                "endpoint_sha256": train["endpoint_sha256"],
                "environment": env_identity,
                "adapter_sha256": _file_sha(Path(__file__).resolve()),
                "design_sha256": FROZEN_DESIGN_SHA256,
                "codes_sha256": _array_sha(np.asarray(ctx.panel.codes)),
                "frozen_universe_sha256": _file_sha(ctx.cfg.frozen_universe),
                "parent_module_sha256": {n: _file_sha(Path(__import__("inspect").getfile(
                    __import__("fea.spec", fromlist=["REGISTRY"]).REGISTRY[n].fn)).resolve())
                    for n in FEATURES},
                "reference_training_source_sha256": FROZEN_TRAINING_SOURCE_SHA256,
            }
            # Per-month artifact reuse is allowed only for exactly matching
            # source/training identities within this phase/year namespace.
            if artifact.exists():
                with np.load(artifact, allow_pickle=False) as saved:
                    prior = json.loads(str(saved["lineage_json"].item()))
                    if prior != lineage:
                        raise RuntimeError(f"monthly model cache lineage differs: {artifact}")
                    weights = {k[len("weight_"):]: saved[k].copy()
                               for k in saved.files if k.startswith("weight_")}
                    median = saved["median"].copy()
                    iqr = saved["iqr"].copy()
                    expected_state_sha = str(saved["state_sha256"].item())
                    saved_train_days = saved["train_days"].copy()
                    saved_exit_days = saved["exit_days"].copy()
                    saved_slot_counts = saved["slot_counts"].copy()
                if (not np.array_equal(median, train["median"])
                        or not np.array_equal(iqr, train["iqr"])
                        or not np.array_equal(saved_train_days, train_days)
                        or not np.array_equal(saved_exit_days, exit_days)
                        or not np.array_equal(saved_slot_counts, train["counts"])):
                    raise RuntimeError(f"monthly cached scales/training rows differ from current frozen input: {artifact}")
                if state_sha(weights) != expected_state_sha:
                    raise RuntimeError(f"monthly model state hash mismatch: {artifact}")
                # Build a fresh exact model using the frozen architecture, then
                # restore the saved state. This initialization is deterministic.
                torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
                model = torch.nn.Sequential(
                    torch.nn.Linear(5, 16, dtype=torch.float64), torch.nn.ReLU(),
                    torch.nn.Linear(16, 8, dtype=torch.float64), torch.nn.ReLU(),
                    torch.nn.Linear(8, 1, dtype=torch.float64),
                ).cuda()
                model.load_state_dict({k.replace("_", ".", 1): torch.as_tensor(v)
                                       for k, v in weights.items()})
            else:
                model, state = fit_once(train["x"], train["y"])
                median, iqr = train["median"], train["iqr"]
                state_sha_value = state_sha(state)
                tmp = artifact.with_suffix(".npz.tmp")
                with tmp.open("wb") as stream:
                    np.savez_compressed(
                        stream, lineage_json=np.asarray(json.dumps(lineage, sort_keys=True)),
                        state_sha256=np.asarray(state_sha_value), median=median, iqr=iqr,
                        train_days=train_days, exit_days=exit_days,
                        slot_counts=train["counts"],
                        **{f"weight_{k}": v for k, v in state.items()})
                    stream.flush(); os.fsync(stream.fileno())
                os.replace(tmp, artifact)
            prediction, prediction_mask = prediction_month(
                model, features, traded, universe, pred_rows, median, iqr)
            prediction_path = model_dir / f"pred_{month_name}_rows_{_array_sha(days[pred_rows])[:16]}.npz"
            if prediction_path.exists():
                with np.load(prediction_path, allow_pickle=False) as saved:
                    prior_lineage = json.loads(str(saved["lineage_json"].item()))
                    prior_dates = saved["dates"].copy()
                    prior_prediction = saved["prediction_f32"].copy()
                    prior_mask = saved["prediction_mask"].copy()
                if (prior_lineage != lineage or not np.array_equal(prior_dates, days[pred_rows])
                        or not np.array_equal(prior_prediction, prediction, equal_nan=True)
                        or not np.array_equal(prior_mask, prediction_mask)):
                    raise RuntimeError(f"cached monthly prediction does not replay exactly: {prediction_path}")
            else:
                pred_tmp = prediction_path.with_suffix(".npz.tmp")
                with pred_tmp.open("wb") as stream:
                    np.savez_compressed(stream, month=np.asarray(month_name), dates=days[pred_rows],
                                        prediction_f32=prediction, prediction_mask=prediction_mask,
                                        model_artifact=np.asarray(artifact.name),
                                        prediction_sha256=np.asarray(_array_sha(prediction)),
                                        lineage_json=np.asarray(json.dumps(lineage, sort_keys=True)),
                                        environment_json=np.asarray(json.dumps(env_identity, sort_keys=True)))
                    stream.flush(); os.fsync(stream.fileno())
                os.replace(pred_tmp, prediction_path)
            if prediction.shape != (len(pred_rows), ctx.panel.C):
                raise RuntimeError("monthly prediction axis does not match current Context")
            out[pred_rows] = prediction
            # One atomic, content-addressed lineage index per annual context.
            manifest_path = model_dir / "monthly_models.manifest.json"
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except FileNotFoundError:
                manifest = {"candidate": "R16ML1", "task_year": task_year,
                            "cutoff": ctx.up.audit_cutoff, "months": {}}
            if manifest.get("candidate") != "R16ML1" or manifest.get("task_year") != task_year:
                raise RuntimeError("monthly model manifest identity mismatch")
            manifest["months"][month_name] = {
                "cutoff": int(days[cutoff]), "model_file": artifact.name,
                "model_file_sha256": _file_sha(artifact),
                "prediction_file": prediction_path.name,
                "prediction_file_sha256": _file_sha(prediction_path),
                "prediction_sha256": _array_sha(prediction),
                "lineage": lineage,
            }
            tmp_manifest = manifest_path.with_suffix(".json.tmp")
            tmp_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                                    encoding="utf-8")
            with tmp_manifest.open("rb") as stream:
                os.fsync(stream.fileno())
            os.replace(tmp_manifest, manifest_path)

        torch.cuda.synchronize()
        if int(torch.cuda.max_memory_reserved(0)) > GPU_BUDGET_BYTES:
            raise MemoryError("R16ML1 CUDA reserved memory exceeded the hard 1-GiB cap")
        if np.isinf(out).any():
            raise RuntimeError("R16ML1 output contains Inf")
    return out


def build(parameters):
    """Catalog builder: accept the one frozen parameter identity only."""
    if json.dumps(dict(parameters), sort_keys=True, separators=(",", ":")) != json.dumps(
            FROZEN_PARAMETERS, sort_keys=True, separators=(",", ":")):
        raise ValueError("R16ML1 single-file builder accepts only the frozen parameter dictionary")
    compute.__name__ = "r16ml1_official_context_frozen"
    return compute
