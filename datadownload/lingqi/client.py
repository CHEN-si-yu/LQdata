"""限速 HTTP 客户端。

设计要点（全部来自实测）：
  * 服务端偶发返回**空结果**（同一请求首次 total=0、重试后 3796 行），
    因此"预期非空"的请求要把空响应当成可重试错误。
  * 响应信封有 4 种形态，统一用 extract_list/extract_total 抹平。
  * 官方限速 280 次/分钟，这里用令牌桶严格执行（并发只影响延迟，不影响速率）。
"""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import requests


class ApiError(RuntimeError):
    """服务端明确返回的业务错误（不可重试）。"""


class RetryableError(RuntimeError):
    """网络/服务端瞬时故障，可重试。"""


class RateLimiter:
    """令牌桶。容量 = 每分钟限额，按时间线性补充。"""

    def __init__(self, per_minute: int):
        self.rate = per_minute / 60.0          # 每秒补充的令牌数
        self.capacity = float(per_minute)
        self._tokens = float(per_minute)
        self._last = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        while True:
            with self._lock:
                now = time.monotonic()
                self._tokens = min(self.capacity, self._tokens + (now - self._last) * self.rate)
                self._last = now
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return
                wait = (1.0 - self._tokens) / self.rate
            time.sleep(min(wait, 0.25))


class GlobalRateLimiter:
    """跨进程共享的滑动窗口限速器。

    为什么需要它：限速是**按 apiKey 全局限**的，而令牌桶是进程内的。
    共享盘上可能同时跑多个实例（换机器、多任务并行、用户手动补跑），
    各自 260/min 叠加起来就会突破 280/min 被服务端限流。
    这里用文件锁 + 最近 60 秒时间戳表，让所有进程共用一个额度。

    代价：每次请求多一次小文件读写（约 0.1ms 量级），相对网络延迟可忽略。
    """

    WINDOW = 60.0

    def __init__(self, path, per_minute: int, batch: int = 20):
        self.path = Path(path)
        self.limit = per_minute
        self.batch = max(1, batch)
        # 进程内缓存的名额 + 节流状态
        self._local_left = 0
        self._local_lock = threading.Lock()
        self._last_req = 0.0
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def acquire(self) -> None:
        """先用进程内配额（便宜），不够了才动一次共享文件（昂贵）。

        为什么这么设计：共享盘是网络文件系统，实测单次 flock 操作
        中位 7~10ms、**最坏 1.7 秒**。每个请求都去锁一次文件，在 20 并发下
        互相争抢，吞吐直接崩掉（实测并发 20 反而比并发 8 慢一倍）。
        改成一次文件操作换 batch 个名额，文件操作次数降一个数量级。

        随后用进程内节流把请求摊平：只做窗口计数会允许"一秒打满 280 次"的
        尖峰，虽然形式上没超限，但对服务端是突发流量，容易触发风控。
        """
        import fcntl
        with self._local_lock:
            if self._local_left <= 0:
                self._refill()
            self._local_left -= 1
            # 均匀节流：距上次请求至少 60/limit 秒（进程内串行，开销可忽略）
            gap = self.WINDOW / self.limit
            wait = self._last_req + gap - time.time()
            if wait > 0:
                time.sleep(wait)
            self._last_req = time.time()

    def _refill(self) -> None:
        """向共享文件申领一批名额。拿不到就等到窗口滑动出空间。"""
        import fcntl
        while True:
            fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o644)
            f = os.fdopen(fd, "r+")
            wait = 0.0
            try:
                fcntl.flock(f, fcntl.LOCK_EX)
                now = time.time()
                try:
                    stamps = [t for t in json.loads(f.read() or "[]")
                              if now - t < self.WINDOW]
                except ValueError:
                    stamps = []
                room = self.limit - len(stamps)
                if room > 0:
                    take = min(self.batch, room)
                    stamps.extend([now] * take)
                    f.seek(0)
                    f.truncate()
                    f.write(json.dumps(stamps))
                    f.flush()
                    self._local_left = take
                    return
                wait = self.WINDOW - (now - stamps[0]) + 0.05
            finally:
                try:
                    fcntl.flock(f, fcntl.LOCK_UN)
                finally:
                    f.close()
            time.sleep(max(0.05, min(wait, 5.0)))


@dataclass
class Stats:
    """全局统计，供进度条展示。"""

    requests: int = 0
    retries: int = 0
    errors: int = 0
    empty_retries: int = 0
    rows: int = 0
    bytes_in: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add(self, **kw: int) -> None:
        with self._lock:
            for k, v in kw.items():
                setattr(self, k, getattr(self, k) + v)

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return {
                "requests": self.requests, "retries": self.retries, "errors": self.errors,
                "empty_retries": self.empty_retries, "rows": self.rows, "bytes_in": self.bytes_in,
            }


def extract_list(data: Any) -> list[dict]:
    """把 4 种响应信封统一取出记录列表。"""
    if data is None:
        return []
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("list", "data"):
            v = data.get(key)
            if isinstance(v, list):
                return v
        # daily_dump 的 Map<股票代码, List<Entry>> 形态由调用方处理
    return []


def extract_total(data: Any, default: int | None = None) -> int | None:
    if isinstance(data, dict):
        t = data.get("total")
        if t is not None:
            return int(t)
        if isinstance(data.get("list"), list):
            return len(data["list"])
    if isinstance(data, list):
        return len(data)
    return default


class LingqiClient:
    def __init__(self, cfg: dict, logger=None):
        api = cfg["api"]
        self.base_url = api["base_url"].rstrip("/")
        self.rate = RateLimiter(api["rate_limit_per_min"])
        self.timeout = api["timeout"]
        self.max_retries = api["max_retries"]
        self.backoff_base = api.get("backoff_base", 1.6)
        self.backoff_max = api.get("backoff_max", 30)
        self.empty_retries = cfg["download"].get("empty_retries", 4)
        self.stats = Stats()
        self.log = logger

        key = self._read_key(cfg)
        self._headers = {"apiKey": key, "Content-Type": "application/json"}
        self._local = threading.local()

        # 全局限速（跨进程）。默认开启，是防止多实例叠加突破 280/min 的关键。
        self.global_rate = None
        if api.get("global_rate_limit", True):
            from pathlib import Path as _P
            root = _P(__file__).resolve().parent.parent
            self.global_rate = GlobalRateLimiter(
                root / cfg["paths"]["state"] / "ratelimit.json",
                api.get("rate_limit_global_per_min", api["rate_limit_per_min"]),
            )

    def _read_key(self, cfg: dict) -> str:
        from pathlib import Path
        root = Path(__file__).resolve().parent.parent
        p = root / cfg["api"]["api_key_file"]
        key = p.read_text(encoding="utf-8").strip()
        if not key:
            raise SystemExit(f"API key 文件为空: {p}")
        return key

    @property
    def session(self) -> requests.Session:
        """每线程独立 Session，复用连接。"""
        s = getattr(self._local, "s", None)
        if s is None:
            s = requests.Session()
            s.headers.update(self._headers)
            self._local.s = s
        return s

    def call(
        self,
        path: str,
        payload: dict | None = None,
        method: str = "POST",
        expect_rows: bool = False,
        raw: bool = False,
        empty_retries: int | None = None,
    ) -> Any:
        """发一次请求并返回 data 字段。

        expect_rows=True 时，空结果会被当作可重试错误（应对服务端偶发空响应）。
        empty_retries 可覆盖默认次数：历史区间常常"合法地为空"（数据还没开始），
        那种情况下多重退避重试纯属浪费 —— 区间模式只重试 1 次，
        剩余的交给 scripts/verify.py 事后复查，不拖慢主干吞吐。
        """
        url = self.base_url + path
        attempt = 0
        n_empty = self.empty_retries if empty_retries is None else empty_retries
        empty_left = n_empty if expect_rows else 0

        while True:
            attempt += 1
            # 先拿全局额度（跨进程），再拿进程内令牌桶，两者都满足才发请求
            if self.global_rate is not None:
                self.global_rate.acquire()
            self.rate.acquire()
            try:
                if method.upper() == "GET":
                    r = self.session.get(url, params=payload or {}, timeout=self.timeout)
                else:
                    r = self.session.post(url, json=payload or {}, timeout=self.timeout)
                self.stats.add(requests=1, bytes_in=len(r.content))

                if r.status_code != 200:
                    raise RetryableError(f"HTTP {r.status_code}: {r.content[:160]!r}")

                try:
                    body = r.json()
                except ValueError:
                    raise RetryableError(f"非 JSON 响应: {r.content[:160]!r}")

                if raw:
                    return body

                code = body.get("code")
                if code != 200:
                    msg = str(body.get("msg"))
                    # 实测错误语义：400=业务规则拒绝，422=参数校验失败（pydantic），
                    # 401/403/404=权限或路由问题。这些都是"重试也没用"的确定性错误。
                    # 注意 422 的 data 是字符串数组，不能当对象解析。
                    if code in (400, 401, 403, 404, 422):
                        detail = ""
                        if isinstance(body.get("data"), list):
                            detail = " | " + "; ".join(str(x) for x in body["data"][:3])
                        raise ApiError(f"code={code} msg={msg}{detail}")
                    raise RetryableError(f"code={code} msg={msg}")

                data = body.get("data")
                if expect_rows and not extract_list(data) and not isinstance(data, dict):
                    if empty_left > 0:
                        empty_left -= 1
                        self.stats.add(empty_retries=1)
                        raise RetryableError("空结果（预期非空），重试")
                if expect_rows and isinstance(data, dict) and not extract_list(data):
                    # ★ 只有"信封里的列表为空"才算空结果。
                    #   2026-09-13 修：原判据是 `not data.get("list")`，对
                    #   daily_dump 的 Map<股票代码, [[...]]> 形态会误判 ——
                    #   extract_list() 找不到 list/data 键返回 []，而
                    #   data.get("list") 恒为 None → 每次都抛"空列表"重试，
                    #   把一个 dump 请求放大成 empty_retries+1 = 5 次真实下载，
                    #   吃掉该日期 10 次配额里的一半（超限会封禁 3 天）。
                    #   新判据：必须真的带 list/data 信封键且它为空，或 dict 整个为空。
                    has_envelope = (isinstance(data.get("list"), list)
                                    or isinstance(data.get("data"), list))
                    if (has_envelope or not data) and empty_left > 0:
                        empty_left -= 1
                        self.stats.add(empty_retries=1)
                        raise RetryableError("空列表（预期非空），重试")
                return data

            except ApiError:
                self.stats.add(errors=1)
                raise
            except Exception as exc:  # noqa: BLE001 - 统一按可重试处理
                if attempt > self.max_retries:
                    self.stats.add(errors=1)
                    raise RetryableError(f"{path} 重试 {attempt - 1} 次仍失败: {exc}") from exc
                self.stats.add(retries=1)
                delay = min(self.backoff_base ** attempt, self.backoff_max)
                if self.log:
                    self.log.debug("重试 %s (第%d次, %.1fs后): %s", path, attempt, delay, exc)
                time.sleep(delay)

    def paginate(
        self,
        path: str,
        payload: dict,
        method: str = "POST",
        page_size: int = 10000,
        max_pages: int = 100000,
        on_page=None,
    ) -> list[dict]:
        """按页拉取直到取完。返回合并后的记录列表。

        用服务端返回的 total 判断结束；若某页返回空则停止（并由调用方校验）。
        """
        out: list[dict] = []
        page = 0
        total = None
        while page < max_pages:
            body = dict(payload)
            body["page"] = page
            body["page_size"] = page_size
            data = self.call(path, body, method=method, expect_rows=(page == 0 or not out))
            rows = extract_list(data)
            if total is None:
                total = extract_total(data)
            out.extend(rows)
            if on_page:
                on_page(len(rows))
            if not rows:
                break
            if total is not None and len(out) >= total:
                break
            if len(rows) < page_size and total is None:
                break
            page += 1
        return out
