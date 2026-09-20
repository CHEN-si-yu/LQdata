"""限速 HTTP 客户端 —— 本工程自建（不 import 旧工程的 lingqi.client）。

为什么必须自建而不是直接调 requests：
    服务端限速是 **280 次/分钟**，且是**按 apiKey 全局**算的。旧工程和我这个日更
    工程会同时在共享盘上跑，各自 260/min 叠加就突破红线 → 被封。
    所以两边必须**共用同一份全局限速台账**（`datadownload/state/ratelimit.json`）。

两级限速（与旧工程同构，照抄其行为而非其代码）：
    请求 → GlobalRateLimiter（跨进程，文件锁 + 60 秒滑窗 + 批量领名额）
         → RateLimiter（进程内令牌桶）
         → 发出

★ 批量领名额：共享盘是网络文件系统，实测单次 flock 中位 7~10ms、最坏 1.7 秒。
  每个请求都锁一次文件，在 12 并发下互相争抢，吞吐反而比 8 并发慢一倍。
  一次领 20 个名额可以把文件操作降一个数量级。
"""
from __future__ import annotations

import fcntl
import json
import logging
import os
import random
import threading
import time
from pathlib import Path
from typing import Any

import requests
from requests.adapters import HTTPAdapter

log = logging.getLogger("daily.client")


# ================================================================ 异常
class ApiError(Exception):
    """业务错误（服务端 HTTP 200 但 body 里 code != 200）。不要重试。"""


class RetryableError(Exception):
    """可重试错误（网络抖动 / 5xx / 空结果）。"""


class QueryLimitError(ApiError):
    """A valid query must be split to remain below the vendor's row limit."""


class FetchRows(list):
    """List-compatible response with per-request pagination completeness.

    Shared counters cannot identify which concurrent entity request was short.
    Preserve returned rows, but never turn an incomplete response into done.
    """
    def __init__(self, rows=(), *, complete=True):
        super().__init__(rows)
        self.complete = complete


# ================================================================ 信封解析
def extract_list(data: Any) -> list[dict]:
    """把 4 种响应信封统一取出记录列表。

    - list            → 直接是记录数组
    - {"list": [...]} → 标准信封
    - {"data": [...]} → 另一种信封
    - {"data": {...}} → Map 形态（daily_dump 的 Map<股票代码, [[...]]>），由调用方处理
    """
    if data is None:
        return []
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("list", "data"):
            v = data.get(key)
            if isinstance(v, list):
                return v
    return []


def extract_total(data: Any, default: int | None = None) -> int | None:
    """取服务端信封里的**总行数**。没有这个字段就返回 `default`（通常是 None）。

    ★★ 2026-09-17 删掉一条会**静默截断分页**的兜底：
       旧实现在信封没有 `total` 时返回 `len(data["list"])`（= **当前这一页的行数**），
       而 `fetch_all` 的终止条件是 `len(rows) >= total` —— 于是**第一页就满足**，
       整个请求只取一页就返回，且不报错、不计 mismatch（因为 `len(rows) == total`）。
       现在返回 None → `fetch_all` 退回"翻到短页为止"的语义（旧工程的行为）。
    """
    if isinstance(data, dict):
        t = data.get("total")
        if t is not None:
            try:
                return int(t)
            except (TypeError, ValueError):
                return default
        return default
    if isinstance(data, list):
        return len(data)          # 裸数组响应：没有分页概念，长度即总数
    return default


# ================================================================ 限速器
class RateLimiter:
    """进程内令牌桶 —— **线程安全、均匀放行**。

    ★★ 2026-09-15 修掉一个会让并发**反而慢 20 倍**的 bug。
       旧实现是：
           wait = self.next_at - now;  if wait>0: sleep(wait)
           self.next_at = max(self.next_at, now) + interval
       **没有锁** —— 12 个线程读到同一个 `next_at`、算出同一个 wait、
       **在同一时刻醒来并一起发出**。对服务端就是一个 12 连发的突刺，
       触发服务端软性节流：实测单请求延迟从 **0.8s 涨到 10s**，
       76 个请求从 ~1 分钟变成 11 分钟。

       ⚠️ 这不是我一个人的发现 —— 参考实现（`/root/scripts`）的注释里
       点名了同一个坑：「睡眠醒来后必须重新检查 token……所有线程会一次通过
       （整批爆发），导致请求按批次突刺触发服务器 429（2026-08-01 全量重拉
       踩坑的根因）」。

       修法：**在锁内一次性预订时间槽**（`_next_at += interval`），
       每个线程拿到互不相同的出发时刻，醒来后各走各的，自然匀速。
    """

    def __init__(self, per_minute: int):
        self.per_minute = max(1, int(per_minute))
        self.interval = 60.0 / self.per_minute
        self._next_at = 0.0
        self._lock = threading.Lock()

    def acquire(self) -> None:
        with self._lock:
            now = time.monotonic()
            t = max(self._next_at, now)
            self._next_at = t + self.interval      # ★ 原子地预订下一个槽位
            wait = t - now
        if wait > 0:
            time.sleep(wait)


class GlobalRateLimiter:
    """跨进程全局限速：文件锁 + 60 秒滑窗 + 一次领 batch 个名额。

    台账文件与旧工程**共用**（`datadownload/state/ratelimit.json`），
    格式是 `[时间戳, ...]` 的 JSON 数组 —— 必须保持这个形状，否则旧工程读不了。
    """

    def __init__(self, per_minute: int, path: Path, batch: int = 20):
        self.per_minute = max(1, int(per_minute))
        self.path = Path(path)
        self.batch = max(1, int(batch))
        self._left = 0                 # 本进程已领到、尚未用完的名额
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            try:
                self.path.write_text("[]", encoding="utf-8")
            except OSError:
                pass

    def _load(self) -> list[float]:
        try:
            v = json.loads(self.path.read_text(encoding="utf-8"))
            return [float(x) for x in v] if isinstance(v, list) else []
        except (json.JSONDecodeError, OSError, TypeError, ValueError):
            return []

    def acquire(self) -> None:
        if self._left > 0:
            self._left -= 1
            return
        fails = 0
        while True:
            try:
                # ★★ 2026-09-17：用 O_CREAT 自建，而不是 `open(...,"r+")`。
                #   旧实现在文件不存在/建不出来时会抛 FileNotFoundError（是 OSError），
                #   被下面的 `except OSError` 吞掉 → sleep(0.2) **无限自旋**：
                #   不报错、不退出、无日志，整个日更卡死在第一个请求上。
                fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o664)
                fails = 0
            except OSError as exc:
                fails += 1
                if fails >= 150:        # ~30s 还打不开：大声报错，别静默空转
                    raise RuntimeError(
                        f"限速台账打不开：{self.path}（{exc}）。共享盘权限/属主可能不一致 ——"
                        f" 修法：`chgrp -R root <state>` + `chmod -R g+rw <state>`") from exc
                time.sleep(0.2)
                continue
            try:
                with os.fdopen(fd, "r+", encoding="utf-8") as f:
                    fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                    try:
                        raw = f.read()
                        try:
                            arr = [float(x) for x in json.loads(raw or "[]")]
                        except (json.JSONDecodeError, TypeError, ValueError):
                            arr = []
                        now = time.time()
                        # ★ 丢掉**遥远的未来时间戳**：跨机/跨容器时钟不同步时写入的
                        #   `t > now`，会让 `now - t < 60` 恒真 → 等待方永远等下去
                        #   （偏差 1 小时就卡 1 小时，且无日志）。
                        arr = [t for t in arr if -5.0 < now - t < 60.0]
                        if len(arr) + self.batch <= self.per_minute:
                            arr.extend([now] * self.batch)
                            self._left = self.batch - 1
                            f.seek(0)
                            f.truncate()
                            f.write(json.dumps(arr))
                            # ★★★ 必须在**解锁之前**把数据交给操作系统。
                            #   `truncate()` 是立即生效的系统调用，而 `write()` 只进
                            #   Python 缓冲；不解锁就 return 的话，[解锁, close] 这段
                            #   窗口里文件是 **0 字节** —— 另一个进程（含旧工程）读到
                            #   空台账会以为额度全空、各自按满额领，一分钟内真实请求
                            #   可能越过 280 红线；进程若在这段窗口里被杀，台账还会
                            #   被清零。旧工程 `lingqi/client.py` 就是锁内 flush 的，
                            #   这里属于移植时丢了（2026-09-17 补回）。
                            f.flush()
                            return
                        # 额度不够：按最老那笔算出还要等多久
                        oldest = min(arr) if arr else now
                        sleep_s = max(0.05, 60.0 - (now - oldest) + 0.02)
                    finally:
                        fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            except OSError:
                time.sleep(0.2)     # 共享盘抖动，退一步重试
                continue
            time.sleep(min(sleep_s, 5.0))


# ================================================================ 客户端
class Client:
    """限速 + 重试 + 信封抹平的 HTTP 客户端。"""

    def __init__(self, cfg: dict, logger: logging.Logger | None = None):
        api = cfg["api"]
        self.base_url = str(api["base_url"]).rstrip("/")
        self.timeout = float(api.get("timeout", 120))
        self.max_retries = int(api.get("max_retries", 6))
        self.backoff_base = float(api.get("backoff_base", 1.6))
        self.backoff_max = float(api.get("backoff_max", 30))
        # ★★ 重试等待档位（用户 2026-09-15 指定）：
        #    `1s → 3s → 5s → 10s → 30s`，共 **5 次重试**（+首次 = 6 次尝试），
        #    五档都失败才记 ⚠️ 交给报告。实测动机：厂商 nginx 的 502 是**分时段发作**的，
        #    旧实现用指数退避（1.6^n + 随机，实测睡 2.5/3.4/4.8/7.3/11s）——
        #    总量差不多但**不好对账**：日志里看不出"这个请求退到第几档了"。
        #    固定档位一句话能说清、能预期，也方便事后复盘。
        #    配成 [] 则退回指数退避（保留旧行为）。
        self.retry_backoff = [float(x) for x in
                              (api.get("retry_backoff_seconds") or [])]
        if self.retry_backoff:
            # 档位数 = 重试次数；总尝试次数 = 档位数 + 1
            self.max_retries = len(self.retry_backoff) + 1
        self.max_rows_per_query = int(api.get("max_rows_per_query", 100_000))
        self.empty_retries = int(api.get("empty_retries", 4))

        self.rate = RateLimiter(api.get("rate_limit_per_min", 260))
        self.global_rate: GlobalRateLimiter | None = None
        if api.get("global_rate_limit", True):
            gp = Path(api.get("global_rate_limit_file", "")) if api.get("global_rate_limit_file") else None
            if gp:
                self.global_rate = GlobalRateLimiter(
                    api.get("rate_limit_per_min", 260), gp)

        key_file = Path(api.get("api_key_file", ""))
        self.api_key = key_file.read_text(encoding="utf-8").strip() if key_file.exists() else ""
        if not self.api_key:
            raise SystemExit(f"读不到 API Key：{key_file}")

        # ★★ 连接池必须 ≥ 并发数（2026-09-15 实测踩到的性能陷阱）：
        #    `requests.Session` 默认 `pool_maxsize = 10`，而我们的并发是 12 ——
        #    多出来的连接会被**丢弃并重建**，每个请求都要重做一次 TCP + TLS 握手。
        #    日志特征：`WARNING Connection pool is full, discarding connection ... size: 10`
        #    实测后果：cyq_chips 并发后仍然 76 请求跑了好几分钟（与串行差不多）。
        #    把池子开到并发的 2 倍，并发才真正生效。
        conc = max(8, int(api.get("concurrency", 12)))
        self.session = requests.Session()
        adapter = HTTPAdapter(pool_connections=conc * 2, pool_maxsize=conc * 2,
                              max_retries=0)          # 重试由我们自己的退避逻辑管
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)
        # ★★ 在途请求闸门（参考实现 /root/scripts 的核心设计）：
        #    「6 个 worker 做流水线 + **在途请求掐在 3 个**」。
        #    实测加大并发会让服务端软性节流：12 并发时单请求延迟从 0.8s 涨到 10s，
        #    总耗时反而慢 20 倍。所以并发度由**这个信号量**决定，而不是 thread pool 大小。
        self.max_inflight = max(1, int(api.get("max_inflight", 3)))
        self._gate = threading.BoundedSemaphore(self.max_inflight)
        self.stats = {"requests": 0, "retries": 0, "errors": 0, "empty_retries": 0, "bytes_in": 0}

    # ------------------------------------------------------------ 底层
    def _retry_delay(self, attempt: int) -> float:
        """第 `attempt` 次失败之后睡多久。

        固定档位（用户 2026-09-15 指定）：1s → 3s → 5s → 10s → 30s；
        没配档位时退回指数退避（旧行为）。
        末尾的 ≤0.5s 抖动：同一时刻最多 3 个在途请求（`max_inflight`），
        让它们别在同一毫秒一起重试 —— 抖动不改变"第几档"这个语义。
        """
        if not self.retry_backoff:
            return min(self.backoff_base ** attempt + random.random(), self.backoff_max)
        i = min(max(0, attempt - 1), len(self.retry_backoff) - 1)
        return self.retry_backoff[i] + random.uniform(0.0, 0.5)

    def _throttle(self) -> None:
        if self.global_rate is not None:
            self.global_rate.acquire()
        self.rate.acquire()

    def call(self, path: str, payload: dict | None = None, method: str = "POST",
             expect_rows: bool = True, empty_retries: int | None = None,
             before_request=None) -> Any:
        """发一次请求，返回**原始信封**（dict 或 list）。

        expect_rows=True 时，空结果会被当成"服务端偶发空响应"重试若干次
        （实测服务端确实会偶发返回 total=0，重试即好）。
        对"合法为空"的接口（早年无数据、退市股、非交易日）必须显式传 False，
        否则每次空响应白烧 4 次请求。
        """
        url = self.base_url + path
        empty_left = self.empty_retries if empty_retries is None else int(empty_retries)
        last_exc: Exception | None = None

        for attempt in range(1, self.max_retries + 1):
            self._throttle()
            try:
                # ★ 在途闸门只罩住**真正的 HTTP 往返**（不含后续解析/重试退避），
                #   用 `with` 保证任何异常路径都会释放 —— 手写 acquire/release
                #   一旦漏掉就会死锁（信号量被永久占满）。
                with self._gate:
                    if before_request is not None:
                        before_request()
                    if method.upper() == "GET":
                        resp = self.session.get(url, params=payload or {},
                                                headers={"apiKey": self.api_key},
                                                timeout=self.timeout)
                    else:
                        resp = self.session.post(url, json=payload or {},
                                                 headers={"apiKey": self.api_key},
                                                 timeout=self.timeout)
                self.stats["requests"] += 1
                self.stats["bytes_in"] += len(resp.content or b"")

                # ★★ 2026-09-17 重排：**先看 HTTP 状态码，再看信封**。
                #   旧实现是 `code = body.get("code", 200)` —— 任何"没有 code 字段的
                #   JSON 错误体"（404/401/403/429 常见的 `{"detail":"Not Found"}`）
                #   都会被当作 code=200 的成功响应；`data` 取不到 → 对
                #   `expect_rows=False` 的调用方就是"合法为空" → `run_range` 直接
                #   `add_coverage()` → **永久空洞**（而且没有任何告警）。
                sc = resp.status_code
                try:
                    body: Any = resp.json()
                except ValueError:
                    body = None
                code = body.get("code") if isinstance(body, dict) else None

                if sc == 429:
                    # 429 = 服务端限流。用**长档**退避（见 call 里的处理），
                    # 否则 49 秒内再打 6 发只会把软节流推成硬封。
                    raise RetryableError("HTTP 429（服务端限流）")
                if sc >= 500:
                    if sc == 502:
                        # 厂商 nginx 的 502 是分时段发作的（周期 15~30 分钟，与我们的限速无关）
                        raise RetryableError(f"HTTP 502（厂商 nginx 抖动）")
                    raise RetryableError(f"HTTP {sc}")
                if sc >= 400:
                    # 4xx 且响应体里没有厂商信封的 code 字段 → 这是真错误，不是"空数据"
                    if sc == 405:
                        raise ApiError("HTTP 405 Method Not Allowed —— 该接口的 method 或参数名不对")
                    raise ApiError(f"HTTP {sc} 且响应体不是标准信封：{str(resp.text)[:120]}")
                if not isinstance(body, (dict, list)):
                    raise RetryableError("HTTP 成功但响应不是有效 JSON 数据，拒绝当成空结果")
                if isinstance(body, dict) and "data" not in body:
                    raise RetryableError("响应缺少 data 信封，拒绝当成空结果")
                if code is None:
                    code = 200
                if code not in (200, 0):
                    detail = ""
                    d = body.get("data")
                    if isinstance(d, list):
                        detail = " | " + "; ".join(str(x) for x in d[:3])
                    msg = f"code={code} msg={body.get('msg')}{detail}"
                    # 400 = 业务拒绝（参数错/超窗口）；422 = 参数校验失败。都不该重试
                    if code in (400, 422):
                        raise ApiError(msg)
                    raise RetryableError(msg)

                data = body.get("data") if isinstance(body, dict) else body

                if expect_rows and not extract_list(data):
                    # ★ Map 形态（daily_dump 的 Map<股票代码, [[...]]>）不能被判成空！
                    #   旧工程在这里踩过坑：判据 `not data.get("list")` 对 Map 恒为 True
                    #   → 一个 dump 请求被放大成 5 次真实下载（配额是雷区）。
                    #   只有"确实带 list/data 信封键且为空，或整个 dict 为空"才算空。
                    is_map = isinstance(data, dict) and bool(data) and "list" not in data and "data" not in data
                    if is_map:
                        pass
                    else:
                        has_envelope = isinstance(data, dict) and (
                            isinstance(data.get("list"), list) or isinstance(data.get("data"), list))
                        if (has_envelope or not data) and empty_left > 0:
                            empty_left -= 1
                            self.stats["empty_retries"] += 1
                            raise RetryableError("空结果（预期非空），重试")
                return data

            except ApiError:
                self.stats["errors"] += 1
                raise
            except (RetryableError, requests.RequestException, ValueError) as exc:
                last_exc = exc
                self.stats["errors"] += 1
                if attempt >= self.max_retries:
                    break
                self.stats["retries"] += 1
                sleep_s = self._retry_delay(attempt)
                if "429" in str(exc):
                    # ★ 2026-09-17：被限流时走**长档**（≥60s）。固定档位最多 30s，
                    #   6 次尝试在 49 秒内打完 —— 对已经进入软节流的服务端
                    #   是把"软节流"推成"硬封禁"的最快方式。
                    sleep_s = max(sleep_s, 60.0)
                log.debug("%s 第 %d 次失败（%s），%.1fs 后重试（档位 %d/%d）",
                          path, attempt, exc, sleep_s, attempt, len(self.retry_backoff) or "-")
                time.sleep(sleep_s)

        raise RetryableError(
            f"{path} 重试 {self.max_retries - 1} 次仍失败"
            + (f"（档位 {'/'.join(f'{x:g}s' for x in self.retry_backoff)}）"
               if self.retry_backoff else "")
            + f": {last_exc}")

    # ------------------------------------------------------------ 分页
    def fetch_all(self, path: str, payload: dict, page_size: int, method: str = "POST",
                  expect_rows: bool = False, max_rows: int | None = None) -> list[dict]:
        """翻页收齐。`page * page_size <= 100000` 是服务端硬限制。

        收不齐时**抛错而不是返回残缺数据** —— 残缺数据落盘会变成永久空洞
        （旧工程 index_daily 缺 159 个交易日就是这么来的）。
        """
        limit = max_rows or self.max_rows_per_query
        rows: list[dict] = []
        total: int | None = None
        page = 0
        ended_naturally = False
        while page < 100_000:
            if page * page_size >= limit:
                raise QueryLimitError(f"{path} 分页达到 {limit:,} 行上限仍未确认结束，请缩小作业窗口")
            body = {**payload, "page": page, "page_size": page_size}
            data = self.call(path, body, method=method,
                             expect_rows=expect_rows and page == 0)
            batch = extract_list(data)
            if total is None:
                total = extract_total(data)
                if total is not None and total > limit:
                    raise QueryLimitError(
                        f"{path} 单次任务要取 {total:,} 行，超过服务端上限 {limit:,} 行"
                        f"（page*page_size ≤ 十万）。请缩小作业窗口或改用批量实体。")
            rows.extend(batch)
            # ★★ 2026-09-17：短页/空页仍然**接受**（服务端的 `total` 本身可能是错的 ——
            #   实测 `399317.SZ` 报 total=13,428 却只给 5,236 行，抛错只会把已经拿到的
            #   数据丢掉），但**必须记账**：下面的 `total_mismatch` 会让
            #   `run_range` **不给这一段标 coverage** → 下一轮重新当缺口取（自愈），
            #   而不是像旧实现那样把被截断的一段直接声明为"已覆盖"（永久空洞）。
            if not batch or len(batch) < page_size:
                ended_naturally = True
                break
            if total is not None and len(rows) >= total:
                break
            page += 1

        # ★★ 2026-09-15：区分「翻页翻到底」与「被截断」。
        #    旧实现只要 `len(rows) < total` 就抛错，**把已经翻完的完整数据也丢掉**。
        #    实测反例：`/index/weight` 的 `399317.SZ` 服务端报 `total=13,428`，
        #    但翻页到第 6 页返回 236 行后就没有了（实测 5,236 行封顶）——
        #    total 字段本身就是错的。旧行为：抛错 → 这 5,236 行**被丢弃** →
        #    该指数在 2026-06 之后整段消失（本地存量数据反而被"升级"弄丢了）。
        #    抛错换不来更多数据，只会丢数据，所以这里**接受自然结束的结果**，
        #    仅计入 `stats["total_mismatch"]` 让上层能看见。
        #    真正需要防的是「没翻完就停了」（页码上限 / max_rows 上限），那条仍然抛错。
        if total is not None and len(rows) < total:
            if ended_naturally:
                self.stats["total_mismatch"] = self.stats.get("total_mismatch", 0) + 1
                log.warning("%s total=%s 但翻页自然结束于 %s 行 —— 以实际取到的为准",
                            path, total, f"{len(rows):,}")
                return FetchRows(rows, complete=False)
            raise RetryableError(f"{path} 分页不足：拿到 {len(rows):,} / 应有 {total:,}")
        return FetchRows(rows)

    def close(self) -> None:
        try:
            self.session.close()
        except Exception:  # noqa: BLE001
            pass
