"""Timestamped, aligned production display; only the coordinator renders it."""
from __future__ import annotations

import datetime as dt
import shutil
import sys
import time


def announce(message, stream=None):
    print(f"[{dt.datetime.now():%H:%M:%S}] {message}", file=stream or sys.stdout, flush=True)


STATES = {'skip': '已对齐，跳过', 'wait': '等待', 'run': '计算中',
          'done': '已完成', 'failed': '失败', 'blocked': '父因子失败，未计算'}


class ProductionProgress:
    def __init__(self, specs, targets, plans, mode='auto', stream=None, manifests=None):
        self.stream = stream or sys.stderr
        self.mode = ('bars' if self.stream.isatty() else 'plain') if mode == 'auto' else mode
        self.names = [s.name for s in specs]
        self.width = max(map(len, self.names), default=6)
        self.group_width = max((len(s.group) for s in specs), default=6)
        self.number_width = len(str(len(self.names)))
        self.index = {n: i for i, n in enumerate(self.names, 1)}
        self.totals = {n: len(targets[n]) for n in self.names}
        self.done = {n: self.totals[n] - sum(map(len, plans[n].values())) for n in self.names}
        self.states = {n: 'skip' if not plans[n] else 'wait' for n in self.names}
        self.active, self.task_started = {}, {}
        self.failed = set()
        self.started = time.monotonic()
        self.last_print = self.started
        self.bars, self.overall = {}, None
        if self.mode == 'bars':
            from tqdm import tqdm
            tqdm.monitor_interval = 0
            columns = shutil.get_terminal_size((140, 30)).columns
            prefix_width = 2*self.number_width+4
            width = min(self.width, max(12, columns - prefix_width - 66))
            fmt = '{desc} {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}'
            kwargs = dict(file=self.stream, ncols=columns, bar_format=fmt,
                          mininterval=0.3, leave=True, unit='天')
            self.overall = tqdm(total=sum(self.totals.values()), initial=sum(self.done.values()),
                                desc='总进度'.ljust(width+prefix_width-3), position=0, **kwargs)
            for pos, n in enumerate(self.names, 1):
                label = n if len(n) <= width else n[:width-1]+'…'
                self.bars[n] = tqdm(total=self.totals[n], initial=self.done[n],
                                    desc=self._number(n)+' '+label.ljust(width), position=pos,
                                    postfix=STATES[self.states[n]], **kwargs)
        elif self.mode == 'plain':
            announce('  因子清单 / 已有结果 / 本次计划', self.stream)
            for spec in specs:
                n = spec.name
                man = (manifests or {}).get(n)
                existing = (f'已有 {man.partition_rows():>10,} 行 / {len(man.partitions):>2} 分区'
                            if man else '已有          0 行 /  0 分区')
                span = f'[{man.coverage[0][0]}→{man.covered_until()}]' if man and man.coverage else '[—]'
                plan = (f'待算 {sum(map(len, plans[n].values()))} 天 / {len(plans[n])} 分区'
                        if plans[n] else '✔ 已对齐，自动跳过')
                announce(f'  {self._number(n)} {n:<{self.width}} ({spec.group:<{self.group_width}}) {existing} {span:<23} · {plan}', self.stream)

    def _number(self, name):
        return f'[{self.index[name]:>{self.number_width}}/{len(self.names)}]'

    def start(self, name, year):
        self.states[name] = 'run'
        self.active.setdefault(name, set()).add(year)
        self.task_started[(name, year)] = time.monotonic()
        years = ','.join(map(str, sorted(self.active[name])))
        if name in self.bars:
            self.bars[name].set_postfix_str('计算中 '+years, refresh=False)
        elif self.mode == 'plain':
            announce(f'  {self._number(name)} {name:<{self.width}} {year} · 开始计算', self.stream)

    def finish(self, name, year, days, error=None, rows=0, added=0):
        elapsed = time.monotonic() - self.task_started.pop((name, year), time.monotonic())
        self.active.get(name, set()).discard(year)
        if error:
            self.failed.add(name)
        else:
            self.done[name] += days
        state = ('failed' if name in self.failed else
                 'done' if self.done[name] == self.totals[name] else
                 'run' if self.active.get(name) else 'wait')
        self.states[name] = state
        detail = f'{STATES[state]} {year} · {rows:,}行 · {elapsed:.1f}s'
        if name in self.bars:
            self.bars[name].update(0 if error else days)
            self.bars[name].set_postfix_str(detail, refresh=False)
            self.overall.update(0 if error else days)
        elif self.mode == 'plain':
            mark = '✘' if error else '✔'
            announce(f'     {mark} {name:<{self.width}} {year} · 新增 {added:>10,} 行 · '
                     f'写入 {rows:>10,} 行 · 进度 {self.done[name]}/{self.totals[name]} 天 · 用时 {elapsed:.1f}s'
                     + (f' · {error}' if error else ''), self.stream)

    def block(self, name):
        self.states[name] = 'blocked'
        if name in self.bars:
            self.bars[name].set_postfix_str(STATES['blocked'])
        elif self.mode == 'plain':
            announce(f'     ✘ {name:<{self.width}} {STATES["blocked"]}', self.stream)

    def heartbeat(self, jobs, memory_gib=None):
        finished = sum(s in ('skip', 'done') for s in self.states.values())
        if self.overall:
            self.overall.set_postfix_str(f'已对齐 {finished}/{len(self.names)} · 在算 {jobs}' +
                                         (f' · 内存 {memory_gib:.1f}GiB' if memory_gib is not None else ''), refresh=False)
            self.overall.refresh()
            for name, bar in self.bars.items():
                if self.active.get(name):
                    bar.refresh()
        elif self.mode == 'plain' and time.monotonic()-self.last_print >= 10:
            self.last_print = time.monotonic()
            active = ', '.join(f'{n}/{",".join(map(str, sorted(ys)))}' for n, ys in self.active.items() if ys)
            announce(f'  ⏳ 已对齐 {finished}/{len(self.names)} 因子 · 在算 {jobs} · '
                     f'已用时 {time.monotonic()-self.started:.0f}s · {active}', self.stream)

    def close(self):
        for bar in reversed(list(self.bars.values())):
            bar.close()
        if self.overall:
            self.overall.close()
