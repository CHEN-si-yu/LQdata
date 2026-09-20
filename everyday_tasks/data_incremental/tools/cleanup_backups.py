"""Replace retired data backups with checksums. Never visits production data.

Called through main.py cleanup-backups, under the shared maintenance lock.
Reports, code backups, manifests and daily_dump's production cache are retained.
Checksums are comparison evidence, not a recoverable data backup.
"""
import hashlib
import json
import os
import stat
import time
from datetime import datetime
from pathlib import Path

from .. import paths

BLOCK_BYTES = 8 * 1024 * 1024


def eligible_backup(p):
    name = p.name
    return (name.endswith(('.parquet', '.parquet.bak', '.parquet.tmp'))
            or (name.endswith('.tar.gz') and 'data' in name)
            or (name.endswith('.json.gz') and p.parent.name == 'vendor'))


def _regular_files(root):
    if root.is_symlink() or not root.exists():
        return
    for parent, dirs, files in os.walk(root, followlinks=False):
        dirs[:] = [d for d in dirs if not (Path(parent) / d).is_symlink()]
        for name in files:
            p = Path(parent) / name
            if not p.is_symlink() and p.is_file():
                yield p


def candidates():
    root = paths.BACKTEST_DIR.resolve()
    if root.is_relative_to(paths.DATA_ROOT.resolve()) or paths.DATA_ROOT.resolve().is_relative_to(root):
        raise ValueError('Backup root overlaps production data')
    result = [(p, root) for p in _regular_files(paths.BACKTEST_DIR) if eligible_backup(p)]
    # A completed full download no longer needs its duplicate Parquet chunks.
    # Keep report.json and chunk metadata; an interrupted download retains all chunks.
    full = paths.STATE_ROOT / 'cyq_perf_full'
    report = full / 'report.json'
    if report.exists() and not full.is_symlink():
        r = json.loads(report.read_text(encoding='utf-8'))
        if r.get('complete') is True and r.get('years'):
            import pyarrow.parquet as pq
            for item in r['years']:
                f = paths.year_partition_path('stock_cyq_perf', int(item['year']))
                if not f.exists() or pq.ParquetFile(f).metadata.num_rows < item['stored_rows']:
                    raise RuntimeError('Completed download cannot be confirmed in production; retaining chunks')
            result.extend((p, full.resolve()) for p in _regular_files(full) if p.suffix == '.parquet')
    return sorted(result, key=lambda pair: str(pair[0]))


def signature(st):
    return [st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns]


def checksum(p, root):
    resolved = p.resolve(strict=True)
    if not resolved.is_relative_to(root) or p.is_symlink():
        raise ValueError(f'Unsafe backup path: {p}')
    fd = os.open(p, os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0))
    md5 = hashlib.md5(usedforsecurity=False)
    sha = hashlib.sha256()
    with os.fdopen(fd, 'rb') as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f'Not a regular file: {p}')
        while block := stream.read(BLOCK_BYTES):
            md5.update(block)
            sha.update(block)
        if signature(before) != signature(os.fstat(stream.fileno())):
            raise RuntimeError(f'Backup changed while hashing: {p}')
    if signature(before) != signature(p.lstat()):
        raise RuntimeError(f'Backup replaced while hashing: {p}')
    return {'path': str(p), 'allowed_root': str(root), 'bytes': before.st_size,
            'md5': md5.hexdigest(), 'sha256': sha.hexdigest(), 'signature': signature(before)}


def durable_json(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix('.tmp')
    with tmp.open('w', encoding='utf-8') as stream:
        json.dump(obj, stream, ensure_ascii=False, indent=1)
        stream.flush()
        os.fsync(stream.fileno())
    paths.chmod_shared(tmp)
    os.replace(tmp, p)


def run(*, apply=False, log=print):
    selected = candidates()
    size = sum(p.stat().st_size for p, _ in selected)
    summary = {'files': len(selected), 'bytes': size, 'applied': False,
               'note': 'Checksums cannot restore deleted backup contents.'}
    log(f'待清理 {len(selected)} 个备份数据文件，{size / 1024**3:.2f} GiB；正式数据/缓存/报告保留。')
    if not apply or not selected:
        return summary
    started = time.monotonic()
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S') + f'_{os.getpid()}'
    archive = paths.BACKTEST_DIR / 'checksum_archives' / stamp
    records = []
    for i, (p, root) in enumerate(selected, 1):
        records.append(checksum(p, root))
        if i % 100 == 0:
            log(f'已计算校验值 {i}/{len(selected)}')
    manifest = archive / 'checksums.json'
    durable_json(manifest, {'created_at': stamp, **summary, 'files': records})
    # Persist the entire checksum manifest before deleting the first byte.
    # On an interrupted cleanup the original hashes still cover every deleted file.
    deleted = 0
    journal = archive / 'deleted.jsonl'
    with journal.open('x', encoding='utf-8') as stream:
        paths.chmod_shared(journal)
        for item in records:
            p = Path(item['path'])
            root = Path(item['allowed_root'])
            if p.is_symlink() or not p.resolve(strict=True).is_relative_to(root):
                raise RuntimeError(f'Path changed before deletion: {p}')
            if signature(p.lstat()) != item['signature']:
                raise RuntimeError(f'File changed before deletion: {p}')
            p.unlink()  # Only the exact hashed file; no recursive directory removal.
            stream.write(json.dumps({'path': str(p), 'bytes': item['bytes']}, ensure_ascii=False) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
            deleted += item['bytes']
    summary.update(applied=True, deleted_bytes=deleted, manifest=str(manifest),
                   seconds=round(time.monotonic() - started, 1))
    durable_json(archive / 'summary.json', summary)
    log(f'已清理 {deleted / 1024**3:.2f} GiB；校验清单：{manifest}')
    return summary
