"""从 V76 派生种子复核单元 V78（seed 3254）与 V79（seed 3255）。

单种子 = 抽奖（R18），2 粒不足以判一个改动（R36）⇒ 胜出臂必须补到三粒、报最差情形。
配方/特征/快照**逐字复制 V76**，只改 `RECIPE['seed']` 与 `name`。
"""
import re
import shutil
import sys
from pathlib import Path

ROOT = Path('/autodl-fs/data/model')
SRC = 'V76'
UNITS = [('V78', 3254, True), ('V79', 3255, True)]

for name, seed, _ in UNITS:
    d = ROOT / name
    if d.exists():
        print(f'✘ {d} 已存在，拒绝覆盖', file=sys.stderr)
        raise SystemExit(1)

for name, seed, _ in UNITS:
    d = ROOT / name
    d.mkdir(parents=True)
    for f in ['model.py', 'analysis.py', 'evaluation_core.py', 'memory_guard.py', 'features_armA_oldcore.json']:
        shutil.copy2(ROOT / SRC / f, d / f)
    shutil.copy2(ROOT / SRC / 'train.sh', d / 'train.sh')

    src = (d / 'model.py').read_text(encoding='utf-8')
    reps = [
        ('"""V76：新快照(trainingdata, 599 列)上的「旧核」臂。',
         f'"""{name}：V76 的**种子复核**单元（seed {seed}），其余与 V76 逐字相同。'),
        ("    name='V76',", f"    name='{name}',"),
        ('    seed=3253,', f'    seed={seed},'),
        (f"VARIANT = '新快照(trainingdata, {599} 列)上的「旧核」臂：",
         f"VARIANT = 'V76（新快照「旧核」331 列）的种子复核单元，seed={seed}；其余配方逐字相同。原 VARIANT："),
    ]
    for old, new in reps:
        if old not in src:
            # VARIANT 那条是模糊匹配，找不到就跳过（不致命）
            if 'VARIANT' in old:
                continue
            print(f'✘ {name}: 找不到 {old[:50]!r}', file=sys.stderr)
            raise SystemExit(1)
        src = src.replace(old, new, 1)
    (d / 'model.py').write_text(src, encoding='utf-8')
    got = re.search(r"seed=(\d+),", src).group(1)
    assert got == str(seed), f'{name} seed 未生效：{got}'
    print(f'✔ {name} 建好（seed={got}）')
