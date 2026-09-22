"""建 V76/V77：新快照(trainingdata, 599 列)上的两条特征臂。

- V76 = 臂 A「旧核」：新快照 ∩ 旧 337 列（旧 337 里有 6 列上游已删，实得 331 列）
- V77 = 臂 B「全量」：新快照全部 599 列

配方逐字沿用 V62（= V32）—— 只换特征清单与快照锁定，其余超参不动。
策略层改动在 analysis.py（止盈 + 5d 口径），由 patch_analysis 施加。

★ 本脚本是**生成器**：只写 V76/V77 两个目录，不碰任何既有版本。
"""
import json
import shutil
from pathlib import Path

ROOT = Path('/autodl-fs/data/model')
SRC = ROOT / 'V62'
COPY = ['model.py', 'analysis.py', 'evaluation_core.py', 'memory_guard.py']
SNAPSHOT = 'trainingdata'

meta = json.loads((ROOT / SNAPSHOT / 'meta.json').read_text())
NEW = list(meta['columns']['features'])
OLD337 = list(json.loads((SRC / 'features_old337.json').read_text()))
OLD656 = list(json.loads((ROOT / 'trainingdata_prev656_20260922' / 'meta.json').read_text())['columns']['features'])

set_new = set(NEW)
arm_a = [c for c in NEW if c in set(OLD337)]                      # 新快照顺序
missing = [c for c in OLD337 if c not in set_new]
arm_b = list(NEW)

print(f'新快照 {len(NEW)} 列 | 旧337 {len(OLD337)} 列 | 旧656 {len(OLD656)} 列')
print(f'臂A 旧核 {len(arm_a)} 列 | 旧337 中已消失 {len(missing)} 列: {missing}')
print(f'臂B 全量 {len(arm_b)} 列')

UNITS = [
    dict(name='V76', arm='A', feats=arm_a, file='features_armA_oldcore.json',
         variant=f'新快照(trainingdata, {len(NEW)}列)上的「旧核」臂：{len(arm_a)} 列 = 旧337 ∩ 新快照（上游删了 {len(missing)} 列）。其余配方与 V62/V32 逐字相同。'),
    dict(name='V77', arm='B', feats=arm_b, file='features_armB_all599.json',
         variant=f'新快照(trainingdata)上的「全量」臂：全部 {len(NEW)} 列。其余配方与 V62/V32 逐字相同，只换输入列 ⇒ 与 V76 构成配对对照。'),
]

for u in UNITS:
    d = ROOT / u['name']
    if d.exists():
        raise SystemExit(f'✘ {d} 已存在，拒绝覆盖（旧版本是当时的结论）')
    d.mkdir(parents=True)
    for f in COPY:
        shutil.copy2(SRC / f, d / f)
    shutil.copy2(ROOT / 'V16' / 'train.sh', d / 'train.sh')
    (d / u['file']).write_text(json.dumps(u['feats'], ensure_ascii=False, indent=1), encoding='utf-8')

    src = (d / 'model.py').read_text(encoding='utf-8')
    reps = [
        ('"""V62：新因子快照(trainingdata, 653 列)上的**旧 337 列对照臂**。',
         f'"""{u["name"]}：新快照(trainingdata, {len(NEW)} 列)上的「{"旧核" if u["arm"]=="A" else "全量"}」臂。'),
        ('配方与 V32 逐字相同，唯一差别是特征清单被锁定成 V32 那份 337 列（features_old337.json）。\n用途：① 验证新快照对旧列是逐位保真的（应逐位复现 V32）；② 作为 V63（全 653 列）的配对对照。',
         f'配方与 V62/V32 逐字相同，唯一差别是特征清单（{u["file"]}，{len(u["feats"])} 列）。'),
        ("VARIANT = '新因子快照对照臂：新快照(trainingdata) + 旧337列清单；其余配方与V32逐字相同。'",
         f"VARIANT = {u['variant']!r}"),
        ("FEATURES_OLD = json.loads((Path(__file__).resolve().parent / 'features_old337.json').read_text())",
         f"FEATURES_OLD = json.loads((Path(__file__).resolve().parent / {u['file']!r}).read_text())"),
        ("    name='V62',", f"    name={u['name']!r},"),
    ]
    for old, new in reps:
        if old not in src:
            raise SystemExit(f'✘ {u["name"]}: 找不到待替换片段 -> {old[:60]!r}')
        src = src.replace(old, new, 1)
    (d / 'model.py').write_text(src, encoding='utf-8')
    print(f'✔ {u["name"]} 建好（{len(u["feats"])} 列）')
