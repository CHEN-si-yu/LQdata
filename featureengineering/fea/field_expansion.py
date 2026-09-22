"""扩字段的显式目录与源字段命名空间，不修改旧字段的归属。"""
from pathlib import Path
import json

CATALOG = json.loads((Path(__file__).resolve().parents[1]/'conf/field_expansion.json').read_text(encoding='utf-8'))


def annual_alias(dataset, field):
    return 'annual__' + dataset + '__' + field


ANNUAL_ALIASES = {}
for entry in CATALOG:
    for dataset, fields in entry['source_fields'].items():
        for field in fields:
            ANNUAL_ALIASES[annual_alias(dataset, field)] = (dataset, field)
ANNUAL_SOURCES = {}
for alias, (dataset, _) in ANNUAL_ALIASES.items():
    ANNUAL_SOURCES.setdefault(dataset, []).append(alias)
