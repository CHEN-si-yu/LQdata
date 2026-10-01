"""小型README目录摘要与机器可读因子字典，避免重复数十万字文档。"""
from pathlib import Path
import csv,json
from .manifest import Manifest
from .spec import all_specs
from .documentation import update_section

def generate(cfg):
    # 因子字典是本模块的产物：放项目内 artifacts/catalog/（原来在 ../CodeX/featureengineering_catalog）。
    out=Path(cfg.root)/"artifacts"/"catalog"
    out.mkdir(parents=True,exist_ok=True)
    records=[]
    for s in all_specs():
        m=Manifest.load(cfg.state_dir,s.name); parts=list(m.partitions.values())
        records.append(dict(name=s.name,kind="market" if s.is_market else "label" if s.is_label else "stock",
            group=s.group,description=s.desc,formula=s.formula,dependencies=list(s.deps),
            start=s.resolved_start(cfg),warmup_calendar_days=s.warmup_days,version=s.version,
            higher_is_better=s.higher_is_better,forward_days=s.forward_days,note=s.note,
            first_date=min((p.get("min_date","") for p in parts),default=""),
            last_date=max((p.get("max_date","") for p in parts),default=""),
            rows=sum(p.get("rows",0) for p in parts),nonnull=sum(p.get("nonnull",0) for p in parts),
            output=str(cfg.factor_root(s)/s.name),source_module=s.fn.__module__))
    (out/"factors.json").write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding="utf-8")
    with (out/"factors.csv").open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader()
        w.writerows([{**r,"dependencies":",".join(r["dependencies"])} for r in records])
    counts={k:sum(r["kind"]==k for r in records) for k in ("stock","market","label")}
    rel=out.relative_to(Path(cfg.root)) if str(out).startswith(str(cfg.root)) else out
    body=(f"注册对象：{counts['stock']} 个股票因子、{counts['market']} 个市场因子、{counts['label']} 个未来收益标签。\n\n"
          f"逐因子的公式、依赖、方向、预热、版本和覆盖率见 [{out.name}/factors.csv]({rel}/factors.csv)，"
          f"同时提供 [JSON]({rel}/factors.json)。运行 `python main.py docs` 刷新。\n")
    update_section(cfg.root,"factor-catalog",body)
    print("因子字典已更新：",out)
    return 0
