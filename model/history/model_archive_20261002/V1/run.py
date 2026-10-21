"""V1 独立单元入口与路径约定；train/pipeline 完成四季度训练、推演和回测。"""
import argparse
import ast
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
sys.dont_write_bytecode = True

UNIT_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = UNIT_ROOT.parents[1]
UNIT = UNIT_ROOT.name


# 独立单元固定布局。开发规则：项目 docs/architecture.md。
SOURCE_FILES = ('run.py', 'model.py', 'analysis.py', 'train.sh')
LAYOUT_QUARTERS = ('2025Q3', '2025Q4', '2026Q1', '2026Q2')
LAYOUT_YEARS = (2025, 2026)
FOLD_FILES = ('best.pt', 'last.pt', 'complete.json', 'history.json',
              'score_predictions.npy', 'test_predictions.npy', 'split.json', 'training_info.json')


def fixed_files():
    """当前单元的固定171个常驻文件；不新增清单文件。"""
    files = set(SOURCE_FILES)
    files.update(f'model_train/{q}/fold{f}/{name}'
                 for q in LAYOUT_QUARTERS for f in range(1, 5) for name in FOLD_FILES)
    files.update(f'logs/{q}_{f}.log' for q in LAYOUT_QUARTERS for f in range(1, 5))
    files.update('logs/' + name for name in ('analysis_0.log', 'resource_summary.json', 'resource_usage.csv'))
    files.update('model_info/' + name for name in ('final_audit.json', 'finished.json', 'launch.json',
                                                   'pipeline.log', 'preflight.json', 'two_strategy_validation.json'))
    files.update(('model_pred/ensemble_curves.png', 'model_pred/picks.md', 'model_pred/ensemble/score_meta.json'))
    files.update(f'model_pred/ensemble/year={y}/data.parquet' for y in LAYOUT_YEARS)
    files.update(f'model_pred/ensemble/CSV/{kind}_{strategy}.csv'
                 for kind in ('cash', 'trades') for strategy in ('top5_5d', 'top1_1d'))
    files.update('model_pred/tables/' + name for name in ('README.md', 'main_metrics_1d.csv',
                                                       'quarterly_summary.csv', 'report.json', 'strategy_summary.csv'))
    return files


# 编辑器与解释器的自动产物：不是单元文件，不参与结构核对，也不代表结构扩张。
TOOL_DIRS = ('.ipynb_checkpoints', '__pycache__')


def unit_paths():
    """扫描单元内的文件与目录（已剔除工具自动产物）。"""
    files, dirs = set(), set()
    for p in UNIT_ROOT.rglob('*'):
        rel = p.relative_to(UNIT_ROOT)
        if any(part in TOOL_DIRS for part in rel.parts):
            continue
        (files if p.is_file() else dirs).add(rel.as_posix())
    return files, dirs


def validate_layout(*, allow_temporary=False, allow_missing=False):
    """只核对文件名和目录位置；开发与运行均须保持当前结构。

    新单元按规则只初始化4个脚本文件，167个常驻产物由运行原位生成，所以起点允许缺失
    （`allow_missing`）；但**任何多余路径永远拒绝**，缺失清单交由调用方比对起终点。
    """
    expected = fixed_files()
    actual, actual_dirs = unit_paths()
    if allow_temporary:
        temporary = set()
        for name in expected:
            path = Path(name)
            if path.suffix == '.json':
                temporary.add(path.with_suffix('.json.tmp').as_posix())
            elif path.suffix in ('.pt', '.npy'):
                temporary.add(path.with_suffix('.tmp').as_posix())
            elif path.suffix == '.parquet':
                temporary.add(path.with_name('data.tmp.parquet').as_posix())
        actual -= temporary
    allowed_dirs = {parent.as_posix() for name in expected for parent in Path(name).parents if str(parent) != '.'}
    extra = sorted((actual - expected) | (actual_dirs - allowed_dirs))
    missing = sorted(expected - actual)
    if extra or (missing and not allow_missing):
        raise ValueError(f'独立单元文件树必须保持固定；额外路径={extra}；缺失文件={missing}；规则见docs/architecture.md')
    return dict(files=len(actual), directories=len(actual_dirs), source_files=list(SOURCE_FILES),
                missing=missing)


def output_file(path):
    """常驻产物只能写在固定清单内的路径上；清单之外一律拒绝。

    不要求文件已存在——新单元第一次运行就是靠这里把产物建出来的。
    """
    path = Path(path).resolve()
    if not path.is_relative_to(UNIT_ROOT) or path.relative_to(UNIT_ROOT).as_posix() not in fixed_files():
        raise ValueError(f'禁止增加或移动独立单元文件: {path}')
    if path.name in SOURCE_FILES:
        raise ValueError(f'源码文件只能由开发者修改，不能当产物覆写: {path}')
    return path


def checked_name(value):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', value):
        raise ValueError(f'非法运行名称: {value}')
    return value


def run_name():
    name = os.environ.get('MODEL_RUN_NAME')
    if not name:
        for filename, key in [('launch.json', 'run_name'), ('final_audit.json', 'run')]:
            path = info_root() / filename
            if path.is_file():
                name = json.loads(path.read_text()).get(key)
                if name:
                    break
    name = checked_name(name or 'current')
    if name in ('legacy', 'before_restructure'):
        raise ValueError('请选择新的运行名称')
    return name


def run_root():
    """固定工作单元；运行名称只标记报告，不再创建额外目录层。"""
    return UNIT_ROOT


def info_root():
    return UNIT_ROOT / 'model_info'


def data_root():
    root = Path(os.environ.get('MX_DATA', PROJECT_ROOT / 'trainingdata' / 'fac_sample')).expanduser().resolve()
    if root.name in ('fac_sample', 'factors') and (root.parent / 'meta.json').is_file():
        return root.parent
    return root


def feature_block():
    explicit = Path(os.environ.get('MX_DATA', 'fac_sample')).name
    block = explicit if explicit in ('fac_sample', 'factors') else os.environ.get('MODEL_FEATURE_BLOCK', 'fac_sample')
    if block not in ('fac_sample', 'factors'):
        raise ValueError('特征块仅支持 fac_sample 或 factors')
    return block


def price_root():
    """回测价格的固定地址（模块① 原始数据，只读）。

    ★ **价格不是因子侧产物**，所以它不在 `trainingdata/` 里：训练数据层只负责对接因子侧
      （股票因子 / 市场因子 / 标签 / 流量四块），价格来自模块① 的原始日线。2026-09-25
      用户定：`prices` 块从训练数据层移出，改由本单元直接读原始数据。

    与 `stock_list_path()`/`calendar_path()` 同一套写法（`MX_*` 可覆盖、从 `PROJECT_ROOT`
    推导，共享盘换挂载点不用改代码）。区别是**这份数据参与数值计算**：可交易性判定
    （停牌/涨跌停/复权）与含费净值都靠它，不是显示用的。读到的值由 `Prices.verify_labels()`
    与 `trainingdata/target` 的标签逐格对拍兜底 —— 口径不一致会直接报错，不会静默跑完。
    """
    return Path(os.environ.get('MX_PRICES', PROJECT_ROOT.parent / 'datadownload' / 'data'))


def price_files(year):
    """某一年的两个价格数据集：`stock_daily` 出 7 列、`stock_adj_factor` 出复权因子。"""
    return [price_root() / name / f'year={year}' / 'data.parquet'
            for name in ('stock_daily', 'stock_adj_factor')]


def stock_list_path():
    """股票中文名对照表的固定地址（用户 2026-09-25 授权，只读）。

    与`calendar_path()`一样只用于 picks.md 的显示，不参与任何数值计算。
    从 PROJECT_ROOT 推出来而不是写死绝对路径，共享盘换挂载点时不用改代码。
    """
    return Path(os.environ.get(
        'MX_STOCK_LIST', PROJECT_ROOT.parent / 'datadownload' / 'data' / 'stock_list' / 'data.parquet'))


def calendar_path():
    """交易日历的固定地址（用户 2026-09-25 授权，只读）。

    用途只有一个：picks.md 里把「T+1 买入 / T+6 卖出」写成**具体日期**。快照内的日期
    取 `panel.days`（与标签口径完全一致），快照之后的日期只能来自这份日历——它含未来
    交易日（实测到 2026-10-25），所以最新一天的买卖日也写得出来。
    """
    return Path(os.environ.get(
        'MX_CALENDAR', PROJECT_ROOT.parent / 'datadownload' / 'data' / 'basic_calendar' / 'data.parquet'))


def feature_columns(meta):
    return list(meta['fac_sample']['columns'] if feature_block() == 'fac_sample' else meta['columns']['features'])


def market_columns(meta):
    """市场门控的输入列：61 个市场因子的 `_z252`（过去252个交易日含当日的 z-score）。

    只取 `_z252` 而不取原值：原值列量纲不可比（指数点位、比率、金额混在一起），
    z-score 已是尺度无关口径，与股票因子块同一套约定。预热期（2018 全年等）为 NaN，
    载入时按「无信息」填 0，门控退化为恒等。
    """
    block = meta['market_factors']
    columns = [c for c in block['columns'] if c.endswith(f"_z{block['z_window']}")]
    if len(columns) != block['n_factors']:
        raise ValueError(f'市场因子 z 窗口列数 {len(columns)} 与 n_factors {block["n_factors"]} 不一致')
    return columns


def analysis_input_root():
    requested = os.environ.get('MODEL_INPUT_RUN')
    if requested and requested != run_name():
        raise ValueError('本单元使用固定目录，不再支持另一个 --input-run；直接读取 model_train')
    return UNIT_ROOT


def inspect_inputs():
    root = data_root()
    meta = json.loads((root / 'meta.json').read_text())
    columns = feature_columns(meta)
    if not columns or len(columns) != len(set(columns)):
        raise ValueError('特征清单为空或有重复')
    market = market_columns(meta)
    if not market or len(market) != len(set(market)):
        raise ValueError('市场因子清单为空或有重复')
    # `trainingdata/` 的四块（特征块 + target/amount/market_factors）：训练数据层只对接因子侧。
    paths = [root / kind / f'year={y}' / 'data.parquet'
             for y in sorted(meta['built_years'])
             for kind in (feature_block(), 'target', 'amount', 'market_factors')]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    # 价格走模块① 原始数据（不是因子侧产物，因此不在 `trainingdata/` 里）。
    prices = [p for y in sorted(meta['built_years']) for p in price_files(y)]
    for path in prices:
        if not path.is_file():
            raise FileNotFoundError(path)
    return {'data_root': str(root), 'feature_block': feature_block(), 'features': columns,
            'market_features': market, 'price_root': str(price_root()),
            'files': [str(p.relative_to(root)) for p in paths],
            'price_files': [str(p) for p in prices]}


def dependency_audit():
    local = {'run', 'model', 'analysis'}
    imports = set()
    for name in sorted(local):
        tree = ast.parse((UNIT_ROOT / f'{name}.py').read_text(encoding='utf-8'))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(x.name.split('.')[0] for x in node.names)
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    raise ValueError('不允许隐式相对模块依赖')
                imports.add(node.module.split('.')[0])
    third_party = {'numpy', 'pandas', 'pyarrow', 'torch', 'scipy', 'matplotlib'}
    unknown = imports - local - set(sys.stdlib_module_names) - third_party
    if unknown:
        raise ValueError(f'非独立依赖: {sorted(unknown)}')
    return {'local_modules': sorted(local), 'third_party': sorted(imports & third_party), 'unknown': []}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', nargs='?', default='train',
                        choices=['pipeline', 'train', 'analyze', 'backtest', 'doctor', 'split', 'rescore', 'check'])
    parser.add_argument('--run-name', default=run_name(), help='报告标签；不改变固定产物目录，也不强制重训')
    parser.add_argument('--input-run', default=os.environ.get('MODEL_INPUT_RUN'),
                        help='兼容旧命令，仅允许与 --run-name 相同；直接读取本单元 model_train')
    parser.add_argument('--data', type=Path, help='trainingdata 根目录或其 fac_sample/factors 子目录；默认 fac_sample')
    parser.add_argument('--epochs', type=int, default=2, help='每个季度每折的最大训练轮数，本次框架测试默认 2')
    parser.add_argument('--jobs', type=int, choices=range(1, 9), default=2)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='cpu')
    parser.add_argument('--dry-run', action='store_true')
    args, extra = parser.parse_known_args()
    if extra[:1] == ['--']:
        extra = extra[1:]
    if args.epochs < 1 or args.threads < 1:
        parser.error('epochs 和 threads 必须为正整数')
    if args.input_run and args.command not in ('analyze', 'backtest'):
        parser.error('--input-run 仅用于 analyze/backtest')
    if args.input_run and args.input_run != args.run_name:
        parser.error('本单元已采用固定目录，不能按 --input-run 选择另一份产物')
    os.environ['MODEL_RUN_NAME'] = checked_name(args.run_name)
    if args.input_run:
        os.environ['MODEL_INPUT_RUN'] = checked_name(args.input_run)
    else:
        os.environ.pop('MODEL_INPUT_RUN', None)
    if args.data:
        os.environ['MX_DATA'] = str(args.data.expanduser().resolve())
    os.environ['MX_EPOCHS'] = str(args.epochs)
    os.environ['MX_THREADS'] = str(args.threads)
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    layout = validate_layout(allow_missing=True)
    dependencies = dependency_audit()
    inputs = inspect_inputs()
    if args.command == 'check':
        print(json.dumps(dict(unit=UNIT, output=str(run_root()),
                              run_name=run_name(), model_info=str(info_root()),
                              model_train=str(run_root()/'model_train'), model_pred=str(run_root()/'model_pred'),
                              logs=str(run_root()/'logs'),
                              data=inputs['data_root'], feature_block=inputs['feature_block'],
                              feature_count=len(inputs['features']),
                              market_count=len(inputs['market_features']), input_files=len(inputs['files']),
                              epochs=args.epochs, quarters=['2025Q3','2025Q4','2026Q1','2026Q2'],
                              folds=4, layout=layout, dependencies=dependencies), ensure_ascii=False, indent=2))
        return 0
    if args.command in ('pipeline', 'train'):
        command = [sys.executable, '-B', '-u', str(UNIT_ROOT/'analysis.py'), '--pipeline', '--jobs', str(args.jobs), '--device', args.device]
    elif args.command in ('analyze', 'backtest'):
        command = [sys.executable, '-B', '-u', str(UNIT_ROOT/'analysis.py')]
        command += {'backtest':['--from-scores'], 'analyze':['--audit']}[args.command]
    else:
        command = [sys.executable, '-B', '-u', str(UNIT_ROOT/'model.py'), '--device', args.device, '--'+args.command]
    command.extend(extra)
    print('单元:', UNIT, '\n特征:', data_root()/feature_block(), '\n产物:', run_root(),
          '\n运行信息:', info_root(), '\n命令:', shlex.join(command), flush=True)
    if args.dry_run:
        return 0
    try:
        return subprocess.call(command, cwd=UNIT_ROOT, env=os.environ.copy())
    finally:
        # 收尾核对两件事：①清单外的多余路径（含子进程被 kill 时留下的 .tmp，也算额外路径）
        # ②起点就有的文件没有被删掉。运行只允许「补齐起点就缺的产物」。
        final = validate_layout(allow_temporary=True, allow_missing=True)
        removed = sorted(set(final['missing']) - set(layout['missing']))
        if removed:
            raise ValueError(f'运行删除了常驻产物: {removed}；规则见docs/architecture.md')


if __name__ == '__main__':
    raise SystemExit(main())
