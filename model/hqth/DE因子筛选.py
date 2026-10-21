import pandas as pd
import numpy as np
from itertools import combinations
from functools import lru_cache

# === 1) 数据读取（与你相同） ===
data = pd.read_feather(rf"/project/model_share/share_1_new/factor_csy/fac20250613_fill_zero_sample/2023q1_num-1558.fea")
data = data.set_index(['date', 'Code'])
all_cols = [col for col in data.columns]

# === 2) 预分组，避免每次候选子集都 groupby 一次 ===
# groups 是每个 date 对应的子 DataFrame（索引仍含 Code）
groups = [g for _, g in data.groupby(level='date', sort=False)]

# === 3) 评分函数（带缓存）===
# 注意：为了可缓存，签名用 tuple；内部再转为 list
@lru_cache(maxsize=200_000)
def score_subset(cols_tuple: tuple) -> float:
    cols_list = list(cols_tuple)
    # 针对每个 date 计算 diversity_entropy，然后取平均
    vals = []
    for g in groups:
        # g 的列包含所有因子列，这里只挑 cols_list
        s = diversity_entropy(g[cols_list])
        vals.append(float(s))
    return float(np.mean(vals)) if vals else float('-inf')

# === 4) 小规模：精确搜索（当列数不大或给定较小的 k）===
def brute_force_best(all_columns, k: int):
    best_s, best_subset = float('-inf'), None
    for subset in combinations(all_columns, k):
        s = score_subset(tuple(subset))
        if s > best_s:
            best_s, best_subset = s, subset
    return list(best_subset), best_s

# === 5) 大规模：贪心前向选择（可设最大子集大小 k）===
def greedy_forward_best(all_columns, k: int, verbose=False):
    chosen = []
    remaining = list(all_columns)
    best_s = float('-inf')
    for step in range(k):
        cand = []
        for c in remaining:
            sub = tuple(sorted(chosen + [c]))
            s = score_subset(sub)
            cand.append((s, c))
        s_star, c_star = max(cand, key=lambda x: x[0])
        if verbose:
            print(f"step={step+1}, add={c_star}, score={s_star:.6f}")
        # 若没有提升则停止（可选）
        if s_star <= best_s:
            break
        best_s = s_star
        chosen.append(c_star)
        remaining.remove(c_star)
    return chosen, best_s

# === 6) 大规模：Beam Search（比纯贪心更稳健）===
def beam_search_best(all_columns, max_k: int, beam_width: int = 5, verbose=False):
    # beam 元素是 (score, subset_tuple)，subset_tuple 要保持排序以利缓存命中
    beams = [(float('-inf'), tuple())]  # 空集起步
    all_columns = list(all_columns)

    for depth in range(1, max_k + 1):
        candidates = []
        for s_prev, subset in beams:
            used = set(subset)
            for c in all_columns:
                if c in used:
                    continue
                new_subset = tuple(sorted(subset + (c,)))
                s = score_subset(new_subset)
                candidates.append((s, new_subset))
        # 选出 top-B
        candidates.sort(key=lambda x: x[0], reverse=True)
        beams = candidates[:beam_width]
        if verbose:
            best_now = beams[0]
            print(f"depth={depth}, best_score={best_now[0]:.6f}, subset_size={len(best_now[1])}")
    # 在最终 beam 中取最优
    best_s, best_subset = max(beams, key=lambda x: x[0])
    return list(best_subset), best_s

# === 7) 自动策略：小规模用精确，大规模用 Beam ===
def find_best_subset(all_columns, k=None, max_k=None, beam_width=8, verbose=False):
    """
    用法：
      - 固定大小：传 k（例如 k=5）
      - 尺寸不定但有上限：传 max_k（例如 max_k=10），返回 [1..max_k] 里分数最高的子集
    """
    n = len(all_columns)

    def try_fixed_k(K):
        # 列数很小时走穷举；否则先 Beam 再可选贪心微调
        if n <= 22 and K is not None:  # 粗略门槛，可调
            return brute_force_best(all_columns, K)
        else:
            # 先 Beam，再用贪心从 beam 的最优继续细化（可选）
            sub, s = beam_search_best(all_columns, max_k=K, beam_width=beam_width, verbose=verbose)
            return sub, s

    if k is not None:
        return try_fixed_k(k)

    if max_k is not None:
        best_s, best_sub = float('-inf'), None
        for K in range(1, max_k + 1):
            sub, s = try_fixed_k(K)
            if verbose:
                print(f"K={K}, score={s:.6f}")
            if s > best_s:
                best_s, best_sub = s, sub
        return best_sub, best_s

    # 若两者都未指定，默认给一个“尽量好”的答案：max_k=10
    return find_best_subset(all_columns, max_k=10, beam_width=beam_width, verbose=verbose)

# === 8) 示例：===
# 1) 固定大小的子集（比如取 5 个因子）：
best_subset_k5, best_score_k5 = find_best_subset(all_cols, k=5, beam_width=12, verbose=True)
print("Best subset (k=5):", best_subset_k5)
print("Best score:", best_score_k5)

# 2) 不固定大小，最多不超过 10 个因子，自动挑平均得分最高的规模：
best_subset_auto, best_score_auto = find_best_subset(all_cols, max_k=10, beam_width=12, verbose=True)
print("Best subset (<=10):", best_subset_auto)
print("Best score:", best_score_auto)

# 3) 复现你之前的示例（取前两个列）以对比：
sample_cols = all_cols[:2]
baseline = score_subset(tuple(sample_cols))
print("Baseline (first 2 cols):", baseline)
