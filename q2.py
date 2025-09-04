# q2.py
# 使用鲸鱼优化算法（WOA）最大化烟幕对真目标的有效遮蔽时长


import math
import random
import numpy as np
from tqdm import tqdm

# ---- 为了静默 calculate_covered_time 中的 tqdm 进度条（仅限本文件运行时） ----
try:
    import utils.calculate_covered_time as cct_module  # 模块对象
    def _silent_tqdm(iterable, **kwargs):
        # 返回原迭代器，不打印进度条；不改变数值逻辑
        return iterable
    cct_module.tqdm = _silent_tqdm  # 动态替换模块内的 tqdm 引用
except Exception:
    # 如果替换失败，也不影响算法的正确性（只是更啰嗦）
    pass

from utils.calculate_covered_time import calculate_covered_time


# -----------------------------
#   固定场景（题目 A 问题 2）
# -----------------------------
ORIGIN_FAKE = np.array([0.0, 0.0, 0.0], dtype=float)  # 假目标在原点（与题目一致）

# 导弹 M1 与无人机 FY1 的初始状态（题面给定）
POS_INIT_MISSILE = np.array([20000.0, 0.0, 2000.0], dtype=float)  # M1
POS_INIT_DRONE   = np.array([17800.0, 0.0, 1800.0], dtype=float)  # FY1


# -----------------------------
#   决策变量与约束边界
# -----------------------------
# 变量顺序: [theta_drone, v_drone, t_drone_fly, t_decoy_delay]
LOWER_BOUNDS = np.array([-math.pi, 70.0, 0.0, 0.0], dtype=float)
UPPER_BOUNDS = np.array([ math.pi, 140.0, 25.0, 10.0], dtype=float)

# 你可以按需要调整以下超参数（尽量保持较小规模，避免仿真时间过长）
POPULATION_SIZE = 16          # 种群规模（鲸鱼数量）
MAX_ITERATIONS  = 200         # 最大迭代次数
SEED            = 42          # 随机种子，便于复现
B_SPIRAL        = 1.0         # 螺旋系数 b，WOA 通常取 1


def clip_to_bounds(x: np.ndarray,
                   lower: np.ndarray = LOWER_BOUNDS,
                   upper: np.ndarray = UPPER_BOUNDS) -> np.ndarray:
    """边界修复：将解向量裁剪回可行区间（逐维裁剪）"""
    return np.minimum(np.maximum(x, lower), upper)


def evaluate_solution(theta_drone: float,
                      v_drone: float,
                      t_drone_fly: float,
                      t_decoy_delay: float) -> float:
    """
    调用一问的仿真核心，返回“负的遮蔽时长”（用于最小化求解）。
    """
    covered_time = calculate_covered_time(
        POS_INIT_MISSILE,
        POS_INIT_DRONE,
        float(v_drone),
        float(theta_drone),
        float(t_drone_fly),
        float(t_decoy_delay),
    )
    # 我们要“最大化” covered_time，故返回其相反数供 WOA 最小化
    return -covered_time


def initialize_population(pop_size: int,
                          lower: np.ndarray = LOWER_BOUNDS,
                          upper: np.ndarray = UPPER_BOUNDS) -> np.ndarray:
    """均匀随机初始化种群（形状: [pop_size, dim]）"""
    dim = lower.shape[0]
    rand01 = np.random.rand(pop_size, dim)
    return lower + rand01 * (upper - lower)


def whale_optimization(population_size: int = POPULATION_SIZE,
                       max_iterations: int = MAX_ITERATIONS,
                       b_spiral: float = B_SPIRAL,
                       seed: int = SEED):
    """
    经典鲸鱼优化算法（WOA）实现：最小化 evaluate_solution 返回值（即最大化遮蔽时长）。
    返回：最优解向量 best_x 及其对应的“负遮蔽时长” best_f。
    """
    np.random.seed(seed)
    random.seed(seed)

    # 初始化鲸鱼群
    X = initialize_population(population_size, LOWER_BOUNDS, UPPER_BOUNDS)

    # 初始评估
    fitness = np.zeros(population_size, dtype=float)
    for i in range(population_size):
        theta, v, t_drop, t_delay = X[i]
        fitness[i] = evaluate_solution(theta, v, t_drop, t_delay)

    best_index = int(np.argmin(fitness))
    best_x = X[best_index].copy()
    best_f = float(fitness[best_index])

    # 线性递减的 a（WOA 标准做法）
    for iteration in tqdm(range(max_iterations), desc="WOA优化进度"):  # 加进度条
        a = 2.0 - 2.0 * (iteration / max_iterations)  # 从 2 线性减到 0

        for i in range(population_size):
            r1 = np.random.rand(X.shape[1])  # 各维独立
            r2 = np.random.rand(X.shape[1])
            A_vec = 2.0 * a * r1 - a
            C_vec = 2.0 * r2

            p = np.random.rand()
            current = X[i].copy()

            if p < 0.5:
                if np.all(np.abs(A_vec) < 1.0):
                    # 包围猎物（向当前全局最优收缩）
                    D_vec = np.abs(C_vec * best_x - current)
                    new_x = best_x - A_vec * D_vec
                else:
                    # 搜索阶段（对随机个体进行包围）
                    rand_index = np.random.randint(population_size)
                    X_rand = X[rand_index]
                    D_vec = np.abs(C_vec * X_rand - current)
                    new_x = X_rand - A_vec * D_vec
            else:
                # 螺旋式气泡网攻击
                D_prime = np.abs(best_x - current)
                l = np.random.uniform(-1.0, 1.0)  # [-1, 1]
                new_x = D_prime * np.exp(b_spiral * l) * np.cos(2.0 * math.pi * l) + best_x

            # 边界修复（角度、速度、时间各维逐一裁剪）
            new_x = clip_to_bounds(new_x, LOWER_BOUNDS, UPPER_BOUNDS)

            # 评估新解
            theta, v, t_drop, t_delay = new_x
            new_f = evaluate_solution(theta, v, t_drop, t_delay)

            # 个体更新（贪婪选择）
            if new_f < fitness[i]:
                X[i] = new_x
                fitness[i] = new_f

                # 刷新全局最优
                if new_f < best_f:
                    best_f = float(new_f)
                    best_x = new_x.copy()

        # 你也可以在此打印轻量日志（避免频繁评估时的输出抖动）
        print(f"[Iter {iteration+1:02d}/{max_iterations}] "
              f"best_covered_time = {-best_f:.4f} s | "
              f"theta={best_x[0]:.4f} rad, v={best_x[1]:.2f} m/s, "
              f"t_drop={best_x[2]:.2f} s, t_delay={best_x[3]:.2f} s")

    return best_x, best_f


def main():
    print("启动鲸鱼优化（WOA）以最大化遮蔽时长 ...")
    best_x, best_f_neg = whale_optimization(
        population_size=POPULATION_SIZE,
        max_iterations=MAX_ITERATIONS,
        b_spiral=B_SPIRAL,
        seed=SEED,
    )
    theta_drone, v_drone, t_drone_fly, t_decoy_delay = best_x.tolist()
    best_covered = -best_f_neg

    print("\n=== 最优结果（问题 2，一枚干扰弹）===")
    print(f"航向 theta_drone      = {theta_drone:.6f} rad "
          f"({theta_drone*180.0/math.pi:.3f} deg)")
    print(f"速度 v_drone          = {v_drone:.3f} m/s")
    print(f"投放时刻 t_drone_fly  = {t_drone_fly:.3f} s")
    print(f"起爆延时 t_decoy_delay= {t_decoy_delay:.3f} s")
    print(f"最大化后的总有效遮蔽时长 = {best_covered:.3f} s")

    # 如需在最优解处再做一次验证或可视化，可在此二次调用 calculate_covered_time（略）


if __name__ == "__main__":
    main()
