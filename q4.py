# q4.py

import numpy as np
import math
import time
from tqdm import tqdm

try:
    import utils.calculate_covered_time as cct_module
    def _silent_tqdm(iterable, **kwargs):
        return iterable
    cct_module.tqdm = _silent_tqdm
except Exception:
    pass

from utils.is_covered import is_covered

# 复用 Q1/Q2 中的物理模型函数与常量
from utils.calculate_covered_time import (
    _missile_pos,              # 导弹轨迹
    _drone_vec,                # 无人机速度向量
    _decoy_state_at_explosion, # 起爆位置
    INTERVAL,
    SINK_SPEED,
    CLOUD_EFFECTIVE,
    SPEED_MISSILE,
    ORIGIN_FAKE
)

# ===============================
# 根据题意定义的初始条件（问题 4）
# ===============================

# 来袭导弹 M1 的初始位置
POS_INIT_MISSILE_M1 = np.array([20000.0,   0.0, 2000.0], dtype=float)

# 三架无人机 FY1、FY2、FY3 的初始位置
POS_INIT_DRONE_FY1  = np.array([17800.0,    0.0, 1800.0], dtype=float)
POS_INIT_DRONE_FY2  = np.array([12000.0, 1400.0, 1400.0], dtype=float)
POS_INIT_DRONE_FY3  = np.array([ 6000.0,-3000.0,  700.0], dtype=float)

# ===============================
# 优化算法（WOA）参数（与 Q2 风格一致）
# ===============================
POPULATION_SIZE = 600
MAX_ITERATIONS  = 20
EARLY_STOP_PATIENCE = 4

# 对于每架无人机需要优化的 4 个变量：
# [v_drone, theta_drone, t_drone_fly, t_decoy_delay]
# 速度：70~140 m/s；航向：0~pi；投放时间：0~60 s；起爆延迟：0~20 s
LB_1 = np.array([70.0, 0.0, 0.0, 0.0], dtype=float)
UB_1 = np.array([140.0, math.pi, 60.0, 20.0], dtype=float)

# 三架无人机，总维度 12
LOWER_BOUNDS = np.hstack([LB_1, LB_1, LB_1])
UPPER_BOUNDS = np.hstack([UB_1, UB_1, UB_1])
DIMENSIONS   = LOWER_BOUNDS.size


# ===============================
# 多无人机：联合遮蔽时间计算（新的函数）
# ===============================
def calculate_covered_time_3drones(pos_init_missile: np.ndarray,
                                  drone_triplets: list,
                                  show_progress: bool = False):
    """
    计算三枚云团对导弹的联合有效遮蔽时长（秒），并给出每枚云团的“单独遮蔽时长”。

    参数:
    pos_init_missile: np.array([x,y,z])，导弹初始位置（如 M1）
    drone_triplets:   list，长度为 3。
                      每个元素为 (pos_init_drone, v_drone, theta_drone, t_drone_fly, t_decoy_delay)
    show_progress:    bool，是否显示仿真进度条

    返回:
      total_union_time: float，三枚云团“联合”（并集）有效遮蔽总时长
      per_decoy_time:   list[float]，长度 3，每枚云团“单独”遮蔽时长（不去重）
      details:          dict，包含每枚云团的投放/起爆时间与坐标等信息
    """
    pos_init_missile = np.asarray(pos_init_missile, dtype=float)

    # 关键时间点与初始几何
    t_drop  = np.zeros(3, dtype=float)
    t_expl  = np.zeros(3, dtype=float)
    t_end   = np.zeros(3, dtype=float)
    v_d_vec = [None]*3
    B_expl  = [None]*3
    p_drop  = [None]*3

    for i, (pos_init_drone, v_drone, theta_drone, t_drone_fly, t_decoy_delay) in enumerate(drone_triplets):
        pos_init_drone = np.asarray(pos_init_drone, dtype=float)
        v_d = _drone_vec(float(v_drone), float(theta_drone))
        v_d_vec[i] = v_d

        # 关键时间
        t_drop[i] = float(t_drone_fly)
        t_expl[i] = float(t_drone_fly) + float(t_decoy_delay)
        t_end[i]  = t_expl[i] + float(CLOUD_EFFECTIVE)

        # 投放与起爆坐标
        p_drop[i] = pos_init_drone + v_d * t_drop[i]
        B_expl[i] = _decoy_state_at_explosion(pos_init_drone, v_d, t_drop[i], float(t_decoy_delay))

    # 导弹飞抵假目标（原点）的时间，上限不必超过此刻
    t_hit_origin = np.linalg.norm(ORIGIN_FAKE - pos_init_missile) / float(SPEED_MISSILE)

    # 全局仿真终止时间：三枚云团有效期末尾 与 导弹命中假目标时间 取较小者
    t_sim_end = float(min(max(t_end.max(), 0.0), t_hit_origin))

    # 时间轴
    times = np.arange(0.0, t_sim_end + 1e-9, float(INTERVAL), dtype=float)
    iterator = tqdm(times, desc="Simulating (3 drones)", unit="step") if show_progress else times

    total_union = 0.0
    per_decoy   = np.zeros(3, dtype=float)

    # 逐时刻仿真（离散并集计时）
    for t in iterator:
        A_t = _missile_pos(pos_init_missile, t)

        covered_any = False
        covered_flags = [False, False, False]

        for i in range(3):
            if t >= t_expl[i] and t <= t_end[i]:
                dt = t - t_expl[i]
                # 起爆后云团仅做竖直匀速下沉，水平坐标不再变化
                B_t = B_expl[i] + np.array([0.0, 0.0, -SINK_SPEED * dt], dtype=float)

                # 调用题中已实现的“完全遮挡”判定
                if is_covered(A_t, B_t):
                    covered_flags[i] = True
                    per_decoy[i] += float(INTERVAL)
                    covered_any = True

        if covered_any:
            total_union += float(INTERVAL)

    # 细节汇总（用于打印和写表）
    details = {
        "t_drop": t_drop.tolist(),
        "t_expl": t_expl.tolist(),
        "t_end":  t_end.tolist(),
        "p_drop": [p_drop[i].tolist() for i in range(3)],
        "p_expl": [B_expl[i].tolist() for i in range(3)],  # 起爆瞬间中心
    }

    return float(total_union), per_decoy.tolist(), details


# ===============================
# WOA 实现（与 Q2 基本一致）
# ===============================
class WhaleOptimizationAlgorithm:
    """
    鲸鱼优化算法 (WOA)
    """
    def __init__(self, obj_func, lower_bounds, upper_bounds, dim, pop_size, max_iter, early_stop_patience=None):
        self.obj_func = obj_func
        self.lb = np.asarray(lower_bounds, dtype=float)
        self.ub = np.asarray(upper_bounds, dtype=float)
        self.dim = int(dim)
        self.pop_size = int(pop_size)
        self.max_iter = int(max_iter)
        self.early_stop_patience = early_stop_patience

        self.leader_pos = np.zeros(self.dim, dtype=float)
        self.leader_score = float('inf')
        
        self.no_improve_counter = 0

        self.positions = np.random.rand(self.pop_size, self.dim) * (self.ub - self.lb) + self.lb

    def optimize(self):
        print("启动鲸鱼优化算法 (WOA)...")
        for t in tqdm(range(self.max_iter), desc="WOA 优化进度"):
            # 记录上一次的最优分数用于比较
            prev_best_score = self.leader_score
            
            # 评价并更新领导者
            for i in range(self.pop_size):
                self.positions[i, :] = np.clip(self.positions[i, :], self.lb, self.ub)
                fitness = self.obj_func(self.positions[i, :])
                if fitness < self.leader_score:
                    self.leader_score = fitness
                    self.leader_pos = self.positions[i, :].copy()
            
            # 早停检查
            if self.early_stop_patience is not None:
                # 如果当前迭代没有提升性能（分数没有变得更小）
                if self.leader_score >= prev_best_score:
                    self.no_improve_counter += 1
                    if self.no_improve_counter >= self.early_stop_patience:
                        print(f"\n早停触发: 连续 {self.early_stop_patience} 次迭代没有性能提升")
                        break
                else:
                    # 有提升则重置计数器
                    self.no_improve_counter = 0

            # a 从 2 线性递减到 0
            a = 2 - t * (2 / self.max_iter)

            # ... 其余代码保持不变 ...

            if (t + 1) % 3 == 0:
                print(f"迭代 {t + 1}/{self.max_iter}，当前最优联合遮蔽时间: {-self.leader_score:.3f} s")
                if self.early_stop_patience is not None:
                    print(f"连续未改进计数: {self.no_improve_counter}/{self.early_stop_patience}")

        return self.leader_pos, -self.leader_score


# ===============================
# 目标函数（三无人机 × 1 弹/机）
# ===============================
def _unpack_params_12(params):
    """
    将长度 12 的参数拆成 3 组 (v, theta, t_fly, t_delay)
    """
    p = np.asarray(params, dtype=float)
    assert p.size == 12
    return p.reshape(3, 4)  # 行：无人机；列：4个参数


def objective_function(params):
    """
    WOA 的适应度函数（最小化）。我们要最大化联合遮蔽时长 -> 返回其相反数。
    """
    triples = _unpack_params_12(params)
    drone_triplets = [
        (POS_INIT_DRONE_FY1, triples[0, 0], triples[0, 1], triples[0, 2], triples[0, 3]),
        (POS_INIT_DRONE_FY2, triples[1, 0], triples[1, 1], triples[1, 2], triples[1, 3]),
        (POS_INIT_DRONE_FY3, triples[2, 0], triples[2, 1], triples[2, 2], triples[2, 3]),
    ]
    total_union, _, _ = calculate_covered_time_3drones(
        pos_init_missile=POS_INIT_MISSILE_M1,
        drone_triplets=drone_triplets,
        show_progress=False  # 优化内循环关闭进度条
    )
    return -float(total_union)


# ===============================
# 主程序
# ===============================
def main():
    # 将优化时的目标函数静默版本（不开启仿真进度条）
    def objective_quiet(params):
        return objective_function(params)

    woa = WhaleOptimizationAlgorithm(
        obj_func=objective_quiet,
        lower_bounds=LOWER_BOUNDS,
        upper_bounds=UPPER_BOUNDS,
        dim=DIMENSIONS,
        pop_size=POPULATION_SIZE,
        max_iter=MAX_ITERATIONS,
        early_stop_patience=EARLY_STOP_PATIENCE
    )

    start = time.time()
    best_params, best_union_time = woa.optimize()
    end = time.time()

    print("\n" + "-"*60)
    print("问题4 优化完成（3 架无人机 × 1 弹/机）")
    print("优化算法：鲸鱼优化算法 (WOA)")
    print(f"总计用时: {end - start:.2f} 秒")
    print("-"*60)

    # 结构化参数
    triples = _unpack_params_12(best_params)

    # 使用最优解开启一次“有进度条”的最终仿真，便于核验与展示
    drone_triplets = [
        (POS_INIT_DRONE_FY1, triples[0, 0], triples[0, 1], triples[0, 2], triples[0, 3]),
        (POS_INIT_DRONE_FY2, triples[1, 0], triples[1, 1], triples[1, 2], triples[1, 3]),
        (POS_INIT_DRONE_FY3, triples[2, 0], triples[2, 1], triples[2, 2], triples[2, 3]),
    ]
    total_union, per_decoy, details = calculate_covered_time_3drones(
        pos_init_missile=POS_INIT_MISSILE_M1,
        drone_triplets=drone_triplets,
        show_progress=True
    )

    # === 打印最优参数 ===
    labels = ["FY1", "FY2", "FY3"]
    for i, lab in enumerate(labels):
        v_opt     = float(triples[i, 0])
        theta_opt = float(triples[i, 1])
        t_drop    = float(details["t_drop"][i])
        t_delay   = float(details["t_expl"][i] - details["t_drop"][i])
        p_drop    = np.array(details["p_drop"][i], dtype=float)
        p_expl    = np.array(details["p_expl"][i], dtype=float)

        print(f"\n[{lab}] 最优参数：")
        print(f"  无人机速度 v_drone: {v_opt:.4f} m/s")
        print(f"  无人机航向 theta_drone: {theta_opt:.4f} rad ({math.degrees(theta_opt):.2f} 度)")
        print(f"  投放时间 t_drone_fly: {t_drop:.4f} s")
        print(f"  起爆延迟 t_decoy_delay: {t_delay:.4f} s")

        print(f"  投放点坐标:  x={p_drop[0]:.2f}  y={p_drop[1]:.2f}  z={p_drop[2]:.2f}")
        print(f"  起爆点坐标:  x={p_expl[0]:.2f}  y={p_expl[1]:.2f}  z={p_expl[2]:.2f}")
        print(f"  该干扰弹单独遮蔽时长: {per_decoy[i]:.4f} s")

    print("\n" + "="*60)
    print(f"三枚云团联合（并集）有效遮蔽总时长: {total_union:.4f} s")
    print("="*60)


if __name__ == "__main__":
    main()
