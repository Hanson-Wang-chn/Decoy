# q3.py

import numpy as np
import math
import time
from tqdm import tqdm

# --- 让 utils.calculate_covered_time 中的 tqdm 静默（若被导入以复用工具函数） ---
try:
    import utils.calculate_covered_time as cct_module
    def _silent_tqdm(iterable, **kwargs):
        return iterable
    cct_module.tqdm = _silent_tqdm
except Exception:
    pass

# 直接使用问题2中已经实现/校验过的基础工具与常量
from utils.calculate_covered_time import (
    _missile_pos, _drone_vec, _decoy_state_at_explosion,
    INTERVAL, G, SINK_SPEED, CLOUD_EFFECTIVE
)
from utils.is_covered import is_covered
from utils.calculate_covered_time import calculate_covered_time

# ---- 根据题目描述定义的常量 ----
# 导弹 M1 的初始位置 [cite: 6]
POS_INIT_MISSILE_M1 = np.array([20000.0, 0.0, 2000.0], dtype=float)
# 无人机 FY1 的初始位置 [cite: 6]
POS_INIT_DRONE_FY1  = np.array([17800.0, 0.0, 1800.0], dtype=float)

# ---- 优化算法配置 ----
POPULATION_SIZE = 10   # 搜索智能体（鲸鱼）的数量
MAX_ITERATIONS  = 3    # 最大迭代次数

# ---- 参数边界 (搜索空间) ----
# 我们需要优化的 8 个变量：
# [v_drone, theta_drone, t_drop1, t_drop2, t_drop3, t_delay1, t_delay2, t_delay3]
LOWER_BOUNDS = np.array([70.0, 0.0, 0.0,  0.0,  0.0,  0.0, 0.0, 0.0], dtype=float)
UPPER_BOUNDS = np.array([140.0, np.pi, 60.0, 60.0, 60.0, 20.0, 20.0, 20.0], dtype=float)
DIMENSIONS    = 8

# ---- 计算三枚干扰弹的遮蔽时长（并集 + 逐枚），直接调用 is_covered ----
def calculate_covered_time_3decoys(pos_init_missile: np.ndarray,
                                   pos_init_drone:   np.ndarray,
                                   v_drone: float,
                                   theta_drone: float,
                                   t_drop_arr,          # 长度为3的 array-like：三次投放时间
                                   t_delay_arr,         # 长度为3的 array-like：三次起爆延迟
                                   interval: float = INTERVAL,
                                   show_progress: bool = False,
                                   return_details: bool = False):
    """
    计算导弹被三枚烟幕云团遮挡的总时长（并集），并可返回每一枚的遮蔽时长（各自统计）。
    说明：
      - 起爆后云团中心以 SINK_SPEED 匀速下沉，存在 CLOUD_EFFECTIVE 秒的有效期。
      - 总时长按“三枚云团遮蔽时间的并集长度”计；每枚遮蔽时长单独统计，不去除重叠。
    """
    pos_init_missile = np.asarray(pos_init_missile, dtype=float)
    pos_init_drone   = np.asarray(pos_init_drone, dtype=float)
    t_drop_arr  = np.asarray(t_drop_arr, dtype=float).reshape(3,)
    t_delay_arr = np.asarray(t_delay_arr, dtype=float).reshape(3,)

    # 三个关键时间：投放、起爆、结束
    t_expl_arr = t_drop_arr + t_delay_arr
    t_end = float(np.max(t_expl_arr) + CLOUD_EFFECTIVE)

    # 无人机速度向量（等高度）
    v_d = _drone_vec(v_drone, theta_drone)

    # 起爆位置（云团初始中心），形状 (3, 3)
    B_expl = np.stack([
        _decoy_state_at_explosion(pos_init_drone, v_d, t_drop_arr[i], t_delay_arr[i])
        for i in range(3)
    ], axis=0)

    # 时间轴
    times = np.arange(0.0, t_end + 1e-9, interval, dtype=float)
    iterator = tqdm(times, desc="Simulating(3 decoys)", unit="step") if show_progress else times

    covered_total = 0.0           # 并集总时长
    covered_each  = np.zeros(3)   # 各自遮蔽时长（不去重）

    for t in iterator:
        # 导弹位置
        A_t = _missile_pos(pos_init_missile, t)

        # 每个云团是否在该时刻有效
        active = t >= t_expl_arr

        # 对于“有效”的云团，计算其时刻位置并判定遮挡
        occluded_flags = []
        for i in range(3):
            if active[i]:
                # 起爆后仅做竖直匀速下沉（水平坐标不再变化）
                dt = t - t_expl_arr[i]
                B_t_i = B_expl[i] + np.array([0.0, 0.0, -SINK_SPEED * dt], dtype=float)
                if is_covered(A_t, B_t_i):
                    occluded_flags.append(True)
                    covered_each[i] += interval
                else:
                    occluded_flags.append(False)
            else:
                occluded_flags.append(False)

        # 并集：任意一枚遮挡即计入总时长
        if any(occluded_flags):
            covered_total += interval

    if return_details:
        return covered_total, covered_each, (t_drop_arr, t_expl_arr, B_expl)
    return covered_total

# ---- 目标函数（WOA 默认最小化，这里返回“负的并集遮蔽时长”）+ 约束惩罚 ----
def objective_function(params, interval_eval=0.02):
    """
    目标：最大化并集遮蔽时长 -> 最小化其相反数。
    约束（以惩罚形式）：
      1) 0 <= t_drop[i] <= 60, 0 <= t_delay[i] <= 20 （由边界保证）
      2) 相邻投放满足：t_drop2 - t_drop1 >= 1, t_drop3 - t_drop2 >= 1
    """
    v_drone      = params[0]
    theta_drone  = params[1]
    t_drop_arr   = params[2:5]
    t_delay_arr  = params[5:8]

    # 约束惩罚（违反越多，惩罚越大）
    penalty = 0.0
    # 投放间隔约束（至少 1s）
    gap12 = t_drop_arr[1] - t_drop_arr[0]
    gap23 = t_drop_arr[2] - t_drop_arr[1]
    if gap12 < 1.0:
        penalty += (1.0 - gap12) * 50.0
    if gap23 < 1.0:
        penalty += (1.0 - gap23) * 50.0
    # 单调性（避免乱序），若乱序，额外强惩罚
    if not (t_drop_arr[0] <= t_drop_arr[1] <= t_drop_arr[2]):
        penalty += 200.0

    # 计算遮蔽并集时长（为速度，将评估步长略放宽）
    covered_total = calculate_covered_time_3decoys(
        pos_init_missile=POS_INIT_MISSILE_M1,
        pos_init_drone=POS_INIT_DRONE_FY1,
        v_drone=v_drone,
        theta_drone=theta_drone,
        t_drop_arr=t_drop_arr,
        t_delay_arr=t_delay_arr,
        interval=interval_eval,     # 加速评估
        show_progress=False,
        return_details=False
    )

    return -(covered_total) + penalty

# ---- 鲸鱼优化算法（沿用问题2版本） ----
class WhaleOptimizationAlgorithm:
    """
    鲸鱼优化算法 (WOA) 的实现，用于寻找最优参数组合以最大化遮蔽时间。
    """
    def __init__(self, obj_func, lower_bounds, upper_bounds, dim, pop_size, max_iter):
        self.obj_func = obj_func
        self.lb = lower_bounds
        self.ub = upper_bounds
        self.dim = dim
        self.pop_size = pop_size
        self.max_iter = max_iter

        # 初始化领导者（当前最优解）
        self.leader_pos = np.zeros(dim)
        self.leader_score = float('inf')  # 初始最优评价值设为无穷大

        # 初始化所有搜索智能体（鲸鱼）的位置
        self.positions = np.random.rand(pop_size, dim) * (self.ub - self.lb) + self.lb

    def optimize(self):
        """
        运行 WOA 优化流程。
        """
        print("启动鲸鱼优化算法（问题3：3枚干扰弹）...")

        for t in tqdm(range(self.max_iter), desc="WOA 优化进度"):
            # 评估 + 更新领导者
            for i in range(self.pop_size):
                self.positions[i, :] = np.clip(self.positions[i, :], self.lb, self.ub)
                fitness = self.obj_func(self.positions[i, :])
                if fitness < self.leader_score:
                    self.leader_score = fitness
                    self.leader_pos = self.positions[i, :].copy()

            # a 从 2 线性递减到 0
            a = 2 - t * (2 / self.max_iter)

            for i in range(self.pop_size):
                r1 = np.random.rand()
                r2 = np.random.rand()

                A = 2 * a * r1 - a
                C = 2 * r2
                b = 1
                l = (a - 1) * np.random.rand() + 1
                p = np.random.rand()

                if p < 0.5:
                    if abs(A) >= 1:
                        # 探索
                        rand_leader_index = np.random.randint(0, self.pop_size)
                        X_rand = self.positions[rand_leader_index, :]
                        D_X_rand = abs(C * X_rand - self.positions[i, :])
                        self.positions[i, :] = X_rand - A * D_X_rand
                    else:
                        # 包围
                        D_Leader = abs(C * self.leader_pos - self.positions[i, :])
                        self.positions[i, :] = self.leader_pos - A * D_Leader
                else:
                    # 螺旋气泡网攻击
                    distance_to_leader = abs(self.leader_pos - self.positions[i, :])
                    self.positions[i, :] = distance_to_leader * np.exp(b * l) * np.cos(l * 2 * np.pi) + self.leader_pos

            if (t + 1) % 5 == 0:
                print(f"迭代 {t + 1}/{self.max_iter}, 当前最优遮蔽时间(估计): {-self.leader_score:.4f} s")

        return self.leader_pos, -self.leader_score

# ---- 结果写入 Excel（result1.xlsx，若无模板则自动创建一个简洁表头） ----
def write_result_to_excel(filepath, v_opt, theta_opt,
                          t_drop, t_expl, P_drop, P_expl,
                          covered_each, covered_total):
    import pandas as pd

    data = {
        "v_drone(m/s)": [v_opt],
        "theta_drone(rad)": [theta_opt],
        "theta_drone(deg)": [math.degrees(theta_opt)],
        # decoy 1
        "decoy1_t_drop(s)": [t_drop[0]],
        "decoy1_drop_x(m)": [P_drop[0][0]],
        "decoy1_drop_y(m)": [P_drop[0][1]],
        "decoy1_drop_z(m)": [P_drop[0][2]],
        "decoy1_t_expl(s)": [t_expl[0]],
        "decoy1_expl_x(m)": [P_expl[0][0]],
        "decoy1_expl_y(m)": [P_expl[0][1]],
        "decoy1_expl_z(m)": [P_expl[0][2]],
        "decoy1_covered_time(s)": [covered_each[0]],
        # decoy 2
        "decoy2_t_drop(s)": [t_drop[1]],
        "decoy2_drop_x(m)": [P_drop[1][0]],
        "decoy2_drop_y(m)": [P_drop[1][1]],
        "decoy2_drop_z(m)": [P_drop[1][2]],
        "decoy2_t_expl(s)": [t_expl[1]],
        "decoy2_expl_x(m)": [P_expl[1][0]],
        "decoy2_expl_y(m)": [P_expl[1][1]],
        "decoy2_expl_z(m)": [P_expl[1][2]],
        "decoy2_covered_time(s)": [covered_each[1]],
        # decoy 3
        "decoy3_t_drop(s)": [t_drop[2]],
        "decoy3_drop_x(m)": [P_drop[2][0]],
        "decoy3_drop_y(m)": [P_drop[2][1]],
        "decoy3_drop_z(m)": [P_drop[2][2]],
        "decoy3_t_expl(s)": [t_expl[2]],
        "decoy3_expl_x(m)": [P_expl[2][0]],
        "decoy3_expl_y(m)": [P_expl[2][1]],
        "decoy3_expl_z(m)": [P_expl[2][2]],
        "decoy3_covered_time(s)": [covered_each[2]],
        # union
        "covered_time_union(s)": [covered_total],
    }

    df = pd.DataFrame(data)
    try:
        df.to_excel(filepath, index=False)
        print(f"\n结果已写入: {filepath}")
    except Exception as e:
        print(f"\n写入 Excel 失败: {e}")

# ---- 主函数 ----
def main():
    """
    问题3：FY1 投放 3 枚烟幕干扰弹，对 M1 实施干扰。
    保持与问题2一致的界面/风格；优化完成后做一次精细仿真验证并打印关键坐标/时长。
    """
    def objective_quiet(params):
        return objective_function(params, interval_eval=0.02)  # 评估加速

    woa = WhaleOptimizationAlgorithm(
        obj_func=objective_quiet,
        lower_bounds=LOWER_BOUNDS,
        upper_bounds=UPPER_BOUNDS,
        dim=DIMENSIONS,
        pop_size=POPULATION_SIZE,
        max_iter=MAX_ITERATIONS
    )

    start_time = time.time()
    best_params, est_max_covered = woa.optimize()
    end_time = time.time()

    print("\n-------------------------------------------------")
    print("问题3 优化完成")
    print("优化算法：鲸鱼优化算法 (WOA)")
    print(f"总计用时: {end_time - start_time:.2f} 秒")
    print("-------------------------------------------------")

    # 解包最优参数
    v_opt         = best_params[0]
    theta_opt     = best_params[1]
    t_drop_opt    = np.array(best_params[2:5], dtype=float)
    t_delay_opt   = np.array(best_params[5:8], dtype=float)
    t_expl_opt    = t_drop_opt + t_delay_opt

    print(f"最优无人机速度 (v_drone): {v_opt:.4f} m/s")
    print(f"最优无人机航向 (theta_drone): {theta_opt:.4f} rad ({math.degrees(theta_opt):.2f} 度)")
    print(f"最优投放时间 (t_drop): {t_drop_opt}")
    print(f"最优起爆延迟 (t_delay): {t_delay_opt}")
    print("\n" + "="*50)
    print(f"估计的最大并集有效遮蔽时间: {est_max_covered:.4f} s")
    print("="*50)

    # ---- 使用最优参数进行一次“精细步长”的最终仿真与验证 ----
    covered_total, covered_each, debug = calculate_covered_time_3decoys(
        pos_init_missile=POS_INIT_MISSILE_M1,
        pos_init_drone=POS_INIT_DRONE_FY1,
        v_drone=v_opt,
        theta_drone=theta_opt,
        t_drop_arr=t_drop_opt,
        t_delay_arr=t_delay_opt,
        interval=INTERVAL,         # 精细验证
        show_progress=False,
        return_details=True
    )
    t_drop_opt, t_expl_opt, B_expl_opt = debug

    # 无人机速度向量 + 三个投放点/起爆点坐标
    v_d = _drone_vec(v_opt, theta_opt)
    P_drop = np.stack([POS_INIT_DRONE_FY1 + v_d * t_drop_opt[i] for i in range(3)], axis=0)
    P_expl = B_expl_opt.copy()

    # ---- 打印关键坐标与逐枚时长 ----
    print("\n" + "-"*50)
    print("最优条件下的关键位置坐标与时序：")
    for i in range(3):
        print(f"[Decoy {i+1}]")
        print(f"  投放时间 t_drop{i+1}: {t_drop_opt[i]:.4f} s")
        print(f"  投放点  P_drop{i+1}: ({P_drop[i,0]:.2f}, {P_drop[i,1]:.2f}, {P_drop[i,2]:.2f}) m")
        print(f"  起爆时间 t_expl{i+1}: {t_expl_opt[i]:.4f} s")
        print(f"  起爆点  P_expl{i+1}: ({P_expl[i,0]:.2f}, {P_expl[i,1]:.2f}, {P_expl[i,2]:.2f}) m")
        print(f"  该弹有效遮蔽时长: {covered_each[i]:.4f} s")
    print("-"*50)
    print(f"三枚烟幕干扰弹‘并集’有效遮蔽总时长: {covered_total:.4f} s")
    print("-"*50)

    # ---- 写出到 result1.xlsx（若需适配官方模板，可在此处按模板列名调整） ----
    write_result_to_excel(
        filepath="result1.xlsx",
        v_opt=v_opt, theta_opt=theta_opt,
        t_drop=t_drop_opt, t_expl=t_expl_opt,
        P_drop=P_drop, P_expl=P_expl,
        covered_each=covered_each, covered_total=covered_total
    )


if __name__ == "__main__":
    main()
