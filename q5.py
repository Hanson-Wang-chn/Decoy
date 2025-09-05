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

# 复用 Q1/Q2 中的物理模型函数与常量
from utils.calculate_covered_time import (
    _missile_pos,              # 导弹轨迹
    _drone_vec,                # 无人机速度向量
    _decoy_state_at_explosion, # 起爆位置
    INTERVAL,
    G,
    SINK_SPEED,
    CLOUD_EFFECTIVE,
    SPEED_MISSILE,
    ORIGIN_FAKE
)
from utils.is_covered import is_covered

# ---- 根据题目描述定义的常量 ----
# 3枚导弹初始位置
POS_INIT_MISSILE_M1 = np.array([20000.0,    0.0, 2000.0], dtype=float)
POS_INIT_MISSILE_M2 = np.array([19000.0,  600.0, 2100.0], dtype=float)
POS_INIT_MISSILE_M3 = np.array([18000.0, -600.0, 1900.0], dtype=float)
POS_INIT_MISSILES = [POS_INIT_MISSILE_M1, POS_INIT_MISSILE_M2, POS_INIT_MISSILE_M3]

# 5架无人机初始位置
POS_INIT_DRONE_FY1 = np.array([17800.0,     0.0, 1800.0], dtype=float)
POS_INIT_DRONE_FY2 = np.array([12000.0,  1400.0, 1400.0], dtype=float)
POS_INIT_DRONE_FY3 = np.array([ 6000.0, -3000.0,  700.0], dtype=float)
POS_INIT_DRONE_FY4 = np.array([11000.0,  2000.0, 1800.0], dtype=float)
POS_INIT_DRONE_FY5 = np.array([13000.0, -2000.0, 1300.0], dtype=float)
POS_INIT_DRONES = [POS_INIT_DRONE_FY1, POS_INIT_DRONE_FY2, POS_INIT_DRONE_FY3, POS_INIT_DRONE_FY4, POS_INIT_DRONE_FY5]

# ---- 优化算法配置 ----
# TODO:
POPULATION_SIZE = 10000   # 搜索智能体（鲸鱼）的数量
MAX_ITERATIONS = 10    # 最大迭代次数

# ---- 参数边界 (搜索空间) ----
# 我们需要优化的 40 个变量：
# 对于每架无人机： [v_drone, theta_drone, t_drop1, t_delta_drop2, t_delta_drop3, t_delay1, t_delay2, t_delay3]
# 修改：t_delta_drop2和t_delta_drop3表示时间间隔，必须≥1秒；theta_drone扩展到0~2π以覆盖全方向
LOWER_BOUNDS = np.tile(np.array([70.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0], dtype=float), 5)
UPPER_BOUNDS = np.tile(np.array([140.0, 2*np.pi, 58.0, 59.0, 59.0, 20.0, 20.0, 20.0], dtype=float), 5)
DIMENSIONS = 40

# 加权系数：强调平衡
BETA = 0.5

# ---- 计算多枚干扰弹对多枚导弹的遮蔽时长（并集 + 逐枚逐导弹） ----
def calculate_covered_time_multi(pos_init_missiles: list,
                                 pos_init_drones: list,
                                 v_drones: np.ndarray,
                                 theta_drones: np.ndarray,
                                 t_drop_arrs: list,
                                 t_delay_arrs: list,
                                 interval: float = INTERVAL,
                                 show_progress: bool = False,
                                 return_details: bool = False):
    """
    计算多枚导弹被所有烟幕云团遮挡的总时长（每个导弹的并集），并可返回每个导弹的遮蔽时长、每个干扰弹对每个导弹的单独遮蔽时长等。
    说明：
      - 起爆后云团中心以 SINK_SPEED 匀速下沉，存在 CLOUD_EFFECTIVE 秒的有效期。
      - 每个导弹的总时长按"所有云团对该导弹遮蔽时间的并集长度"计；每个干扰弹对每个导弹的遮蔽时长单独统计，不去除重叠。
    """
    num_missiles = len(pos_init_missiles)
    num_drones = len(pos_init_drones)
    num_decoys_per_drone = 3
    num_decoys = num_drones * num_decoys_per_drone

    # 收集所有干扰弹的信息
    all_t_expl = []
    all_p_expl = []
    all_p_drop = []
    all_t_drop = []

    for d_id in range(num_drones):
        pos_drone = np.asarray(pos_init_drones[d_id], dtype=float)
        v_d = _drone_vec(v_drones[d_id], theta_drones[d_id])
        t_drop_arr = np.asarray(t_drop_arrs[d_id], dtype=float)
        t_delay_arr = np.asarray(t_delay_arrs[d_id], dtype=float)

        for i in range(num_decoys_per_drone):
            t_drop = t_drop_arr[i]
            t_delay = t_delay_arr[i]
            t_expl = t_drop + t_delay

            p_drop = pos_drone + v_d * t_drop
            p_expl = _decoy_state_at_explosion(pos_drone, v_d, t_drop, t_delay)

            all_t_drop.append(t_drop)
            all_t_expl.append(t_expl)
            all_p_drop.append(p_drop)
            all_p_expl.append(p_expl)

    all_t_expl = np.array(all_t_expl, dtype=float)
    all_p_expl = np.stack(all_p_expl, axis=0)
    all_p_drop = np.stack(all_p_drop, axis=0)

    # 对于每枚导弹，独立仿真其遮蔽情况
    union_times = np.zeros(num_missiles)
    per_decoy_per_missile = np.zeros((num_missiles, num_decoys))

    for m_id in range(num_missiles):
        pos_missile = np.asarray(pos_init_missiles[m_id], dtype=float)
        t_hit = np.linalg.norm(pos_missile) / SPEED_MISSILE
        times = np.arange(0.0, t_hit + 1e-9, interval, dtype=float)
        iterator = tqdm(times, desc=f"Simulating missile M{m_id+1}", unit="step") if show_progress else times

        covered_union = 0.0
        covered_per_decoy = np.zeros(num_decoys)

        for t in iterator:
            A_t = _missile_pos(pos_missile, t)
            occluded_flags = np.full(num_decoys, False)

            for d_id in range(num_decoys):
                if t >= all_t_expl[d_id] and t < all_t_expl[d_id] + CLOUD_EFFECTIVE:
                    dt = t - all_t_expl[d_id]
                    B_t = all_p_expl[d_id] + np.array([0.0, 0.0, -SINK_SPEED * dt], dtype=float)
                    if is_covered(A_t, B_t):
                        occluded_flags[d_id] = True
                        covered_per_decoy[d_id] += interval

            if np.any(occluded_flags):
                covered_union += interval

        union_times[m_id] = covered_union
        per_decoy_per_missile[m_id, :] = covered_per_decoy

    sum_union = np.sum(union_times)
    min_union = np.min(union_times)

    if return_details:
        details = {
            "union_times": union_times,
            "sum_union": sum_union,
            "min_union": min_union,
            "per_decoy_per_missile": per_decoy_per_missile,
            "all_p_drop": all_p_drop,
            "all_p_expl": all_p_expl,
            "all_t_drop": np.array(all_t_drop, dtype=float),
            "all_t_expl": all_t_expl
        }
        return sum_union, min_union, details
    return sum_union, min_union

# ---- 目标函数（WOA 默认最小化，这里返回"负的(总并集时长 + BETA * 最低并集时长)"） ----
def objective_function(params, interval_eval=0.02):
    """
    目标：最大化(所有导弹遮蔽并集时长之和 + BETA * 最低导弹遮蔽并集时长) -> 最小化其相反数。
    参数：
      params: 长度40，5架无人机各8个参数 [v, theta, t_drop1, delta2, delta3, delay1, delay2, delay3]
    """
    params = np.asarray(params, dtype=float).reshape(5, 8)
    v_drones = params[:, 0]
    theta_drones = params[:, 1]
    t_drop_arrs = []
    t_delay_arrs = []

    for d in range(5):
        t_drop1 = params[d, 2]
        t_delta2 = params[d, 3]  # 已确保 ≥1.0
        t_delta3 = params[d, 4]  # 已确保 ≥1.0
        t_drop2 = t_drop1 + t_delta2
        t_drop3 = t_drop2 + t_delta3
        t_drop_arrs.append([t_drop1, t_drop2, t_drop3])
        t_delay_arrs.append(params[d, 5:8].tolist())

    # 计算遮蔽时长（为速度，将评估步长略放宽）
    sum_union, min_union = calculate_covered_time_multi(
        pos_init_missiles=POS_INIT_MISSILES,
        pos_init_drones=POS_INIT_DRONES,
        v_drones=v_drones,
        theta_drones=theta_drones,
        t_drop_arrs=t_drop_arrs,
        t_delay_arrs=t_delay_arrs,
        interval=interval_eval,     # 加速评估
        show_progress=False,
        return_details=False
    )

    return - (sum_union + BETA * min_union)

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
        print("启动鲸鱼优化算法（问题5：5架无人机对3枚导弹）...")

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

            if (t + 1) % 1 == 0:
                print(f"迭代 {t + 1}/{self.max_iter}, 当前最优目标值(估计): {-self.leader_score:.4f}")

        return self.leader_pos, -self.leader_score

# ---- 主函数 ----
def main():
    """
    问题5：利用5架无人机，每架投放3枚烟幕干扰弹，对M1、M2、M3实施干扰。
    保持与问题2一致的界面/风格；优化完成后做一次精细仿真验证并打印关键坐标/时长，并保存到result3.xlsx。
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
    best_params, est_opt_value = woa.optimize()
    end_time = time.time()

    print("\n-------------------------------------------------")
    print("问题5 优化完成")
    print("优化算法：鲸鱼优化算法 (WOA)")
    print(f"总计用时: {end_time - start_time:.2f} 秒")
    print("-------------------------------------------------")

    # 解包最优参数并转换为绝对时间
    best_params = np.asarray(best_params, dtype=float).reshape(5, 8)
    v_opts = best_params[:, 0]
    theta_opts = best_params[:, 1]
    t_drop_arrs = []
    t_delay_arrs = []

    for d in range(5):
        t_drop1 = best_params[d, 2]
        t_delta2 = best_params[d, 3]
        t_delta3 = best_params[d, 4]
        t_drop2 = t_drop1 + t_delta2
        t_drop3 = t_drop2 + t_delta3
        t_drop_arrs.append(np.array([t_drop1, t_drop2, t_drop3], dtype=float))
        t_delay_arrs.append(best_params[d, 5:8])

    # ---- 使用最优参数进行一次"精细步长"的最终仿真与验证 ----
    _, _, details = calculate_covered_time_multi(
        pos_init_missiles=POS_INIT_MISSILES,
        pos_init_drones=POS_INIT_DRONES,
        v_drones=v_opts,
        theta_drones=theta_opts,
        t_drop_arrs=t_drop_arrs,
        t_delay_arrs=t_delay_arrs,
        interval=INTERVAL,         # 精细验证
        show_progress=True,
        return_details=True
    )

    union_times = details["union_times"]
    sum_union = details["sum_union"]
    min_union = details["min_union"]
    per_decoy_per_missile = details["per_decoy_per_missile"]
    all_p_drop = details["all_p_drop"]
    all_p_expl = details["all_p_expl"]
    all_t_drop = details["all_t_drop"]
    all_t_expl = details["all_t_expl"]

    # ---- 事后为每个干扰弹分配导弹（贡献最大者）并获取有效时长 ----
    assigned_missiles = []
    effective_times = []

    for decoy_id in range(15):
        per_m_times = per_decoy_per_missile[:, decoy_id]
        assigned_m = np.argmax(per_m_times) + 1  # M1=1, etc.
        eff_time = per_m_times[assigned_m - 1]
        assigned_missiles.append(f"M{assigned_m}")
        effective_times.append(eff_time)

    # ---- 打印关键信息 ----
    labels = ["FY1", "FY2", "FY3", "FY4", "FY5"]
    for d_id, lab in enumerate(labels):
        print(f"\n[{lab}] 最优参数：")
        print(f"  无人机速度 v_drone: {v_opts[d_id]:.4f} m/s")
        print(f"  无人机航向 theta_drone: {theta_opts[d_id]:.4f} rad ({math.degrees(theta_opts[d_id]):.2f} 度)")

        for i in range(3):
            decoy_id = d_id * 3 + i
            print(f"  [Decoy {i+1}]")
            print(f"    投放时间 t_drop: {all_t_drop[decoy_id]:.4f} s")
            print(f"    投放点 P_drop: ({all_p_drop[decoy_id,0]:.2f}, {all_p_drop[decoy_id,1]:.2f}, {all_p_drop[decoy_id,2]:.2f}) m")
            print(f"    起爆时间 t_expl: {all_t_expl[decoy_id]:.4f} s")
            print(f"    起爆点 P_expl: ({all_p_expl[decoy_id,0]:.2f}, {all_p_expl[decoy_id,1]:.2f}, {all_p_expl[decoy_id,2]:.2f}) m")
            print(f"    有效干扰时长: {effective_times[decoy_id]:.4f} s")
            print(f"    干扰的导弹编号: {assigned_missiles[decoy_id]}")

    print("\n" + "-"*50)
    for m_id in range(3):
        print(f"导弹 M{m_id+1} 的并集有效遮蔽时长: {union_times[m_id]:.4f} s")
    print(f"所有导弹并集有效遮蔽总时长: {sum_union:.4f} s")
    print(f"最低导弹并集有效遮蔽时长: {min_union:.4f} s")
    print("-"*50)

    # ---- 保存到 result3.xlsx ----
    headers = [
        "无人机编号", "无人机运动方向", "无人机运动速度 (m/s)", "烟幕干扰弹编号",
        "烟幕干扰弹投放点的x坐标 (m)", "烟幕干扰弹投放点的y坐标 (m)", "烟幕干扰弹投放点的z坐标 (m)",
        "烟幕干扰弹起爆点的x坐标 (m)", "烟幕干扰弹起爆点的y坐标 (m)", "烟幕干扰弹起爆点的z坐标 (m)",
        "有效干扰时长 (s)", "干扰的导弹编号"
    ]
    rows = []

    for d_id in range(5):
        direction_deg = math.degrees(theta_opts[d_id]) % 360
        speed = v_opts[d_id]
        for i in range(3):
            decoy_id = d_id * 3 + i
            row = [
                labels[d_id], direction_deg, speed, i+1,
                all_p_drop[decoy_id, 0], all_p_drop[decoy_id, 1], all_p_drop[decoy_id, 2],
                all_p_expl[decoy_id, 0], all_p_expl[decoy_id, 1], all_p_expl[decoy_id, 2],
                effective_times[decoy_id], assigned_missiles[decoy_id]
            ]
            rows.append(row)

    # 添加空行和注
    rows.append([None] * len(headers))
    rows.append([None, "注：以x轴为正向，逆时针方向为正，取值0~360（度）。"] + [None] * (len(headers) - 2))


if __name__ == "__main__":
    main()
