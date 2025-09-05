import numpy as np
import math
from tqdm import tqdm

# 导入必要的函数和常量
from utils.is_covered import is_covered
from utils.calculate_covered_time import (
    _missile_pos, _drone_vec, _decoy_state_at_explosion,
    INTERVAL, SINK_SPEED, CLOUD_EFFECTIVE, SPEED_MISSILE, ORIGIN_FAKE
)

# 导弹初始位置
POS_INIT_MISSILE = np.array([20000.0, 0.0, 2000.0], dtype=float)

# 三架无人机初始位置
POS_INIT_DRONE_FY1 = np.array([17800.0,    0.0, 1800.0], dtype=float)
POS_INIT_DRONE_FY2 = np.array([12000.0, 1400.0, 1400.0], dtype=float)
POS_INIT_DRONE_FY3 = np.array([ 6000.0,-3000.0,  700.0], dtype=float)

def calculate_covered_time_3drones(pos_init_missile, drone_triplets, show_progress=True):
    """
    计算三枚云团对导弹的联合有效遮蔽时长（秒）
    
    参数:
    pos_init_missile: 导弹初始位置
    drone_triplets: 包含三个元组 (pos_init_drone, v_drone, theta_drone, t_drone_fly, t_decoy_delay)
    show_progress: 是否显示进度条
    
    返回:
    total_union: 三枚云团并集遮蔽时长
    per_decoy: 每枚云团单独遮蔽时长
    details: 详细数据（投放点、起爆点等）
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

    # 全局仿真终止时间
    t_sim_end = float(min(max(t_end.max(), 0.0), t_hit_origin))

    # 时间轴
    times = np.arange(0.0, t_sim_end + 1e-9, float(INTERVAL), dtype=float)
    iterator = tqdm(times, desc="模拟计算中", unit="step") if show_progress else times

    total_union = 0.0
    per_decoy   = np.zeros(3, dtype=float)

    # 逐时刻仿真
    for t in iterator:
        A_t = _missile_pos(pos_init_missile, t)

        covered_any = False
        covered_flags = [False, False, False]

        for i in range(3):
            if t >= t_expl[i] and t <= t_end[i]:
                dt = t - t_expl[i]
                # 起爆后云团仅做竖直匀速下沉
                B_t = B_expl[i] + np.array([0.0, 0.0, -SINK_SPEED * dt], dtype=float)

                # 判断是否完全遮挡
                if is_covered(A_t, B_t):
                    covered_flags[i] = True
                    per_decoy[i] += float(INTERVAL)
                    covered_any = True

        if covered_any:
            total_union += float(INTERVAL)

    # 结果汇总
    details = {
        "t_drop": t_drop.tolist(),
        "t_expl": t_expl.tolist(),
        "t_end":  t_end.tolist(),
        "p_drop": [p_drop[i].tolist() for i in range(3)],
        "p_expl": [B_expl[i].tolist() for i in range(3)],
    }

    return float(total_union), per_decoy.tolist(), details

def test():
    """测试函数：输入参数，计算并显示结果"""
    print("\n===== 烟幕干扰弹遮蔽时长测试 =====")
    print("请选择输入方式:")
    print("1. 使用预设参数")
    print("2. 手动输入参数")
    
    # choice = input("请选择 (1/2): ")
    choice = "1"
    
    if choice == "2":
        # 手动输入模式
        drone_params = []
        drone_positions = [POS_INIT_DRONE_FY1, POS_INIT_DRONE_FY2, POS_INIT_DRONE_FY3]
        
        for i, name in enumerate(["FY1", "FY2", "FY3"]):
            print(f"\n==== {name} 参数 ====")
            v = float(input(f"无人机速度 v_drone (m/s, 70-140): "))
            theta_deg = float(input(f"无人机航向 (度, 0-180): "))
            theta = math.radians(theta_deg)
            t_fly = float(input(f"投放时间 t_drone_fly (s): "))
            t_delay = float(input(f"起爆延迟 t_decoy_delay (s): "))
            
            drone_params.append((drone_positions[i], v, theta, t_fly, t_delay))
    else:
        # 预设参数模式
        print("\n使用预设参数:")
        drone_params = [
            (POS_INIT_DRONE_FY1, 110.0, math.radians(90.0), 23.8, 4.2),  # FY1: 速度, 航向, 投放时间, 起爆延迟
            (POS_INIT_DRONE_FY2, 140.0, math.radians(-45.0), 11.0, 3.0),  # FY2
            (POS_INIT_DRONE_FY3, 70.0, math.radians(-180.0), 1.0, 3.0)   # FY3
        ]
        
        # 显示预设参数
        for i, (_, v, theta, t_fly, t_delay) in enumerate(drone_params):
            name = ["FY1", "FY2", "FY3"][i]
            print(f"{name}: 速度={v}m/s, 航向={math.degrees(theta):.1f}°, 投放时间={t_fly}s, 起爆延迟={t_delay}s")
    
    # 执行计算
    print("\n开始计算遮蔽时长...")
    total_time, individual_times, details = calculate_covered_time_3drones(
        pos_init_missile=POS_INIT_MISSILE,
        drone_triplets=drone_params,
        show_progress=True
    )
    
    # 显示结果
    print("\n==== 计算结果 ====")
    for i, name in enumerate(["FY1", "FY2", "FY3"]):
        params = drone_params[i]
        p_drop = np.array(details["p_drop"][i])
        p_expl = np.array(details["p_expl"][i])
        
        print(f"\n[{name}] 参数与结果:")
        print(f"  无人机速度: {params[1]:.2f} m/s")
        print(f"  无人机航向: {math.degrees(params[2]):.2f}°")
        print(f"  投放时间: {params[3]:.2f} s")
        print(f"  起爆延迟: {params[4]:.2f} s")
        print(f"  投放点: ({p_drop[0]:.2f}, {p_drop[1]:.2f}, {p_drop[2]:.2f})")
        print(f"  起爆点: ({p_expl[0]:.2f}, {p_expl[1]:.2f}, {p_expl[2]:.2f})")
        print(f"  单独遮蔽时长: {individual_times[i]:.4f} 秒")
    
    print("\n" + "="*60)
    print(f"三枚云团联合有效遮蔽总时长: {total_time:.4f} 秒")
    print("="*60)

if __name__ == "__main__":
    test()
