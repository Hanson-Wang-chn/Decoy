# test4.py

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from utils.is_covered import is_covered

# -----------------------------
# 常量参数
# -----------------------------
V_MISSILE = 300.0       # m/s, 导弹速度
G = 9.80                # m/s^2, 重力加速度
V_SINK = 3.0            # m/s, 云团下沉速度
DT = 0.01               # s, 仿真时间步长

# 坐标定义
# 假目标作为坐标原点 (0,0,0), xOy 为水平面, z 为高度
M1_INIT = np.array([20000.0,    0.0, 2000.0])   # 导弹 M1 初始位置
TARGET = np.array([0.0, 0.0, 0.0])              # 假目标原点

# 无人机初始状态（固定）
FY_INIT = {
    1: np.array([17800.0,      0.0, 1800.0]),   # FY1
    2: np.array([12000.0,   1400.0, 1400.0]),   # FY2
    3: np.array([ 6000.0,  -3000.0,  700.0]),   # FY3
}

# -----------------------------
# 参数区（请按需修改）
# -----------------------------

# 无人机航向（度，x轴正向为0，逆时针为正）
# theta_drone_deg = {
#     1: 4.59,   # FY1 航向
#     2: 4.59,   # FY2 航向
#     3: 10.09,   # FY3 航向
# }
# # 无人机速度（m/s, 要求 70~140）
# v_drone = {
#     1: 91.8321,   # FY1 速度
#     2: 91.9806,   # FY2 速度
#     3: 91.9793,   # FY3 速度
# }
# # 投放时间（s，自 t=0 起计）
# t_drone_fly = {
#     1: 1.2886,   # FY1 投放时间
#     2: 1.0653,   # FY2 投放时间
#     3: 1.2068,   # FY3 投放时间
# }
# # 可选：时间引信延迟（s），起爆时间 = 投放时间 + 引信延迟。
# # 若不需要延迟，保持 0.0 即“投放即起爆”。
# t_fuze = {
#     1: ,    # FY1 延迟
#     2: ,    # FY2 延迟
#     3: ,    # FY3 延迟
# }

# theta_drone_deg = {
#     1: -180.0,   # FY1 航向
#     2: -45.0,   # FY2 航向
#     3: 90.0,   # FY3 航向
# }
# # 无人机速度（m/s, 要求 70~140）
# v_drone = {
#     1: 70.0,   # FY1 速度
#     2: 140.0,   # FY2 速度
#     3: 110.0,   # FY3 速度
# }
# # 投放时间（s，自 t=0 起计）
# t_drone_fly = {
#     1: 1.0,   # FY1 投放时间
#     2: 11.0,   # FY2 投放时间
#     3: 23.8,   # FY3 投放时间
# }
# # 可选：时间引信延迟（s），起爆时间 = 投放时间 + 引信延迟。
# # 若不需要延迟，保持 0.0 即“投放即起爆”。
# t_fuze = {
#     1: 3.0,    # FY1 延迟
#     2: 3.0,    # FY2 延迟
#     3: 4.2,    # FY3 延迟
# }

# TODO:
theta_drone_deg = {
    1: 5.69,   # FY1 航向
    2: 243.61,   # FY2 航向
    3: 103.86,   # FY3 航向
}
# 无人机速度（m/s, 要求 70~140）
v_drone = {
    1: 107.43,   # FY1 速度
    2: 78.14,   # FY2 速度
    3: 117.89,   # FY3 速度
}
# 投放时间（s，自 t=0 起计）
t_drone_fly = {
    1: 0.72,   # FY1 投放时间
    2: 12.34,   # FY2 投放时间
    3: 21.50,   # FY3 投放时间
}
# 可选：时间引信延迟（s），起爆时间 = 投放时间 + 引信延迟。
# 若不需要延迟，保持 0.0 即"投放即起爆"。
t_fuze = {
    1: 0.33,    # FY1 延迟
    2: 7.25,    # FY2 延迟
    3: 5.49,    # FY3 延迟
}
# 12.05
# 10.88

# -----------------------------
# 工具函数
# -----------------------------
def unit(v):
    n = np.linalg.norm(v)
    if n == 0.0:
        return v
    return v / n

# -----------------------------
# 导弹轨迹（直线至原点）
# -----------------------------
dM = unit(TARGET - M1_INIT)              # 单位方向
dist_to_target = np.linalg.norm(TARGET - M1_INIT)
T_END = dist_to_target / V_MISSILE       # 命中时间

# -----------------------------
# 预计算三枚烟幕的起爆信息与中心轨迹
# -----------------------------
decoys = []  # 每项: {'t_exp':..., 'P_exp':..., 'v_sink':..., 'id':i}

for i in (1, 2, 3):
    # 参数校验与航向 -> 速度向量
    vi = float(v_drone[i])
    if not (70.0 <= vi <= 140.0):
        raise ValueError(f"FY{i} 速度必须在 [70, 140] m/s 之间，当前={vi}")
    theta_rad = np.deg2rad(float(theta_drone_deg[i]))
    dir_xy = np.array([np.cos(theta_rad), np.sin(theta_rad), 0.0])  # 水平单位向量
    v_vec = vi * dir_xy

    # 无人机初始与投放时刻位置（等高度直线）
    P0 = FY_INIT[i]
    t_drop = float(t_drone_fly[i])
    P_drop = P0 + v_vec * t_drop  # 水平等高运动

    # 时间引信
    t_delay = float(t_fuze[i])
    t_exp = t_drop + t_delay

    # 抛体运动：从投放到起爆，初速度 = 无人机速度，竖直初速=0，加速度 a=(0,0,-G)
    # 起爆位置
    P_exp = P_drop + v_vec * t_delay + np.array([0.0, 0.0, -0.5 * G * t_delay**2])

    decoys.append({
        'id': i,
        't_exp': t_exp,
        'P_exp': P_exp,
        'v_sink': V_SINK,
    })

# -----------------------------
# 主仿真：0.01 s 步长，至命中为止
# -----------------------------
times = np.arange(0.0, T_END + 1e-9, DT)       # 包含末端
covered = np.zeros(times.shape[0], dtype=bool)

for k, t in enumerate(times):
    # 导弹位置
    Pm = M1_INIT + V_MISSILE * dM * t

    # 判断当前时刻是否被任一烟幕完全遮蔽（并集）
    occluded = False
    for d in decoys:
        t_exp = d['t_exp']
        # 仅在起爆后 20 s 内有效
        if t < t_exp or (t - t_exp) > 20.0:
            continue
        # 云团中心：起爆点 + 匀速下沉
        Pc = d['P_exp'] + np.array([0.0, 0.0, -d['v_sink'] * (t - t_exp)])
        if is_covered(Pm, Pc):
            occluded = True
            break

    covered[k] = occluded

total_covered_time = covered.sum() * DT

# -----------------------------
# 输出
# -----------------------------
print("===== Problem 4 Verification (FY1, FY2, FY3; one decoy each) =====")
print(f"Time step: {DT:.2f} s, Simulation horizon: {T_END:.3f} s")
for d in decoys:
    print(f"FY{d['id']}: t_exp = {d['t_exp']:.3f} s, "
          f"P_exp = ({d['P_exp'][0]:.3f}, {d['P_exp'][1]:.3f}, {d['P_exp'][2]:.3f})")

print(f"Total completely-occluded time for M1 (union): {total_covered_time:.2f} s")
