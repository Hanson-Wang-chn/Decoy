# q1.py

import math
import numpy as np
from tqdm import tqdm

from utils.is_covered import is_covered


# ---- 常量（题面给定/默认） ----
SPEED_MISSILE = 300.0      # 导弹速率 (m/s)
INTERVAL = 0.03            # 仿真步长 (s)
G = 9.80                   # 重力加速度 (m/s^2)
SINK_SPEED = 3.0           # 云团下沉速度 (m/s)
CLOUD_EFFECTIVE = 20.0     # 起爆后有效遮蔽时长 (s)

ORIGIN_FAKE = np.array([0.0, 0.0, 0.0], dtype=float)  # 假目标在原点


def _missile_pos(pos_init_missile: np.ndarray, t: float) -> np.ndarray:
    """导弹按直线匀速飞向假目标（原点）。"""
    m0 = pos_init_missile
    vhat = (ORIGIN_FAKE - m0).astype(float)
    vhat /= np.linalg.norm(vhat)
    return m0 + SPEED_MISSILE * t * vhat


def _drone_vec(v_drone: float, theta_drone: float) -> np.ndarray:
    """无人机等高度匀速飞行的速度向量（z 分量为 0）。"""
    return np.array([v_drone * math.cos(theta_drone),
                     v_drone * math.sin(theta_drone),
                     0.0], dtype=float)


def _decoy_state_at_explosion(pos_init_drone: np.ndarray,
                              v_drone_vec: np.ndarray,
                              t_drop: float,
                              t_delay: float) -> np.ndarray:
    """
    计算起爆瞬间云团中心（即干扰弹位置）。
    干扰弹在投放后仅受重力下落，水平速度保持与无人机一致，初始竖直速度取 0。
    """
    # 投放点
    p_drop = pos_init_drone + v_drone_vec * t_drop
    # 从投放到起爆的抛体位移（水平匀速 + 竖直位移）
    tau = t_delay
    horiz = v_drone_vec * tau
    vert = np.array([0.0, 0.0, -0.5 * G * tau * tau], dtype=float)
    return p_drop + horiz + vert


def calculate_covered_time(pos_init_missile: np.ndarray,
                           pos_init_drone: np.ndarray,
                           v_drone: float,
                           theta_drone: float,
                           t_drone_fly: float,
                           t_decoy_delay: float) -> float:
    """
    计算导弹被烟幕云团遮挡的时间（秒）

    参数:
    pos_init_missile: np.array([x, y, z])，导弹初始（被探测到）时的位置
    pos_init_drone:   np.array([x, y, z])，拦截该导弹的无人机在接收到指令时的位置
    v_drone:          float，无人机速度 (m/s)
    theta_drone:      float，无人机航向 (弧度)，x 轴正向为 0，逆时针为正
    t_drone_fly:      float，从接到指令到投放烟幕干扰弹的时间 (s)
    t_decoy_delay:    float，从投放到起爆的时间延迟 (s)

    返回:
    float，总的有效遮挡时间 (s)
    """
    pos_init_missile = np.asarray(pos_init_missile, dtype=float)
    pos_init_drone   = np.asarray(pos_init_drone, dtype=float)

    # 关键时间点
    t_drop = float(t_drone_fly)
    t_expl = t_drop + float(t_decoy_delay)
    t_end  = t_expl + CLOUD_EFFECTIVE

    # 无人机速度向量（等高度）
    v_d = _drone_vec(v_drone, theta_drone)

    # 起爆位置（云团初始中心）
    B_expl = _decoy_state_at_explosion(pos_init_drone, v_d, t_drop, t_decoy_delay)

    # 进度条
    times = np.arange(0.0, t_end + 1e-9, INTERVAL, dtype=float)
    iterator = tqdm(times, desc="Simulating", unit="step")

    # 遮挡累计
    covered_time = 0.0

    for t in iterator:
        # 导弹位置（全时段均匀速直线）
        A_t = _missile_pos(pos_init_missile, t)

        # 仅在“起爆后且有效期内”才存在云团并参与遮挡判定
        if t >= t_expl:
            # 云团中心：起爆后只做竖直匀速下沉（水平坐标不再变化）
            dt = t - t_expl
            B_t = B_expl + np.array([0.0, 0.0, -SINK_SPEED * dt], dtype=float)

            # 判定是否“完全遮挡”（你的 is_covered 已定义为：A 点到整根真目标柱面的
            # 所有视线均被半径为 10 m 的球体优先命中）
            if is_covered(A_t, B_t):
                covered_time += INTERVAL

    return covered_time


def main():
    """
    按“问题 1”的固定场景直接运行一次并打印结果：
      - 导弹 M1 初始位置: (20000, 0, 2000)
      - 无人机 FY1 初始位置: (17800, 0, 1800)
      - FY1 以 120 m/s 朝向假目标（原点）飞行
      - 1.5 s 后投放，3.6 s 后起爆
    """
    # 初始条件（题面给定）
    pos_init_missile = np.array([20000.0, 0.0, 2000.0], dtype=float)
    pos_init_drone   = np.array([17800.0, 0.0, 1800.0], dtype=float)
    v_drone = 120.0
    # 航向：指向假目标（原点）的平面方位角
    theta_drone = math.atan2(ORIGIN_FAKE[1] - pos_init_drone[1],
                             ORIGIN_FAKE[0] - pos_init_drone[0])
    t_drone_fly   = 1.5
    t_decoy_delay = 3.6

    covered = calculate_covered_time(
        pos_init_missile, pos_init_drone,
        v_drone, theta_drone,
        t_drone_fly, t_decoy_delay
    )
    print(f"问题 1：总有效遮蔽时长 = {covered:.2f} s")


if __name__ == "__main__":
    main()
