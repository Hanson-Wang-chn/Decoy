# q1.py

import numpy as np
import math

from utils.calculate_covered_time import calculate_covered_time


ORIGIN_FAKE = np.array([0.0, 0.0, 0.0], dtype=float)  # 假目标在原点


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
