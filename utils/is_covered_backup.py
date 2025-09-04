# utils/is_covered_backup.py

import math
import numpy as np

r = 10.0     # 烟幕云团半径
K = 100      # 锥面离散条数

def is_covered(a, b, c, d, e, f):
    # TODO: 输入参数直接使用np.array
    """
    判断导弹是否在烟幕云团范围内

    参数:
    a, b, c: 导弹（点A）的x, y, z坐标
    d, e, f: 烟幕云团中心（点B）的x, y, z坐标

    返回:
    布尔值，表示导弹是否在烟幕云团范围内
    """
    A = np.array([a, b, c], dtype=float)
    B = np.array([d, e, f], dtype=float)
    AB = B - A
    BA = A - B
    g = float(np.linalg.norm(AB))  # |AB| = g

    # A 与 B 重合或 A 在球内：显然被覆盖
    if g == 0.0 or g <= r:
        return True

    sin_alpha = r / g
    alpha = float(np.arcsin(sin_alpha))

    u_hat = BA / g
    
    # TODO: 检查一下这里对不对
    # ---- 取一个与 û 不共线的向量并正交化得到 v̂ ----
    # 选择 û 绝对值最小的分量对应的基向量，避免共线
    idx = int(np.argmin(np.abs(u_hat)))
    t = np.zeros(3, dtype=float)
    t[idx] = 1.0

    # 施密特正交化
    v = t - np.dot(t, u_hat) * u_hat
    nv = np.linalg.norm(v)
    if nv < 1e-12:
        # 极端退化再换一个 t
        t = np.array([1.0, 0.0, 0.0]) if idx != 0 else np.array([0.0, 1.0, 0.0])
        v = t - np.dot(t, u_hat) * u_hat
        nv = np.linalg.norm(v)
    v_hat = v / nv

    # ---- ŵ = (û × v̂) / ||û × v̂|| ----
    w = np.cross(u_hat, v_hat)
    w_hat = w / np.linalg.norm(w)

    # ---- φ_n 与锥面母线方向 d_n ----
    n = np.arange(K, dtype=float)
    phi_n = 2.0 * np.pi * n / K  # φ_n = 2πn/K, 0 ≤ n ≤ K-1

    # 每条母线的单位方向：d_n
    # d_n = cosα * û + sinα * (cosφ_n * v̂ + sinφ_n * ŵ)
    dir_n = (np.cos(alpha) * u_hat[None, :] +
             np.sin(alpha) * (np.cos(phi_n)[:, None] * v_hat[None, :] +
                              np.sin(phi_n)[:, None] * w_hat[None, :]))

    # ---- K 条射线的参数方程 ----
    # 对第 n 条射线：
    #   r_n(t) = (1 - t) * A + t * A_n[n],   t >= 0
    # 等价：r_n(t) = A + t * (A_n[n] - A)
    
    # TODO: 计算由点A和dir_n确定的每一条直线（母线）与柱面x^2 + (y - 200)^2 = 49的交点。如果每一条直线都

    pass
