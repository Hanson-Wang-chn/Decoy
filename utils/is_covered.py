# utils/is_covered.py

import numpy as np


# 烟幕云团参数
R_DECOY = 10.0  # 半径

# ---- 圆柱参数：x^2 + (y - 200)^2 = 49,  z ∈ [0, 10] ----
R_CYL = 7.0
CYL_CY = 200.0
Z_MIN, Z_MAX = 0.0, 10.0

# 采样密度
SIDE_THETA = 360     # 侧面角向采样
SIDE_Z     = 1000    # 侧面高度采样
CAP_THETA  = 360     # 上/下盖角向采样
CAP_RADIAL = 700     # 上/下盖径向采样

_TOL = 1e-9

def _sample_cylinder_surface():
    """离散侧面 + 上下底，返回 (N,3)"""
    # 侧面
    thetas = np.linspace(0.0, 2.0*np.pi, SIDE_THETA, endpoint=False)
    zs = np.linspace(Z_MIN, Z_MAX, SIDE_Z)
    T, Z = np.meshgrid(thetas, zs, indexing='xy')
    Xs = R_CYL * np.cos(T)
    Ys = CYL_CY + R_CYL * np.sin(T)
    P_side = np.stack([Xs, Ys, Z], axis=-1).reshape(-1, 3)

    # 上/下底（极坐标均匀网格）
    thetas_c = np.linspace(0.0, 2.0*np.pi, CAP_THETA, endpoint=False)
    rs = np.linspace(0.0, R_CYL, CAP_RADIAL)
    Tc, Rc = np.meshgrid(thetas_c, rs, indexing='xy')
    Xc = Rc * np.cos(Tc)
    Yc = CYL_CY + Rc * np.sin(Tc)

    P_top = np.stack([Xc, Yc, np.full_like(Xc, Z_MAX)], axis=-1).reshape(-1, 3)
    # 不必采样下底面，影响效率
    # P_bot = np.stack([Xc, Yc, np.full_like(Xc, Z_MIN)], axis=-1).reshape(-1, 3)

    # return np.vstack([P_side, P_top, P_bot])
    return np.vstack([P_side, P_top])

def is_covered(A, B):
    """
    判断导弹是否在烟幕云团范围内

    参数:
    A: np.array([x, y, z])，导弹（点A）的位置
    B: np.array([x, y, z])，烟幕云团中心（点B）的位置

    返回:
    布尔值，表示导弹是否在烟幕云团范围内
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)

    BA = B - A
    g = float(np.linalg.norm(BA))
    if g <= R_DECOY + _TOL:
        return True  # A 在球内或与 B 重合，显然遮挡

    # 锥参数
    sin_alpha = np.clip(R_DECOY / g, -1.0, 1.0)
    alpha = float(np.arcsin(sin_alpha))
    cos_alpha = float(np.cos(alpha))
    a_hat = BA / g

    # 采样柱体表面
    P = _sample_cylinder_surface()

    # 向量化计算
    V = P - A[None, :]                       # 指向每个采样点的向量
    dist = np.linalg.norm(V, axis=1)
    valid = dist > 1e-12                     # 避免除零
    if not np.any(valid):
        return False

    d_hat = np.zeros_like(V)
    d_hat[valid] = V[valid] / dist[valid, None]

    # (1) 在锥内：<d_hat, a_hat> >= cos(alpha)
    inside_cone = (d_hat @ a_hat) >= (cos_alpha - 1e-12)
    if not np.all(inside_cone[valid]):
        return False

    # (2) 球先于柱：解 |A + t d_hat - B|^2 = R_DECOY^2 的最近正根 t_hit
    #    t_hit = dot - sqrt(dot^2 - (g^2 - R_DECOY^2)),  其中 dot = <d_hat, BA>
    dot = np.einsum('ij,j->i', d_hat, BA)           # (N,)
    disc = dot**2 - (g**2 - R_DECOY**2)
    disc = np.maximum(disc, 0.0)                    # 数值稳健
    t_hit = dot - np.sqrt(disc)                     # 最近正根

    # 命中应发生在到达柱体之前
    occluded = (t_hit >= -1e-9) & (t_hit <= dist + 1e-9)

    return bool(np.all(occluded[valid]))

