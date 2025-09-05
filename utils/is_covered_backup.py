# # utils/is_covered_backup.py

# import math
# import numpy as np

# r = 10.0     # 烟幕云团半径
# K = 100      # 锥面离散条数

# def is_covered(A, B):
#     """
#     判断导弹是否在烟幕云团范围内

#     参数:
#     A: np.array([x, y, z])，导弹（点A）的位置
#     B: np.array([x, y, z])，烟幕云团中心（点B）的位置

#     返回:
#     布尔值，表示导弹是否在烟幕云团范围内
#     """
#     A = np.asarray(A, dtype=float)
#     B = np.asarray(B, dtype=float)
#     AB = B - A
#     BA = A - B
#     g = float(np.linalg.norm(AB))  # |AB| = g

#     # A 与 B 重合或 A 在球内：显然被覆盖
#     if g == 0.0 or g <= r:
#         return True

#     sin_alpha = r / g
#     alpha = float(np.arcsin(sin_alpha))

#     u_hat = BA / g
    
#     # ---- 取一个与 û 不共线的向量并正交化得到 v̂ ----
#     # 选择 û 绝对值最小的分量对应的基向量，避免共线
#     idx = int(np.argmin(np.abs(u_hat)))
#     t = np.zeros(3, dtype=float)
#     t[idx] = 1.0

#     # 施密特正交化
#     v = t - np.dot(t, u_hat) * u_hat
#     nv = np.linalg.norm(v)
#     if nv < 1e-12:
#         # 极端退化再换一个 t
#         t = np.array([1.0, 0.0, 0.0]) if idx != 0 else np.array([0.0, 1.0, 0.0])
#         v = t - np.dot(t, u_hat) * u_hat
#         nv = np.linalg.norm(v)
#     v_hat = v / nv

#     # ---- ŵ = (û × v̂) / ||û × v̂|| ----
#     w = np.cross(u_hat, v_hat)
#     w_hat = w / np.linalg.norm(w)

#     # ---- φ_n 与锥面母线方向 d_n ----
#     n = np.arange(K, dtype=float)
#     phi_n = 2.0 * np.pi * n / K  # φ_n = 2πn/K, 0 ≤ n ≤ K-1

#     # 每条母线的单位方向：d_n
#     # d_n = cosα * û + sinα * (cosφ_n * v̂ + sinφ_n * ŵ)
#     dir_n = (np.cos(alpha) * u_hat[None, :] +
#              np.sin(alpha) * (np.cos(phi_n)[:, None] * v_hat[None, :] +
#                               np.sin(phi_n)[:, None] * w_hat[None, :]))

#     # ---- K 条射线的参数方程 ----
#     # 对第 n 条射线：
#     #   r_n(t) = (1 - t) * A + t * A_n[n],   t >= 0
#     # 等价：r_n(t) = A + t * (A_n[n] - A)
    
#     # TODO: 计算由点A和dir_n确定的每一条直线（母线）与柱面x^2 + (y - 200)^2 = 49的交点。

#     pass
















































# # utils/is_covered_backup.py

# import math
# import numpy as np

# # ----- 题面参数 -----
# R_CYL   = 7.0        # 圆柱半径
# CYL_CY  = 200.0      # 圆柱在 y 轴方向的偏移：x^2 + (y - 200)^2 = R_CYL^2
# Z_MIN   = 0.0        # 圆柱底面 z
# Z_MAX   = 10.0       # 圆柱顶面 z

# # ----- 云团/锥面离散参数（可按需调整） -----
# r = 10.0     # 烟幕云团半径（题面给定，起爆后 10 m 范围有效）
# K = 360      # 锥面离散条数（越大越精细，耗时略增）

# _TOL = 1e-9


# def _orthonormal_basis_from_axis(axis_hat: np.ndarray):
#     """
#     给定轴向单位向量 axis_hat，构造与之正交的一组正交单位基 (v_hat, w_hat)。
#     """
#     # 选一个尽量不共线的基向量
#     idx = int(np.argmin(np.abs(axis_hat)))
#     t = np.zeros(3, dtype=float)
#     t[idx] = 1.0

#     # 施密特正交化：先得到 v，再单位化
#     v = t - np.dot(t, axis_hat) * axis_hat
#     nv = np.linalg.norm(v)
#     if nv < 1e-12:
#         # 极端退化时更换一个 t
#         t = np.array([1.0, 0.0, 0.0]) if idx != 0 else np.array([0.0, 1.0, 0.0])
#         v = t - np.dot(t, axis_hat) * axis_hat
#         nv = np.linalg.norm(v)
#     v_hat = v / nv

#     # w = axis_hat x v_hat
#     w = np.cross(axis_hat, v_hat)
#     w_hat = w / np.linalg.norm(w)
#     return v_hat, w_hat


# def _ray_hits_open_z_interval(Az: float, dz: float) -> bool:
#     """
#     一条半直线 z(t) = Az + t*dz, t >= 0 是否与开区间 (Z_MIN, Z_MAX) 相交。
#     """
#     if abs(dz) <= _TOL:
#         return (Z_MIN + _TOL) < Az < (Z_MAX - _TOL)
#     if dz > 0:
#         return Az < (Z_MAX - _TOL)
#     else:
#         return Az > (Z_MIN + _TOL)


# def is_covered(A, B) -> bool:
#     """
#     判断导弹（点 A）对圆柱体目标是否被球状云团（中心 B、半径 r）完全遮挡。

#     按思路：
#     1) 用 K 条锥面母线与无限圆柱 x^2 + (y - 200)^2 = R_CYL^2 求交，
#        若存在交点的 z 落在 (0, 10) 内，则不是完全遮挡；
#     2) 若 1) 不成立，再判断圆柱中心 Q 是否在锥内，若在则完全遮挡，否则不是。

#     参数:
#         A: np.ndarray shape (3,)
#         B: np.ndarray shape (3,)
#     返回:
#         bool: True = 完全遮挡；False = 否。
#     """
#     A = np.asarray(A, dtype=float)
#     B = np.asarray(B, dtype=float)

#     AB = B - A
#     g = float(np.linalg.norm(AB))  # |AB|
#     if g <= _TOL:
#         # A 与 B 几乎重合，或 A 在球内 → 显然完全遮挡
#         return True
#     if g <= (r + _TOL):
#         return True

#     # 锥半角 alpha： sin(alpha) = r / g
#     sin_alpha = min(1.0, max(0.0, r / g))
#     alpha = float(np.arcsin(sin_alpha))
#     cos_alpha = float(np.sqrt(max(0.0, 1.0 - sin_alpha * sin_alpha)))

#     # 锥轴单位向量：从 A 指向 B
#     axis_hat = AB / g

#     # 构造与轴正交的两个单位向量
#     v_hat, w_hat = _orthonormal_basis_from_axis(axis_hat)

#     # K 条母线方向：d_n = cosα * axis_hat + sinα * (cosφ * v_hat + sinφ * w_hat)
#     n = np.arange(K, dtype=float)
#     phi = 2.0 * np.pi * n / K
#     dir_n = (cos_alpha * axis_hat[None, :]
#              + sin_alpha * (np.cos(phi)[:, None] * v_hat[None, :]
#                             + np.sin(phi)[:, None] * w_hat[None, :]))  # (K,3)

#     # ----- 与无限圆柱侧面求交 -----
#     Ax, Ay, Az = float(A[0]), float(A[1]), float(A[2])
#     dx = dir_n[:, 0]
#     dy = dir_n[:, 1]
#     dz = dir_n[:, 2]

#     a2 = dx * dx + dy * dy
#     b1 = 2.0 * (Ax * dx + (Ay - CYL_CY) * dy)
#     c0 = Ax * Ax + (Ay - CYL_CY) * (Ay - CYL_CY) - R_CYL * R_CYL

#     # 处理一般情形：a2 > 0
#     mask_general = a2 > _TOL
#     D = b1[mask_general] * b1[mask_general] - 4.0 * a2[mask_general] * c0
#     has_real = D >= -1e-12  # 允许微小负值的数值误差
#     if np.any(has_real):
#         sqrtD = np.zeros_like(D)
#         sqrtD[has_real] = np.sqrt(np.maximum(0.0, D[has_real]))
#         a2g = a2[mask_general]

#         t1 = (-b1[mask_general] - sqrtD) / (2.0 * a2g)
#         t2 = (-b1[mask_general] + sqrtD) / (2.0 * a2g)

#         # 只保留 t >= 0 的交点
#         t_candidates = []
#         if t1.size:
#             t_candidates.append(t1)
#         if t2.size:
#             t_candidates.append(t2)
#         if t_candidates:
#             T = np.vstack(t_candidates)  # (2, Ngeneral)
#             # 对每个根，检查 z 是否落在 (Z_MIN, Z_MAX)
#             dz_g = dz[mask_general]
#             Az_g = Az  # scalar
#             for row in range(T.shape[0]):
#                 t = T[row]
#                 # 只看 t >= 0
#                 valid = t >= -1e-12
#                 if np.any(valid):
#                     z_hit = Az_g + t[valid] * dz_g[valid]
#                     if np.any((z_hit > (Z_MIN + _TOL)) & (z_hit < (Z_MAX - _TOL))):
#                         return False  # 母线穿过侧面且 z 在 (0,10) 内 → 非完全遮挡

#     # 处理退化：a2 ≈ 0（母线几乎平行 z 轴，x,y 基本不变）
#     mask_deg = ~mask_general
#     if np.any(mask_deg):
#         # 该半直线在投影平面上的路径是一个点 (Ax, Ay)
#         rho2 = Ax * Ax + (Ay - CYL_CY) * (Ay - CYL_CY)
#         dz_d = dz[mask_deg]
#         # 1) 若恰好位于圆柱侧面（切线）：认为可能“擦到侧面”
#         if abs(rho2 - R_CYL * R_CYL) <= 1e-8:
#             # 只要沿半直线方向能进入 z∈(0,10) 的区间，就视作会触侧面
#             if _ray_hits_open_z_interval(Az, float(np.mean(dz_d))):
#                 return False
#         # 2) 若在圆柱截面内部（rho2 < R^2），严格来说母线会穿过截面，
#         #    但这通常会在端面检查中更直观体现；这里先不判侧面为穿越。

#     # ----- 与端面（z=0, z=10）求交，且检查是否落在圆盘内 -----
#     # z = z_plane → t = (z_plane - Az) / dz
#     for z_plane in (Z_MIN, Z_MAX):
#         # 避免 dz ~ 0 的分母
#         mask_dz = np.abs(dz) > _TOL
#         if not np.any(mask_dz):
#             continue
#         t_plane = (z_plane - Az) / dz[mask_dz]
#         valid = t_plane >= -1e-12  # 只要在半直线上
#         if np.any(valid):
#             x_hit = Ax + t_plane[valid] * dx[mask_dz][valid]
#             y_hit = Ay + t_plane[valid] * dy[mask_dz][valid]
#             # 是否落在端面圆盘内
#             inside_disk = (x_hit * x_hit + (y_hit - CYL_CY) * (y_hit - CYL_CY)) <= (R_CYL * R_CYL + 1e-9)
#             if np.any(inside_disk):
#                 return False  # 与端面相交 → 非完全遮挡

#     # ----- 若侧面/端面均未被母线穿过，再检验圆柱中心方向是否在锥内 -----
#     Q = np.array([0.0, CYL_CY, 0.5 * (Z_MIN + Z_MAX)], dtype=float)  # (0,200,5)
#     AQ = Q - A
#     nAQ = float(np.linalg.norm(AQ))
#     if nAQ <= _TOL:
#         # A 就在 Q 上，视作在锥内（也一定被遮）
#         return True

#     AQ_hat = AQ / nAQ
#     # 角度条件： angle(AQ_hat, axis_hat) <= alpha  ⇔ dot >= cos(alpha)
#     dot_ = float(np.dot(AQ_hat, axis_hat))
#     if dot_ >= (cos_alpha - 1e-12):
#         return True  # 完全遮挡
#     else:
#         return False





































# utils/is_covered_backup.py

import math
import numpy as np

# ---- 定义目标圆柱体参数 ----
# [cite_start]根据题意：半径7m、高10m的圆柱形固定目标，下底面的圆心为(0,200,0) [cite: 6]
R_CYLINDER = 7.0   # 圆柱半径
H_CYLINDER = 10.0  # 圆柱高
# 圆柱轴线在 xy 平面的投影点 (即底面圆心)
POS_CYLINDER_XY = np.array([0.0, 200.0], dtype=float)

# ---- 烟幕和离散化参数 ----
r = 10.0     # 烟幕云团半径
K = 10000      # 锥面离散条数 (可根据精度要求调整)

def is_covered(A, B):
    """
    判断圆柱体目标是否被以B为中心的烟幕云团对导弹A完全遮蔽。

    参数:
    A: np.array([x, y, z])，导弹（视点）的位置
    B: np.array([x, y, z])，烟幕云团中心的位置

    返回:
    布尔值，表示目标是否被完全遮蔽
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    AB = B - A
    g = float(np.linalg.norm(AB))  # |AB| = g

    # 情况1: 导弹与烟幕中心重合或在烟幕内部，视线被完全阻挡
    if g <= r:
        return True

    # ---- 构建以导弹A为顶点，与烟幕球相切的视线锥 ----
    sin_alpha = r / g
    # 浮点数精度问题可能导致 1-sin_alpha**2 < 0，取max确保非负
    cos_alpha = math.sqrt(max(0.0, 1.0 - sin_alpha**2))
    alpha = float(np.arcsin(sin_alpha))

    # 锥轴的单位方向向量 u_hat (从烟幕中心B指向导弹A)
    u_hat = (A - B) / g
    
    # ---- 构造与 u_hat 正交的基向量 v_hat, w_hat (Gram-Schmidt) ----
    # (此部分沿用您原有的稳健写法)
    idx = int(np.argmin(np.abs(u_hat)))
    t = np.zeros(3, dtype=float)
    t[idx] = 1.0

    v = t - np.dot(t, u_hat) * u_hat
    nv = np.linalg.norm(v)
    if nv < 1e-12:
        t = np.array([1.0, 0.0, 0.0]) if idx != 0 else np.array([0.0, 1.0, 0.0])
        v = t - np.dot(t, u_hat) * u_hat
        nv = np.linalg.norm(v)
    v_hat = v / nv
    w_hat = np.cross(u_hat, v_hat)

    # ---- 核心逻辑1: 遍历K条母线，判断是否与目标圆柱相交 ----
    x_c, y_c = POS_CYLINDER_XY[0], POS_CYLINDER_XY[1]

    for n in range(K):
        phi_n = 2.0 * math.pi * n / K
        
        # 计算当前母线的单位方向向量 d_n
        d_n = (cos_alpha * u_hat +
               sin_alpha * (math.cos(phi_n) * v_hat + math.sin(phi_n) * w_hat))

        # 求解直线 A + t*d_n 与无限长圆柱 (x-xc)^2 + (y-yc)^2 = R_c^2 的交点
        # 整理为关于参数 t 的二次方程: a*t^2 + b*t + c = 0
        a_quad = d_n[0]**2 + d_n[1]**2
        b_quad = 2 * (d_n[0] * (A[0] - x_c) + d_n[1] * (A[1] - y_c))
        c_quad = (A[0] - x_c)**2 + (A[1] - y_c)**2 - R_CYLINDER**2

        delta = b_quad**2 - 4 * a_quad * c_quad

        if delta >= 0:  # delta >= 0 表示直线与无限圆柱有交点
            sqrt_delta = math.sqrt(delta)
            
            # 两个潜在解
            t1 = (-b_quad - sqrt_delta) / (2 * a_quad)
            t2 = (-b_quad + sqrt_delta) / (2 * a_quad)

            # 检查 t1 对应的交点
            # t > 0 意味着交点在导弹视线前方
            if t1 > 1e-9: # 使用一个小的容差避免t=0的情况
                z_intersect = A[2] + t1 * d_n[2]
                # 如果交点的z坐标在圆柱高度范围内，说明视线锥的边缘
                # "扫过"了目标实体，目标因此没有被完全遮挡
                if 0 < z_intersect < H_CYLINDER:
                    return False

            # 检查 t2 对应的交点
            if t2 > 1e-9:
                z_intersect = A[2] + t2 * d_n[2]
                if 0 < z_intersect < H_CYLINDER:
                    return False
    
    # ---- 核心逻辑2: 若所有母线均未与目标相交，则判断目标中心是否在锥内 ----
    # 目标中心点 Q
    Q = np.array([x_c, y_c, H_CYLINDER / 2.0], dtype=float)
    
    # 从导弹A到目标中心Q的向量
    vec_AQ = Q - A
    
    # 计算 vec_AQ 与锥轴 u_hat 的夹角(beta)的余弦值
    cos_beta = np.dot(vec_AQ, u_hat) / np.linalg.norm(vec_AQ)

    # beta 是 vec_AQ 和 u_hat 的夹角，alpha 是视线锥的半角。
    # 如果 beta < alpha，则点Q在锥内。
    # 因为cos在[0, pi]上是减函数，所以等价于 cos(beta) > cos(alpha)。
    if cos_beta > cos_alpha:
        # 目标中心在锥内，且锥的边界没有穿过目标，
        # 因此可以判定整个(凸)目标被完全遮挡。
        return True
    
    # 若母线未相交，且中心点在锥外，则目标完全未被遮挡。
    return False
