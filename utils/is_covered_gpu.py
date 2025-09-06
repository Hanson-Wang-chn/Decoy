import torch
import numpy as np

# 烟幕云团参数
R_DECOY = 10.0  # 半径

# ---- 圆柱参数：x^2 + (y - 200)^2 = 49,  z ∈ [0, 10] ----
R_CYL = 7.0
CYL_CY = 200.0
Z_MIN, Z_MAX = 0.0, 10.0

# 采样密度
SIDE_THETA = 8     # 侧面角向采样
SIDE_Z     = 10    # 侧面高度采样
CAP_THETA  = 8     # 上/下盖角向采样
CAP_RADIAL = 10    # 上/下盖径向采样

_TOL = 1e-9

# 保存预计算的采样点
_cylinder_samples = {}

def _sample_cylinder_surface(device):
    """离散侧面 + 上下底，返回 (N,3)"""
    # 侧面
    thetas = torch.linspace(0.0, 2.0*torch.pi, SIDE_THETA, device=device)
    zs = torch.linspace(Z_MIN, Z_MAX, SIDE_Z, device=device)
    T, Z = torch.meshgrid(thetas, zs, indexing='xy')
    Xs = R_CYL * torch.cos(T)
    Ys = CYL_CY + R_CYL * torch.sin(T)
    P_side = torch.stack([Xs, Ys, Z], dim=-1).reshape(-1, 3)

    # 上/下底（极坐标均匀网格）
    thetas_c = torch.linspace(0.0, 2.0*torch.pi, CAP_THETA, device=device)
    rs = torch.linspace(0.0, R_CYL, CAP_RADIAL, device=device)
    Tc, Rc = torch.meshgrid(thetas_c, rs, indexing='xy')
    Xc = Rc * torch.cos(Tc)
    Yc = CYL_CY + Rc * torch.sin(Tc)

    P_top = torch.stack([Xc, Yc, torch.full_like(Xc, Z_MAX)], dim=-1).reshape(-1, 3)
    return torch.cat([P_side, P_top], dim=0)

def is_covered(A, B, device='cuda'):
    """
    判断导弹是否在烟幕云团范围内

    参数:
    A: array/tensor([x, y, z])，导弹（点A）的位置
    B: array/tensor([x, y, z])，烟幕云团中心（点B）的位置
    device: 计算设备，'cuda'或'cpu'

    返回:
    布尔值，表示导弹是否在烟幕云团范围内
    """
    # 转换输入到torch.tensor并指定设备
    if isinstance(A, np.ndarray):
        A = torch.tensor(A, dtype=torch.float32, device=device)
    if isinstance(B, np.ndarray):
        B = torch.tensor(B, dtype=torch.float32, device=device)
    
    # 使用缓存的采样点，若没有则创建
    if device not in _cylinder_samples:
        _cylinder_samples[device] = _sample_cylinder_surface(device)
    
    P = _cylinder_samples[device]

    BA = B - A
    g = torch.norm(BA)
    
    # A在球内或与B重合，显然遮挡
    if g <= R_DECOY + _TOL:
        return True

    # 锥参数计算
    sin_alpha = torch.clamp(R_DECOY / g, -1.0, 1.0)
    alpha = torch.asin(sin_alpha)
    cos_alpha = torch.cos(alpha)
    a_hat = BA / g

    # 向量化计算所有采样点到导弹的方向向量
    V = P - A.unsqueeze(0)  # [n_samples, 3]
    dist = torch.norm(V, dim=1)
    valid = dist > 1e-12     # 避免除零问题
    
    if not torch.any(valid):
        return False

    # 计算单位方向向量
    d_hat = torch.zeros_like(V, device=device)
    d_hat[valid] = V[valid] / dist[valid].unsqueeze(1)

    # 检查(1): 所有采样点是否都在锥内: <d_hat, a_hat> >= cos(alpha)
    inside_cone = (torch.matmul(d_hat, a_hat)) >= (cos_alpha - 1e-12)
    
    if not torch.all(inside_cone[valid]):
        return False

    # 检查(2): 球是否先于柱体相交: |A + t*d_hat - B|^2 = R_DECOY^2
    dot = torch.sum(d_hat * BA, dim=1)  # [n_samples]
    disc = dot**2 - (g**2 - R_DECOY**2)
    disc = torch.clamp(disc, min=0.0)  # 数值稳健
    t_hit = dot - torch.sqrt(disc)    # 最近正根

    # 检查命中是否发生在到达柱体之前
    occluded = (t_hit >= -1e-9) & (t_hit <= dist + 1e-9)

    return bool(torch.all(occluded[valid]).item())

# 批处理版本
def is_covered_batch(A_batch, B_batch, device='cuda'):
    """
    批量判断多个导弹位置和云团组合是否遮挡
    
    参数:
    A_batch: tensor[batch_size, 3], 多个导弹位置
    B_batch: tensor[batch_size, 3], 多个云团位置
    
    返回:
    tensor[batch_size], 布尔值张量
    """
    batch_size = A_batch.shape[0]
    results = torch.zeros(batch_size, dtype=torch.bool, device=device)
    
    # 由于is_covered函数中存在条件执行路径，逐个处理更为高效
    for i in range(batch_size):
        results[i] = is_covered(A_batch[i], B_batch[i], device)
    
    return results
