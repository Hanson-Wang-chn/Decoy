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
    判断导弹是否在烟幕云团范围内 (向量化版本)

    参数:
    A: tensor([N, 3])，N个导弹（点A）的位置
    B: tensor([N, 3])，N个烟幕云团中心（点B）的位置
    device: 计算设备

    返回:
    tensor([N])，布尔值，表示每个对应的A、B对是否遮挡
    """
    # 确保输入是torch tensor
    if not isinstance(A, torch.Tensor):
        A = torch.tensor(A, dtype=torch.float32, device=device)
    if not isinstance(B, torch.Tensor):
        B = torch.tensor(B, dtype=torch.float32, device=device)
    
    if A.dim() == 1: A = A.unsqueeze(0)
    if B.dim() == 1: B = B.unsqueeze(0)
    
    batch_size = A.shape[0]
    results = torch.zeros(batch_size, dtype=torch.bool, device=device)

    # 使用缓存的采样点
    if device not in _cylinder_samples:
        _cylinder_samples[device] = _sample_cylinder_surface(device)
    P = _cylinder_samples[device] # [n_samples, 3]
    
    BA = B - A # [N, 3]
    g = torch.norm(BA, dim=1) # [N]

    # 情况1: A在球内或与B重合，显然遮挡
    in_sphere_mask = g <= R_DECOY + _TOL
    results[in_sphere_mask] = True
    
    # 只处理不在球内的点
    process_mask = ~in_sphere_mask
    if not torch.any(process_mask):
        return results

    A_proc, B_proc, BA_proc, g_proc = A[process_mask], B[process_mask], BA[process_mask], g[process_mask]
    
    # 锥参数
    sin_alpha = torch.clamp(R_DECOY / g_proc, -1.0, 1.0)
    cos_alpha = torch.cos(torch.asin(sin_alpha)) # [N_proc]
    a_hat = BA_proc / g_proc.unsqueeze(1) # [N_proc, 3]

    # 向量化计算
    # V: [N_proc, n_samples, 3], P: [n_samples, 3], A_proc: [N_proc, 3]
    V = P.unsqueeze(0) - A_proc.unsqueeze(1)
    dist = torch.norm(V, dim=2) # [N_proc, n_samples]
    
    valid = dist > 1e-12
    
    d_hat = torch.zeros_like(V, device=device)
    d_hat[valid] = V[valid] / dist[valid].unsqueeze(1)

    # (1) 在锥内: <d_hat, a_hat> >= cos(alpha)
    # a_hat: [N_proc, 3] -> [N_proc, 1, 3]
    # d_hat: [N_proc, n_samples, 3]
    # dot product -> [N_proc, n_samples]
    inside_cone_dot = torch.sum(d_hat * a_hat.unsqueeze(1), dim=2)
    inside_cone = inside_cone_dot >= (cos_alpha.unsqueeze(1) - 1e-12)
    
    # (2) 球先于柱
    # BA_proc: [N_proc, 3] -> [N_proc, 1, 3]
    dot = torch.sum(d_hat * BA_proc.unsqueeze(1), dim=2) # [N_proc, n_samples]
    g_sq = g_proc**2 # [N_proc]
    disc = dot**2 - (g_sq.unsqueeze(1) - R_DECOY**2)
    disc = torch.clamp(disc, min=0.0)
    t_hit = dot - torch.sqrt(disc)

    occluded = (t_hit >= -1e-9) & (t_hit <= dist + 1e-9)
    
    # 对每个批次内的所有采样点进行判断
    # 必须所有采样点都满足条件才算遮挡
    final_occluded_mask = torch.all( (inside_cone & occluded) | ~valid, dim=1)
    
    # 更新最终结果
    results[process_mask] = final_occluded_mask
    
    return results

# 批处理版本 (现在只是一个别名，因为is_covered本身就是批处理的)
def is_covered_batch(A_batch, B_batch, device='cuda'):
    """
    批量判断多个导弹位置和云团组合是否遮挡
    
    参数:
    A_batch: tensor[batch_size, 3], 多个导弹位置
    B_batch: tensor[batch_size, 3], 多个云团位置
    
    返回:
    tensor[batch_size], 布尔值张量
    """
    return is_covered(A_batch, B_batch, device=device)
