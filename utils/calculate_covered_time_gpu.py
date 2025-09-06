# utils/calculate_covered_time_gpu.py

import math
import torch
import numpy as np
from tqdm import tqdm

from utils.is_covered_gpu import is_covered, is_covered_batch

# ---- 常量（题面给定/默认） ----
SPEED_MISSILE = 300.0      # 导弹速率 (m/s)
INTERVAL = 0.01            # 仿真步长 (s)
G = 9.80                   # 重力加速度 (m/s^2)
SINK_SPEED = 3.0           # 云团下沉速度 (m/s)
CLOUD_EFFECTIVE = 20.0     # 起爆后有效遮蔽时长 (s)

# 转换为PyTorch tensor常量
ORIGIN_FAKE = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)


def _missile_pos(pos_init_missile, t, device='cuda'):
    """
    导弹按直线匀速飞向假目标（原点）。
    
    参数:
    pos_init_missile: tensor或ndarray，导弹初始位置
    t: float或tensor，时间点
    device: 计算设备
    
    返回:
    tensor，导弹在时间t的位置
    """
    # 确保输入是torch tensor
    if isinstance(pos_init_missile, np.ndarray):
        pos_init_missile = torch.tensor(pos_init_missile, dtype=torch.float32, device=device)
    elif isinstance(pos_init_missile, torch.Tensor) and pos_init_missile.device != device:
        pos_init_missile = pos_init_missile.to(device)
    
    # 处理批量时间t (可能是标量或向量)
    is_scalar = not isinstance(t, torch.Tensor)
    if is_scalar:
        t = torch.tensor([t], dtype=torch.float32, device=device)
    elif t.device != device:
        t = t.to(device)
    
    # 确保t是2D张量 [batch_size, 1]
    if t.dim() == 1:
        t = t.unsqueeze(1)
    
    # 计算单位方向向量 (从导弹指向原点)
    vhat = (ORIGIN_FAKE.to(device) - pos_init_missile)
    vhat = vhat / torch.norm(vhat)
    
    # 计算导弹位置: p0 + v*t
    # 广播t以匹配每个方向分量
    missile_positions = pos_init_missile + SPEED_MISSILE * t * vhat
    
    # 如果输入是标量，返回标量结果
    if is_scalar:
        return missile_positions[0]
    return missile_positions


def _drone_vec(v_drone, theta_drone, device='cuda'):
    """
    无人机等高度匀速飞行的速度向量（z 分量为 0）。
    
    参数:
    v_drone: float或tensor，无人机速度
    theta_drone: float或tensor，航向角度（弧度）
    device: 计算设备
    
    返回:
    tensor，无人机速度向量
    """
    # 检查是否为标量输入
    is_scalar = (not isinstance(v_drone, torch.Tensor) or v_drone.dim() == 0) and \
                (not isinstance(theta_drone, torch.Tensor) or theta_drone.dim() == 0)
    
    # 转换为tensor
    if not isinstance(v_drone, torch.Tensor):
        v_drone = torch.tensor(v_drone, dtype=torch.float32, device=device)
    elif v_drone.device != device:
        v_drone = v_drone.to(device)
        
    if not isinstance(theta_drone, torch.Tensor):
        theta_drone = torch.tensor(theta_drone, dtype=torch.float32, device=device)
    elif theta_drone.device != device:
        theta_drone = theta_drone.to(device)
    
    # 计算速度向量分量
    vx = v_drone * torch.cos(theta_drone)
    vy = v_drone * torch.sin(theta_drone)
    
    # 标量情况处理
    if is_scalar:
        return torch.tensor([float(vx), float(vy), 0.0], dtype=torch.float32, device=device)
    
    # 批量情况处理
    # 确保形状兼容
    if v_drone.dim() == 0 and theta_drone.dim() > 0:
        v_drone = v_drone.expand_as(theta_drone)
    if theta_drone.dim() == 0 and v_drone.dim() > 0:
        theta_drone = theta_drone.expand_as(v_drone)
    
    # 重新计算向量分量(因为可能已经展开了张量)
    vx = v_drone * torch.cos(theta_drone)
    vy = v_drone * torch.sin(theta_drone)
    
    # 确定批量大小(安全地获取)
    if v_drone.dim() > 0:
        batch_size = v_drone.shape[0]
    else:  # 这种情况理论上不会发生，但为了代码健壮性
        batch_size = theta_drone.shape[0]
    
    vz = torch.zeros(batch_size, device=device, dtype=torch.float32)
    return torch.stack([vx, vy, vz], dim=-1)


def _decoy_state_at_explosion(pos_init_drone, v_drone_vec, t_drop, t_delay, device='cuda'):
    """
    计算起爆瞬间云团中心（即干扰弹位置）。
    干扰弹在投放后仅受重力下落，水平速度保持与无人机一致，初始竖直速度取 0。
    
    参数:
    pos_init_drone: tensor或ndarray，无人机初始位置
    v_drone_vec: tensor，无人机速度向量
    t_drop: float或tensor，投放时间
    t_delay: float或tensor，延迟起爆时间
    device: 计算设备
    
    返回:
    tensor，起爆位置
    """
    # 确保输入是torch tensor
    if isinstance(pos_init_drone, np.ndarray):
        pos_init_drone = torch.tensor(pos_init_drone, dtype=torch.float32, device=device)
    elif isinstance(pos_init_drone, torch.Tensor) and pos_init_drone.device != device:
        pos_init_drone = pos_init_drone.to(device)
    
    if isinstance(v_drone_vec, np.ndarray):
        v_drone_vec = torch.tensor(v_drone_vec, dtype=torch.float32, device=device)
    elif isinstance(v_drone_vec, torch.Tensor) and v_drone_vec.device != device:
        v_drone_vec = v_drone_vec.to(device)
    
    # 标量转tensor
    if not isinstance(t_drop, torch.Tensor):
        t_drop = torch.tensor(t_drop, dtype=torch.float32, device=device)
    elif t_drop.device != device:
        t_drop = t_drop.to(device)
        
    if not isinstance(t_delay, torch.Tensor):
        t_delay = torch.tensor(t_delay, dtype=torch.float32, device=device)
    elif t_delay.device != device:
        t_delay = t_delay.to(device)
    
    # 计算投放点
    p_drop = pos_init_drone + v_drone_vec * t_drop
    
    # 从投放到起爆的抛体位移（水平匀速 + 竖直位移）
    horiz = v_drone_vec * t_delay
    
    # 重力导致的竖直位移
    if v_drone_vec.dim() > 1:
        # 批处理情况
        batch_size = v_drone_vec.shape[0]
        vert = torch.zeros_like(v_drone_vec)
        vert[:, 2] = -0.5 * G * t_delay * t_delay
    else:
        # 单个向量情况
        vert = torch.tensor([0.0, 0.0, -0.5 * G * t_delay * t_delay], 
                           dtype=torch.float32, device=device)
    
    return p_drop + horiz + vert


def calculate_covered_time(pos_init_missile, pos_init_drone,
                          v_drone, theta_drone,
                          t_drone_fly, t_decoy_delay,
                          device='cuda', show_progress=True):
    """
    计算导弹被烟幕云团遮挡的时间（秒）- GPU加速版本
    
    参数:
    pos_init_missile: ndarray或tensor，导弹初始位置
    pos_init_drone: ndarray或tensor，无人机初始位置
    v_drone: float，无人机速度
    theta_drone: float，无人机航向（弧度）
    t_drone_fly: float，投放时间
    t_decoy_delay: float，起爆延迟
    device: 计算设备
    show_progress: 是否显示进度条
    
    返回:
    float，总的有效遮挡时间
    """
    # 转换为torch tensor
    if isinstance(pos_init_missile, np.ndarray):
        pos_init_missile = torch.tensor(pos_init_missile, dtype=torch.float32, device=device)
    elif isinstance(pos_init_missile, torch.Tensor) and pos_init_missile.device != device:
        pos_init_missile = pos_init_missile.to(device)
    
    if isinstance(pos_init_drone, np.ndarray):
        pos_init_drone = torch.tensor(pos_init_drone, dtype=torch.float32, device=device)
    elif isinstance(pos_init_drone, torch.Tensor) and pos_init_drone.device != device:
        pos_init_drone = pos_init_drone.to(device)

    # 关键时间点
    t_drop = float(t_drone_fly)
    t_expl = t_drop + float(t_decoy_delay)
    t_end = t_expl + CLOUD_EFFECTIVE

    # 无人机速度向量（等高度）
    v_d = _drone_vec(v_drone, theta_drone, device=device)

    # 起爆位置（云团初始中心）
    B_expl = _decoy_state_at_explosion(pos_init_drone, v_d, t_drop, t_decoy_delay, device=device)

    # 时间轴
    times = torch.arange(0.0, t_end + 1e-9, INTERVAL, dtype=torch.float32, device=device)
    iterator = tqdm(times, desc="Simulating (GPU)", unit="step") if show_progress else times

    # 遮挡累计
    covered_time = 0.0

    # 批处理大小（根据GPU内存调整）
    batch_size = 256
    num_batches = (len(times) + batch_size - 1) // batch_size

    # 导弹位置预计算
    A_all = _missile_pos(pos_init_missile, times, device=device)

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(times))
        
        # 当前批次的时间点和导弹位置
        batch_times = times[start_idx:end_idx]
        A_batch = A_all[start_idx:end_idx]
        
        # 遮挡判断
        mask = batch_times >= t_expl
        if torch.any(mask):
            valid_idx = torch.where(mask)[0]
            valid_times = batch_times[valid_idx]
            valid_A = A_batch[valid_idx]
            
            # 计算每个时间点的云团位置
            dt = valid_times - t_expl
            sink_offset = torch.zeros((len(valid_idx), 3), device=device)
            sink_offset[:, 2] = -SINK_SPEED * dt
            
            B_t = B_expl + sink_offset
            
            # 批量判断是否被遮挡
            for i in range(len(valid_idx)):
                if is_covered(valid_A[i], B_t[i], device=device):
                    covered_time += INTERVAL

    return covered_time


def calculate_covered_time_batch(pos_init_missile_batch, pos_init_drone_batch,
                               v_drone_batch, theta_drone_batch,
                               t_drone_fly_batch, t_decoy_delay_batch,
                               device='cuda', show_progress=True):
    """
    批量计算多组参数的遮挡时间（适用于优化算法内部）
    
    参数:
    所有参数都是批量形式 [batch_size, ...]
    
    返回:
    tensor，每组参数的遮挡时间
    """
    batch_size = len(v_drone_batch)
    results = torch.zeros(batch_size, dtype=torch.float32, device=device)
    
    # 优化：将时间线计算和导弹轨迹预计算一次
    for i in range(batch_size):
        pos_init_missile = pos_init_missile_batch[i]
        pos_init_drone = pos_init_drone_batch[i]
        v_drone = v_drone_batch[i]
        theta_drone = theta_drone_batch[i]
        t_drone_fly = t_drone_fly_batch[i]
        t_decoy_delay = t_decoy_delay_batch[i]
        
        results[i] = calculate_covered_time(
            pos_init_missile, pos_init_drone,
            v_drone, theta_drone,
            t_drone_fly, t_decoy_delay,
            device=device, show_progress=False
        )
    
    return results
