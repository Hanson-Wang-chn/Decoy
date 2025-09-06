# q4_gpu.py

import torch
import numpy as np
import math
import time
from tqdm import tqdm
import argparse

# 复用GPU优化版本的物理模型函数与常量
from utils.calculate_covered_time_gpu import (
    _missile_pos,              # 导弹轨迹
    _drone_vec,                # 无人机速度向量
    _decoy_state_at_explosion, # 起爆位置
    INTERVAL,
    G,
    SINK_SPEED,
    CLOUD_EFFECTIVE,
    SPEED_MISSILE,
    ORIGIN_FAKE
)
from utils.is_covered_gpu import is_covered, is_covered_batch


def parse_args():
    parser = argparse.ArgumentParser(description="问题4 GPU优化版")
    parser.add_argument('--use_cpu', action='store_true', help='强制使用CPU')
    return parser.parse_args()

# 解析命令行参数
args = parse_args()

# 配置是否使用GPU - 根据命令行参数决定
USE_GPU = not args.use_cpu

# 依次检查不同的硬件加速选项
if USE_GPU and torch.cuda.is_available():
    DEVICE = 'cuda'
    print(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
elif USE_GPU and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    DEVICE = 'mps'
    print("Using MPS (Metal Performance Shaders) on Mac")
else:
    DEVICE = 'cpu'
    if USE_GPU:
        print("GPU加速不可用，回退到CPU模式")
    else:
        print("使用CPU模式")

# ===============================
# 根据题意定义的初始条件（问题 4）
# ===============================

# 来袭导弹 M1 的初始位置
POS_INIT_MISSILE_M1 = torch.tensor([20000.0,   0.0, 2000.0], dtype=torch.float32, device=DEVICE)

# 三架无人机 FY1、FY2、FY3 的初始位置
POS_INIT_DRONE_FY1  = torch.tensor([17800.0,    0.0, 1800.0], dtype=torch.float32, device=DEVICE)
POS_INIT_DRONE_FY2  = torch.tensor([12000.0, 1400.0, 1400.0], dtype=torch.float32, device=DEVICE)
POS_INIT_DRONE_FY3  = torch.tensor([ 6000.0,-3000.0,  700.0], dtype=torch.float32, device=DEVICE)
POS_INIT_DRONES = [POS_INIT_DRONE_FY1, POS_INIT_DRONE_FY2, POS_INIT_DRONE_FY3]

# ===============================
# 优化算法（WOA）参数（与 Q2 风格一致）
# ===============================
POPULATION_SIZE = 600
MAX_ITERATIONS  = 30
EARLY_STOP_PATIENCE = 8

# 对于每架无人机需要优化的 4 个变量：
# [v_drone, theta_drone, t_drone_fly, t_decoy_delay]
# 速度：70~140 m/s；航向：0~pi；投放时间：0~60 s；起爆延迟：0~20 s
LB_1 = torch.tensor([70.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=DEVICE)
UB_1 = torch.tensor([140.0, math.pi, 60.0, 20.0], dtype=torch.float32, device=DEVICE)

LB_2 = torch.tensor([70.0, math.pi, 0.0, 0.0], dtype=torch.float32, device=DEVICE)
UB_2 = torch.tensor([140.0, 2 * math.pi, 60.0, 20.0], dtype=torch.float32, device=DEVICE)

LB_3 = torch.tensor([70.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=DEVICE)
UB_3 = torch.tensor([140.0, 5 / 6 * math.pi, 60.0, 20.0], dtype=torch.float32, device=DEVICE)

# 三架无人机，总维度 12
LOWER_BOUNDS = torch.cat([LB_1, LB_2, LB_3])
UPPER_BOUNDS = torch.cat([UB_1, UB_2, UB_3])
DIMENSIONS = LOWER_BOUNDS.size(0)


# ===============================
# 多无人机：联合遮蔽时间计算 - GPU优化版
# ===============================
def calculate_covered_time_3drones(pos_init_missile, drone_triplets,
                                 show_progress=False, device=DEVICE):
    """
    GPU优化版：计算三枚云团对导弹的联合有效遮蔽时长（秒），并给出每枚云团的"单独遮蔽时长"。

    参数:
    pos_init_missile: tensor([x,y,z])，导弹初始位置（如 M1）
    drone_triplets:   list，长度为 3。
                      每个元素为 (pos_init_drone, v_drone, theta_drone, t_drone_fly, t_decoy_delay)
    show_progress:    bool，是否显示仿真进度条
    device:           计算设备

    返回:
      total_union_time: float，三枚云团"联合"（并集）有效遮蔽总时长
      per_decoy_time:   list[float]，长度 3，每枚云团"单独"遮蔽时长（不去重）
      details:          dict，包含每枚云团的投放/起爆时间与坐标等信息
    """
    # 确保输入是torch tensor
    if not isinstance(pos_init_missile, torch.Tensor):
        pos_init_missile = torch.tensor(pos_init_missile, dtype=torch.float32, device=device)
    elif pos_init_missile.device != device:
        pos_init_missile = pos_init_missile.to(device)

    # 关键时间点与初始几何
    t_drop = torch.zeros(3, dtype=torch.float32, device=device)
    t_expl = torch.zeros(3, dtype=torch.float32, device=device)
    t_end = torch.zeros(3, dtype=torch.float32, device=device)
    v_d_vec = []
    B_expl = []
    p_drop = []

    for i, (pos_init_drone, v_drone, theta_drone, t_drone_fly, t_decoy_delay) in enumerate(drone_triplets):
        # 确保输入是torch tensor
        if not isinstance(pos_init_drone, torch.Tensor):
            pos_init_drone = torch.tensor(pos_init_drone, dtype=torch.float32, device=device)
        elif pos_init_drone.device != device:
            pos_init_drone = pos_init_drone.to(device)
            
        # 无人机速度向量
        v_d = _drone_vec(float(v_drone), float(theta_drone), device=device)
        v_d_vec.append(v_d)

        # 关键时间点
        t_drop[i] = float(t_drone_fly)
        t_expl[i] = float(t_drone_fly) + float(t_decoy_delay)
        t_end[i] = t_expl[i] + CLOUD_EFFECTIVE

        # 投放与起爆坐标
        p_drop_i = pos_init_drone + v_d * t_drop[i]
        p_drop.append(p_drop_i)
        B_expl_i = _decoy_state_at_explosion(pos_init_drone, v_d, t_drop[i], float(t_decoy_delay), device=device)
        B_expl.append(B_expl_i)

    # 导弹飞抵假目标（原点）的时间，上限不必超过此刻
    t_hit_origin = torch.norm(ORIGIN_FAKE.to(device) - pos_init_missile) / SPEED_MISSILE

    # 全局仿真终止时间：三枚云团有效期末尾 与 导弹命中假目标时间 取较小者
    t_sim_end = float(min(max(t_end.max().item(), 0.0), t_hit_origin.item()))

    # 预计算时间轴
    times = torch.arange(0.0, t_sim_end + 1e-9, INTERVAL, dtype=torch.float32, device=device)
    if show_progress:
        print(f"Simulating (GPU Vectorized)... {len(times)} steps.")

    # 预计算所有时间点的导弹位置
    A_t = _missile_pos(pos_init_missile, times, device=device)

    # 为每个干扰弹创建遮挡标记矩阵 [T, 3]
    occluded_matrix = torch.zeros(len(times), 3, dtype=torch.bool, device=device)

    # 向量化计算所有干扰弹的遮挡情况
    # [1, 3]
    t_expl_exp = t_expl.unsqueeze(0)
    # [T, 1]
    times_exp = times.unsqueeze(1)
    
    # [T, 3] bool, 标记每个时间点，每个干扰弹是否在其有效期内
    valid_time_mask = (times_exp >= t_expl_exp) & (times_exp < t_expl_exp + CLOUD_EFFECTIVE)
    
    # 获取所有有效时间点的索引 [N_valid, 2], N_valid是所有有效(t, decoy)对的数量
    valid_indices = torch.where(valid_time_mask)
    time_indices, decoy_indices = valid_indices
    
    if time_indices.numel() > 0:
        # 提取所有有效的 A, B, t
        valid_A = A_t[time_indices]  # [N_valid, 3]
        valid_t = times[time_indices]  # [N_valid]
        
        # 处理每个干扰弹
        is_covered_results = torch.zeros(len(time_indices), dtype=torch.bool, device=device)
        
        for i in range(3):
            mask = decoy_indices == i
            if not torch.any(mask):
                continue
                
            idx = torch.where(mask)[0]
            t_vals = valid_t[idx]
            A_vals = valid_A[idx]
            
            # 计算云团位置
            dt = t_vals - t_expl[i]
            sink_offset = torch.zeros_like(A_vals)
            sink_offset[:, 2] = -SINK_SPEED * dt
            B_vals = B_expl[i] + sink_offset
            
            # 判断是否被遮挡
            covered = is_covered(A_vals, B_vals, device=device)
            is_covered_results[idx] = covered
        
        # 将结果填回遮挡矩阵
        occluded_matrix[time_indices, decoy_indices] = is_covered_results

    # 计算每个干扰弹的单独遮蔽时长
    per_decoy = torch.sum(occluded_matrix, dim=0) * INTERVAL
    
    # 计算并集遮蔽时长
    any_occluded = torch.any(occluded_matrix, dim=1)
    total_union = torch.sum(any_occluded).item() * INTERVAL

    # 细节汇总（用于打印和写表）
    details = {
        "t_drop": t_drop.cpu().numpy(),
        "t_expl": t_expl.cpu().numpy(),
        "t_end": t_end.cpu().numpy(),
        "p_drop": [p.cpu().numpy() for p in p_drop],
        "p_expl": [B.cpu().numpy() for B in B_expl],  # 起爆瞬间中心
    }

    return total_union, per_decoy.cpu().numpy().tolist(), details


# ===============================
# 目标函数 - GPU优化版
# ===============================
def _unpack_params_12(params, device=DEVICE):
    """
    将长度 12 的参数拆成 3 组 (v, theta, t_fly, t_delay)
    """
    # 确保输入是torch tensor
    if not isinstance(params, torch.Tensor):
        p = torch.tensor(params, dtype=torch.float32, device=device)
    else:
        p = params
    
    assert p.size(0) == 12
    return p.reshape(3, 4)  # 行：无人机；列：4个参数


def objective_function(params, device=DEVICE):
    """
    GPU优化版：WOA 的适应度函数（最小化）。
    我们要最大化联合遮蔽时长 -> 返回其相反数。
    """
    # 确保输入是torch tensor
    if not isinstance(params, torch.Tensor):
        params = torch.tensor(params, dtype=torch.float32, device=device)
    elif params.device != device:
        params = params.to(device)
        
    triples = _unpack_params_12(params, device=device)
    drone_triplets = [
        (POS_INIT_DRONE_FY1, triples[0, 0], triples[0, 1], triples[0, 2], triples[0, 3]),
        (POS_INIT_DRONE_FY2, triples[1, 0], triples[1, 1], triples[1, 2], triples[1, 3]),
        (POS_INIT_DRONE_FY3, triples[2, 0], triples[2, 1], triples[2, 2], triples[2, 3]),
    ]
    total_union, _, _ = calculate_covered_time_3drones(
        pos_init_missile=POS_INIT_MISSILE_M1,
        drone_triplets=drone_triplets,
        show_progress=False,  # 优化内循环关闭进度条
        device=device
    )
    return -float(total_union)


# ===============================
# 鲸鱼优化算法 - GPU优化版
# ===============================
class WhaleOptimizationAlgorithm:
    """
    GPU优化版：鲸鱼优化算法 (WOA)
    """
    def __init__(self, obj_func, lower_bounds, upper_bounds, dim, pop_size, max_iter, 
                early_stop_patience=None, device=DEVICE):
        self.obj_func = obj_func
        self.device = device
        
        # 确保边界是tensor并在正确设备上
        if not isinstance(lower_bounds, torch.Tensor):
            self.lb = torch.tensor(lower_bounds, dtype=torch.float32, device=device)
        else:
            self.lb = lower_bounds.to(device)
            
        if not isinstance(upper_bounds, torch.Tensor):
            self.ub = torch.tensor(upper_bounds, dtype=torch.float32, device=device)
        else:
            self.ub = upper_bounds.to(device)
        
        self.dim = dim
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.early_stop_patience = early_stop_patience

        # 初始化领导者（当前最优解）
        self.leader_pos = torch.zeros(dim, dtype=torch.float32, device=device)
        self.leader_score = float('inf')
        
        # 早停相关变量
        self.no_improve_counter = 0

        # 初始化所有搜索智能体（鲸鱼）的位置
        self.positions = torch.rand(pop_size, dim, dtype=torch.float32, device=device) * (self.ub - self.lb) + self.lb

    def optimize(self):
        """
        运行 WOA 优化流程 - GPU优化版本
        """
        print(f"启动鲸鱼优化算法（GPU加速，设备：{self.device}）...")

        for t in tqdm(range(self.max_iter), desc="WOA 优化进度"):
            # 记录上一次的最优分数用于比较
            prev_best_score = self.leader_score
            
            # 批量评估所有鲸鱼
            self.positions = torch.clamp(self.positions, self.lb, self.ub)
            
            # 批量评估（根据GPU内存分批处理）
            batch_size = 5  # 可调整
            num_batches = (self.pop_size + batch_size - 1) // batch_size
            
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, self.pop_size)
                
                batch_positions = self.positions[start_idx:end_idx]
                
                # 并行评估此批次
                for i in range(end_idx - start_idx):
                    fitness = self.obj_func(batch_positions[i])
                    
                    if fitness < self.leader_score:
                        self.leader_score = fitness
                        self.leader_pos = batch_positions[i].clone()
            
            # 早停检查
            if self.early_stop_patience is not None:
                if self.leader_score >= prev_best_score:
                    self.no_improve_counter += 1
                    if self.no_improve_counter >= self.early_stop_patience:
                        print(f"\n早停触发: 连续 {self.early_stop_patience} 次迭代没有性能提升")
                        break
                else:
                    self.no_improve_counter = 0

            # a 从 2 线性递减到 0
            a = 2 - t * (2 / self.max_iter)

            # 批量更新所有鲸鱼位置
            r1 = torch.rand(self.pop_size, 1, device=self.device)
            r2 = torch.rand(self.pop_size, 1, device=self.device)
            A = 2 * a * r1 - a
            C = 2 * r2
            b = 1
            l = (a - 1) * torch.rand(self.pop_size, 1, device=self.device) + 1
            p = torch.rand(self.pop_size, 1, device=self.device)

            # 使用掩码操作实现条件选择
            explore_mask = (p < 0.5) & (torch.abs(A) >= 1)
            exploit_encircle_mask = (p < 0.5) & (torch.abs(A) < 1)
            exploit_spiral_mask = p >= 0.5

            # 探索阶段
            if torch.any(explore_mask):
                explore_indices = torch.where(explore_mask.squeeze())[0]
                
                for idx in explore_indices:
                    # 随机选择领导者
                    rand_leader_index = torch.randint(0, self.pop_size, (1,), device=self.device)
                    X_rand = self.positions[rand_leader_index].squeeze()
                    
                    # 更新位置
                    D_X_rand = torch.abs(C[idx] * X_rand - self.positions[idx])
                    self.positions[idx] = X_rand - A[idx] * D_X_rand

            # 利用阶段 - 包围猎物
            if torch.any(exploit_encircle_mask):
                encircle_indices = torch.where(exploit_encircle_mask.squeeze())[0]
                
                for idx in encircle_indices:
                    D_Leader = torch.abs(C[idx] * self.leader_pos - self.positions[idx])
                    self.positions[idx] = self.leader_pos - A[idx] * D_Leader

            # 利用阶段 - 螺旋攻击
            if torch.any(exploit_spiral_mask):
                spiral_indices = torch.where(exploit_spiral_mask.squeeze())[0]
                
                for idx in spiral_indices:
                    distance_to_leader = torch.abs(self.leader_pos - self.positions[idx])
                    self.positions[idx] = distance_to_leader * torch.exp(b * l[idx]) * torch.cos(l[idx] * 2 * torch.pi) + self.leader_pos

            if (t + 1) % 3 == 0:
                print(f"迭代 {t + 1}/{self.max_iter}，当前最优联合遮蔽时间: {-self.leader_score:.3f} s")
                if self.early_stop_patience is not None:
                    print(f"连续未改进计数: {self.no_improve_counter}/{self.early_stop_patience}")

        return self.leader_pos.cpu().numpy(), -self.leader_score


# ===============================
# 主程序
# ===============================
def main():
    """
    问题4：三架无人机联合对一枚导弹实施干扰 - GPU优化版
    """
    # 将优化时的目标函数静默版本
    def objective_quiet(params):
        return objective_function(params, device=DEVICE)

    # 初始化WOA优化器
    woa = WhaleOptimizationAlgorithm(
        obj_func=objective_quiet,
        lower_bounds=LOWER_BOUNDS,
        upper_bounds=UPPER_BOUNDS,
        dim=DIMENSIONS,
        pop_size=POPULATION_SIZE,
        max_iter=MAX_ITERATIONS,
        early_stop_patience=EARLY_STOP_PATIENCE,
        device=DEVICE
    )

    # 运行优化
    print(f"开始优化 (使用{DEVICE})...")
    start = time.time()
    best_params, best_union_time = woa.optimize()
    end = time.time()

    print("\n" + "-"*60)
    print(f"问题4 优化完成（3 架无人机 × 1 弹/机，GPU优化版本，设备：{DEVICE}）")
    print("优化算法：鲸鱼优化算法 (WOA)")
    print(f"总计用时: {end - start:.2f} 秒")
    print("-"*60)

    # 结构化参数
    best_params_tensor = torch.tensor(best_params, dtype=torch.float32, device=DEVICE)
    triples = _unpack_params_12(best_params_tensor, device=DEVICE)

    # 使用最优解开启一次"有进度条"的最终仿真，便于核验与展示
    drone_triplets = [
        (POS_INIT_DRONE_FY1, triples[0, 0].item(), triples[0, 1].item(), triples[0, 2].item(), triples[0, 3].item()),
        (POS_INIT_DRONE_FY2, triples[1, 0].item(), triples[1, 1].item(), triples[1, 2].item(), triples[1, 3].item()),
        (POS_INIT_DRONE_FY3, triples[2, 0].item(), triples[2, 1].item(), triples[2, 2].item(), triples[2, 3].item()),
    ]
    
    print("\n执行最终验证 (使用精细步长)...")
    total_union, per_decoy, details = calculate_covered_time_3drones(
        pos_init_missile=POS_INIT_MISSILE_M1,
        drone_triplets=drone_triplets,
        show_progress=True,
        device=DEVICE
    )

    # === 打印最优参数 ===
    labels = ["FY1", "FY2", "FY3"]
    for i, lab in enumerate(labels):
        v_opt     = float(triples[i, 0].item())
        theta_opt = float(triples[i, 1].item())
        t_drop    = float(details["t_drop"][i])
        t_delay   = float(details["t_expl"][i] - details["t_drop"][i])
        p_drop    = np.array(details["p_drop"][i], dtype=float)
        p_expl    = np.array(details["p_expl"][i], dtype=float)

        print(f"\n[{lab}] 最优参数：")
        print(f"  无人机速度 v_drone: {v_opt:.4f} m/s")
        print(f"  无人机航向 theta_drone: {theta_opt:.4f} rad ({math.degrees(theta_opt):.2f} 度)")
        print(f"  投放时间 t_drone_fly: {t_drop:.4f} s")
        print(f"  起爆延迟 t_decoy_delay: {t_delay:.4f} s")

        print(f"  投放点坐标:  x={p_drop[0]:.2f}  y={p_drop[1]:.2f}  z={p_drop[2]:.2f}")
        print(f"  起爆点坐标:  x={p_expl[0]:.2f}  y={p_expl[1]:.2f}  z={p_expl[2]:.2f}")
        print(f"  该干扰弹单独遮蔽时长: {per_decoy[i]:.4f} s")

    print("\n" + "="*60)
    print(f"三枚云团联合（并集）有效遮蔽总时长: {total_union:.4f} s")
    print("="*60)


if __name__ == "__main__":
    main()
