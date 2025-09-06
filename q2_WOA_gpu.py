# q2_WOA_gpu.py

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
    parser = argparse.ArgumentParser(description="问题2 GPU优化版")
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

# ---- 根据题目描述定义的常量 ----
# 导弹 M1 的初始位置 [cite: 6]
POS_INIT_MISSILE_M1 = torch.tensor([20000.0, 0.0, 2000.0], dtype=torch.float32, device=DEVICE)
# 无人机 FY1 的初始位置 [cite: 6]
POS_INIT_DRONE_FY1 = torch.tensor([17800.0, 0.0, 1800.0], dtype=torch.float32, device=DEVICE)

# ---- 优化算法配置 ----
POPULATION_SIZE = 200  # 搜索智能体（鲸鱼）的数量
MAX_ITERATIONS = 30    # 最大迭代次数

# 参数边界 (搜索空间)
# 我们需要优化的四个变量：
# [无人机速度 v_drone, 无人机航向 theta_drone, 无人机飞行时间 t_drone_fly, 干扰弹延迟起爆时间 t_decoy_delay]
# 无人机速度范围：70~140m/s [cite: 7]
LOWER_BOUNDS = torch.tensor([70.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=DEVICE) # 下界
UPPER_BOUNDS = torch.tensor([140.0, np.pi, 60.0, 20.0], dtype=torch.float32, device=DEVICE) # 上界
DIMENSIONS = 4 # 待优化变量的数量

def calculate_covered_time(pos_init_missile, pos_init_drone,
                          v_drone, theta_drone,
                          t_drone_fly, t_decoy_delay,
                          interval=INTERVAL, show_progress=True, device=DEVICE):
    """
    GPU优化版：计算烟幕云团对导弹的有效遮蔽时长
    """
    # 确保输入是torch tensors
    if not isinstance(pos_init_missile, torch.Tensor):
        pos_init_missile = torch.tensor(pos_init_missile, dtype=torch.float32, device=device)
    elif pos_init_missile.device != device:
        pos_init_missile = pos_init_missile.to(device)
        
    if not isinstance(pos_init_drone, torch.Tensor):
        pos_init_drone = torch.tensor(pos_init_drone, dtype=torch.float32, device=device)
    elif pos_init_drone.device != device:
        pos_init_drone = pos_init_drone.to(device)
        
    # 无人机速度向量（等高度）
    v_d = _drone_vec(v_drone, theta_drone, device=device)
    
    # 投放点
    p_drop = pos_init_drone + v_d * t_drone_fly
    
    # 起爆时间
    t_expl = t_drone_fly + t_decoy_delay
    
    # 起爆位置（云团初始中心）
    p_expl = _decoy_state_at_explosion(pos_init_drone, v_d, t_drone_fly, t_decoy_delay, device=device)
    
    # 云团坐标有效时间区间
    t_end = t_expl + CLOUD_EFFECTIVE
    
    # 导弹到达假目标时间
    t_hit = torch.norm(pos_init_missile) / SPEED_MISSILE
    
    # 如果云团消散时间晚于导弹击中时间，调整结束时间
    if t_end > t_hit:
        t_end = t_hit
    
    # 如果云团起爆时间晚于导弹击中时间，无法遮蔽
    if t_expl > t_hit:
        return 0.0
    
    # 时间轴
    times = torch.arange(0.0, t_end.item() + 1e-9, interval, dtype=torch.float32, device=device)
    iterator = tqdm(times, desc="Simulating (GPU)", unit="step") if show_progress else times
    
    # 遮挡累计
    covered_time = 0.0
    
    # 批处理大小（根据GPU内存调整）
    batch_size = 256
    num_batches = (len(times) + batch_size - 1) // batch_size
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(times))
        time_batch = times[start_idx:end_idx]
        
        # 计算这批次时间点上的导弹位置
        A_batch = _missile_pos(pos_init_missile, time_batch, device=device)
        
        # 计算这批次时间点上的云团位置
        dt_batch = time_batch - t_expl
        valid_mask = (time_batch >= t_expl) & (time_batch < t_end)
        
        if torch.any(valid_mask):
            valid_indices = torch.where(valid_mask)[0]
            valid_times = time_batch[valid_indices]
            valid_dt = valid_times - t_expl
            
            # 计算对应时间的导弹位置
            valid_A = A_batch[valid_indices]
            
            # 计算对应时间的云团位置
            sink_offset = torch.zeros_like(p_expl)
            sink_offset[2] = -SINK_SPEED
            valid_B = p_expl + sink_offset * valid_dt.unsqueeze(1)
            
            # 检查是否遮蔽
            is_covered_results = is_covered(valid_A, valid_B, device=device)
            
            # 累加遮蔽时间
            covered_time += torch.sum(is_covered_results).item() * interval
            
    return covered_time

def objective_function(params, device=DEVICE):
    """
    GPU优化版：优化算法的目标函数（适应度函数）。
    它接收一组参数，运行仿真，并返回需要被最小化的值。
    由于我们的目标是最大化有效遮蔽时间，因此返回其相反数。
    """
    # 确保输入是torch tensor
    if not isinstance(params, torch.Tensor):
        params = torch.tensor(params, dtype=torch.float32, device=device)
    elif params.device != device:
        params = params.to(device)
    
    v_drone, theta_drone, t_drone_fly, t_decoy_delay = params
    
    covered_time = calculate_covered_time(
        pos_init_missile=POS_INIT_MISSILE_M1,
        pos_init_drone=POS_INIT_DRONE_FY1,
        v_drone=v_drone,
        theta_drone=theta_drone,
        t_drone_fly=t_drone_fly,
        t_decoy_delay=t_decoy_delay,
        show_progress=False,
        device=device
    )
    
    # WOA 算法默认求解最小值，因此我们返回时间的负值
    return -covered_time

class WhaleOptimizationAlgorithm:
    """
    GPU优化版：鲸鱼优化算法 (WOA) 的实现，用于寻找最优参数组合以最大化遮蔽时间。
    """
    def __init__(self, obj_func, lower_bounds, upper_bounds, dim, pop_size, max_iter, device=DEVICE):
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

        # 初始化领导者（当前最优解）
        self.leader_pos = torch.zeros(dim, dtype=torch.float32, device=device)
        self.leader_score = float('inf')  # 初始最优评价值设为无穷大

        # 初始化所有搜索智能体（鲸鱼）的位置
        self.positions = torch.rand(pop_size, dim, dtype=torch.float32, device=device) * (self.ub - self.lb) + self.lb

    def optimize(self):
        """
        运行 WOA 优化流程 - GPU优化版本
        """
        print(f"启动鲸鱼优化算法（GPU加速，设备：{self.device}）...")

        for t in tqdm(range(self.max_iter), desc="WOA 优化进度"):
            # 批量评估所有鲸鱼
            self.positions = torch.clamp(self.positions, self.lb, self.ub)
            
            # 批量评估（根据GPU内存分批处理）
            batch_size = 64  # 可调整
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

            if (t + 1) % 5 == 0:
                print(f"迭代 {t + 1}/{self.max_iter}, 当前最优遮蔽时间: {-self.leader_score:.4f} s")

        return self.leader_pos.cpu().numpy(), -self.leader_score

def main():
    """
    主函数，为问题2运行优化过程。
    """
    def objective_function_quiet(params):
        return objective_function(params, device=DEVICE)
    
    # 初始化并运行 WOA 优化器
    woa = WhaleOptimizationAlgorithm(
        obj_func=objective_function_quiet,
        lower_bounds=LOWER_BOUNDS,
        upper_bounds=UPPER_BOUNDS,
        dim=DIMENSIONS,
        pop_size=POPULATION_SIZE,
        max_iter=MAX_ITERATIONS,
        device=DEVICE
    )
    
    start_time = time.time()
    best_params, max_covered_time = woa.optimize()
    end_time = time.time()
    
    print("\n-------------------------------------------------")
    print(f"问题2 优化完成 (GPU优化版本，设备：{DEVICE})")
    print("优化算法：鲸鱼优化算法 (WOA)")
    print(f"总计用时: {end_time - start_time:.2f} 秒")
    print("-------------------------------------------------")
    
    # 解包并展示找到的最优参数
    v_opt, theta_opt, t_fly_opt, t_delay_opt = best_params
    
    print(f"最优无人机速度 (v_drone): {v_opt:.4f} m/s")
    print(f"最优无人机航向 (theta_drone): {theta_opt:.4f} rad ({math.degrees(theta_opt):.2f} 度)")
    print(f"最优投放时间 (t_drone_fly): {t_fly_opt:.4f} s")
    print(f"最优起爆延迟 (t_decoy_delay): {t_delay_opt:.4f} s")
    print("\n" + "="*50)
    print(f"求得的最大有效遮蔽时间: {max_covered_time:.4f} s")
    print("="*50)
    
    # 使用最优参数重新运行一次仿真，并显示进度条以进行验证
    print("\n使用最优结果进行最终仿真验证 (GPU):")
    covered_time = calculate_covered_time(
        pos_init_missile=POS_INIT_MISSILE_M1,
        pos_init_drone=POS_INIT_DRONE_FY1,
        v_drone=torch.tensor(v_opt, dtype=torch.float32, device=DEVICE),
        theta_drone=torch.tensor(theta_opt, dtype=torch.float32, device=DEVICE),
        t_drone_fly=torch.tensor(t_fly_opt, dtype=torch.float32, device=DEVICE),
        t_decoy_delay=torch.tensor(t_delay_opt, dtype=torch.float32, device=DEVICE),
        device=DEVICE
    )
    print(f"最终验证的有效遮蔽时间: {covered_time:.4f} s")
    
    # 计算并输出最优条件下的投放点和起爆点坐标
    # 计算无人机速度向量
    v_d = _drone_vec(v_opt, theta_opt, device=DEVICE)
    
    # 计算投放点坐标
    p_drop = POS_INIT_DRONE_FY1 + v_d * t_fly_opt
    
    # 计算起爆点坐标
    p_explosion = _decoy_state_at_explosion(POS_INIT_DRONE_FY1, v_d, t_fly_opt, t_delay_opt, device=DEVICE)
    
    print("\n" + "-"*50)
    print("最优条件下的关键位置坐标:")
    print(f"烟幕干扰弹投放点的x坐标: {p_drop[0].item():.2f} m")
    print(f"烟幕干扰弹投放点的y坐标: {p_drop[1].item():.2f} m")
    print(f"烟幕干扰弹投放点的z坐标: {p_drop[2].item():.2f} m")
    print(f"烟幕干扰弹起爆点的x坐标: {p_explosion[0].item():.2f} m")
    print(f"烟幕干扰弹起爆点的y坐标: {p_explosion[1].item():.2f} m")
    print(f"烟幕干扰弹起爆点的z坐标: {p_explosion[2].item():.2f} m")
    print("-"*50)

if __name__ == "__main__":
    main()
