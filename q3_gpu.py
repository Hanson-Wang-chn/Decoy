# q3_gpu.py

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
    parser = argparse.ArgumentParser(description="问题3 GPU优化版")
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
POPULATION_SIZE = 400   # 搜索智能体（鲸鱼）的数量
MAX_ITERATIONS = 30     # 最大迭代次数
EARLY_STOP_PATIENCE = 5  # 早停阈值：连续多少次迭代没有性能提升就停止

# ---- 参数边界 (搜索空间) ----
# 我们需要优化的 8 个变量：
# [v_drone, theta_drone, t_drop1, t_delta_drop2, t_delta_drop3, t_delay1, t_delay2, t_delay3]
# 修改：t_delta_drop2和t_delta_drop3表示时间间隔，必须≥1秒
LOWER_BOUNDS = torch.tensor([70.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=DEVICE)
UPPER_BOUNDS = torch.tensor([140.0, np.pi, 58.0, 59.0, 59.0, 20.0, 20.0, 20.0], dtype=torch.float32, device=DEVICE)
DIMENSIONS = 8


# ---- 计算三枚干扰弹的遮蔽时长 - GPU优化版本 ----
def calculate_covered_time_3decoys(pos_init_missile,
                                  pos_init_drone,
                                  v_drone,
                                  theta_drone,
                                  t_drop_arr,          # 长度为3的 array-like：三次投放时间
                                  t_delay_arr,         # 长度为3的 array-like：三次起爆延迟
                                  interval=INTERVAL,
                                  show_progress=False,
                                  return_details=False,
                                  device=DEVICE):
    """
    GPU优化版：计算导弹被三枚烟幕云团遮挡的总时长（并集），并可返回每一枚的遮蔽时长（各自统计）。
    说明：
      - 起爆后云团中心以 SINK_SPEED 匀速下沉，存在 CLOUD_EFFECTIVE 秒的有效期。
      - 总时长按"三枚云团遮蔽时间的并集长度"计；每枚遮蔽时长单独统计，不去除重叠。
    """
    # 确保输入是torch tensor
    if isinstance(pos_init_missile, np.ndarray):
        pos_init_missile = torch.tensor(pos_init_missile, dtype=torch.float32, device=device)
    elif isinstance(pos_init_missile, torch.Tensor) and pos_init_missile.device != device:
        pos_init_missile = pos_init_missile.to(device)
    
    if isinstance(pos_init_drone, np.ndarray):
        pos_init_drone = torch.tensor(pos_init_drone, dtype=torch.float32, device=device)
    elif isinstance(pos_init_drone, torch.Tensor) and pos_init_drone.device != device:
        pos_init_drone = pos_init_drone.to(device)
        
    if not isinstance(t_drop_arr, torch.Tensor):
        t_drop_arr = torch.tensor(t_drop_arr, dtype=torch.float32, device=device)
    elif t_drop_arr.device != device:
        t_drop_arr = t_drop_arr.to(device)
        
    if not isinstance(t_delay_arr, torch.Tensor):
        t_delay_arr = torch.tensor(t_delay_arr, dtype=torch.float32, device=device)
    elif t_delay_arr.device != device:
        t_delay_arr = t_delay_arr.to(device)
    
    # 三个关键时间：投放、起爆、结束
    t_expl_arr = t_drop_arr + t_delay_arr
    t_end = float(torch.max(t_expl_arr + CLOUD_EFFECTIVE).item())

    # 无人机速度向量（等高度）
    v_d = _drone_vec(v_drone, theta_drone, device=device)

    # 起爆位置（云团初始中心），形状 (3, 3)
    B_expl_list = []
    for i in range(3):
        B_expl_list.append(_decoy_state_at_explosion(pos_init_drone, v_d, t_drop_arr[i], t_delay_arr[i], device=device))
    B_expl = torch.stack(B_expl_list, dim=0)

    # 时间轴
    times = torch.arange(0.0, t_end + 1e-9, interval, dtype=torch.float32, device=device)
    iterator = tqdm(times, desc="Simulating(3 decoys GPU)", unit="step") if show_progress else times

    # 预计算所有时间点的导弹位置
    A_t_all = _missile_pos(pos_init_missile, times, device=device)  # [T, 3]

    # 创建遮挡矩阵，表示每个时间点，每个云团是否遮挡
    # [T, 3] 每一行是一个时间点，每一列是一个云团
    occluded_matrix = torch.zeros(len(times), 3, dtype=torch.bool, device=device)

    # 向量化计算所有云团的遮挡情况
    # [1, 3]
    t_expl_exp = t_expl_arr.unsqueeze(0)
    # [T, 1]
    times_exp = times.unsqueeze(1)
    
    # [T, 3] bool, 标记每个时间点，每个云团是否在其有效期内
    valid_time_mask = (times_exp >= t_expl_exp) & (times_exp < t_expl_exp + CLOUD_EFFECTIVE)
    
    # 获取所有有效时间点的索引 [N_valid, 2], N_valid是所有有效(t, decoy)对的数量
    valid_indices = torch.where(valid_time_mask)
    time_indices, decoy_indices = valid_indices
    
    if time_indices.numel() > 0:
        # 提取所有有效的 A, B, t
        valid_A = A_t_all[time_indices]  # [N_valid, 3]
        valid_t = times[time_indices]  # [N_valid]
        valid_decoy_expl_t = t_expl_arr[decoy_indices]  # [N_valid]
        valid_decoy_expl_p = B_expl[decoy_indices]  # [N_valid, 3]
        
        # 批量计算云团位置
        dt = valid_t - valid_decoy_expl_t  # [N_valid]
        sink_offset = torch.zeros_like(valid_decoy_expl_p)
        sink_offset[:, 2] = -SINK_SPEED * dt
        valid_B = valid_decoy_expl_p + sink_offset  # [N_valid, 3]
        
        # 核心：一次性调用向量化的is_covered
        is_covered_results = is_covered(valid_A, valid_B, device=device)  # [N_valid]
        
        # 将结果填充回遮挡矩阵
        occluded_matrix[time_indices, decoy_indices] = is_covered_results

    # 计算每个云团的单独遮蔽时长
    covered_each = torch.sum(occluded_matrix, dim=0) * interval
    
    # 计算并集遮蔽时长
    any_occluded = torch.any(occluded_matrix, dim=1)
    covered_total = torch.sum(any_occluded) * interval

    if return_details:
        details = {
            "t_drop_arr": t_drop_arr.cpu().numpy(),
            "t_expl_arr": t_expl_arr.cpu().numpy(),
            "B_expl": B_expl.cpu().numpy()
        }
        return covered_total.item(), covered_each.cpu().numpy(), details
    
    return covered_total.item()


# ---- 目标函数 - GPU优化版本 ----
def objective_function(params, interval_eval=0.02, device=DEVICE):
    """
    GPU优化版：目标函数
    目标：最大化并集遮蔽时长 -> 最小化其相反数。
    """
    # 确保输入是torch tensor
    if not isinstance(params, torch.Tensor):
        params = torch.tensor(params, dtype=torch.float32, device=device)
    elif params.device != device:
        params = params.to(device)
    
    v_drone = params[0]
    theta_drone = params[1]
    
    # 转换增量时间为绝对时间
    t_drop1 = params[2]
    t_delta_drop2 = params[3]  # 已确保 ≥ 1.0
    t_delta_drop3 = params[4]  # 已确保 ≥ 1.0
    
    t_drop2 = t_drop1 + t_delta_drop2
    t_drop3 = t_drop2 + t_delta_drop3
    
    t_drop_arr = torch.tensor([t_drop1, t_drop2, t_drop3], dtype=torch.float32, device=device)
    t_delay_arr = params[5:8]

    # 计算遮蔽并集时长（为速度，将评估步长略放宽）
    covered_total = calculate_covered_time_3decoys(
        pos_init_missile=POS_INIT_MISSILE_M1,
        pos_init_drone=POS_INIT_DRONE_FY1,
        v_drone=v_drone,
        theta_drone=theta_drone,
        t_drop_arr=t_drop_arr,
        t_delay_arr=t_delay_arr,
        interval=interval_eval,     # 加速评估
        show_progress=False,
        return_details=False,
        device=device
    )

    return -covered_total


# ---- 鲸鱼优化算法 - GPU优化版本 ----
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
        print(f"启动鲸鱼优化算法（问题3：3枚干扰弹，GPU加速，设备：{self.device}）...")

        for t in tqdm(range(self.max_iter), desc="WOA 优化进度"):
            # 记录上一次的最优分数用于比较
            prev_best_score = self.leader_score
            
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
                print(f"迭代 {t + 1}/{self.max_iter}, 当前最优遮蔽时间(估计): {-self.leader_score:.4f} s")
                if self.early_stop_patience is not None:
                    print(f"连续未改进计数: {self.no_improve_counter}/{self.early_stop_patience}")

        return self.leader_pos.cpu().numpy(), -self.leader_score


# ---- 主函数 ----
def main():
    """
    问题3：FY1 投放 3 枚烟幕干扰弹，对 M1 实施干扰。
    保持与问题2一致的界面/风格；优化完成后做一次精细仿真验证并打印关键坐标/时长。
    GPU优化版本
    """
    # 用于目标函数的简化包装
    def objective_quiet(params):
        return objective_function(params, interval_eval=0.02, device=DEVICE)

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
    start_time = time.time()
    best_params, est_max_covered = woa.optimize()
    end_time = time.time()

    print("\n-------------------------------------------------")
    print(f"问题3 优化完成 (GPU优化版本，设备：{DEVICE})")
    print("优化算法：鲸鱼优化算法 (WOA)")
    print(f"总计用时: {end_time - start_time:.2f} 秒")
    print("-------------------------------------------------")

    # 解包最优参数并转换为绝对时间
    v_opt = best_params[0]
    theta_opt = best_params[1]
    
    # 计算真实的投放时间点
    t_drop1 = best_params[2]
    t_delta_drop2 = best_params[3]
    t_delta_drop3 = best_params[4]
    
    t_drop2 = t_drop1 + t_delta_drop2
    t_drop3 = t_drop2 + t_delta_drop3
    
    t_drop_opt = np.array([t_drop1, t_drop2, t_drop3], dtype=float)
    t_delay_opt = np.array(best_params[5:8], dtype=float)
    t_expl_opt = t_drop_opt + t_delay_opt

    print(f"最优无人机速度 (v_drone): {v_opt:.4f} m/s")
    print(f"最优无人机航向 (theta_drone): {theta_opt:.4f} rad ({math.degrees(theta_opt):.2f} 度)")
    print(f"最优投放时间 (t_drop): {t_drop_opt}")
    print(f"最优起爆延迟 (t_delay): {t_delay_opt}")
    print("\n" + "="*50)
    print(f"估计的最大并集有效遮蔽时间: {est_max_covered:.4f} s")
    print("="*50)

    # 转换为PyTorch张量用于最终验证
    t_drop_opt_tensor = torch.tensor(t_drop_opt, dtype=torch.float32, device=DEVICE)
    t_delay_opt_tensor = torch.tensor(t_delay_opt, dtype=torch.float32, device=DEVICE)

    # ---- 使用最优参数进行一次"精细步长"的最终仿真与验证 ----
    print("\n执行最终验证 (使用精细步长)...")
    covered_total, covered_each, debug = calculate_covered_time_3decoys(
        pos_init_missile=POS_INIT_MISSILE_M1,
        pos_init_drone=POS_INIT_DRONE_FY1,
        v_drone=v_opt,
        theta_drone=theta_opt,
        t_drop_arr=t_drop_opt_tensor,
        t_delay_arr=t_delay_opt_tensor,
        interval=INTERVAL,         # 精细验证
        show_progress=True,
        return_details=True,
        device=DEVICE
    )

    # 无人机速度向量 + 三个投放点/起爆点坐标
    v_d = _drone_vec(v_opt, theta_opt, device='cpu').cpu().numpy()
    
    # 使用CPU上的numpy数组计算投放点坐标（用于打印）
    P_drop = np.stack([np.array(POS_INIT_DRONE_FY1.cpu()) + v_d * t_drop_opt[i] for i in range(3)], axis=0)
    P_expl = debug["B_expl"]

    # ---- 打印关键坐标与逐枚时长 ----
    print("\n" + "-"*50)
    print("最优条件下的关键位置坐标与时序：")
    for i in range(3):
        print(f"[Decoy {i+1}]")
        print(f"  投放时间 t_drop{i+1}: {t_drop_opt[i]:.4f} s")
        print(f"  投放点  P_drop{i+1}: ({P_drop[i,0]:.2f}, {P_drop[i,1]:.2f}, {P_drop[i,2]:.2f}) m")
        print(f"  起爆时间 t_expl{i+1}: {t_expl_opt[i]:.4f} s")
        print(f"  起爆点  P_expl{i+1}: ({P_expl[i,0]:.2f}, {P_expl[i,1]:.2f}, {P_expl[i,2]:.2f}) m")
        print(f"  该弹有效遮蔽时长: {covered_each[i]:.4f} s")
    print("-"*50)
    print(f"三枚烟幕干扰弹'并集'有效遮蔽总时长: {covered_total:.4f} s")
    print("-"*50)


if __name__ == "__main__":
    main()
