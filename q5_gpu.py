# q5_gpu.py

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
    parser = argparse.ArgumentParser(description="问题5 GPU优化版")
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
# 3枚导弹初始位置
POS_INIT_MISSILE_M1 = torch.tensor([20000.0,    0.0, 2000.0], dtype=torch.float32, device=DEVICE)
POS_INIT_MISSILE_M2 = torch.tensor([19000.0,  600.0, 2100.0], dtype=torch.float32, device=DEVICE)
POS_INIT_MISSILE_M3 = torch.tensor([18000.0, -600.0, 1900.0], dtype=torch.float32, device=DEVICE)
POS_INIT_MISSILES = [POS_INIT_MISSILE_M1, POS_INIT_MISSILE_M2, POS_INIT_MISSILE_M3]

# 5架无人机初始位置
POS_INIT_DRONE_FY1 = torch.tensor([17800.0,     0.0, 1800.0], dtype=torch.float32, device=DEVICE)
POS_INIT_DRONE_FY2 = torch.tensor([12000.0,  1400.0, 1400.0], dtype=torch.float32, device=DEVICE)
POS_INIT_DRONE_FY3 = torch.tensor([ 6000.0, -3000.0,  700.0], dtype=torch.float32, device=DEVICE)
POS_INIT_DRONE_FY4 = torch.tensor([11000.0,  2000.0, 1800.0], dtype=torch.float32, device=DEVICE)
POS_INIT_DRONE_FY5 = torch.tensor([13000.0, -2000.0, 1300.0], dtype=torch.float32, device=DEVICE)
POS_INIT_DRONES = [POS_INIT_DRONE_FY1, POS_INIT_DRONE_FY2, POS_INIT_DRONE_FY3, POS_INIT_DRONE_FY4, POS_INIT_DRONE_FY5]

# ---- 优化算法配置 ----
# TODO:
POPULATION_SIZE = 200   # 搜索智能体（鲸鱼）的数量
MAX_ITERATIONS = 10      # 最大迭代次数
EARLY_STOP_PATIENCE = 5  # 早停阈值：连续多少次迭代没有性能提升就停止

# ---- 参数边界 (搜索空间) ----
# 对于每架无人机： [v_drone, theta_drone, t_drop1, t_delta_drop2, t_delta_drop3, t_delay1, t_delay2, t_delay3]
LOWER_BOUNDS = torch.tensor(np.tile(np.array([70.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0], dtype=float), 5), 
                           dtype=torch.float32, device=DEVICE)
UPPER_BOUNDS = torch.tensor(np.tile(np.array([140.0, 2*np.pi, 58.0, 59.0, 59.0, 20.0, 20.0, 20.0], dtype=float), 5), 
                           dtype=torch.float32, device=DEVICE)
DIMENSIONS = 40

# 加权系数
W_SUM_TIME = 0.7 # 所有导弹遮蔽并集时长之和
W_MIN_TIME = 0.3 # 最低导弹遮蔽并集时长

# ---- 计算多枚干扰弹对多枚导弹的遮蔽时长 - GPU优化版本 ----
def calculate_covered_time_multi(pos_init_missiles, pos_init_drones,
                                v_drones, theta_drones,
                                t_drop_arrs, t_delay_arrs,
                                interval=INTERVAL, show_progress=False,
                                return_details=False, device=DEVICE):
    """
    GPU优化版：计算多枚导弹被所有烟幕云团遮挡的总时长 (完全向量化)
    """
    num_missiles = len(pos_init_missiles)
    num_drones = len(pos_init_drones)
    num_decoys_per_drone = 3
    num_decoys = num_drones * num_decoys_per_drone

    # 收集所有干扰弹的信息
    all_t_expl_list = []
    all_p_expl_list = []
    all_p_drop_list = []
    all_t_drop_list = []

    for d_id in range(num_drones):
        pos_drone = pos_init_drones[d_id]
        v_d = _drone_vec(v_drones[d_id], theta_drones[d_id], device=device)
        t_drop_arr = t_drop_arrs[d_id]
        t_delay_arr = t_delay_arrs[d_id]

        for i in range(num_decoys_per_drone):
            t_drop = t_drop_arr[i]
            t_delay = t_delay_arr[i]
            t_expl = t_drop + t_delay

            p_drop = pos_drone + v_d * t_drop
            p_expl = _decoy_state_at_explosion(pos_drone, v_d, t_drop, t_delay, device=device)

            all_t_drop_list.append(t_drop)
            all_t_expl_list.append(t_expl)
            all_p_drop_list.append(p_drop)
            all_p_expl_list.append(p_expl)

    all_t_expl = torch.stack(all_t_expl_list) # [15]
    all_p_expl = torch.stack(all_p_expl_list) # [15, 3]
    all_p_drop = torch.stack(all_p_drop_list) # [15, 3]
    all_t_drop = torch.stack(all_t_drop_list) # [15]

    # 对于每枚导弹，独立仿真其遮蔽情况
    union_times = torch.zeros(num_missiles, dtype=torch.float32, device=device)
    per_decoy_per_missile = torch.zeros((num_missiles, num_decoys), dtype=torch.float32, device=device)

    for m_id in range(num_missiles):
        pos_missile = pos_init_missiles[m_id]
        t_hit = torch.norm(pos_missile) / SPEED_MISSILE
        times = torch.arange(0.0, t_hit.item() + 1e-9, interval, dtype=torch.float32, device=device) # [T]
        
        if show_progress:
            print(f"Simulating missile M{m_id+1} (GPU Vectorized)... {len(times)} steps.")

        # 预计算所有时间点的导弹位置
        A_t = _missile_pos(pos_missile, times, device=device) # [T, 3]

        # [T, 15] 每一行是一个时间点，每一列是一个干扰弹
        occluded_matrix = torch.zeros(len(times), num_decoys, dtype=torch.bool, device=device)

        # 向量化计算所有干扰弹的遮挡情况
        # [1, 15]
        t_expl_exp = all_t_expl.unsqueeze(0)
        # [T, 1]
        times_exp = times.unsqueeze(1)
        
        # [T, 15] bool, 标记每个时间点，每个干扰弹是否在其有效期内
        valid_time_mask = (times_exp >= t_expl_exp) & (times_exp < t_expl_exp + CLOUD_EFFECTIVE)
        
        # 获取所有有效时间点的索引 [N_valid, 2], N_valid是所有有效(t, decoy)对的数量
        valid_indices = torch.where(valid_time_mask)
        time_indices, decoy_indices = valid_indices
        
        if time_indices.numel() > 0:
            # 提取所有有效的 A, B, t
            valid_A = A_t[time_indices] # [N_valid, 3]
            valid_t = times[time_indices] # [N_valid]
            valid_decoy_expl_t = all_t_expl[decoy_indices] # [N_valid]
            valid_decoy_expl_p = all_p_expl[decoy_indices] # [N_valid, 3]
            
            # 批量计算云团位置
            dt = valid_t - valid_decoy_expl_t # [N_valid]
            sink_offset = torch.zeros_like(valid_decoy_expl_p)
            sink_offset[:, 2] = -SINK_SPEED * dt
            valid_B = valid_decoy_expl_p + sink_offset # [N_valid, 3]
            
            # *** 核心：一次性调用向量化的is_covered ***
            is_covered_results = is_covered(valid_A, valid_B, device=device) # [N_valid]
            
            # 将结果填充回遮挡矩阵
            occluded_matrix[time_indices, decoy_indices] = is_covered_results

        # 计算每个干扰弹的单独遮蔽时长
        per_decoy_per_missile[m_id] = torch.sum(occluded_matrix, dim=0) * interval
        
        # 计算并集遮蔽时长
        any_occluded = torch.any(occluded_matrix, dim=1)
        union_times[m_id] = torch.sum(any_occluded) * interval

    sum_union = torch.sum(union_times).item()
    min_union = torch.min(union_times).item()

    if return_details:
        # 转换为CPU numpy数组以便打印和保存
        details = {
            "union_times": union_times.cpu().numpy(),
            "sum_union": sum_union,
            "min_union": min_union,
            "per_decoy_per_missile": per_decoy_per_missile.cpu().numpy(),
            "all_p_drop": all_p_drop.cpu().numpy(),
            "all_p_expl": all_p_expl.cpu().numpy(),
            "all_t_drop": all_t_drop.cpu().numpy(),
            "all_t_expl": all_t_expl.cpu().numpy()
        }
        return sum_union, min_union, details
    return sum_union, min_union

# ---- 目标函数 - GPU优化版本 ----
def objective_function(params, interval_eval=0.02, device=DEVICE):
    """
    GPU优化版：目标函数
    """
    # 确保输入是torch tensor
    if not isinstance(params, torch.Tensor):
        params = torch.tensor(params, dtype=torch.float32, device=device)
    elif params.device != device:
        params = params.to(device)
    
    params = params.reshape(5, 8)
    v_drones = params[:, 0]
    theta_drones = params[:, 1]
    t_drop_arrs = []
    t_delay_arrs = []

    for d in range(5):
        t_drop1 = params[d, 2]
        t_delta2 = params[d, 3]  # 已确保 ≥1.0
        t_delta3 = params[d, 4]  # 已确保 ≥1.0
        t_drop2 = t_drop1 + t_delta2
        t_drop3 = t_drop2 + t_delta3
        t_drop_arrs.append(torch.tensor([t_drop1, t_drop2, t_drop3], dtype=torch.float32, device=device))
        t_delay_arrs.append(params[d, 5:8])

    # 计算遮蔽时长（为速度，将评估步长略放宽）
    sum_union, min_union = calculate_covered_time_multi(
        pos_init_missiles=POS_INIT_MISSILES,
        pos_init_drones=POS_INIT_DRONES,
        v_drones=v_drones,
        theta_drones=theta_drones,
        t_drop_arrs=t_drop_arrs,
        t_delay_arrs=t_delay_arrs,
        interval=interval_eval,     # 加速评估
        show_progress=False,
        return_details=False,
        device=device
    )

    return - (W_SUM_TIME * sum_union + W_MIN_TIME * min_union)

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
        print(f"启动鲸鱼优化算法（GPU加速，设备：{self.device}）...")

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

            if (t + 1) % 1 == 0:
                print(f"迭代 {t + 1}/{self.max_iter}, 当前最优目标值: {-self.leader_score:.4f}")
                if self.early_stop_patience is not None:
                    print(f"连续未改进计数: {self.no_improve_counter}/{self.early_stop_patience}")

        return self.leader_pos.cpu().numpy(), -self.leader_score


# ---- 主函数 ----
def main():
    """
    问题5：利用5架无人机，每架投放3枚烟幕干扰弹，对M1、M2、M3实施干扰。
    GPU优化版本
    """
    # 用于目标函数的简化包装
    def objective_quiet(params):
        return objective_function(params, interval_eval=INTERVAL, device=DEVICE)

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
    best_params, est_opt_value = woa.optimize()
    end_time = time.time()

    print("\n-------------------------------------------------")
    print(f"问题5 优化完成 (GPU优化版本，设备：{DEVICE})")
    print("优化算法：鲸鱼优化算法 (WOA)")
    print(f"总计用时: {end_time - start_time:.2f} 秒")
    print("-------------------------------------------------")

    # 解包最优参数并转换为绝对时间
    best_params = np.asarray(best_params, dtype=float).reshape(5, 8)
    v_opts = best_params[:, 0]
    theta_opts = best_params[:, 1]
    t_drop_arrs = []
    t_delay_arrs = []

    for d in range(5):
        t_drop1 = best_params[d, 2]
        t_delta2 = best_params[d, 3]
        t_delta3 = best_params[d, 4]
        t_drop2 = t_drop1 + t_delta2
        t_drop3 = t_drop2 + t_delta3
        t_drop_arrs.append(np.array([t_drop1, t_drop2, t_drop3], dtype=float))
        t_delay_arrs.append(best_params[d, 5:8])

    # 转换为PyTorch张量用于最终验证
    v_opts_tensor = torch.tensor(v_opts, dtype=torch.float32, device=DEVICE)
    theta_opts_tensor = torch.tensor(theta_opts, dtype=torch.float32, device=DEVICE)
    t_drop_arrs_tensor = [torch.tensor(arr, dtype=torch.float32, device=DEVICE) for arr in t_drop_arrs]
    t_delay_arrs_tensor = [torch.tensor(arr, dtype=torch.float32, device=DEVICE) for arr in t_delay_arrs]

    # ---- 使用最优参数进行一次"精细步长"的最终仿真与验证 ----
    print("\n执行最终验证 (使用精细步长)...")
    _, _, details = calculate_covered_time_multi(
        pos_init_missiles=POS_INIT_MISSILES,
        pos_init_drones=POS_INIT_DRONES,
        v_drones=v_opts_tensor,
        theta_drones=theta_opts_tensor,
        t_drop_arrs=t_drop_arrs_tensor,
        t_delay_arrs=t_delay_arrs_tensor,
        interval=INTERVAL,         # 精细验证
        show_progress=True,
        return_details=True,
        device=DEVICE
    )

    union_times = details["union_times"]
    sum_union = details["sum_union"]
    min_union = details["min_union"]
    per_decoy_per_missile = details["per_decoy_per_missile"]
    all_p_drop = details["all_p_drop"]
    all_p_expl = details["all_p_expl"]
    all_t_drop = details["all_t_drop"]
    all_t_expl = details["all_t_expl"]

    # ---- 事后为每个干扰弹分配导弹（贡献最大者）并获取有效时长 ----
    assigned_missiles = []
    effective_times = []

    for decoy_id in range(15):
        per_m_times = per_decoy_per_missile[:, decoy_id]
        assigned_m = np.argmax(per_m_times) + 1  # M1=1, etc.
        eff_time = per_m_times[assigned_m - 1]
        assigned_missiles.append(f"M{assigned_m}")
        effective_times.append(eff_time)

    # ---- 打印关键信息 ----
    labels = ["FY1", "FY2", "FY3", "FY4", "FY5"]
    for d_id, lab in enumerate(labels):
        print(f"\n[{lab}] 最优参数：")
        print(f"  无人机速度 v_drone: {v_opts[d_id]:.4f} m/s")
        print(f"  无人机航向 theta_drone: {theta_opts[d_id]:.4f} rad ({math.degrees(theta_opts[d_id]):.2f} 度)")

        for i in range(3):
            decoy_id = d_id * 3 + i
            print(f"  [Decoy {i+1}]")
            print(f"    投放时间 t_drop: {all_t_drop[decoy_id]:.4f} s")
            print(f"    投放点 P_drop: ({all_p_drop[decoy_id,0]:.2f}, {all_p_drop[decoy_id,1]:.2f}, {all_p_drop[decoy_id,2]:.2f}) m")
            print(f"    起爆时间 t_expl: {all_t_expl[decoy_id]:.4f} s")
            print(f"    起爆点 P_expl: ({all_p_expl[decoy_id,0]:.2f}, {all_p_expl[decoy_id,1]:.2f}, {all_p_expl[decoy_id,2]:.2f}) m")
            print(f"    有效干扰时长: {effective_times[decoy_id]:.4f} s")
            print(f"    干扰的导弹编号: {assigned_missiles[decoy_id]}")

    print("\n" + "-"*50)
    for m_id in range(3):
        print(f"导弹 M{m_id+1} 的并集有效遮蔽时长: {union_times[m_id]:.4f} s")
    print(f"所有导弹并集有效遮蔽总时长: {sum_union:.4f} s")
    print(f"最低导弹并集有效遮蔽时长: {min_union:.4f} s")
    print("-"*50)


if __name__ == "__main__":
    main()
