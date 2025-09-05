# q2_PSO.py

import numpy as np
import math
import time
from tqdm import tqdm

try:
    import utils.calculate_covered_time as cct_module
    def _silent_tqdm(iterable, **kwargs):
        return iterable
    cct_module.tqdm = _silent_tqdm
except Exception:
    pass

from utils.calculate_covered_time import calculate_covered_time

# ---- 根据题目描述定义的常量 ----
# 导弹 M1 的初始位置 [cite: 6]
POS_INIT_MISSILE_M1 = np.array([20000.0, 0.0, 2000.0], dtype=float)
# 无人机 FY1 的初始位置 [cite: 6]
POS_INIT_DRONE_FY1 = np.array([17800.0, 0.0, 1800.0], dtype=float)

# ---- 优化算法配置 ----
# PSO 算法参数
POPULATION_SIZE = 50  # 搜索智能体（粒子）的数量
MAX_ITERATIONS = 30   # 最大迭代次数

# 参数边界 (搜索空间)
# 我们需要优化的四个变量：
# [无人机速度 v_drone, 无人机航向 theta_drone, 无人机飞行时间 t_drone_fly, 干扰弹延迟起爆时间 t_decoy_delay]
# 无人机速度范围：70~140m/s [cite: 7]
LOWER_BOUNDS = np.array([70.0, 0.0, 0.0, 0.0]) # 下界
UPPER_BOUNDS = np.array([140.0, np.pi, 60.0, 20.0]) # 上界
DIMENSIONS = 4 # 待优化变量的数量

def objective_function(params):
    """
    优化算法的目标函数（适应度函数）。
    它接收一组参数，运行仿真，并返回需要被最小化的值。
    由于我们的目标是最大化有效遮蔽时间，因此返回其相反数。
    """
    v_drone, theta_drone, t_drone_fly, t_decoy_delay = params
    
    covered_time = calculate_covered_time(
        pos_init_missile=POS_INIT_MISSILE_M1,
        pos_init_drone=POS_INIT_DRONE_FY1,
        v_drone=v_drone,
        theta_drone=theta_drone,
        t_drone_fly=t_drone_fly,
        t_decoy_delay=t_decoy_delay
    )
    
    # PSO 算法默认求解最小值，因此我们返回时间的负值
    return -covered_time

class ParticleSwarmOptimization:
    """
    粒子群优化算法 (PSO) 的实现，用于寻找最优参数组合以最大化遮蔽时间。
    """
    def __init__(self, obj_func, lower_bounds, upper_bounds, dim, pop_size, max_iter):
        self.obj_func = obj_func
        self.lb = lower_bounds
        self.ub = upper_bounds
        self.dim = dim
        self.pop_size = pop_size
        self.max_iter = max_iter

        # PSO 算法参数
        self.w = 0.9  # 惯性权重
        self.c1 = 2.0  # 个体学习因子
        self.c2 = 2.0  # 社会学习因子

        # 初始化全局最优解
        self.gbest_pos = np.zeros(dim)
        self.gbest_score = float('inf') # 初始最优评价值设为无穷大

        # 初始化所有搜索智能体（粒子）的位置和速度
        self.positions = np.random.rand(pop_size, dim) * (self.ub - self.lb) + self.lb
        self.velocities = np.random.rand(pop_size, dim) * (self.ub - self.lb) * 0.1  # 初始速度较小
        
        # 初始化个体最优位置和评价值
        self.pbest_pos = self.positions.copy()
        self.pbest_score = np.full(pop_size, float('inf'))

    def optimize(self):
        """
        运行 PSO 优化流程。
        """
        print("启动粒子群优化算法...")
        
        # 主循环
        for t in tqdm(range(self.max_iter), desc="PSO 优化进度"):
            for i in range(self.pop_size):
                # 检查并确保所有粒子的位置都在边界范围内
                self.positions[i, :] = np.clip(self.positions[i, :], self.lb, self.ub)
                
                # 为每个粒子计算其目标函数值
                fitness = self.obj_func(self.positions[i, :])
                
                # 更新个体最优解
                if fitness < self.pbest_score[i]:
                    self.pbest_score[i] = fitness
                    self.pbest_pos[i, :] = self.positions[i, :].copy()
                
                # 更新全局最优解
                if fitness < self.gbest_score:
                    self.gbest_score = fitness
                    self.gbest_pos = self.positions[i, :].copy()

            # 更新所有粒子的速度和位置
            for i in range(self.pop_size):
                # 生成随机数
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)
                
                # 更新速度
                self.velocities[i, :] = (self.w * self.velocities[i, :] + 
                                       self.c1 * r1 * (self.pbest_pos[i, :] - self.positions[i, :]) + 
                                       self.c2 * r2 * (self.gbest_pos - self.positions[i, :]))
                
                # 限制速度范围
                v_max = (self.ub - self.lb) * 0.2  # 最大速度为搜索空间的20%
                self.velocities[i, :] = np.clip(self.velocities[i, :], -v_max, v_max)
                
                # 更新位置
                self.positions[i, :] = self.positions[i, :] + self.velocities[i, :]
            
            # 动态调整惯性权重（线性递减）
            self.w = 0.9 - (0.9 - 0.4) * t / self.max_iter
            
            # 打印进度
            if (t + 1) % 5 == 0:
                print(f"迭代 {t + 1}/{self.max_iter}, 当前最优遮蔽时间: {-self.gbest_score:.4f} s")

        return self.gbest_pos, -self.gbest_score


def main():
    """
    主函数，为问题2运行优化过程。
    """
    def objective_function_quiet(params):
        v_drone, theta_drone, t_drone_fly, t_decoy_delay = params
        covered_time = calculate_covered_time(
            pos_init_missile=POS_INIT_MISSILE_M1,
            pos_init_drone=POS_INIT_DRONE_FY1,
            v_drone=v_drone,
            theta_drone=theta_drone,
            t_drone_fly=t_drone_fly,
            t_decoy_delay=t_decoy_delay,
            # 将这个新参数传递给您的函数
            # show_progress=False 
        )
        return -covered_time

    # 初始化并运行 PSO 优化器
    pso = ParticleSwarmOptimization(
        obj_func=objective_function_quiet,
        lower_bounds=LOWER_BOUNDS,
        upper_bounds=UPPER_BOUNDS,
        dim=DIMENSIONS,
        pop_size=POPULATION_SIZE,
        max_iter=MAX_ITERATIONS
    )
    
    start_time = time.time()
    best_params, max_covered_time = pso.optimize()
    end_time = time.time()
    
    print("\n-------------------------------------------------")
    print("问题2 优化完成")
    print("优化算法：粒子群优化算法 (PSO)")
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
    
    # 可选：使用最优参数重新运行一次仿真，并显示进度条以进行验证
    print("\n使用最优结果进行最终仿真验证:")
    covered_time = calculate_covered_time(
        pos_init_missile=POS_INIT_MISSILE_M1,
        pos_init_drone=POS_INIT_DRONE_FY1,
        v_drone=v_opt,
        theta_drone=theta_opt,
        t_drone_fly=t_fly_opt,
        t_decoy_delay=t_delay_opt
    )
    print(covered_time)


if __name__ == "__main__":
    main()
