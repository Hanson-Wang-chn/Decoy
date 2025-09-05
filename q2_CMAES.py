# q2_CMAES.py

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
# CMA-ES 算法参数
POPULATION_SIZE = 20  # CMA-ES推荐的种群大小（通常为 4 + 3*ln(n)，这里n=4）
MAX_GENERATIONS = 50  # 最大代数

# 参数边界 (搜索空间)
# 我们需要优化的四个变量：
# [无人机速度 v_drone, 无人机航向 theta_drone, 无人机飞行时间 t_drone_fly, 干扰弹延迟起爆时间 t_decoy_delay]
# 无人机速度范围：70~140m/s [cite: 7]
LOWER_BOUNDS = np.array([70.0, 0.0, 0.0, 0.0]) # 下界
UPPER_BOUNDS = np.array([140.0, np.pi, 50.0, 20.0]) # 上界
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
    
    # CMA-ES 算法默认求解最小值，因此我们返回时间的负值
    return -covered_time

class CMAEvolutionStrategy:
    """
    协方差矩阵自适应进化策略 (CMA-ES) 的实现，用于寻找最优参数组合以最大化遮蔽时间。
    """
    def __init__(self, obj_func, initial_mean, initial_sigma, lower_bounds, upper_bounds, pop_size, max_generations):
        self.obj_func = obj_func
        self.lb = lower_bounds
        self.ub = upper_bounds
        self.dim = len(initial_mean)
        self.pop_size = pop_size
        self.max_generations = max_generations
        
        # CMA-ES 参数初始化
        self.mean = initial_mean.copy()  # 分布均值
        self.sigma = initial_sigma       # 步长（全局标准差）
        self.C = np.eye(self.dim)        # 协方差矩阵
        self.pc = np.zeros(self.dim)     # 进化路径（协方差）
        self.ps = np.zeros(self.dim)     # 进化路径（步长）
        
        # 策略参数设置
        self.mu = pop_size // 2  # 父代个数
        self.weights = np.log(self.mu + 1/2) - np.log(np.arange(1, self.mu + 1))
        self.weights = self.weights / np.sum(self.weights)  # 归一化权重
        self.mueff = 1.0 / np.sum(self.weights**2)  # 有效选择质量数
        
        # 时间常数
        self.cc = (4 + self.mueff/self.dim) / (self.dim + 4 + 2*self.mueff/self.dim)  # 协方差时间常数
        self.cs = (self.mueff + 2) / (self.dim + self.mueff + 5)  # 步长时间常数
        self.c1 = 2 / ((self.dim + 1.3)**2 + self.mueff)  # 学习率（rank-1更新）
        self.cmu = min(1 - self.c1, 2 * (self.mueff - 2 + 1/self.mueff) / ((self.dim + 2)**2 + self.mueff))  # 学习率（rank-μ更新）
        self.damps = 1 + 2*max(0, np.sqrt((self.mueff-1)/(self.dim+1))-1) + self.cs  # 步长阻尼
        
        # 期望值
        self.chiN = np.sqrt(self.dim) * (1 - 1.0/(4*self.dim) + 1.0/(21*self.dim**2))
        
        # 记录最优解
        self.best_solution = None
        self.best_fitness = float('inf')
        
        # 特征值分解相关
        self.eigeneval = 0
        self.B = np.eye(self.dim)
        self.D = np.ones(self.dim)
        self.invsqrtC = np.eye(self.dim)

    def _boundary_handling(self, x):
        """边界处理：将解限制在可行域内"""
        return np.clip(x, self.lb, self.ub)

    def _update_eigensystem(self):
        """更新协方差矩阵的特征系统"""
        if self.eigeneval - 1 > self.pop_size / (self.c1 + self.cmu) / self.dim / 10:
            self.eigeneval = 1
            
            # 确保协方差矩阵是对称的
            self.C = np.triu(self.C) + np.triu(self.C, 1).T
            
            # 特征值分解
            eigenvalues, eigenvectors = np.linalg.eigh(self.C)
            
            # 确保特征值为正
            eigenvalues = np.maximum(eigenvalues, 1e-14)
            
            self.D = np.sqrt(eigenvalues)
            self.B = eigenvectors
            self.invsqrtC = self.B @ np.diag(1.0 / self.D) @ self.B.T
        else:
            self.eigeneval += 1

    def optimize(self):
        """
        运行 CMA-ES 优化流程。
        """
        print("启动协方差矩阵自适应进化策略算法...")
        
        for generation in tqdm(range(self.max_generations), desc="CMA-ES 优化进度"):
            # 更新特征系统
            self._update_eigensystem()
            
            # 生成候选解
            population = []
            fitness_values = []
            
            for _ in range(self.pop_size):
                # 生成标准正态分布随机向量
                z = np.random.randn(self.dim)
                # 应用协方差矩阵变换
                y = self.B @ (self.D * z)
                # 生成候选解
                x = self.mean + self.sigma * y
                # 边界处理
                x = self._boundary_handling(x)
                
                population.append(x)
                
                # 评估适应度
                fitness = self.obj_func(x)
                fitness_values.append(fitness)
                
                # 更新最优解
                if fitness < self.best_fitness:
                    self.best_fitness = fitness
                    self.best_solution = x.copy()
            
            population = np.array(population)
            fitness_values = np.array(fitness_values)
            
            # 排序
            sorted_indices = np.argsort(fitness_values)
            population = population[sorted_indices]
            fitness_values = fitness_values[sorted_indices]
            
            # 选择父代
            selected = population[:self.mu]
            
            # 更新分布均值
            old_mean = self.mean.copy()
            self.mean = np.sum(self.weights[:, np.newaxis] * selected, axis=0)
            
            # 更新进化路径
            ps_old = self.ps.copy()
            self.ps = (1 - self.cs) * self.ps + np.sqrt(self.cs * (2 - self.cs) * self.mueff) * self.invsqrtC @ (self.mean - old_mean) / self.sigma
            
            # 计算||ps||
            hsig = (np.linalg.norm(self.ps) / np.sqrt(1 - (1 - self.cs)**(2 * generation + 2)) / self.chiN) < (1.4 + 2.0 / (self.dim + 1))
            
            self.pc = (1 - self.cc) * self.pc + hsig * np.sqrt(self.cc * (2 - self.cc) * self.mueff) * (self.mean - old_mean) / self.sigma
            
            # 更新协方差矩阵
            artmp = (selected - old_mean) / self.sigma
            self.C = (1 - self.c1 - self.cmu) * self.C + self.c1 * (np.outer(self.pc, self.pc) + (1 - hsig) * self.cc * (2 - self.cc) * self.C) + self.cmu * np.sum(self.weights[:, np.newaxis, np.newaxis] * artmp[:, :, np.newaxis] * artmp[:, np.newaxis, :], axis=0)
            
            # 更新步长
            self.sigma = self.sigma * np.exp((self.cs / self.damps) * (np.linalg.norm(self.ps) / self.chiN - 1))
            
            # 打印进度
            if (generation + 1) % 10 == 0:
                print(f"代数 {generation + 1}/{self.max_generations}, 当前最优遮蔽时间: {-self.best_fitness:.4f} s")
        
        return self.best_solution, -self.best_fitness


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

    # 设置初始参数
    initial_mean = (UPPER_BOUNDS + LOWER_BOUNDS) / 2.0  # 搜索空间中心作为初始均值
    initial_sigma = 0.3  # 初始步长（相对于搜索范围）
    
    # 初始化并运行 CMA-ES 优化器
    cmaes = CMAEvolutionStrategy(
        obj_func=objective_function_quiet,
        initial_mean=initial_mean,
        initial_sigma=initial_sigma,
        lower_bounds=LOWER_BOUNDS,
        upper_bounds=UPPER_BOUNDS,
        pop_size=POPULATION_SIZE,
        max_generations=MAX_GENERATIONS
    )
    
    start_time = time.time()
    best_params, max_covered_time = cmaes.optimize()
    end_time = time.time()
    
    print("\n-------------------------------------------------")
    print("问题2 优化完成")
    print("优化算法：协方差矩阵适应进化策略 (CMA-ES)")
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
