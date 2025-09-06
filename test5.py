import numpy as np
from utils.is_covered import is_covered

# Gravity constant (m/s^2)
g = 9.8

# Time step for simulation (s)
dt = 0.01

# Missile initial positions
missile_pos0 = {
    'M1': np.array([20000.0, 0.0, 2000.0]),
    'M2': np.array([19000.0, 600.0, 2100.0]),
    'M3': np.array([18000.0, -600.0, 1900.0]),
}

# Drone initial positions
drone_pos0 = {
    'FY1': np.array([17800.0, 0.0, 1800.0]),
    'FY2': np.array([12000.0, 1400.0, 1400.0]),
    'FY3': np.array([6000.0, -3000.0, 700.0]),
    'FY4': np.array([11000.0, 2000.0, 1800.0]),
    'FY5': np.array([13000.0, -2000.0, 1300.0]),
}

# Input parameters: modify these values as per your strategy
# theta_drone_i: flight direction in degrees (0 is positive x, counterclockwise increasing)
# v_drone_i: flight speed in m/s
# t_drop_i_j: drop time for smoke j of drone i (from t=0), set to -1 if not used
# t_explode_i_j: explode time for smoke j of drone i, must > t_drop_i_j if used

# theta_drone_1 = 14.18
# v_drone_1 = 106.6019
# t_drop_1_1 = 0.0000
# t_explode_1_1 = 0.3911
# t_drop_1_2 = 16.6732
# t_explode_1_2 = 18.7967
# t_drop_1_3 = 38.9718
# t_explode_1_3 = 41.7227

# theta_drone_2 = 287.52
# v_drone_2 = 136.8857
# t_drop_2_1 = 5.6832
# t_explode_2_1 = 10.6251
# t_drop_2_2 = 36.0395
# t_explode_2_2 = 41.2725
# t_drop_2_3 = 37.0813
# t_explode_2_3 = 50.9454

# theta_drone_3 = 28.28
# v_drone_3 = 76.3145
# t_drop_3_1 = 0.0000
# t_explode_3_1 = 6.8346
# t_drop_3_2 = 18.8298
# t_explode_3_2 = 20.2160
# t_drop_3_3 = 37.0082
# t_explode_3_3 = 39.7457

# theta_drone_4 = 345.14
# v_drone_4 = 136.8857
# t_drop_4_1 = 56.7133
# t_explode_4_1 = 56.7133
# t_drop_4_2 = 66.7518
# t_explode_4_2 = 68.0260
# t_drop_4_3 = 76.0378
# t_explode_4_3 = 86.0979

# theta_drone_5 = 129.49
# v_drone_5 = 106.6278
# t_drop_5_1 = 15.1756
# t_explode_5_1 = 23.0816
# t_drop_5_2 = 16.2015
# t_explode_5_2 = 19.6596
# t_drop_5_3 = 28.8745
# t_explode_5_3 = 34.1075

# TODO:
theta_drone_1 = -180.00
v_drone_1 = 70.00
t_drop_1_1 = 1.000
t_explode_1_1 = 4.000
t_drop_1_2 = 2.400
t_explode_1_2 = 6.000
t_drop_1_3 = 4.400
t_explode_1_3 = 8.000

theta_drone_2 = -90.00
v_drone_2 = 100.00
t_drop_2_1 = 7.000
t_explode_2_1 = 10.000
t_drop_2_2 = 100.0
t_explode_2_2 = 100.0
t_drop_2_3 = 100.0
t_explode_2_3 = 100.0

theta_drone_3 = 90.00
v_drone_3 = 130.00
t_drop_3_1 = 19.000
t_explode_3_1 = 22.000
t_drop_3_2 = 100.0
t_explode_3_2 = 100.0
t_drop_3_3 = 100.0
t_explode_3_3 = 100.0

theta_drone_4 = 0.00
v_drone_4 = 0.00
t_drop_4_1 = 100.0
t_explode_4_1 = 100.0
t_drop_4_2 = 100.0
t_explode_4_2 = 100.0
t_drop_4_3 = 100.0
t_explode_4_3 = 100.0

theta_drone_5 = 0.00
v_drone_5 = 0.00
t_drop_5_1 = 100.0
t_explode_5_1 = 100.0
t_drop_5_2 = 100.0
t_explode_5_2 = 100.0
t_drop_5_3 = 100.0
t_explode_5_3 = 100.0

# Main simulation
for m_name, m_pos0 in missile_pos0.items():
    # Compute total flight time for missile to reach fake target
    dist = np.linalg.norm(m_pos0)
    T = dist / 300.0
    times = np.arange(0, T + dt / 2, dt)  # Time points
    num_t = len(times)
    
    # Missile velocity vector
    dir_m = -m_pos0 / dist
    vel_m = 300.0 * dir_m
    
    # Missile positions over time
    m_pos_t = m_pos0[None, :] + vel_m[None, :] * times[:, None]
    
    print(f"For missile {m_name}:")
    
    # Dictionary to store covered arrays for each smoke
    covered_per_smoke = {}
    
    # Array for union covered (any smoke covering at each time step)
    total_covered = np.zeros(num_t, dtype=bool)
    
    for i in range(1, 6):
        # Get drone parameters
        theta = eval(f'theta_drone_{i}')
        theta_rad = np.deg2rad(theta)
        v = eval(f'v_drone_{i}')
        vel_d = v * np.array([np.cos(theta_rad), np.sin(theta_rad), 0.0])
        pos0_d = drone_pos0[f'FY{i}']
        
        for j in range(1, 4):
            # Get drop and explode times
            t_d = eval(f't_drop_{i}_{j}')
            t_e = eval(f't_explode_{i}_{j}')
            
            label = f'FY{i}_sm{j}'
            
            if t_d < 0 or t_e <= t_d:
                # Invalid: no coverage
                covered = np.zeros(num_t, dtype=bool)
            else:
                # Compute drop position
                pos_drop = pos0_d + vel_d * t_d
                
                # Relative time to explode
                rel_e = t_e - t_d
                
                # Explode position (projectile motion)
                pos_e = pos_drop + vel_d * rel_e + np.array([0.0, 0.0, -0.5 * g * rel_e**2])
                
                # Compute covered at each time step
                covered = np.zeros(num_t, dtype=bool)
                for idx, t in enumerate(times):
                    if t < t_e or t >= t_e + 20.0:
                        continue  # Not active
                    s = t - t_e
                    cloud_pos = pos_e.copy()
                    cloud_pos[2] -= 3.0 * s  # Cloud sinking
                    A = m_pos_t[idx]
                    B = cloud_pos
                    if is_covered(A, B):
                        covered[idx] = True
            
            # Store
            covered_per_smoke[label] = covered
            
            # Update total covered (union)
            total_covered = np.logical_or(total_covered, covered)
    
    # Print effective times for each smoke against this missile
    for label, cov in covered_per_smoke.items():
        dur = np.sum(cov) * dt
        print(f"{label} effective interference time: {dur:.2f} s")
    
    # Print total covered time (union) for this missile
    total_dur = np.sum(total_covered) * dt
    print(f"Total covered time: {total_dur:.2f} s")
    print()
