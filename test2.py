# test2.py

import numpy as np
from utils.is_covered import is_covered

# Constants
G = 9.8  # gravity m/s^2
DECOY_SINK_SPEED = 3.0  # m/s
DECOY_EFFECTIVE_TIME = 20.0  # s
MISSILE_SPEED = 300.0  # m/s
DRONE_ALTITUDE_CHANGE = False  # drone flies at constant height
TIME_STEP = 0.01  # s

# Initial positions
MISSILE_INIT = np.array([20000.0, 0.0, 2000.0])
DRONE_INIT = np.array([17800.0, 0.0, 1800.0])
FAKE_TARGET = np.array([0.0, 0.0, 0.0])

# Missile direction and velocity
missile_dist_init = np.linalg.norm(MISSILE_INIT)
missile_dir = -MISSILE_INIT / missile_dist_init
missile_vel = missile_dir * MISSILE_SPEED

# Total time to simulate: until missile hits fake target
total_time = missile_dist_init / MISSILE_SPEED
num_steps = int(total_time / TIME_STEP) + 1

# Input parameters
# TODO:
# theta_drone = 180.0  # degrees, x positive is 0, counterclockwise
# v_drone = 120.0  # m/s
# t_drone_fly = 1.5  # s, time flying before drop
# t_decoy_delay = 3.6  # s, delay from drop to explode

theta_drone = 8.522
v_drone = 70.607
t_drone_fly = 0.041
t_decoy_delay = 1.006
# 4.74
# 4.58

# Convert theta to radians
theta_rad = np.deg2rad(theta_drone)

# Drone velocity components (in xy plane, z=0)
drone_vel = np.array([
    v_drone * np.cos(theta_rad),
    v_drone * np.sin(theta_rad),
    0.0
])

# Drop time (from t=0)
t_drop = t_drone_fly

# Drone position at drop
pos_drop = DRONE_INIT + drone_vel * t_drop

# Decoy initial velocity (same as drone)
decoy_vel = drone_vel.copy()  # horizontal only

# Explode time (global)
t_explode = t_drop + t_decoy_delay

# Decoy position at explode
decoy_fall_time = t_decoy_delay
pos_explode = pos_drop + decoy_vel * decoy_fall_time - np.array([0.0, 0.0, 0.5 * G * decoy_fall_time**2])

# Simulation
covered_time = 0.0

for step in range(num_steps):
    t = step * TIME_STEP
    
    # Missile position at t
    pos_missile = MISSILE_INIT + missile_vel * t
    
    # Check if within effective time
    if t < t_explode or t - t_explode > DECOY_EFFECTIVE_TIME:
        continue
    
    # Decoy cloud position at t (sinking)
    t_since_explode = t - t_explode
    pos_cloud = pos_explode - np.array([0.0, 0.0, DECOY_SINK_SPEED * t_since_explode])
    
    # Check coverage
    if is_covered(pos_missile, pos_cloud):
        covered_time += TIME_STEP

print(f"Completely covered time: {covered_time:.2f} s")
