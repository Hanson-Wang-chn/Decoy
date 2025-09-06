# theta_drone = 8.522
# v_drone = 70.607
# t_drone_fly = 0.0
# t_delta_drop2 = 1.0
# t_delta_drop3 = 1.587
# t_decoy_delay_1 = 0.0
# t_decoy_delay_2 = 1.38
# t_decoy_delay_3 = 3.133

# theta_drone = 5.69
# v_drone = 123.6350
# t_drone_fly = 0.0
# t_delta_drop2 = 2.15713826
# t_delta_drop3 = 1.587
# t_decoy_delay_1 = 0.0
# t_decoy_delay_2 = 0.0
# t_decoy_delay_3 = 8.04550099



theta_drone_deg = {
    1: 4.59,   # FY1 航向
    2: 4.59,   # FY2 航向
    3: 10.09,   # FY3 航向
}
# 无人机速度（m/s, 要求 70~140）
v_drone = {
    1: 91.8321,   # FY1 速度
    2: 91.9806,   # FY2 速度
    3: 91.9793,   # FY3 速度
}
# 投放时间（s，自 t=0 起计）
t_drone_fly = {
    1: 1.2886,   # FY1 投放时间
    2: 1.0653,   # FY2 投放时间
    3: 1.2068,   # FY3 投放时间
}
# 可选：时间引信延迟（s），起爆时间 = 投放时间 + 引信延迟。
# 若不需要延迟，保持 0.0 即“投放即起爆”。
t_fuze = {
    1: 0.0827,    # FY1 延迟
    2: 0.3282,    # FY2 延迟
    3: 0.7065,    # FY3 延迟
}


# TODO:
theta_drone_1 = 14.18
v_drone_1 = 106.6019
t_drop_1_1 = 0.0000
t_explode_1_1 = 0.3911
t_drop_1_2 = 16.6732
t_explode_1_2 = 18.7967
t_drop_1_3 = 38.9718
t_explode_1_3 = 41.7227

theta_drone_2 = 287.52
v_drone_2 = 136.8857
t_drop_2_1 = 5.6832
t_explode_2_1 = 10.6251
t_drop_2_2 = 36.0395
t_explode_2_2 = 41.2725
t_drop_2_3 = 37.0813
t_explode_2_3 = 50.9454

theta_drone_3 = 28.28
v_drone_3 = 76.3145
t_drop_3_1 = 0.0000
t_explode_3_1 = 6.8346
t_drop_3_2 = 18.8298
t_explode_3_2 = 20.2160
t_drop_3_3 = 37.0082
t_explode_3_3 = 39.7457

theta_drone_4 = 345.14
v_drone_4 = 136.8857
t_drop_4_1 = 56.7133
t_explode_4_1 = 56.7133
t_drop_4_2 = 66.7518
t_explode_4_2 = 68.0260
t_drop_4_3 = 76.0378
t_explode_4_3 = 86.0979

theta_drone_5 = 129.49
v_drone_5 = 106.6278
t_drop_5_1 = 15.1756
t_explode_5_1 = 23.0816
t_drop_5_2 = 16.2015
t_explode_5_2 = 19.6596
t_drop_5_3 = 28.8745
t_explode_5_3 = 34.1075
