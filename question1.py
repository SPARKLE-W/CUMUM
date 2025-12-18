import numpy as np

# 时间参数
t = np.linspace(5.1, 25.1, 2000)  # 云团有效时间段（单位: 秒），共20秒

# 真目标中心（圆柱底面圆心）
T_center = np.array([0, 200, 5])  # 注意z取0，题意中底面圆心

遮蔽时长 = 0
dt = t[1] - t[0]  # 步长，大约0.01秒

for t_val in t:
    tau = t_val - 5.1  # 起爆后经过的时间

    # 云团中心坐标
    C = np.array([17188, 0, 1736.496 - 3 * tau])

    # M1 当前位置
    m1_x = 20000 - 3000 / np.sqrt(101) * t_val
    m1_y = 0
    m1_z = 2000 - 300 / np.sqrt(101) * t_val
    P_m1 = np.array([m1_x, m1_y, m1_z])

    # 构造向量
    PT = T_center - P_m1
    PC = C - P_m1

    # 叉乘，点到直线距离
    cross = np.cross(PT, PC)
    norm_PT = np.linalg.norm(PT)
    d = np.linalg.norm(cross) / norm_PT if norm_PT > 0 else np.inf

    # 投影参数，判断最近点是否在线段上
    proj = np.dot(PC, PT) / (norm_PT ** 2 + 1e-7)

    if d <= 10 and 0 <= proj <= 1:
        遮蔽时长 += dt

print(f"有效遮蔽时长：{遮蔽时长:.1f} 秒")