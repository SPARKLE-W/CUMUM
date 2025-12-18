import numpy as np

# 位置参数
x = 12000
y = 1400
z = 1400

# 速度参数
v0 = 70

# 角度参数
angle = 0

# 时间参数

# 投放时间间隔
tq = np.linspace(9, 14, 100)
# 爆炸延时
tp = np.linspace(14, 20, 100)
# 云团有效时间段（单位: 秒），共20秒
t = np.linspace(20, 40, 100)
# 速度范围
v = np.linspace(80, 100, 100)
# 角度范围
agl = np.linspace(-20, -60, 100)

# 真目标中心（圆柱底面圆心）
T_center = np.array([0, 200, 5])  # 注意z取0，题意中底面圆心

# 时间步长
dt = t[1] - t[0]  # 步长，大约0.01秒

# 定义目标函数
def objective_function(params, dt):
    # 解码参数索引
    t_val0_idx = int(np.round(params[0]))
    t_val1_idx = int(np.round(params[1]))
    t_val_idx = int(np.round(params[2]))
    v_0_idx = int(np.round(params[3]))
    a_idx = int(np.round(params[4]))

    # 确保索引不越界
    t_val0_idx = min(t_val0_idx, len(tq) - 1)
    t_val1_idx = min(t_val1_idx, len(tp) - 1)
    t_val_idx = min(t_val_idx, len(t) - 1)
    v_0_idx = min(v_0_idx, len(v) - 1)
    a_idx = min(a_idx, len(agl) - 1)

    # 获取实际值
    t_val0 = tq[t_val0_idx]
    t_val1 = tp[t_val1_idx]
    t_val = t[t_val_idx]
    v_0 = v[v_0_idx]
    a = agl[a_idx]

    # 初始化遮蔽时长
    shielding_time = 0
    start_time = None
    for current_t in np.arange(t_val1, t_val, dt):
        # 计算云团和导弹位置
        x0 = x - v_0 * np.cos(a) * t_val0
        y0 = v_0 * np.sin(a) * t_val0
        z0 = z

        x1 = x0 - v_0 * np.cos(a) * (t_val1 - t_val0)
        y1 = y0 + v_0 * np.sin(a) * (t_val1 - t_val0)
        z1 = z - 0.5 * 9.8 * (t_val1 - t_val0) * (t_val1 - t_val0)
        tau = t_val - t_val1

        C = np.array([x1, y1, z1 - 3 * tau])  # 云团中心坐标

        # 计算导弹位置
        m1_x = 20000 - 3000 / np.sqrt(101) * current_t
        m1_z = 2000 - 300 / np.sqrt(101) * current_t
        P_m1 = np.array([m1_x, 0, m1_z])

        # 几何判断
        PT = T_center - P_m1
        PC = C - P_m1
        cross = np.cross(PT, PC)
        d = np.linalg.norm(cross) / (np.linalg.norm(PT) + 1e-7)
        proj = np.dot(PC, PT) / (np.linalg.norm(PT) ** 2 + 1e-7)

        # 遮蔽条件判断
        if d <= 10 and 0 <= proj <= 1:
            shielding_time += dt
            if start_time is None:  # 首次满足条件时记录
                start_time = current_t
                print(f"First shielding at {current_t:.2f}s")  # 实时打印

    return -shielding_time  # 注意返回负值

# 粒子群优化算法
class PSO:
    def __init__(self, objective_function, bounds, num_particles, max_iter, dt):
        self.objective_function = objective_function
        self.bounds = bounds
        self.num_particles = num_particles
        self.max_iter = max_iter
        self.dt = dt
        self.particles = []
        self.velocities = []
        self.pbest_positions = []
        self.pbest_scores = []
        self.gbest_position = None
        self.gbest_score = float('inf')
        self.w = 0.9  # 惯性权重
        self.c1 = 2.0  # 个体学习因子
        self.c2 = 1.6  # 群体学习因子

        # 初始化粒子
        for _ in range(num_particles):
            position = np.random.uniform(bounds[0], bounds[1], size=len(bounds[0]))
            velocity = np.random.uniform(-0.05, 0.05, size=len(bounds[0]))
            self.particles.append(position)
            self.velocities.append(velocity)
            score = objective_function(position, dt)
            self.pbest_positions.append(position)
            self.pbest_scores.append(score)
            if score < self.gbest_score:
                self.gbest_score = score
                self.gbest_position = position

    def optimize(self):
        for iteration in range(self.max_iter):
            # 动态调整惯性权重
            self.w = 0.9 - (0.9 - 0.4) * (iteration / self.max_iter)
            for i in range(self.num_particles):
                # 更新速度
                r1, r2 = np.random.rand(), np.random.rand()
                self.velocities[i] = (self.w * self.velocities[i] +
                                      self.c1 * r1 * (self.pbest_positions[i] - self.particles[i]) +
                                      self.c2 * r2 * (self.gbest_position - self.particles[i]))
                # 更新位置
                self.particles[i] += self.velocities[i]
                # 限制位置在边界内
                self.particles[i] = np.clip(self.particles[i], self.bounds[0], self.bounds[1])
                # 计算新位置的得分
                score = self.objective_function(self.particles[i], self.dt)
                # 更新个体最优解
                if score < self.pbest_scores[i]:
                    self.pbest_positions[i] = self.particles[i]
                    self.pbest_scores[i] = score
                    # 更新全局最优解
                    if score < self.gbest_score:
                        self.gbest_score = score
                        self.gbest_position = self.particles[i]
            print(f"Iteration {iteration+1}/{self.max_iter}, Best Score: {-self.gbest_score}")
        return self.gbest_position, self.gbest_score

# 定义搜索空间的边界
bounds = (np.array([0, 0, 0, 0, 0]), np.array([len(tq)-1, len(tp)-1, len(t)-1, len(v)-1, len(agl)-1]))

# 初始化 PSO
pso = PSO(objective_function, bounds, num_particles=300, max_iter=80, dt=dt)

# 执行优化
best_position, best_score = pso.optimize()

# 解码最佳位置
best_t_val0_idx = min(int(np.round(best_position[0])), len(tq) - 1)
best_t_val1_idx = min(int(np.round(best_position[1])), len(tp) - 1)
best_t_val_idx = min(int(np.round(best_position[2])), len(t) - 1)
best_v_0_idx = min(int(np.round(best_position[3])), len(v) - 1)
best_a_idx = min(int(np.round(best_position[4])), len(agl) - 1)

best_t_val0 = tq[best_t_val0_idx]
best_t_val1 = tp[best_t_val1_idx]
best_t_val = t[best_t_val_idx]
best_v_0 = v[best_v_0_idx]
best_a = agl[best_a_idx]

print(f"Best Parameters: t_val0={best_t_val0}, t_val1={best_t_val1}, t_val={best_t_val}, v_0={best_v_0}, a={best_a}")
print(f"Best Score: {best_score}")