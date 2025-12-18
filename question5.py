import numpy as np
import random
# 参数表

x0, y0, z0 = 17800, 0, 1800
x1, y1, z1 = 12000, 1400, 1400
x2, y2, z2 = 6000, -3000, 700
x3, y3, z3 = 11000, 2000, 1800
x4, y4, z4 = 13000, -2000, 1300

tq1 = np.linspace(0, 5, 50)
tp1 = np.linspace(5, 8, 100)
tq2 = np.linspace(6, 11, 50)
tp2 = np.linspace(11, 14, 100)
tq3 = np.linspace(12, 17, 50)
tp3 = np.linspace(17, 20, 100)
t1  = np.linspace(8, 28, 200)
t2  = np.linspace(14, 34, 200)
t3  = np.linspace(20, 40, 200)
v   = np.linspace(70, 140, 140)
agl = np.linspace(0, np.pi/2, 180)


T_center = np.array([0, 200, 5])
M1_init = np.array([20000, 0, 2000])

RADIUS = 10
dt = 0.1
T_max = 80
# 固定视线方向
PT = T_center - M1_init
norm_PT = np.linalg.norm(PT)

for i in range (0,5):
    x0, y0, z0 = eval(f"x{i}, y{i}, z{i}")

 #  适应度函数 
    def fitness(params):
        tq1_val = tq1[params[0]]
        tp1_val = tp1[params[1]]
        tq2_val = tq2[params[3]]
        tp2_val = tp2[params[4]]
        tq3_val = tq3[params[6]]
        tp3_val = tp3[params[7]]
        v0  = v[params[9]]
        ang = agl[params[10]]
        UAV_v = np.array([-v0*np.cos(ang), v0*np.sin(ang), 0])
        smokes = [(tq1_val, tp1_val), (tq2_val, tp2_val), (tq3_val, tp3_val)]
        times = np.arange(0, T_max, dt)
        covered = np.zeros_like(times, dtype=bool)
        for tq_val, tp_val in smokes:
            texplode = tq_val + tp_val
            for idx, current_t in enumerate(times):
                if not (texplode <= current_t <= texplode + 20):
                    continue
                launch_pos = np.array([x0, y0, z0]) + UAV_v * tq_val
                explode_pos = launch_pos + UAV_v * tp_val
                explode_pos[2] -= 0.5 * 9.8 * (tp_val**2)
                pos_current = explode_pos.copy()
                pos_current[2] -= 3.0 * (current_t - texplode)
                PC = pos_current - M1_init
                d = np.linalg.norm(np.cross(PT, PC)) / (norm_PT + 1e-9)
                proj = np.dot(PC, PT) / (norm_PT**2 + 1e-9)
                if d <= RADIUS and 0 <= proj <= 1:
                    covered[idx] = True
        return max(np.sum(covered) * dt, 1e-6)  # 防全零
    # GA 参数
    pop_size = 200
    gens = 250
    mutation_rate = 0.45
    mutation_genes = 3
    elite_fraction = 0.1  # 精英比例
    bounds = [
        len(tq1)-1, len(tp1)-1, len(t1)-1,
        len(tq2)-1, len(tp2)-1, len(t2)-1,
        len(tq3)-1, len(tp3)-1, len(t3)-1,
        len(v)-1,   len(agl)-1
    ]
    # 初始化（50%中值附近，50%随机）
    pop = []
    for i in range(pop_size):
        if i < pop_size//2:
            ind = [bounds[j]//2 + random.randint(-5,5) for j in range(len(bounds))]
            ind = [min(max(vv,0), bounds[j]) for j,vv in enumerate(ind)]
        else:
            ind = [random.randint(0, ub) for ub in bounds]
        pop.append(np.array(ind, dtype=int))
    best = None
    best_fit = -1
    # 遗传搜索
    for g in range(gens):
        fits = [fitness(ind) for ind in pop]
        # 更新全局最优
        for ind, fit in zip(pop, fits):
            if fit > best_fit:
                best_fit = fit
                best = ind.copy()
        if (g+1) % 20 == 0:
            print(f"Gen {g+1}: Best shielding = {best_fit:.4f}s")
        # 精英保留
        elite_count = max(1, int(elite_fraction * pop_size))
        elite_idx = np.argsort(fits)[-elite_count:]
        new_pop = [pop[i].copy() for i in elite_idx]
        # 锦标赛选择+交叉
        while len(new_pop) < pop_size:
            i1, i2 = random.randint(0, pop_size-1), random.randint(0, pop_size-1)
            p1 = pop[i1] if fits[i1] > fits[i2] else pop[i2]
            i3, i4 = random.randint(0, pop_size-1), random.randint(0, pop_size-1)
            p2 = pop[i3] if fits[i3] > fits[i4] else pop[i4]
            c1, c2 = p1.copy(), p2.copy()
            if random.random() < 0.85:  # 高交叉率
                point = random.randint(1, len(bounds)-2)
                c1[:point], c2[:point] = p2[:point].copy(), p1[:point].copy()
            new_pop.append(c1)
            if len(new_pop) < pop_size:
                new_pop.append(c2)
        # 变异
        for i in range(elite_count, pop_size):
            if random.random() < mutation_rate:
                changes = mutation_genes if i > pop_size//2 else 1
                for _ in range(changes):
                    m_idx = random.randint(0, len(bounds)-1)
                    new_pop[i][m_idx] = random.randint(0, bounds[m_idx])
        pop = new_pop
    # 局部爬山优化
    def hill_climb(ind, steps=3):
        global best_fit
        current = ind.copy()
        cur_fit = fitness(current)
        for _ in range(steps):
            for j in range(len(bounds)):
                new = current.copy()
                new[j] = max(0, min(bounds[j], new[j] + random.choice([-1, 1])))
                new_fit = fitness(new)
                if new_fit > cur_fit:
                    current, cur_fit = new, new_fit
        return current, cur_fit
    best, best_fit = hill_climb(best, steps=10)
    #  输出结果 
    print("\n=== 最优方案 ===")
    print("无人机i的结果")
    print(f"最大视线存在遮蔽时长（秒）：{best_fit:.4f}")
    print("tq1 =", tq1[best[0]], "tp1 =", tp1[best[1]])
    print("tq2 =", tq2[best[3]], "tp2 =", tp2[best[4]])
    print("tq3 =", tq3[best[6]], "tp3 =", tp3[best[7]])
    print("速度 =", v[best[9]], "m/s")
    print("角度 =", np.degrees(agl[best[10]]), "°")

