import numpy as np
import random
import pandas as pd

# ===== 基础参数 =====
UAV_init = [
    (17800, 0, 1800),
    (12000, 1400, 1400),
    (6000, -3000, 700),
    (11000, 2000, 1800),
    (13000, -2000, 1300)
]
missile_init = [
    (20000, 0, 2000),
    (19000, 600, 2100),
    (18000, -600, 1900)
]
missile_speed = 300.0
T_center = np.array([0, 200, 5])
RADIUS = 10
dt = 0.2
T_max = 65

# ===== 决策变量离散表 =====
tq = np.linspace(0, 15, 30)       # 投放延迟
tp = np.linspace(4, 10, 60)       # 起爆延迟
vel = np.linspace(70, 140, 70)    # UAV速度
agl = np.linspace(0, np.pi/2, 90) # 航向角（0~90度）

bounds = []
for _ in range(5):
    bounds.append(len(vel)-1)
    bounds.append(len(agl)-1)
    for _ in range(3):
        bounds.append(len(tq)-1)
        bounds.append(len(tp)-1)

# ===== 单烟幕干扰信息（严格按题目判定） =====
def smoke_cover_info(ui_idx, smoke_idx, params):
    vi = vel[params[ui_idx*8]]
    ang = agl[params[ui_idx*8+1]]
    UAV_vec = np.array([-vi*np.cos(ang), vi*np.sin(ang), 0.0])
    ui_pos = np.array(UAV_init[ui_idx])

    tq_val = tq[params[ui_idx*8 + 2 + smoke_idx*2]]
    tp_val = tp[params[ui_idx*8 + 3 + smoke_idx*2]]
    texplode = tq_val + tp_val

    times = np.arange(0, T_max, dt)
    mask_explode = (times >= texplode) & (times <= texplode+20)
    if not np.any(mask_explode):
        return 0.0, ""

    # 投放点和起爆点
    launch_pos = ui_pos + UAV_vec * tq_val
    explode_pos = launch_pos + UAV_vec * tp_val
    explode_pos[2] -= 0.5*9.8*(tp_val**2)
    cloud_pos_t = np.repeat(explode_pos[None,:], len(times), axis=0)
    cloud_pos_t[:,2] -= 3.0*(times - texplode)

    covered_any = np.zeros_like(times, dtype=bool)
    missiles_hit = []

    for m_idx in range(3):
        Mpos_init = np.array(missile_init[m_idx])
        PT0 = T_center - Mpos_init
        norm_PT0 = np.linalg.norm(PT0)
        Mpos_t = Mpos_init - (missile_speed * times[:,None]) * (PT0 / norm_PT0)

        PT = T_center - Mpos_t
        normPT = np.linalg.norm(PT, axis=1)
        PC = cloud_pos_t - Mpos_t
        d = np.linalg.norm(np.cross(PT, PC), axis=1) / (normPT + 1e-9)
        proj = np.sum(PC*PT, axis=1) / (normPT**2 + 1e-9)

        covered_m = mask_explode & (d <= RADIUS) & (proj >= 0) & (proj <= 1)
        if np.any(covered_m):
            missiles_hit.append(f"M{m_idx+1}")
        covered_any |= covered_m

    duration = np.sum(covered_any) * dt
    return duration, ",".join(missiles_hit)

# ===== 通用遮蔽计算，返回三枚导弹的时长 =====
def cover_times_for_all(params):
    times = np.arange(0, T_max, dt)
    UAV_v = []
    for i in range(5):
        vi = vel[params[i*8]]
        ang = agl[params[i*8+1]]
        UAV_v.append(np.array([-vi*np.cos(ang), vi*np.sin(ang), 0.0]))

    cover_times = []

    for m_idx in range(3):
        Mpos_init = np.array(missile_init[m_idx])
        PT0 = T_center - Mpos_init
        norm_PT0 = np.linalg.norm(PT0)
        covered_m = np.zeros_like(times, dtype=bool)

        for ui in range(5):
            ui_pos = np.array(UAV_init[ui])
            for k in range(3):
                tq_val = tq[params[ui*8 + 2 + k*2]]
                tp_val = tp[params[ui*8 + 3 + k*2]]
                texplode = tq_val + tp_val
                mask = (times >= texplode) & (times <= texplode+20)
                if not np.any(mask):
                    continue

                launch_pos = ui_pos + UAV_v[ui] * tq_val
                explode_pos = launch_pos + UAV_v[ui] * tp_val
                explode_pos[2] -= 0.5*9.8*(tp_val**2)
                pos_now = np.repeat(explode_pos[None,:], len(times), axis=0)
                pos_now[:,2] -= 3.0*(times - texplode)

                Mpos_t = Mpos_init - (missile_speed * times[:,None]) * (PT0 / norm_PT0)
                PT = T_center - Mpos_t
                normPT = np.linalg.norm(PT, axis=1)
                PC = pos_now - Mpos_t
                d = np.linalg.norm(np.cross(PT, PC), axis=1) / (normPT + 1e-9)
                proj = np.sum(PC*PT, axis=1) / (normPT**2 + 1e-9)

                covered_m |= mask & (d <= RADIUS) & (proj >= 0) & (proj <= 1)

        cover_times.append(np.sum(covered_m) * dt)

    return cover_times

# ===== 适应度函数（带惩罚，鼓励三导弹都有遮蔽） =====
def fitness_with_penalty(params):
    cts = cover_times_for_all(params)
    penalty = 0
    for t in cts:
        if t < 0.1:
            penalty -= 50   # 遮蔽不足惩罚
    return sum(cts) + penalty

# ===== GA 运行函数 =====
def run_ga(fitness_func, init_pop=None):
    pop_size = 100
    gens = 50
    mutation_rate = 0.4
    mutation_genes = 4
    elite_fraction = 0.1

    # 初始化种群
    if init_pop is None:
        pop = []
        for i in range(pop_size):
            ind = []
            for u in range(5):
                ind.append(random.randint(0, len(vel)-1))
                ind.append(random.randint(0, len(agl)-1))
                for k in range(3):
                    ind.append(random.randint(0, len(tq)-1))
                    ind.append(random.randint(0, len(tp)-1))
            # 种子朝向，确保覆盖不同导弹
            if i % 3 == 0:  # 朝 M1
                ind[1] = random.randint(0, len(agl)//3)
            elif i % 3 == 1:  # 朝 M2
                ind[1] = random.randint(len(agl)//3, 2*len(agl)//3)
            else:  # 朝 M3
                ind[1] = random.randint(2*len(agl)//3, len(agl)-1)
            pop.append(np.array(ind, dtype=int))
    else:
        pop = init_pop

    best = None
    best_fit = float("-inf")

    for g in range(gens):
        fits = [fitness_func(ind) for ind in pop]
        for ind, fit in zip(pop, fits):
            if fit > best_fit:
                best_fit = fit
                best = ind.copy()

        elite_count = max(1, int(elite_fraction * pop_size))
        elite_idx = np.argsort(fits)[-elite_count:]
        new_pop = [pop[i].copy() for i in elite_idx]

        while len(new_pop) < pop_size:
            p1 = pop[random.randint(0, pop_size-1)]
            p2 = pop[random.randint(0, pop_size-1)]
            if random.random() < 0.85:
                point = random.randint(1, len(bounds)-2)
                c1 = np.concatenate((p1[:point], p2[point:]))
                c2 = np.concatenate((p2[:point], p1[point:]))
            else:
                c1, c2 = p1.copy(), p2.copy()
            new_pop.append(c1)
            if len(new_pop) < pop_size:
                new_pop.append(c2)

        for i in range(elite_count, len(new_pop)):
            if random.random() < mutation_rate:
                for _ in range(mutation_genes):
                    m_idx = random.randint(0, len(bounds)-1)
                    new_pop[i][m_idx] = random.randint(0, bounds[m_idx])

        pop = new_pop
        if (g+1) % 10 == 0:
            print(f"Gen {g+1}: Best fitness = {best_fit:.2f}")

    return best, pop

# === 运行 GA ===
best_sol, _ = run_ga(fitness_with_penalty)

# === 输出结果到 Excel ===
rows = []
for i in range(5):
    v_i = vel[best_sol[i*8]]
    ang_i_rad = agl[best_sol[i*8+1]]
    ang_i_deg = np.degrees(ang_i_rad)
    UAV_vec = np.array([-v_i*np.cos(ang_i_rad), v_i*np.sin(ang_i_rad), 0.0])
    ui_pos = np.array(UAV_init[i])

    for k in range(3):
        tq_val = tq[best_sol[i*8+2+k*2]]
        tp_val = tp[best_sol[i*8+3+k*2]]
        launch_pos = ui_pos + UAV_vec * tq_val
        explode_pos = launch_pos + UAV_vec * tp_val
        explode_pos[2] -= 0.5*9.8*(tp_val**2)
        duration, missiles_hit = smoke_cover_info(i, k, best_sol)

        rows.append([
            f"FY{i+1}", round(ang_i_deg, 2), round(v_i, 2),
            f"烟幕{k+1}",
            round(launch_pos[0], 2), round(launch_pos[1], 2), round(launch_pos[2], 2),
            round(explode_pos[0], 2), round(explode_pos[1], 2), round(explode_pos[2], 2),
            round(duration, 2), missiles_hit
        ])

df = pd.DataFrame(rows, columns=[
    "无人机编号", "无人机运动方向", "无人机运动速度 (m/s)",
    "烟幕干扰弹编号",
    "烟幕干扰弹投放点x坐标 (m)", "烟幕干扰弹投放点y坐标 (m)", "烟幕干扰弹投放点z坐标 (m)",
    "烟幕干扰弹起爆点x坐标 (m)", "烟幕干扰弹起爆点y坐标 (m)", "烟幕干扰弹起爆点z坐标 (m)",
    "有效干扰时长 (s)", "干扰的导弹编号"
])
df.to_excel("result.xlsx", index=False)
print("\n最优策略已保存到 result.xlsx")

