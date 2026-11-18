import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')   # 改用tk后端，避开pycharm interagg后端bug
import numpy as np
import matplotlib.pyplot as plt

# ---------------------- CEC2017 基准测试函数 ----------------------
def fun_F1(x):
    return np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (x[:-1] - 1) ** 2)

def fun_F2(x):
    return np.sum(np.abs(x) ** 2)

def fun_F3(x):
    dim = len(x)
    return np.sum(np.cumsum(x)**2)

def get_benchmark(fun_id):
    funs = {
        1: fun_F1,
        2: fun_F2,
        3: fun_F3
    }
    return funs[fun_id]

# ---------------------- 原始 AOO 算法 ----------------------
def AOO(N, MaxIter, Dim, lb, ub, fobj):
    pop = np.random.uniform(lb, ub, (N, Dim))
    fit = np.array([fobj(pop[i]) for i in range(N)])
    best_idx = np.argmin(fit)
    best_pos = pop[best_idx].copy()
    best_fit = fit[best_idx]
    curve = np.zeros(MaxIter)

    for t in range(MaxIter):
        for i in range(N):
            alpha = np.random.uniform(0, 1)
            beta = np.random.uniform(0, 1)
            if alpha < 0.5:
                pop[i] = pop[i] + beta * (best_pos - pop[i])
            else:
                pop[i] = best_pos + beta * (pop[i] - best_pos)
            pop[i] = np.clip(pop[i], lb, ub)
            new_fit = fobj(pop[i])
            if new_fit < fit[i]:
                fit[i] = new_fit
                if new_fit < best_fit:
                    best_fit = new_fit
                    best_pos = pop[i].copy()
        curve[t] = best_fit
    return best_fit, best_pos, curve

# ---------------------- EAOO 增强燕麦优化算法 ----------------------
def EAOO(N, MaxIter, Dim, lb, ub, fobj):
    # 1. 正弦混沌初始化
    pop = np.zeros((N, Dim))
    x0 = np.random.rand()
    for i in range(N):
        for d in range(Dim):
            x0 = np.sin(np.pi * x0)
            pop[i, d] = lb + (ub - lb) * x0

    fit = np.array([fobj(pop[i]) for i in range(N)])
    best_idx = np.argmin(fit)
    best_pos = pop[best_idx].copy()
    best_fit = fit[best_idx]
    curve = np.zeros(MaxIter)

    for t in range(MaxIter):
        # 非线性扰动因子
        a = 2 - (t / MaxIter) ** 2
        for i in range(N):
            alpha = np.random.uniform(0, 1)
            beta = np.random.uniform(0, 1)
            if alpha < 0.5:
                pop[i] = pop[i] + a * beta * (best_pos - pop[i])
            else:
                pop[i] = best_pos + a * beta * (pop[i] - best_pos)
            # 自适应t分布变异
            df = MaxIter - t + 1
            if np.random.rand() < 0.3:
                pop[i] = pop[i] + np.random.standard_t(df, size=Dim) * 0.05
            pop[i] = np.clip(pop[i], lb, ub)
            new_fit = fobj(pop[i])
            if new_fit < fit[i]:
                fit[i] = new_fit
                if new_fit < best_fit:
                    best_fit = new_fit
                    best_pos = pop[i].copy()
        curve[t] = best_fit
    return best_fit, best_pos, curve

# ---------------------- GWO 灰狼优化（对比算法） ----------------------
def GWO(N, MaxIter, Dim, lb, ub, fobj):
    pop = np.random.uniform(lb, ub, (N, Dim))
    fit = np.array([fobj(pop[i]) for i in range(N)])
    idx = np.argsort(fit)
    alpha_pos, beta_pos, delta_pos = pop[idx[0]].copy(), pop[idx[1]].copy(), pop[idx[2]].copy()
    alpha_fit, beta_fit, delta_fit = fit[idx[0]], fit[idx[1]], fit[idx[2]]
    curve = np.zeros(MaxIter)

    for t in range(MaxIter):
        a = 2 - t * (2 / MaxIter)
        for i in range(N):
            r1, r2 = np.random.rand(2)
            A1 = 2 * a * r1 - a
            C1 = 2 * r2
            D_alpha = np.abs(C1 * alpha_pos - pop[i])
            X1 = alpha_pos - A1 * D_alpha

            r1, r2 = np.random.rand(2)
            A2 = 2 * a * r1 - a
            C2 = 2 * r2
            D_beta = np.abs(C2 * beta_pos - pop[i])
            X2 = beta_pos - A2 * D_beta

            r1, r2 = np.random.rand(2)
            A3 = 2 * a * r1 - a
            C3 = 2 * r2
            D_delta = np.abs(C3 * delta_pos - pop[i])
            X3 = delta_pos - A3 * D_delta

            pop[i] = (X1 + X2 + X3) / 3
            pop[i] = np.clip(pop[i], lb, ub)
            new_fit = fobj(pop[i])
            if new_fit < fit[i]:
                fit[i] = new_fit
        idx = np.argsort(fit)
        alpha_pos, beta_pos, delta_pos = pop[idx[0]].copy(), pop[idx[1]].copy(), pop[idx[2]].copy()
        alpha_fit, beta_fit, delta_fit = fit[idx[0]], fit[idx[1]], fit[idx[2]]
        curve[t] = alpha_fit
    return alpha_fit, alpha_pos, curve

# ---------------------- 批量实验 + 统计 ----------------------
def run_exp(fun_id=2, N=30, MaxIter=500, Dim=30, lb=-100, ub=100, run_times=30):
    fobj = get_benchmark(fun_id)
    res_ea = []
    res_aoo = []
    res_gwo = []
    curve_ea_avg = np.zeros(MaxIter)
    curve_aoo_avg = np.zeros(MaxIter)
    curve_gwo_avg = np.zeros(MaxIter)

    for r in range(run_times):
        f_ea, p_ea, c_ea = EAOO(N, MaxIter, Dim, lb, ub, fobj)
        f_aoo, p_aoo, c_aoo = AOO(N, MaxIter, Dim, lb, ub, fobj)
        f_gwo, p_gwo, c_gwo = GWO(N, MaxIter, Dim, lb, ub, fobj)
        res_ea.append(f_ea)
        res_aoo.append(f_aoo)
        res_gwo.append(f_gwo)
        curve_ea_avg += c_ea
        curve_aoo_avg += c_aoo
        curve_gwo_avg += c_gwo
    curve_ea_avg /= run_times
    curve_aoo_avg /= run_times
    curve_gwo_avg /= run_times

    print("==== 统计结果 ({}次独立运行) ====".format(run_times))
    print(f"EAOO  | Mean:{np.mean(res_ea):.4e}, Std:{np.std(res_ea):.4e}, Min:{np.min(res_ea):.4e}")
    print(f"AOO   | Mean:{np.mean(res_aoo):.4e}, Std:{np.std(res_aoo):.4e}, Min:{np.min(res_aoo):.4e}")
    print(f"GWO   | Mean:{np.mean(res_gwo):.4e}, Std:{np.std(res_gwo):.4e}, Min:{np.min(res_gwo):.4e}")

    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.plot(curve_ea_avg, label="EAOO", linewidth=2)
    plt.plot(curve_aoo_avg, label="AOO", linewidth=2)
    plt.plot(curve_gwo_avg, label="GWO", linewidth=2)
    plt.xlabel("迭代次数")
    plt.ylabel("适应度值")
    plt.title("收敛曲线对比")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

if __name__ == "__main__":
    # fun_id:1=F1,2=F2,3=F3
    run_exp(fun_id=2, N=30, MaxIter=500, Dim=30, lb=-100, ub=100, run_times=30)
