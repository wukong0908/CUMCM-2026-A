# -*- coding: utf-8 -*-
"""
2026 A题 问题三：全程强耦合 + Kirchhoff 势界面通量 + 烘干终点判据
输出：
  - result3.xlsx
  - 表 5 打印
  - 5 张可视化图
"""

import os
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from openpyxl import Workbook
import matplotlib.font_manager as fm

# 设置 Mac 上的中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Heiti TC', 'Hiragino Sans GB']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
# ============================================================
# 0. 路径
# ============================================================
ROOT = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(ROOT, 'output_q3')
os.makedirs(OUTPUT_DIR, exist_ok=True)

PCHIP_FILE = os.path.join(ROOT, '插值结果_PCHIP.xlsx')   # 附件1插值结果

# ============================================================
# 1. 物理参数
# ============================================================
R = 0.02          # 半径 m
L = 0.25          # 长度 m
T0 = 28.0         # 初始温度 °C
C0 = 2.55         # 初始水分浓度 kg/kg

h_T = 25.0        # 对流换热系数 W/(m^2·K)
h_m = 8.0e-7      # 对流传质系数 m/s

# 烘干终点判据
C_DRY = 0.15      # kg/kg

# 时间步长与网格
DT = 1.0          # s
DR = 1.0e-3       # m
N = int(round(R / DR))   # 20

# 全程时长（3 天）
T_END = 3 * 24 * 3600.0  # 259200 s


# ============================================================
# 2. 附录 3 物性
# ============================================================
def rho_fun(C):
    return 650.0 + 128.0 * C

def cp_fun(C):
    return 1450.0 + 2736.0 * C / (C + 1.0)

def k_fun(C):
    return 0.21 + 0.38 * C / (C + 1.0)

def D_fun(C, T):
    """T 单位 K"""
    C = np.maximum(C, 1e-8)
    return 2.4e-3 * np.exp(-0.45 / C) * np.exp(-3850.0 / T)


# ============================================================
# 3. Kirchhoff 势
# ============================================================
def H_fun(C):
    """
    H(C) = ∫_0^C e^{-0.45/s} ds
         = C e^{-0.45/C} - 0.45 E1(0.45/C)
    用 scipy 的 exp1 实现；若没有 scipy，用近似展开。
    """
    C = np.asarray(C, dtype=float)
    C = np.maximum(C, 1e-12)
    try:
        from scipy.special import exp1
        return C * np.exp(-0.45 / C) - 0.45 * exp1(0.45 / C)
    except ImportError:
        # 简易近似：小 C 时 e^{-0.45/C} 极小，H ≈ C e^{-0.45/C}
        return C * np.exp(-0.45 / C)


# ============================================================
# 4. 环境边界
# ============================================================
def load_environment(path=PCHIP_FILE):
    """
    读取附件1插值结果，返回 t_env, T_env, C_env
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f'找不到文件：{path}')
    df = pd.read_excel(path)
    cols = df.columns.tolist()
    if '时间(s)' in cols:
        t = df['时间(s)'].values.astype(float)
        Ta = df['温度(°C)'].values.astype(float)
        Ca = df['水分浓度(kg/kg)'].values.astype(float)
    else:
        t = df.iloc[:, 0].values.astype(float)
        Ta = df.iloc[:, 1].values.astype(float)
        Ca = df.iloc[:, 2].values.astype(float)
    idx = np.argsort(t)
    return t[idx], Ta[idx], Ca[idx]


def make_env_funcs(t_env, T_env, C_env, t_switch=4 * 3600.0,
                   T_hold=50.165, C_hold=0.04986):
    """
    0 ~ t_switch：附件1线性插值
    t_switch 以后：恒定值
    """
    def T_inf(t):
        if t <= t_switch:
            return float(np.interp(t, t_env, T_env))
        else:
            return T_hold

    def C_inf(t):
        if t <= t_switch:
            return float(np.interp(t, t_env, C_env))
        else:
            return C_hold

    return T_inf, C_inf


# ============================================================
# 5. 网格与几何
# ============================================================
def make_grid(R, N, L):
    dr = R / N
    r = np.linspace(0.0, R, N + 1)
    V = np.empty(N + 1)
    V[0] = math.pi * (0.5 * dr) ** 2 * L
    V[1:N] = 2.0 * math.pi * r[1:N] * dr * L
    V[N] = math.pi * (R ** 2 - (R - 0.5 * dr) ** 2) * L
    r_face = 0.5 * (r[:-1] + r[1:])
    A_face = 2.0 * math.pi * r_face * L
    A_surf = 2.0 * math.pi * R * L
    return r, dr, V, A_face, A_surf


# ============================================================
# 6. 单步显式推进（顺序：先热后质，Kirchhoff 势）
# ============================================================
def step_explicit(T, C, Ta, Ca, dt, r, dr, V, A_face, A_surf,
                  scheme='kirchhoff'):
    """
    scheme: 'kirchhoff' / 'harmonic' / 'arithmetic'
    """
    N = len(T) - 1
    T_new = T.copy()

    # ---------- 温度场 ----------
    k_node = k_fun(C)
    k_face = np.zeros(N)
    for j in range(N):
        kj, kj1 = k_node[j], k_node[j + 1]
        k_face[j] = 2.0 * kj * kj1 / (kj + kj1 + 1e-30)

    def flux_T(j):
        return -k_face[j] * A_face[j] * (T[j + 1] - T[j]) / dr

    rho_cp = rho_fun(C) * cp_fun(C)

    T_new[0] = T[0] + dt / (rho_cp[0] * V[0]) * (-flux_T(0))
    for j in range(1, N):
        T_new[j] = T[j] + dt / (rho_cp[j] * V[j]) * (flux_T(j - 1) - flux_T(j))
    q_in = flux_T(N - 1)
    q_conv = h_T * A_surf * (Ta - T[N])
    T_new[N] = T[N] + dt / (rho_cp[N] * V[N]) * (q_in + q_conv)

    # ---------- 水分场 ----------
    C_new = C.copy()

    if scheme == 'kirchhoff':
        H_node = H_fun(C)
        T_K = T_new + 273.15
        B_node = 2.4e-3 * np.exp(-3850.0 / T_K)
        B_face = 0.5 * (B_node[:-1] + B_node[1:])

        def flux_C(j):
            return -B_face[j] * A_face[j] * (H_node[j + 1] - H_node[j]) / dr
    else:
        D_node = D_fun(C, T_new + 273.15)
        D_face = np.zeros(N)
        for j in range(N):
            Dj, Dj1 = D_node[j], D_node[j + 1]
            if scheme == 'harmonic':
                D_face[j] = 2.0 * Dj * Dj1 / (Dj + Dj1 + 1e-30)
            else:
                D_face[j] = 0.5 * (Dj + Dj1)

        def flux_C(j):
            return -D_face[j] * A_face[j] * (C[j + 1] - C[j]) / dr

    C_new[0] = C[0] + dt / V[0] * (-flux_C(0))
    for j in range(1, N):
        C_new[j] = C[j] + dt / V[j] * (flux_C(j - 1) - flux_C(j))
    q_in_C = flux_C(N - 1)
    q_conv_C = h_m * A_surf * (Ca - C[N])
    C_new[N] = C[N] + dt / V[N] * (q_in_C + q_conv_C)

    return T_new, C_new


# ============================================================
# 7. 全程求解 + 烘干终点搜索
# ============================================================
def solve_full(scheme='kirchhoff', dt=DT, t_end=T_END,
               store_every=60):
    """
    返回 dict：
      t_hist, r, T_hist, C_hist, C_max_hist, t_dry, N, dr
    """
    t_env, T_env, C_env = load_environment()
    T_inf, C_inf = make_env_funcs(t_env, T_env, C_env)

    r, dr, V, A_face, A_surf = make_grid(R, N, L)

    n_steps = int(round(t_end / dt))
    n_store = n_steps // store_every + 1

    T = np.full(N + 1, T0)
    C = np.full(N + 1, C0)

    t_hist = np.zeros(n_store)
    T_hist = np.zeros((n_store, N + 1))
    C_hist = np.zeros((n_store, N + 1))

    t_hist[0] = 0.0
    T_hist[0] = T
    C_hist[0] = C

    t_dry = None
    C_max_hist = np.zeros(n_store)
    C_max_hist[0] = C.max()

    k_store = 1
    print(f'[INFO] 全程求解 scheme={scheme}, dt={dt}s, t_end={t_end}s')

    for n in range(1, n_steps + 1):
        t_now = n * dt
        Ta = T_inf(t_now)
        Ca = C_inf(t_now)

        T, C = step_explicit(T, C, Ta, Ca, dt, r, dr, V, A_face, A_surf,
                             scheme=scheme)

        C_max = C.max()

        if n % store_every == 0:
            t_hist[k_store] = t_now
            T_hist[k_store] = T
            C_hist[k_store] = C
            C_max_hist[k_store] = C_max
            k_store += 1

        if t_dry is None and C_max <= C_DRY:
            t_dry = t_now
            print(f'[INFO] 烘干终点命中 t_dry = {t_dry:.1f} s = {t_dry/3600:.2f} h')

        if n % 3600 == 0:
            print(f'  t={t_now/3600:6.2f} h  C_max={C_max:.4f}  '
                  f'C_center={C[0]:.4f}  C_surf={C[-1]:.4f}')

    t_hist = t_hist[:k_store]
    T_hist = T_hist[:k_store]
    C_hist = C_hist[:k_store]
    C_max_hist = C_max_hist[:k_store]

    return dict(t_hist=t_hist, r=r, T_hist=T_hist, C_hist=C_hist,
                C_max_hist=C_max_hist, t_dry=t_dry,
                N=N, dr=dr)


# ============================================================
# 8. 打印表 5
# ============================================================
def print_table5(t_hist, r, C_hist, t_dry):
    times_h = [6, 12, 18, 24, 30, 36, 42, 48, 54]
    dists_cm = [0.0, 0.5, 1.0, 1.5, 2.0]

    dr_cm = r[1] * 100 - r[0] * 100
    idx = [int(round(d / dr_cm)) for d in dists_cm]

    print('\n' + '=' * 78)
    print('表 5  药材烘干过程的水分浓度（kg/kg）')
    print('=' * 78)

    header = f'{"时间/h":>8} | ' + ' | '.join([f'{"r=" + str(d) + "cm":>8}'
                                                for d in dists_cm])
    print(header)
    print('-' * len(header))

    for th in times_h:
        t_target = th * 3600.0
        n = int(np.argmin(np.abs(t_hist - t_target)))
        vals = [C_hist[n, i] for i in idx]
        line = f'{th:>8} | ' + ' | '.join([f'{v:8.4f}' for v in vals])
        print(line)

    if t_dry is not None:
        n = int(np.argmin(np.abs(t_hist - t_dry)))
        vals = [C_hist[n, i] for i in idx]
        line = f'{"结束":>8} | ' + ' | '.join([f'{v:8.4f}' for v in vals])
        print(line)
        print('-' * len(header))
        print(f'烘干所需时间：{t_dry/3600:.2f} h = {t_dry/3600/24:.2f} 天')

    print('=' * 78 + '\n')


# ============================================================
# 9. 保存 result3.xlsx
# ============================================================
def write_result3(t_hist, r, C_hist, t_dry, out_path='result3.xlsx'):
    wb = Workbook()
    ws = wb.active
    ws.title = '水分浓度'

    dist_cm = np.round(r * 100, 1)
    header = ['时间\\到药材中心的距离'] + [float(d) for d in dist_cm]
    ws.append(header)

    for n in range(len(t_hist)):
        row = [int(round(t_hist[n]))] + [round(float(v), 4) for v in C_hist[n]]
        ws.append(row)

    if t_dry is not None:
        n = int(np.argmin(np.abs(t_hist - t_dry)))
        row = [int(round(t_dry))] + [round(float(v), 4) for v in C_hist[n]]
        ws.append(row)

    wb.save(out_path)
    print(f'[INFO] 已保存 {out_path}')


# ============================================================
# 10. 可视化 1：全场最大水分浓度 + 首达放大
# ============================================================
def plot_endpoint(res):
    t_hist = res['t_hist']
    C_max = res['C_max_hist']
    t_dry = res['t_dry']

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    ax = axes[0]
    ax.plot(t_hist / 3600.0, C_max, 'b-', lw=1.8, label='max C(r,t)')
    ax.axhline(C_DRY, color='r', ls='--', lw=1.2, label='判据 0.15')
    if t_dry is not None:
        ax.axvline(t_dry / 3600.0, color='k', ls=':', lw=1.2,
                   label=f't_dry={t_dry/3600:.2f} h')
    ax.set_xlabel('时间 t / h')
    ax.set_ylabel('全场最大水分浓度 / (kg/kg)')
    ax.set_title('全场最大水分浓度全程演化')
    ax.legend()
    ax.grid(True, ls='--', alpha=0.5)

    ax = axes[1]
    if t_dry is not None:
        mask = (t_hist >= t_dry - 3600 * 0.6) & (t_hist <= t_dry + 60)
        ax.plot(t_hist[mask] / 3600.0, C_max[mask], 'b-', lw=1.8)
        ax.axhline(C_DRY, color='r', ls='--', lw=1.2)
        ax.axvline(t_dry / 3600.0, color='k', ls=':', lw=1.2)
        ax.set_xlabel('时间 t / h')
        ax.set_ylabel('max C')
        ax.set_title('首达时刻放大窗（最后 0.6 h）')
        ax.grid(True, ls='--', alpha=0.5)

    fig.tight_layout()
    out = os.path.join(OUTPUT_DIR, 'q3_endpoint.png')
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f'[INFO] 已保存 {out}')


# ============================================================
# 11. 可视化 2：三种界面通量取法对比
# ============================================================
def plot_scheme_compare():
    schemes = ['kirchhoff', 'harmonic', 'arithmetic']
    labels = {'kirchhoff': 'Kirchhoff 势（本文）',
              'harmonic': '调和平均',
              'arithmetic': '算术平均'}
    colors = {'kirchhoff': 'b', 'harmonic': 'r', 'arithmetic': 'g'}

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for s in schemes:
        print(f'\n[INFO] 求解 scheme={s}')
        res = solve_full(scheme=s, dt=DT, t_end=T_END, store_every=600)
        t_hist = res['t_hist']
        C_hist = res['C_hist']
        ax.plot(t_hist / 3600.0, C_hist[:, 0],
                color=colors[s], lw=1.8, label=labels[s])

    ax.set_xlabel('时间 t / h')
    ax.set_ylabel('中心水分浓度 C(0,t) / (kg/kg)')
    ax.set_title('三种界面通量取法的全程对比')
    ax.legend()
    ax.grid(True, ls='--', alpha=0.5)
    fig.tight_layout()
    out = os.path.join(OUTPUT_DIR, 'q3_scheme_compare.png')
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f'[INFO] 已保存 {out}')


# ============================================================
# 12. 可视化 3：极轴稳定性探针
# ============================================================
def plot_stab_probe():
    C_freeze = 0.15
    alpha = k_fun(C_freeze) / (rho_fun(C_freeze) * cp_fun(C_freeze))
    dr = DR

    def run_probe(dt):
        x = alpha * dt / dr ** 2
        Nn = N
        T = np.zeros(Nn + 1)
        T[:] = 1e-6 * ((-1) ** np.arange(Nn + 1))
        T_hist = [T.copy()]
        for _ in range(1000):
            T_new = T.copy()
            for j in range(1, Nn):
                T_new[j] = T[j] + x * (T[j - 1] - 2 * T[j] + T[j + 1])
            T_new[0] = T[0] + 4 * x * (T[1] - T[0])
            T_new[Nn] = T[Nn] + x * (T[Nn - 1] - T[Nn])
            T = T_new
            T_hist.append(T.copy())
        return x, np.array(T_hist)

    x1, Th1 = run_probe(1.0)
    x2, Th2 = run_probe(2.0)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.semilogy(np.max(np.abs(Th1), axis=1), 'b-', lw=1.8,
                label=f'dt=1 s, x={x1:.3f}')
    ax.semilogy(np.max(np.abs(Th2), axis=1), 'r-', lw=1.8,
                label=f'dt=2 s, x={x2:.3f}')
    ax.set_xlabel('时间步 n')
    ax.set_ylabel('全场最大偏差 (对数坐标)')
    ax.set_title('极轴稳定性探针（冻结 C=0.15）')
    ax.legend()
    ax.grid(True, which='both', ls='--', alpha=0.5)
    fig.tight_layout()
    out = os.path.join(OUTPUT_DIR, 'q3_stab_probe.png')
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f'[INFO] 已保存 {out}')


# ============================================================
# 13. 可视化 4：不同时刻水分浓度剖面
# ============================================================
def plot_profiles(res):
    t_hist = res['t_hist']
    r = res['r']
    C_hist = res['C_hist']

    times_h = [6, 12, 18, 24, 30, 36, 42, 48, 54]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for th in times_h:
        t_target = th * 3600.0
        n = int(np.argmin(np.abs(t_hist - t_target)))
        ax.plot(r * 100, C_hist[n], lw=1.5, label=f'{th} h')

    ax.axhline(C_DRY, color='r', ls='--', lw=1.2, label='判据 0.15')
    ax.set_xlabel('到药材中心的距离 r / cm')
    ax.set_ylabel('水分浓度 C / (kg/kg)')
    ax.set_title('不同时刻水分浓度剖面')
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, ls='--', alpha=0.5)
    fig.tight_layout()
    out = os.path.join(OUTPUT_DIR, 'q3_profiles.png')
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f'[INFO] 已保存 {out}')


# ============================================================
# 14. 可视化 5：全程干燥时间序列
# ============================================================
def plot_process(res):
    t_hist = res['t_hist']
    C_hist = res['C_hist']
    C_max = res['C_max_hist']

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(t_hist / 3600.0, C_hist[:, 0], 'b-', lw=1.8, label='中心 r=0')
    ax.plot(t_hist / 3600.0, C_hist[:, -1], 'r-', lw=1.8, label='表面 r=2 cm')
    ax.plot(t_hist / 3600.0, C_max, 'k--', lw=1.5, label='全场最大')
    ax.axhline(C_DRY, color='g', ls=':', lw=1.2, label='判据 0.15')
    ax.set_xlabel('时间 t / h')
    ax.set_ylabel('水分浓度 C / (kg/kg)')
    ax.set_title('全程干燥时间序列')
    ax.legend()
    ax.grid(True, ls='--', alpha=0.5)
    fig.tight_layout()
    out = os.path.join(OUTPUT_DIR, 'q3_process.png')
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f'[INFO] 已保存 {out}')


# ============================================================
# 15. 主程序
# ============================================================
if __name__ == '__main__':
    # ---------- 主求解（Kirchhoff 势） ----------
    res = solve_full(scheme='kirchhoff', dt=DT, t_end=T_END, store_every=60)

    t_hist = res['t_hist']
    r      = res['r']
    C_hist = res['C_hist']
    t_dry  = res['t_dry']

    # ---------- 打印表 5 ----------
    print_table5(t_hist, r, C_hist, t_dry)

    # ---------- 保存 result3.xlsx ----------
    write_result3(t_hist, r, C_hist, t_dry,
                  out_path=os.path.join(OUTPUT_DIR, 'result3.xlsx'))

    # ---------- 可视化 ----------
    plot_endpoint(res)
    plot_profiles(res)
    plot_process(res)

    # ---------- 三种界面通量对比 ----------
    plot_scheme_compare()

    # ---------- 极轴稳定性探针 ----------
    plot_stab_probe()

    print('\n[INFO] 问题三全部完成。')