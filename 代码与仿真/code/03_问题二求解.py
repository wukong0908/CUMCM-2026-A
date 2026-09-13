import os
import math
import numpy as np
import pandas as pd
from openpyxl import Workbook
from scipy.special import exp1, j0, j1, jn_zeros
# 路径与基本参数
ROOT = os.path.dirname(os.path.abspath(__file__))
PCHIP_FILE = os.path.join(ROOT, '..', '..', '结果', '插值结果_PCHIP.xlsx')
ATT1_FILE = os.path.join(ROOT, '..', '附件', '附件1.xlsx')
OUTPUT_DIR = os.path.join(ROOT, 'newpage2')
os.makedirs(OUTPUT_DIR, exist_ok=True)

R = 0.02          # 半径 m
L = 0.25          # 长度 m
T0 = 28.0         # 初始温度 °C
C0 = 2.55         # 初始水分浓度 kg/kg
h_T = 25.0        # 对流换热 W/(m^2 K)
h_m = 8.0e-7      # 对流传质 m/s



# 1. 环境数据读取

def load_environment(path=PCHIP_FILE, fallback=ATT1_FILE):
    """返回 (t, Ta, Ca)，单位 s, °C, kg/kg"""
    if os.path.exists(path):
        print(f'读取环境数据：{path}')
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

    print(f'未找到 {path}，回退到 {fallback}，线性插值到 1 s')
    df = pd.read_excel(fallback)
    t = df['时间'].values.astype(float)
    Ta = df['温度'].values.astype(float)
    Ca = df['水分浓度'].values.astype(float)
    t_fine = np.arange(t[0], t[-1] + 1, 1.0)
    return t_fine, np.interp(t_fine, t, Ta), np.interp(t_fine, t, Ca)



# 2. 附录 3 经验公式

def rho_of_C(C):
    return 650.0 + 128.0 * C

def cp_of_C(C):
    return 1450.0 + 2736.0 * C / (C + 1.0)

def k_of_C(C):
    return 0.21 + 0.38 * C / (C + 1.0)

def B_of_T(T_K):
    return 2.4e-3 * np.exp(-3850.0 / np.maximum(T_K, 1.0))

def D_of_C_T(C, T_K):
    return B_of_T(T_K) * np.exp(-0.45 / np.maximum(C, 1e-8))


# ---------- Kirchhoff 势 H(C) = ∫_0^C e^{-0.45/s} ds ----------
def H_of_C(C):
    """
    H(C) = C * exp(-0.45/C) - 0.45 * E1(0.45/C)
    对小 C 用 H(C) ≈ C 避免数值问题
    """
    C = np.asarray(C, dtype=float)
    H = np.zeros_like(C)
    small = C < 1e-6
    H[small] = C[small]
    Cl = C[~small]
    z = 0.45 / Cl
    H[~small] = Cl * np.exp(-z) - 0.45 * exp1(z)
    return H



# 3. 网格

def make_grid(R, N):
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



# 4. 单步显式推进（Kirchhoff 势界面通量）

def step_explicit_q2(T, C, Ta, Ca, dt, r, dr, V, A_face, A_surf):
    Nn = len(T) - 1

    # ---------- 温度场：用 C^n 计算物性 ----------
    T_new = T.copy()
    rho_n = rho_of_C(C)
    cp_n = cp_of_C(C)
    k_n = k_of_C(C)

    # 界面导热系数：调和平均
    k_face = np.zeros(Nn)
    for j in range(Nn):
        kj, kj1 = k_n[j], k_n[j + 1]
        k_face[j] = 2.0 * kj * kj1 / (kj + kj1 + 1e-30)

    def flux_T(j):
        return -k_face[j] * A_face[j] * (T[j + 1] - T[j]) / dr

    # 中心节点
    T_new[0] = T[0] + dt / (rho_n[0] * cp_n[0] * V[0]) * (-flux_T(0))
    # 内部节点
    for j in range(1, Nn):
        T_new[j] = T[j] + dt / (rho_n[j] * cp_n[j] * V[j]) * (
            flux_T(j - 1) - flux_T(j))
    # 表面节点
    q_in = flux_T(Nn - 1)
    q_conv = h_T * A_surf * (Ta - T[Nn])
    T_new[Nn] = T[Nn] + dt / (rho_n[Nn] * cp_n[Nn] * V[Nn]) * (q_in + q_conv)

    # ---------- 水分场：用 T^{n+1} 计算 D，Kirchhoff 势界面通量 ----------
    C_new = C.copy()
    T_K = T_new + 273.15
    B_face = 0.5 * (B_of_T(T_K[:-1]) + B_of_T(T_K[1:]))

    H_node = H_of_C(C)

    def flux_C(j):
        return -B_face[j] * A_face[j] * (H_node[j + 1] - H_node[j]) / dr

    C_new[0] = C[0] + dt / V[0] * (-flux_C(0))
    for j in range(1, Nn):
        C_new[j] = C[j] + dt / V[j] * (flux_C(j - 1) - flux_C(j))
    q_in_C = flux_C(Nn - 1)
    q_conv_C = h_m * A_surf * (Ca - C[Nn])
    C_new[Nn] = C[Nn] + dt / V[Nn] * (q_in_C + q_conv_C)

    return T_new, C_new



# 5. 备选界面格式（调和 / 算术平均）

def step_explicit_q2_alt(T, C, Ta, Ca, dt, r, dr, V, A_face, A_surf,
                         scheme='harmonic', D_const=None):
    Nn = len(T) - 1
    T_new = T.copy()
    rho_n = rho_of_C(C)
    cp_n = cp_of_C(C)
    k_n = k_of_C(C)
    k_face = np.zeros(Nn)
    for j in range(Nn):
        kj, kj1 = k_n[j], k_n[j + 1]
        if scheme == 'harmonic':
            k_face[j] = 2.0 * kj * kj1 / (kj + kj1 + 1e-30)
        else:
            k_face[j] = 0.5 * (kj + kj1)

    def flux_T(j):
        return -k_face[j] * A_face[j] * (T[j + 1] - T[j]) / dr

    T_new[0] = T[0] + dt / (rho_n[0] * cp_n[0] * V[0]) * (-flux_T(0))
    for j in range(1, Nn):
        T_new[j] = T[j] + dt / (rho_n[j] * cp_n[j] * V[j]) * (
            flux_T(j - 1) - flux_T(j))
    q_in = flux_T(Nn - 1)
    q_conv = h_T * A_surf * (Ta - T[Nn])
    T_new[Nn] = T[Nn] + dt / (rho_n[Nn] * cp_n[Nn] * V[Nn]) * (q_in + q_conv)

    C_new = C.copy()
    T_K = T_new + 273.15
    if D_const is not None:
        D_node = D_const * np.ones_like(C)
    else:
        D_node = D_of_C_T(C, T_K)

    D_face = np.zeros(Nn)
    for j in range(Nn):
        Dj, Dj1 = D_node[j], D_node[j + 1]
        if scheme == 'harmonic':
            D_face[j] = 2.0 * Dj * Dj1 / (Dj + Dj1 + 1e-30)
        else:
            D_face[j] = 0.5 * (Dj + Dj1)

    def flux_C(j):
        return -D_face[j] * A_face[j] * (C[j + 1] - C[j]) / dr

    C_new[0] = C[0] + dt / V[0] * (-flux_C(0))
    for j in range(1, Nn):
        C_new[j] = C[j] + dt / V[j] * (flux_C(j - 1) - flux_C(j))
    q_in_C = flux_C(Nn - 1)
    q_conv_C = h_m * A_surf * (Ca - C[Nn])
    C_new[Nn] = C[Nn] + dt / V[Nn] * (q_in_C + q_conv_C)
    return T_new, C_new



# 6. 问题二主求解

def solve_problem2(N=20, dt=1.0, t_end=10800.0,
                   t_env=None, Ta_env=None, Ca_env=None,
                   interface_scheme='kirchhoff',
                   verbose=True):
    if t_env is None:
        t_env, Ta_env, Ca_env = load_environment()

    r, dr, V, A_face, A_surf = make_grid(R, N)
    n_steps = int(round(t_end / dt))
    n_rows = n_steps + 1

    T = np.full(N + 1, T0)
    C = np.full(N + 1, C0)
    T_hist = np.zeros((n_rows, N + 1))
    C_hist = np.zeros((n_rows, N + 1))
    T_hist[0] = T
    C_hist[0] = C

    if verbose:
        print(f'问题二求解：N={N}, dr={dr*1000:.2f} mm, '
              f'dt={dt:.4f}s, t_end={t_end}s, scheme={interface_scheme}')

    for n in range(1, n_rows):
        t_now = n * dt
        Ta = float(np.interp(t_now, t_env, Ta_env))
        Ca = float(np.interp(t_now, t_env, Ca_env))

        if interface_scheme == 'kirchhoff':
            T, C = step_explicit_q2(T, C, Ta, Ca, dt, r, dr, V, A_face, A_surf)
        else:
            T, C = step_explicit_q2_alt(T, C, Ta, Ca, dt, r, dr, V,
                                        A_face, A_surf, scheme=interface_scheme)

        T_hist[n] = T
        C_hist[n] = C

        if verbose and n % 1800 == 0:
            print(f'  t={t_now:7.1f}s  T_surf={T[-1]:.4f}°C  '
                  f'C_surf={C[-1]:.4f}  C_center={C[0]:.4f}')

    t_arr = np.arange(n_rows) * dt
    return t_arr, r, T_hist, C_hist



# 7. 模型验证

# ---------- 解析解：无限长圆柱，恒定表面值 ----------
def analytic_cylinder(r, t, R, V0, Vinf, alpha, n_terms=300):
    lambdas = jn_zeros(0, n_terms)
    s = 0.0
    for lam in lambdas:
        s += (j0(lam * r / R) / (lam * j1(lam))) * \
             np.exp(-alpha * lam ** 2 * t / R ** 2)
    return Vinf + 2.0 * (V0 - Vinf) * s


# ---------- V1：热方程退化解析 ----------

# V1（修正版）：退化解析，第一类边界 T(R,t)=50 °C

def verify_V1_fixed():
    print('\n' + '=' * 60)
    print('V1 退化解析：冻结 C=2.55，表面恒定 T=50 °C（第一类边界）')
    print('=' * 60)
    C_freeze = 2.55
    rho = rho_of_C(C_freeze)
    cp = cp_of_C(C_freeze)
    k = k_of_C(C_freeze)
    alpha = k / (rho * cp)

    r, dr, V, A_face, A_surf = make_grid(R, 20)
    T = np.full(21, T0)
    C = np.full(21, C_freeze)

    # 强制第一类边界：每步把表面温度设为 50
    dt = 1.0
    T_snap = {}
    for n in range(1, 1801):
        T, _ = step_explicit_q2(T, C, 50.0, 0.02, dt, r, dr, V, A_face, A_surf)
        # 关键：强制表面温度为 50，模拟第一类边界
        T[-1] = 50.0
        if n in (100, 600, 1800):
            T_snap[n] = T.copy()

    print(f'{"t/s":>6} {"T_num(0)":>12} {"T_ana(0)":>12} '
          f'{"T_num(R)":>12} {"T_ana(R)":>12} {"err_max":>10}')
    err_max_all = 0.0
    for t_check in [100, 600, 1800]:
        T_num = T_snap[t_check]
        T_ana0 = analytic_cylinder(0.0, t_check, R, T0, 50.0, alpha)
        T_anaR = analytic_cylinder(R, t_check, R, T0, 50.0, alpha)
        err = max(abs(T_num[0] - T_ana0), abs(T_num[-1] - T_anaR))
        err_max_all = max(err_max_all, err)
        print(f'{t_check:>6} {T_num[0]:>12.6f} {T_ana0:>12.6f} '
              f'{T_num[-1]:>12.6f} {T_anaR:>12.6f} {err:>10.2e}')
    print(f'V1 最大误差 = {err_max_all:.3e} K')
    return err_max_all



# V2（修正版）：退化解析，第一类边界 C(R,t)=0.02，D=1e-8 常数

def verify_V2_fixed():
    print('\n' + '=' * 60)
    print('V2 退化解析：冻结 T=323.315 K，D=1e-8 常数，'
          '表面恒定 C=0.02（第一类边界）')
    print('=' * 60)
    D_const = 1e-8
    C_inf = 0.02

    r, dr, V, A_face, A_surf = make_grid(R, 20)
    T = np.full(21, 323.315 - 273.15)
    C = np.full(21, C0)

    dt = 1.0
    C_snap = {}
    for n in range(1, 1801):
        _, C = step_explicit_q2_alt(T, C, 50.0, C_inf, dt, r, dr, V,
                                    A_face, A_surf, scheme='harmonic',
                                    D_const=D_const)
        # 关键：强制表面浓度为 0.02，模拟第一类边界
        C[-1] = C_inf
        if n in (100, 600, 1800):
            C_snap[n] = C.copy()

    print(f'{"t/s":>6} {"C_num(0)":>12} {"C_ana(0)":>12} '
          f'{"C_num(R)":>12} {"C_ana(R)":>12} {"err_max":>10}')
    err_max_all = 0.0
    for t_check in [100, 600, 1800]:
        C_num = C_snap[t_check]
        C_ana0 = analytic_cylinder(0.0, t_check, R, C0, C_inf, D_const)
        C_anaR = analytic_cylinder(R, t_check, R, C0, C_inf, D_const)
        err = max(abs(C_num[0] - C_ana0), abs(C_num[-1] - C_anaR))
        err_max_all = max(err_max_all, err)
        print(f'{t_check:>6} {C_num[0]:>12.6f} {C_ana0:>12.6f} '
              f'{C_num[-1]:>12.6f} {C_anaR:>12.6f} {err:>10.2e}')
    print(f'V2 最大误差 = {err_max_all:.3e} kg/kg')
    return err_max_all

# ---------- V4：网格收敛性 ----------
def verify_V4():
    print('\n' + '=' * 60)
    print('V4 网格收敛性（0–3 h）')
    print('=' * 60)
    t_env, Ta_env, Ca_env = load_environment()

    x_target = 0.182
    alpha_max = k_of_C(1.0) / (rho_of_C(1.0) * cp_of_C(1.0))

    print(f'{"dr/mm":>8} {"dt/s":>8} {"T(0)":>12} {"T(R)":>12} '
          f'{"C(0)":>10} {"C(R)":>10}')
    rows = []
    for N in [20, 40, 80]:
        dr = R / N
        dt = x_target * dr ** 2 / alpha_max
        _, r, T_hist, C_hist = solve_problem2(
            N=N, dt=dt, t_end=10800.0,
            t_env=t_env, Ta_env=Ta_env, Ca_env=Ca_env,
            interface_scheme='kirchhoff', verbose=False)
        T_end = T_hist[-1]
        C_end = C_hist[-1]
        rows.append([dr * 1000, dt, T_end[0], T_end[-1], C_end[0], C_end[-1]])
        print(f'{dr*1000:>8.2f} {dt:>8.4f} {T_end[0]:>12.4f} '
              f'{T_end[-1]:>12.4f} {C_end[0]:>10.4f} {C_end[-1]:>10.4f}')

    print('\n相邻网格偏差：')
    for i in range(1, len(rows)):
        dr_p, dt_p, T0p, TRp, C0p, CRp = rows[i - 1]
        dr_c, dt_c, T0c, TRc, C0c, CRc = rows[i]
        dT = max(abs(T0c - T0p), abs(TRc - TRp))
        dC = max(abs(C0c - C0p), abs(CRc - CRp))
        print(f'dr: {dr_p:.2f} -> {dr_c:.2f} mm, '
              f'ΔT_max = {dT:.2e} °C, ΔC_max = {dC:.2e} kg/kg')


# ---------- V6：界面格式对比 ----------
def verify_V6():
    print('\n' + '=' * 60)
    print('V6 界面通量取法对比（0–3 h）')
    print('=' * 60)
    t_env, Ta_env, Ca_env = load_environment()

    _, _, _, C_k = solve_problem2(N=20, dt=1.0, t_end=10800.0,
                                  t_env=t_env, Ta_env=Ta_env, Ca_env=Ca_env,
                                  interface_scheme='kirchhoff', verbose=False)
    _, _, _, C_h = solve_problem2(N=20, dt=1.0, t_end=10800.0,
                                  t_env=t_env, Ta_env=Ta_env, Ca_env=Ca_env,
                                  interface_scheme='harmonic', verbose=False)
    _, _, _, C_a = solve_problem2(N=20, dt=1.0, t_end=10800.0,
                                  t_env=t_env, Ta_env=Ta_env, Ca_env=Ca_env,
                                  interface_scheme='arithmetic', verbose=False)

    dC_kh = np.max(np.abs(C_k - C_h))
    dC_ka = np.max(np.abs(C_k - C_a))
    print(f'Kirchhoff vs 调和平均：max|ΔC| = {dC_kh:.3e} kg/kg')
    print(f'Kirchhoff vs 算术平均：max|ΔC| = {dC_ka:.3e} kg/kg')
    print(f'3h 中心 C（Kirchhoff）= {C_k[-1][0]:.4f}')
    print(f'3h 表面 C（Kirchhoff）= {C_k[-1][-1]:.4f}')
    print(f'3h 中心 C（调和平均）= {C_h[-1][0]:.4f}')
    print(f'3h 表面 C（调和平均）= {C_h[-1][-1]:.4f}')
    print(f'3h 中心 C（算术平均）= {C_a[-1][0]:.4f}')
    print(f'3h 表面 C（算术平均）= {C_a[-1][-1]:.4f}')



# 8. 结果输出

def write_result2(t_arr, r, T_hist, C_hist):
    wb = Workbook()
    ws_T = wb.active
    ws_T.title = '温度'
    ws_C = wb.create_sheet('水分浓度')

    dist_cm = np.round(r * 100, 1)
    header = ['时间\\到药材中心的距离'] + [float(d) for d in dist_cm]

    for ws, data in [(ws_T, T_hist), (ws_C, C_hist)]:
        ws.append(header)
        for n in range(len(t_arr)):
            row = [int(round(t_arr[n]))] + [round(float(v), 4)
                                            for v in data[n]]
            ws.append(row)

    out_path = os.path.join(ROOT, '..', '..', '结果', 'result2.xlsx')
    wb.save(out_path)
    print(f'已保存 {out_path}')


def print_tables(t_arr, r, T_hist, C_hist):
    times_req_h = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    dists_req_cm = [0.0, 0.5, 1.0, 1.5, 2.0]
    dr_cm = r[1] * 100 - r[0] * 100
    idx = [int(round(d / dr_cm)) for d in dists_req_cm]

    print('\n表 3  3 小时内药材的温度（°C）')
    print(f'{"t/h":>6} | ' + ' | '.join([f'r={d}cm' for d in dists_req_cm]))
    for t_h in times_req_h:
        n = int(round(t_h * 3600 / (t_arr[1] - t_arr[0])))
        vals = [T_hist[n, i] for i in idx]
        print(f'{t_h:>6} | ' + ' | '.join([f'{v:10.4f}' for v in vals]))

    print('\n表 4  3 小时内药材的水分浓度（kg/kg）')
    print(f'{"t/h":>6} | ' + ' | '.join([f'r={d}cm' for d in dists_req_cm]))
    for t_h in times_req_h:
        n = int(round(t_h * 3600 / (t_arr[1] - t_arr[0])))
        vals = [C_hist[n, i] for i in idx]
        print(f'{t_h:>6} | ' + ' | '.join([f'{v:10.4f}' for v in vals]))

# 问题二可视化

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

OUTPUT_DIR = os.path.join(ROOT, 'outs')
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Songti SC',
                                   'Heiti SC', 'Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False


from scipy.interpolate import PchipInterpolator


def plot_q2(t_arr, r, T_hist, C_hist):
    """
    生成 4 张图：
    1) 温度随时间曲线
    2) 水分浓度随时间曲线
    3) 3h 横截面温度云图
    4) 3h 横截面水分浓度云图
    """
    # 选 5 个径向位置
    dists_cm = [0.0, 0.5, 1.0, 1.5, 2.0]
    dr_cm = r[1] * 100 - r[0] * 100
    idx = [int(round(d / dr_cm)) for d in dists_cm]

    # ---------- 图 1：温度随时间 ----------
    plt.figure(figsize=(7, 4.5))
    for d, i in zip(dists_cm, idx):
        plt.plot(t_arr / 3600.0, T_hist[:, i], label=f'r={d} cm')
    plt.xlabel('时间 t / h')
    plt.ylabel('温度 T / $^\\circ$C')
    #plt.title('3 小时内不同径向位置的温度')
    plt.legend()
    plt.grid(True, ls='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig_q2_T_time.png'), dpi=300)
    plt.close()

    # ---------- 图 2：水分浓度随时间 ----------
    plt.figure(figsize=(7, 4.5))
    for d, i in zip(dists_cm, idx):
        plt.plot(t_arr / 3600.0, C_hist[:, i], label=f'r={d} cm')
    plt.xlabel('时间 t / h')
    plt.ylabel('水分浓度 C / (kg/kg)')
    #plt.title('3 小时内不同径向位置的水分浓度')
    plt.legend()
    plt.grid(True, ls='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig_q2_C_time.png'), dpi=300)
    plt.close()

    # ---------- 图 3、4：3h 横截面云图 ----------
    T_snap = T_hist[-1]
    C_snap = C_hist[-1]

    # 关键改动 1：先把径向剖面插值到细网格，消除折线感
    r_fine = np.linspace(0.0, R, 400)
    T_fine = PchipInterpolator(r, T_snap)(r_fine)
    C_fine = PchipInterpolator(r, C_snap)(r_fine)

    Ngrid = 600
    x = np.linspace(-R, R, Ngrid)
    y = np.linspace(-R, R, Ngrid)
    X, Y = np.meshgrid(x, y)
    Rr = np.sqrt(X ** 2 + Y ** 2)

    # 关键改动 2：用细网格插值后的剖面构造二维场
    T2 = np.interp(Rr.ravel(), r_fine, T_fine).reshape(Rr.shape)
    C2 = np.interp(Rr.ravel(), r_fine, C_fine).reshape(Rr.shape)

    # 关键改动 3：圆外掩膜，边界更干净
    mask = Rr > R
    T2[mask] = np.nan
    C2[mask] = np.nan

    # 图 3：温度云图
    plt.figure(figsize=(5.5, 5))
    cf = plt.pcolormesh(X * 100, Y * 100, T2,
                        shading='gouraud', cmap='hot')
    plt.colorbar(cf, label='温度 T / $^\\circ$C')
    plt.gca().add_patch(Circle((0, 0), R * 100, fill=False,
                               color='k', lw=1.0))
    plt.axis('equal')
    plt.xlabel('x / cm')
    plt.ylabel('y / cm')
    #plt.title('3 h 药材横截面温度分布')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig_q2_T_disk.png'), dpi=300)
    plt.close()

    # 图 4：水分云图
    plt.figure(figsize=(5.5, 5))
    cf = plt.pcolormesh(X * 100, Y * 100, C2,
                        shading='gouraud', cmap='Blues')
    plt.colorbar(cf, label='水分浓度 C / (kg/kg)')
    plt.gca().add_patch(Circle((0, 0), R * 100, fill=False,
                               color='k', lw=1.0))
    plt.axis('equal')
    plt.xlabel('x / cm')
    plt.ylabel('y / cm')
    #plt.title('3 h 药材横截面水分浓度分布')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig_q2_C_disk.png'), dpi=300)
    plt.close()

    print(' 问题二 4 张图已保存到', OUTPUT_DIR)


# 9. 主程序

if __name__ == '__main__':
    # ---------- 主求解 ----------
    t_env, Ta_env, Ca_env = load_environment()
    t_arr, r, T_hist, C_hist = solve_problem2(
        N=20, dt=1.0, t_end=10800.0,
        t_env=t_env, Ta_env=Ta_env, Ca_env=Ca_env,
        interface_scheme='kirchhoff')

    write_result2(t_arr, r, T_hist, C_hist)
    print_tables(t_arr, r, T_hist, C_hist)
    plot_q2(t_arr, r, T_hist, C_hist)
    # ---------- 模型验证 ----------
    verify_V1_fixed()
    verify_V2_fixed()
    verify_V4()
    verify_V6()
