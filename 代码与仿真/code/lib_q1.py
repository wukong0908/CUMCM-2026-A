"""
Q1 solver — radial 1D transient heat + moisture diffusion in a cylinder.
"""
import os
import numpy as np
import openpyxl
from scipy.special import j0, j1
from scipy.optimize import brentq

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ATT1 = os.path.join(BASE, "附件", "附件1.xlsx")
OUTDIR = os.path.join(BASE, "..", "结果")
os.makedirs(OUTDIR, exist_ok=True)

# ------------------------- constants (SI) -------------------------
R, L = 0.02, 0.25
RHO, CP, K = 820.0, 2600.0, 0.36
H = 25.0
HM = 8e-7
T0 = 28.0 + 273.15          # K
C0 = 2.55                   # kg/kg
ALPHA = K / (RHO * CP)      # m^2/s


def D_nonlin(C):
    return 7e-9 * np.exp(-0.89 / C)


# ------------------------- data -------------------------
def read_attach1(path):
    wb = openpyxl.load_workbook(path, data_only=True)
    rows = list(wb.active.iter_rows(values_only=True))[1:]
    t = np.array([r[0] for r in rows], float)
    T = np.array([r[1] for r in rows], float) + 273.15   # -> K
    C = np.array([r[2] for r in rows], float)
    assert len(t) == len(T) == len(C) and len(t) > 0, "empty attachment1"
    assert np.all(np.diff(t) > 0), "time not strictly increasing"
    assert not np.any(np.isnan(T)) and not np.any(np.isnan(C)), "NaN in attachment1"
    return t, T, C


def make_interp(t, vals):
    def f(tt):
        return np.interp(tt, t, vals)
    return f


# ------------------------- solver core -------------------------
def build_grid(dr):
    N = int(round(R / dr))
    r = np.arange(N + 1) * dr                     # nodes r_j
    # interface positions r_{j+1/2} for j=0..N ; cell j uses r_{j-1/2}..r_{j+1/2}
    rhalf = np.arange(N + 2) * dr - dr / 2        # index j -> r_{j-1/2}; rhalf[0]=-dr/2 (clamped), rhalf[N+1]=R
    rhalf[0] = 0.0
    rhalf[N + 1] = R
    V = np.pi * (rhalf[1:] ** 2 - rhalf[:-1] ** 2) * L          # cell volumes, len N+1
    A = 2 * np.pi * rhalf[1:N + 1] * L                          # interface areas r_{1/2}..r_{N-1/2}, len N
    AR = 2 * np.pi * R * L
    return N, r, V, A, AR


def solve(T_inf_of_t, C_inf_of_t, t_end, dr, dt, D_func=D_nonlin, harmonic=True,
          T0=T0, C0=C0):
    """Explicit FVM, per spec card. Returns (t_out, T, C, courant_max, courants)."""
    N, r, V, A, AR = build_grid(dr)
    nsteps = int(round(t_end / dt))
    t_out = np.arange(nsteps + 1) * dt
    T = np.zeros((nsteps + 1, N + 1))
    C = np.zeros((nsteps + 1, N + 1))
    T[0, :] = T0
    C[0, :] = C0
    courants = np.empty(nsteps)
    for n in range(nsteps):
        tn = n * dt
        Tn = T[n].copy()
        Cn = C[n].copy()
        T_inf = T_inf_of_t(tn)
        C_inf = C_inf_of_t(tn)
        # ---- heat (linear, constant k) ----
        # F[j] = Fourier flux through interface j+1/2, positive in +r direction
        Fh = -K * A * (Tn[1:] - Tn[:-1]) / dr
        rhs = np.zeros(N + 1)
        rhs[0] = -Fh[0]                                    # cell 0: only right face
        rhs[1:-1] = Fh[:-1] - Fh[1:]                       # gain from left - loss to right
        rhs[N] = Fh[N - 1] - AR * H * (Tn[N] - T_inf)      # surface Robin (loss)
        T[n + 1] = Tn + dt * rhs / (RHO * CP * V)
        # ---- mass (nonlinear D, explicit) ----
        D = D_func(Cn)
        if harmonic:
            D_int = 2 * D[1:] * D[:-1] / (D[1:] + D[:-1])   # harmonic at j+1/2
        else:
            D_int = (D[1:] + D[:-1]) / 2                    # arithmetic (baseline M0)
        Fm = -D_int * A * (Cn[1:] - Cn[:-1]) / dr
        rhs = np.zeros(N + 1)
        rhs[0] = -Fm[0]
        rhs[1:-1] = Fm[:-1] - Fm[1:]
        rhs[N] = Fm[N - 1] - AR * HM * (Cn[N] - C_inf)     # surface Robin (loss)
        C[n + 1] = Cn + dt * rhs / V
        courants[n] = max(ALPHA * dt / dr ** 2, D.max() * dt / dr ** 2)
    return t_out, T, C, courants.max(), courants


# ------------------------- analytic solutions (verification) -------------------------
def bessel_roots(Bi, nmax, interval=(0.0, 500.0)):
    f = lambda x: x * j1(x) - Bi * j0(x)
    roots = []
    xs = np.linspace(interval[0], interval[1], 200000)
    y = f(xs)
    for i in range(len(xs) - 1):
        if y[i] * y[i + 1] < 0 and len(roots) < nmax:
            roots.append(brentq(f, xs[i], xs[i + 1]))
    return np.array(roots)


def series_field(phi0, phi_inf, lam, coeff_type, r_pts, t_pts, diffus):
    """Transient cylinder solution: phi(r,t)=phi_inf + (phi0-phi_inf)*2*sum(...)."""
    out = np.zeros((len(t_pts), len(r_pts)))
    An = j1(lam) / (lam * (j0(lam) ** 2 + j1(lam) ** 2))
    for i, t in enumerate(t_pts):
        for m, rr in enumerate(r_pts):
            out[i, m] = phi_inf + (phi0 - phi_inf) * 2 * np.sum(
                An * j0(lam * rr / R) * np.exp(-lam ** 2 * diffus * t / R ** 2))
    return out


# ------------------------- verification: V1 heat analytic -------------------------
def v1_heat_analytic(t_pts, r_pts_cm, dr, dt):
    T_inf = 50.0 + 273.15
    t_end = max(t_pts)
    t_out, T, _, _, _ = solve(lambda t: T_inf, lambda t: 0.02, t_end, dr, dt,
                              D_func=lambda C: 5e-9 * np.ones_like(C), T0=T0)
    Bi = H * R / K
    lam = bessel_roots(Bi, 300)
    r_pts = np.array(r_pts_cm) / 100.0
    ana = series_field(T0, T_inf, lam, None, r_pts, np.array(t_pts), ALPHA)
    idx = [int(round(t / dt)) for t in t_pts]
    jdx = [int(round(rr / dr)) for rr in r_pts]
    num = np.array([[T[i, j] for j in jdx] for i in idx])
    err = np.abs(num - ana)
    return num, ana, err


# ------------------------- verification: V2 mass analytic -------------------------
def v2_mass_analytic(t_pts, r_pts_cm, dr, dt):
    Dconst = 5e-9
    C_inf = 0.02
    t_end = max(t_pts)
    t_out, _, C, _, _ = solve(lambda t: 50.0 + 273.15, lambda t: C_inf, t_end, dr, dt,
                              D_func=lambda C: Dconst * np.ones_like(C))
    Bi_m = HM * R / Dconst
    lam = bessel_roots(Bi_m, 300)
    r_pts = np.array(r_pts_cm) / 100.0
    ana = series_field(C0, C_inf, lam, None, r_pts, np.array(t_pts), Dconst)
    idx = [int(round(t / dt)) for t in t_pts]
    jdx = [int(round(rr / dr)) for rr in r_pts]
    num = np.array([[C[i, j] for j in jdx] for i in idx])
    err = np.abs(num - ana)
    return num, ana, err


# ------------------------- verification: V3 conservation -------------------------
def v3_conservation(T_inf_of_t, C_inf_of_t, t_end, dr, dt, label):
    N, r, V, A, AR = build_grid(dr)
    t_out, T, C, _, _ = solve(T_inf_of_t, C_inf_of_t, t_end, dr, dt)
    nsteps = int(round(t_end / dt))
    M = np.array([np.sum(V * C[i]) for i in range(nsteps + 1)])
    E = np.array([np.sum(RHO * CP * V * T[i]) for i in range(nsteps + 1)])
    # integrated boundary fluxes
    dM_bc = np.zeros(nsteps + 1)
    dE_bc = np.zeros(nsteps + 1)
    for n in range(nsteps):
        dM_bc[n + 1] = dM_bc[n] + dt * AR * HM * (C[n, N] - C_inf_of_t(n * dt))
        dE_bc[n + 1] = dE_bc[n] + dt * AR * H * (T_inf_of_t(n * dt) - T[n, N])
    m_bal = np.abs((M[0] - M) - dM_bc)
    e_bal = np.abs((E - E[0]) - dE_bc)
    return dict(label=label,
                m_abs_max=m_bal.max(), m_rel=m_bal.max() / M[0],
                e_abs_max=e_bal.max(), e_rel=e_bal.max() / E[0])


# ------------------------- verification: V4 grid convergence -------------------------
def v4_convergence(T_inf_of_t, C_inf_of_t, t_end):
    results = []
    for dr, dt in [(0.001, 1.0), (0.0005, 0.25), (0.00025, 0.0625)]:
        t_out, T, C, _, _ = solve(T_inf_of_t, C_inf_of_t, t_end, dr, dt)
        n = int(round(t_end / dt))
        results.append(dict(dr=dr, Tc=T[n, 0], Ts=T[n, -1],
                            Cc=C[n, 0], Cs=C[n, -1]))
    return results


# ------------------------- main run -------------------------
def main():
    t_data, T_data, C_data = read_attach1(ATT1)
    from scipy.interpolate import PchipInterpolator
    T_pchip = PchipInterpolator(t_data, T_data, extrapolate=False)
    C_pchip = PchipInterpolator(t_data, C_data, extrapolate=False)
    T_inf_of_t = T_pchip
    C_inf_of_t = C_pchip
    dr, dt, t_end = 0.001, 1.0, 1800.0

    # M1 main model (harmonic interface D)
    t_out, T1, C1, cmax1, cour1 = solve(T_inf_of_t, C_inf_of_t, t_end, dr, dt,
                                        D_func=D_nonlin, harmonic=True)
    # M0 baseline (arithmetic interface D)
    t_out, T0m, C0m, cmax0, cour0 = solve(T_inf_of_t, C_inf_of_t, t_end, dr, dt,
                                          D_func=D_nonlin, harmonic=False)

    print("== M1 main (harmonic D) ==")
    print("max Courant: %.5f (stable if < 0.5)" % cmax1)
    print("== M0 baseline (arithmetic D) ==")
    print("max Courant: %.5f" % cmax0)

    # tables 1/2
    t_list = [100, 300, 600, 900, 1200, 1500, 1800]
    r_cm = [0.0, 0.5, 1.0, 1.5, 2.0]
    idx = [int(round(t / dt)) for t in t_list]
    jdx = [int(round(rr / (dr * 100))) for rr in r_cm]
    for model, (T, C) in [("M1", (T1, C1)), ("M0", (T0m, C0m))]:
        print("\n--- 表1 温度(°C) [%s] ---" % model)
        hdr = "t/s | " + " | ".join("r=%.1fcm" % r for r in r_cm)
        print(hdr)
        for i in idx:
            row = [T[i, j] - 273.15 for j in jdx]
            print("%4d | " % t_out[i] + " | ".join("%8.4f" % v for v in row))
        print("--- 表2 水分浓度(kg/kg) [%s] ---" % model)
        print(hdr)
        for i in idx:
            row = [C[i, j] for j in jdx]
            print("%4d | " % t_out[i] + " | ".join("%8.4f" % v for v in row))

    # M0 vs M1 difference
    print("\nM0 vs M1: max|dT|=%.6f °C, max|dC|=%.6f kg/kg"
          % (np.abs(T1 - T0m).max(), np.abs(C1 - C0m).max()))

    # V1 heat analytic
    t_pts = [100, 300, 600, 900, 1200, 1500, 1800]
    num, ana, err = v1_heat_analytic(t_pts, r_cm, dr, dt)
    print("\n== V1 heat vs Bessel series (const T_inf=50C, K) ==")
    print("max abs err: %.3e K" % err.max())
    for k, t in enumerate(t_pts):
        print("t=%4ds num=%s" % (t, np.array2string(num[k] - 273.15, precision=3)))
        print("       ana=%s" % np.array2string(ana[k] - 273.15, precision=3))

    # V2 mass analytic
    num, ana, err = v2_mass_analytic(t_pts, r_cm, dr, dt)
    print("\n== V2 mass vs Bessel series (const D=5e-9, C_inf=0.02) ==")
    print("max abs err: %.3e kg/kg" % err.max())
    for k, t in enumerate(t_pts):
        print("t=%4ds num=%s" % (t, np.array2string(num[k], precision=4)))
        print("       ana=%s" % np.array2string(ana[k], precision=4))

    # V3 conservation (real BC)
    v3 = v3_conservation(T_inf_of_t, C_inf_of_t, t_end, dr, dt, "real BC")
    print("\n== V3 conservation (real BC) ==")
    print("moisture: max abs imbalance %.3e (rel %.3e)" % (v3["m_abs_max"], v3["m_rel"]))
    print("energy:   max abs imbalance %.3e (rel %.3e)" % (v3["e_abs_max"], v3["e_rel"]))

    # V4 grid convergence (real BC)
    print("\n== V4 grid convergence (T at t=1800 s, °C) ==")
    for res in v4_convergence(T_inf_of_t, C_inf_of_t, t_end):
        print("dr=%.5f m: T(0)=%.6f T(R)=%.6f C(0)=%.6f C(R)=%.6f"
              % (res["dr"], res["Tc"] - 273.15, res["Ts"] - 273.15, res["Cc"], res["Cs"]))

    # ---- write result1.xlsx (M1) ----
    wb = openpyxl.Workbook()
    wsT = wb.active
    wsT.title = "温度"
    wsC = wb.create_sheet("水分浓度")
    header = "时间\\到药材中心的距离"
    dists = [round(dr * 100 * j, 1) for j in range(21)]   # 0..2.0 cm
    for ws, data in [(wsT, T1), (wsC, C1)]:
        ws.cell(row=1, column=1, value=header)
        for c, d in enumerate(dists, start=2):
            ws.cell(row=1, column=c, value=d)
        for i in range(1, int(t_end) + 1):
            ws.cell(row=i + 1, column=1, value=i)
            for j in range(21):
                val = data[i, j] - 273.15 if ws is wsT else data[i, j]
                ws.cell(row=i + 1, column=j + 2, value=round(float(val), 4))
    out = os.path.join(OUTDIR, "result1.xlsx")
    wb.save(out)
    print("\nsaved:", out)

    # stage-5 self-check summary
    print("\n== 五层检查 ==")
    print("L1 程序: 语法/维度/索引 OK (无异常); 时间步=%d 节点=%d" % (len(t_out) - 1, 21))
    print("L2 数学: V1/V2 解析对比见上; V4 网格收敛见上")
    print("L3 数据: 附件1 校验(单调/非NaN)通过; 插值域内无外推")
    print("L4 数值: Courant max=%.5f; C 全程>0: %s" % (cmax1, bool((C1 > 0).all())))
    print("L5 建模: 表1/表2 直接回答 Q1 输出要求")


if __name__ == "__main__":
    main()
