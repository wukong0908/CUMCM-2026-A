# -*- coding: utf-8 -*-
"""
Q1 Stage 6 verification / robustness suite.
R1: 2D axisymmetric finite cylinder (r,z) vs 1D infinite cylinder (midplane)
R2: parameter perturbation +-10%
R3: smoothed BC vs raw linear-interp BC
R4: attachment-1 data perturbation
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import lib_q1 as S

R, L = S.R, S.L
dr1, dt1 = 0.001, 1.0
t_pts = [100, 300, 600, 900, 1200, 1500, 1800]
r_cm = [0.0, 0.5, 1.0, 1.5, 2.0]
idx = [int(round(t / dt1)) for t in t_pts]
jdx = [int(round(rr / (dr1 * 100))) for rr in r_cm]

t_data, T_data, C_data = S.read_attach1(S.ATT1)
T_inf_of_t = S.make_interp(t_data, T_data)
C_inf_of_t = S.make_interp(t_data, C_data)

# base solution
_, Tb, Cb, _, _ = S.solve(T_inf_of_t, C_inf_of_t, 1800.0, dr1, dt1, harmonic=True)
base = dict(T=Tb, C=Cb)


# ---------------- R1: 2D axisymmetric finite cylinder ----------------
def solve_2d(T_inf_of_t, C_inf_of_t, t_end, dr, dz, dt,
             D_func=S.D_nonlin, harmonic=True):
    """Explicit FVM, axisymmetric (r,z). z in [0, L/2], j=0 = midplane (sym),
    j=NZ = end face (Robin). r in [0, R], i=0 axis, i=NR surface (Robin)."""
    NR = int(round(R / dr))
    NZ = int(round((L / 2) / dz))
    rr = np.arange(NR + 1) * dr
    zz = np.arange(NZ + 1) * dz
    rh = np.arange(NR + 2) * dr - dr / 2
    rh[0] = 0.0
    rh[NR + 1] = R
    Vr = np.pi * (rh[1:] ** 2 - rh[:-1] ** 2)          # ring areas, len NR+1
    Ar = 2 * np.pi * rh[1:NR + 1]                      # r-interface perimeters, len NR
    AR_side = 2 * np.pi * R
    V = Vr[:, None] * dz                               # cell volumes (NR+1, NZ+1)
    Arz = Ar[:, None] * dz                             # r-face areas
    Az = Vr[:, None]                                   # z-face areas (both faces same)
    A_end = Vr.copy()                                  # end face areas
    nsteps = int(round(t_end / dt))
    T = np.full((NR + 1, NZ + 1), S.T0)
    C = np.full((NR + 1, NZ + 1), S.C0)
    for n in range(nsteps):
        tn = n * dt
        T_inf = T_inf_of_t(tn)
        C_inf = C_inf_of_t(tn)
        # --- heat ---
        Fr = -S.K * Arz * (T[1:, :] - T[:-1, :]) / dr          # r-flux at i+1/2, len NR
        Fz = -S.K * Az * (T[:, 1:] - T[:, :-1]) / dz           # z-flux at j+1/2, len NZ
        rhs = np.zeros_like(T)
        rhs[0, :] = -Fr[0, :]
        rhs[1:-1, :] = Fr[:-1, :] - Fr[1:, :]
        rhs[NR, :] = Fr[NR - 1, :] - AR_side * S.H * (T[NR, :] - T_inf) * dz
        rhs[:, 0] += -Fz[:, 0]
        rhs[:, 1:-1] += Fz[:, :-1] - Fz[:, 1:]
        rhs[:, NZ] += Fz[:, NZ - 1] - A_end * S.H * (T[:, NZ] - T_inf)
        T = T + dt * rhs / (S.RHO * S.CP * V)
        # --- mass ---
        D = D_func(C)
        if harmonic:
            Dr = 2 * D[1:, :] * D[:-1, :] / (D[1:, :] + D[:-1, :])
        else:
            Dr = (D[1:, :] + D[:-1, :]) / 2
        Dz = (D[:, 1:] + D[:, :-1]) / 2
        Fr = -Dr * Arz * (C[1:, :] - C[:-1, :]) / dr
        Fz = -Dz * Az * (C[:, 1:] - C[:, :-1]) / dz
        rhs = np.zeros_like(C)
        rhs[0, :] = -Fr[0, :]
        rhs[1:-1, :] = Fr[:-1, :] - Fr[1:, :]
        rhs[NR, :] = Fr[NR - 1, :] - AR_side * S.HM * (C[NR, :] - C_inf) * dz
        rhs[:, 0] += -Fz[:, 0]
        rhs[:, 1:-1] += Fz[:, :-1] - Fz[:, 1:]
        rhs[:, NZ] += Fz[:, NZ - 1] - A_end * S.HM * (C[:, NZ] - C_inf)
        C = C + dt * rhs / V
    return rr, zz, T, C


def r1_end_effect():
    dr, dz, dt = 0.002, 0.005, 0.5
    rr, zz, T2, C2 = solve_2d(T_inf_of_t, C_inf_of_t, 1800.0, dr, dz, dt)
    n = int(round(1800.0 / dt))
    NR, NZ = rr.shape[0] - 1, zz.shape[0] - 1
    # compare midplane (z=0) radial profiles at t=1800
    diffT = np.abs(T2[:, 0] - base["T"][-1, ::2])       # base nodes at even indices (dr1=1mm vs dr=2mm)
    diffC = np.abs(C2[:, 0] - base["C"][-1, ::2])
    print("== R1 end effect: 2D finite cylinder vs 1D (midplane z=0, t=1800 s) ==")
    print("max |dT| = %.3e K, max |dC| = %.3e kg/kg" % (diffT.max(), diffC.max()))
    print("end face surface T (2D) = %.4f C vs side surface T = %.4f C"
          % (T2[NR, NZ] - 273.15, T2[NR, 0] - 273.15))
    return diffT.max(), diffC.max()


# ---------------- R2: parameter perturbation ----------------
def r2_params():
    print("\n== R2 parameter perturbation +-10% (t=1800 s) ==")
    print("param | dT(0) C | dT(R) C | dC(0) | dC(R)")
    baseT0 = base["T"][-1, 0] - 273.15
    baseTR = base["T"][-1, -1] - 273.15
    baseC0 = base["C"][-1, 0]
    baseCR = base["C"][-1, -1]
    perturbs = {
        "k": ("K", 0.36 * 0.9, 0.36 * 1.1),
        "h": ("H", 25 * 0.9, 25 * 1.1),
        "h_m": ("HM", 8e-7 * 0.9, 8e-7 * 1.1),
        "rho*cp": ("RHO", 820 * 0.9, 820 * 1.1),
        "D0": ("D0", 0.9, 1.1),
    }
    worst = {}
    for name, (attr, lo, hi) in perturbs.items():
        rows = []
        for factor in (lo, hi):
            if attr == "D0":
                Dfunc = lambda C, f=factor: f * 7e-9 * np.exp(-0.89 / C)
            else:
                setattr(S, attr, factor)
                Dfunc = S.D_nonlin
            _, T, C, _, _ = S.solve(T_inf_of_t, C_inf_of_t, 1800.0, dr1, dt1,
                                    D_func=Dfunc, harmonic=True)
            rows.append((T[-1, 0] - 273.15, T[-1, -1] - 273.15, C[-1, 0], C[-1, -1]))
            if attr != "D0":
                setattr(S, attr, {"K": 0.36, "H": 25.0, "HM": 8e-7, "RHO": 820.0}[attr])
        dT0 = max(abs(r[0] - baseT0) for r in rows)
        dTR = max(abs(r[1] - baseTR) for r in rows)
        dC0 = max(abs(r[2] - baseC0) for r in rows)
        dCR = max(abs(r[3] - baseCR) for r in rows)
        worst[name] = max(dT0, dTR, dC0, dCR)
        print("%s | %+.4f | %+.4f | %+.6f | %+.6f" % (name, dT0, dTR, dC0, dCR))
    print("worst-case response:", worst)
    return worst


# ---------------- R3: smoothed BC ----------------
def r3_smooth_bc():
    print("\n== R3 smoothed BC (window 5) vs raw linear interp ==")
    w = 5
    Tp = np.pad(T_data, (w // 2, w // 2), mode="edge")
    Cp = np.pad(C_data, (w // 2, w // 2), mode="edge")
    Ts = np.convolve(Tp, np.ones(w) / w, mode="valid")
    Cs = np.convolve(Cp, np.ones(w) / w, mode="valid")
    T_inf_s = S.make_interp(t_data, Ts)
    C_inf_s = S.make_interp(t_data, Cs)
    _, T, C, _, _ = S.solve(T_inf_s, C_inf_s, 1800.0, dr1, dt1, harmonic=True)
    dT = np.abs(T - base["T"]).max()
    dC = np.abs(C - base["C"]).max()
    print("max |dT| = %.3e K, max |dC| = %.3e kg/kg" % (dT, dC))
    return dT, dC


# ---------------- R4: attachment1 perturbation ----------------
def r4_data_perturb():
    print("\n== R4 attachment1 perturbation ==")
    print("case | max|dT| K | max|dC| kg/kg")
    cases = {
        "T_inf +0.5C": (T_data + 0.5, C_data),
        "T_inf -0.5C": (T_data - 0.5, C_data),
        "C_inf +5%": (T_data, C_data * 1.05),
        "C_inf -5%": (T_data, C_data * 0.95),
    }
    res = {}
    for name, (Td, Cd) in cases.items():
        _, T, C, _, _ = S.solve(S.make_interp(t_data, Td), S.make_interp(t_data, Cd),
                                1800.0, dr1, dt1, harmonic=True)
        dT = np.abs(T - base["T"]).max()
        dC = np.abs(C - base["C"]).max()
        res[name] = (dT, dC)
        print("%s | %.3e | %.3e" % (name, dT, dC))
    return res


if __name__ == "__main__":
    r1_end_effect()
    r2_params()
    r3_smooth_bc()
    r4_data_perturb()
