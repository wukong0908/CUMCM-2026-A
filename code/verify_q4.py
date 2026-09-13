# -*- coding: utf-8 -*-
"""
Q4 Stage 6 verification / robustness suite (per 规格卡_Q4.md §11).
V1: transform correctness — A(freeze_R) vs B(freeze_R) pointwise (machine precision)
V2: model comparison — A(moving) vs B(moving truncated) endpoints + sampled profiles
V3: grid convergence — A N=20 dt=1 vs N=40 dt=0.25 (x constant), endpoint
V5: degenerate analytic — freeze_R + freeze C/T vs Bessel series (appendix-4 props)
R1: parameter perturbation +-10% (7 knobs, full 3 days) -> endpoint
R2: attachment2 processing — smoothed R, R +-1% -> endpoint
R3: BC asymptote cross-check (50.15 C / 0.0507)
Results -> 结果/verify_q4_results.json
"""
import os
import sys
import json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import solve_q1 as S
import solve_q4 as Q4

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, "结果", "verify_q4_results.json")
T_END = 259200.0
r_cm = [0.0, 0.5, 1.0]
jdx = [int(round(rr / 0.1)) for rr in r_cm]
t_pts = [1800, 3600, 5400, 7200, 9000, 10800]

t_data, T_data, C_data = S.read_attach1(S.ATT1)
t2, R2 = Q4.read_attach2(Q4.ATT2)


def bc(nsteps, dt, T_tail=None, C_tail=None):
    return Q4.bc_arrays(t_data, T_data, C_data, nsteps, dt, T_tail, C_tail)


def endp(cmax, dt):
    hit = np.where(cmax <= 0.15)[0]
    return hit[0] * dt / 3600.0 if hit.size else None


def run_A(dt=1.0, N=20, DXI=0.05, t_end=T_END, store_every=1, **kw):
    nsteps = int(round(t_end / dt))
    T_inf, C_inf = bc(nsteps, dt)
    return Q4.solve_A(T_inf, C_inf, t_end, dt, N, DXI, store_every=store_every, **kw)


# ================= V1: A(freeze_R) vs B(freeze_R) =================
def v1_transform():
    print("== V1 transform correctness (A frozen vs B frozen, 3 days) ==", flush=True)
    nsteps = int(round(T_END / 1.0))
    T_inf, C_inf = bc(nsteps, 1.0)
    TA, CA, RA, RdA, xA, lA, cmA = Q4.solve_A(T_inf, C_inf, T_END, 1.0, freeze_R=True)
    TB, CB, RB = Q4.solve_B(T_inf, C_inf, T_END, 1.0, freeze_R=True)
    i72 = int(round(72 * 3600))
    dT = np.abs(TA - TB).max()
    dC = np.abs(CA - CB).max()
    print("  max|dT|=%.3e K  max|dC|=%.3e | Cc72 A=%.6f B=%.6f"
          % (dT, dC, CA[i72, 0], CB[i72, 0]), flush=True)
    return dict(dT=dT, dC=dC, Cc72_A=CA[i72, 0], Cc72_B=CB[i72, 0])


# ================= V2: A(moving) vs B(moving) =================
def v2_models():
    print("== V2 model comparison (A moving vs B moving, 3 days) ==", flush=True)
    nsteps = int(round(T_END / 1.0))
    T_inf, C_inf = bc(nsteps, 1.0)
    TA, CA, RA, RdA, xA, lA, cmA = Q4.solve_A(T_inf, C_inf, T_END, 1.0)
    TB, CB, RB = Q4.solve_B(T_inf, C_inf, T_END, 1.0)
    eA = endp(cmA, 1.0)
    eB = endp(np.nanmax(CB, axis=1), 1.0)
    print("  A: t_dry=%.4f h | B: t_dry=%s h" % (eA, "%.4f" % eB if eB else "None"),
          flush=True)
    diffs = []
    for th in (3.0, 24.0, 52.0):
        i = int(th * 3600)
        for rr in r_cm:
            a = Q4.interp_at(CA[i], RA[i], rr / 100.0)
            b = CB[i, jdx[r_cm.index(rr)]]
            diffs.append((th, rr, a, b, abs(a - b)))
            print("  t=%4.1fh r=%.1f: A=%.5f B=%.5f diff=%.2e"
                  % (th, rr, a, b, abs(a - b)), flush=True)
    return dict(t_dry_A=eA, t_dry_B=eB, max_diff=max(d[4] for d in diffs))


# ================= V3: grid convergence (N=40, dt=0.25) =================
def v3_convergence():
    print("== V3 grid convergence (N=40 dxi=0.025 dt=0.25, t_end=190000 s) ==",
          flush=True)
    t_end3 = 190000.0
    nsteps = int(round(t_end3 / 0.25))
    T_inf, C_inf = bc(nsteps, 0.25)
    T, C, Rarr, Rdot, xmax, led, cmax = Q4.solve_A(
        T_inf, C_inf, t_end3, 0.25, N=40, DXI=0.025, store_every=4)
    e40 = endp(cmax, 0.25)
    i52 = int(round(52 * 3600 / 0.25))
    print("  N=40: t_dry=%.4f h | Cc(52h)=%.6f | x_max=%.4f | ledger=%.2e"
          % (e40, C[i52 // 4, 0], xmax.max(), led.max()), flush=True)
    return dict(t_dry_N40=e40, Cc52_N40=C[i52 // 4, 0], x_max=xmax.max(),
                ledger=led.max())


# ================= V5: degenerate analytic (Bessel series) =================
def v5_analytic():
    print("== V5 degenerate analytic (freeze_R, frozen props, 3 h) ==", flush=True)
    r_m = [rr / 100.0 for rr in r_cm]
    out = {}
    for C_f in (2.55, 0.15):
        k = float(Q4.k_c(np.array(C_f)))
        rho = float(Q4.rho_c(np.array(C_f)))
        cp = float(Q4.cp_c(np.array(C_f)))
        alpha = k / (rho * cp)
        Bi = Q4.P['H'] * 0.02 / k
        T_inf_c = np.full(10801, 323.315)
        C_inf_c = np.full(10801, 0.04986)
        T, C, Rarr, Rd, xm, ld, cm = Q4.solve_A(
            T_inf_c, C_inf_c, 10800.0, 1.0, freeze_R=True, freeze_C=True, C_init=C_f)
        lam = S.bessel_roots(Bi, 300)
        ana = S.series_field(301.15, 323.315, lam, None, r_m, t_pts, alpha)
        num = np.array([[T[t, j] for j in jdx] for t in t_pts])
        err = float(np.abs(num - ana).max())
        out["heat_C%.2f" % C_f] = dict(Bi=Bi, alpha=alpha, max_err=err)
        print("  heat C_f=%.2f: Bi=%.4f alpha=%.3e max err=%.3e K"
              % (C_f, Bi, alpha, err), flush=True)
    Dc = 1e-8
    T_inf_c = np.full(10801, 323.315)
    C_inf_c = np.full(10801, 0.04986)
    T, C, Rarr, Rd, xm, ld, cm = Q4.solve_A(
        T_inf_c, C_inf_c, 10800.0, 1.0, freeze_R=True, freeze_T=True,
        D_func=lambda Cn: Dc * np.ones_like(Cn))
    Bi_m = Q4.P['HM'] * 0.02 / Dc
    lam = S.bessel_roots(Bi_m, 300)
    ana = S.series_field(2.55, 0.04986, lam, None, r_m, t_pts, Dc)
    num = np.array([[C[t, j] for j in jdx] for t in t_pts])
    err = float(np.abs(num - ana).max())
    out["mass_D1e-8"] = dict(Bi_m=Bi_m, max_err=err)
    print("  mass D=1e-8: Bi_m=%.1f max err=%.3e" % (Bi_m, err), flush=True)
    return out


# ================= R1: parameter perturbation =================
def r1_params():
    print("== R1 parameter perturbation +-10% (7 knobs, full 3 days) ==", flush=True)
    knobs = [("D0", ["D0"]), ("A_C", ["A_C"]), ("A_T", ["A_T"]),
             ("H", ["H"]), ("HM", ["HM"]),
             ("cp", ["cp0", "cp1"]), ("k", ["k0", "k1"])]
    nsteps = int(round(T_END / 1.0))
    T_inf, C_inf = bc(nsteps, 1.0)
    saved = dict(Q4.P)
    rows = []
    for name, keys in knobs:
        for f in (0.9, 1.1):
            for key in keys:
                Q4.P[key] = saved[key] * f
            try:
                T, C, Rarr, Rd, xm, ld, cm = Q4.solve_A(T_inf, C_inf, T_END, 1.0)
                hit = endp(cm, 1.0)
                rows.append(dict(knob=name, f=f, t_dry=hit,
                                 x_max=float(xm.max()), ledger=float(ld.max())))
                print("  %s x%.1f: t_dry=%s h | x_max=%.3f ledger=%.1e"
                      % (name, f, "%.3f" % hit if hit else "None",
                         xm.max(), ld.max()), flush=True)
            finally:
                for key in keys:
                    Q4.P[key] = saved[key]
    return rows


# ================= R2: attachment2 processing =================
def r2_attach2():
    print("== R2 attachment2 processing (smooth w5 / +-1%) ==", flush=True)
    w = 5
    Rp = np.pad(R2, (w // 2, w // 2), mode="edge")
    R2s = np.convolve(Rp, np.ones(w) / w, mode="valid")
    cases = {"smooth5": R2s, "plus1pct": R2 * 1.01, "minus1pct": R2 * 0.99}
    nsteps = int(round(T_END / 1.0))
    T_inf, C_inf = bc(nsteps, 1.0)
    rows = []
    for name, Rm in cases.items():
        T, C, Rarr, Rd, xm, ld, cm = Q4.solve_A(
            T_inf, C_inf, T_END, 1.0, R_override=(t2, Rm))
        hit = endp(cm, 1.0)
        rows.append(dict(case=name, t_dry=hit))
        print("  %s: t_dry=%s h" % (name, "%.3f" % hit if hit else "None"),
              flush=True)
    return rows


# ================= R3: BC asymptote cross-check =================
def r3_asymptote():
    print("== R3 asymptote BC cross-check (50.15 C / 0.0507) ==", flush=True)
    nsteps = int(round(T_END / 1.0))
    T_inf, C_inf = bc(nsteps, 1.0, 50.15 + 273.15, 0.0507)
    T, C, Rarr, Rd, xm, ld, cm = Q4.solve_A(T_inf, C_inf, T_END, 1.0)
    hit = endp(cm, 1.0)
    print("  t_dry=%s h" % ("%.3f" % hit if hit else "None"), flush=True)
    return dict(t_dry=hit)


if __name__ == "__main__":
    results = {}
    results["v1"] = v1_transform()
    results["v2"] = v2_models()
    results["v3"] = v3_convergence()
    results["v5"] = v5_analytic()
    results["r1"] = r1_params()
    results["r2"] = r2_attach2()
    results["r3"] = r3_asymptote()
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1, default=str)
    print("saved:", OUT, flush=True)
    print("ALL Q4 STAGE-6 CHECKS DONE", flush=True)
