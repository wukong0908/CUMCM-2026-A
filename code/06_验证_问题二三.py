# -*- coding: utf-8 -*-
"""
Q2 Stage 6 verification / robustness suite (per 规格卡_Q2.md v1.1).
V1: degenerate heat (C frozen at 2.55 / 0.15) vs Bessel-series analytic
V2: degenerate mass (T frozen, constant D) vs Bessel-series analytic
V4: grid convergence (dr, dr/2, dr/4) over full 3 days: endpoint + 72h values
V5: asymptotic consistency (constant BC from t=0, 5-day run)
V6: flux-scheme comparison (kirchhoff / harmonic / arithmetic, full 3 days)
R1: parameter perturbation +-10% (7 knobs, dt=2 s)
R2: smoothed attachment BC vs raw
R3: attachment perturbation (T +-0.5 C, C +-5%)
R4: D01 asymptote cross-check (T=50.15 C, C=0.0507)
All full-run results collected into 结果/verify_q2_results.json.
"""
import os
import sys
import json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import lib_q1 as S
import lib_q2 as Q

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, "结果", "verify_q2_results.json")
T_END = 259200.0
r_cm = [0.0, 0.5, 1.0, 1.5, 2.0]
jdx = [int(round(rr / 0.1)) for rr in r_cm]
t_pts = [1800, 3600, 5400, 7200, 9000, 10800]

t_data, T_data, C_data = S.read_attach1(S.ATT1)


def base_arrays(dt):
    nsteps = int(round(T_END / dt))
    return Q.bc_arrays(t_data, T_data, C_data, nsteps, dt)


def endpoint(C, dt):
    hit = np.where(C.max(axis=1) <= 0.15)[0]
    return (hit[0] * dt / 3600.0) if hit.size else None


def full_run(T_inf, C_inf, dr, dt, scheme="kirchhoff", t_end=T_END, **kw):
    T, C, cour, bal = Q.solve(T_inf, C_inf, t_end, dr, dt, flux_scheme=scheme, **kw)
    i = int(round(t_end / dt))
    return T, C, cour, bal, endpoint(C, dt), (C[i, 0], C[i, -1],
                                              T[i, 0] - 273.15, T[i, -1] - 273.15)


# ================= V1: degenerate heat, C frozen =================
def v1_heat_analytic(C_f, label):
    k = Q.k_c(np.array(C_f)); rho = Q.rho_c(np.array(C_f)); cp = Q.cp_c(np.array(C_f))
    alpha = float(k / (rho * cp))
    Bi = Q.P['H'] * S.R / float(k)
    T_inf_c = 323.315
    nsteps = 10800
    T_inf = np.full(nsteps + 1, T_inf_c)
    C_inf = np.full(nsteps + 1, 0.04986)
    T, C, cour, bal = Q.solve(T_inf, C_inf, 10800.0, 0.001, 1.0,
                              freeze_C=True, C_init=C_f)
    lam = S.bessel_roots(Bi, 300)
    ana = S.series_field(301.15, T_inf_c, lam, None,
                         [r / 100.0 for r in r_cm], t_pts, alpha)
    num = np.array([[T[t, j] for j in jdx] for t in t_pts])
    err = np.abs(num - ana)
    print("V1 heat analytic (C frozen=%.2f): Bi=%.4f alpha=%.3e | max err %.3e K | err@3h %.3e K"
          % (C_f, Bi, alpha, err.max(), err[-1].max()), flush=True)
    return err.max(), err[-1].max()


# ================= V2: degenerate mass, T frozen + constant D =================
def v2_mass_analytic():
    Dc = 1e-8
    T_f = 323.315
    nsteps = 10800
    T_inf = np.full(nsteps + 1, T_f)
    C_inf = np.full(nsteps + 1, 0.04986)
    T, C, cour, bal = Q.solve(T_inf, C_inf, 10800.0, 0.001, 1.0,
                              flux_scheme="harmonic", freeze_T=True, T_init=T_f,
                              D_func=lambda Cn: Dc * np.ones_like(Cn))
    Bi_m = Q.P['HM'] * S.R / Dc
    lam = S.bessel_roots(Bi_m, 300)
    ana = S.series_field(2.55, 0.04986, lam, None,
                         [r / 100.0 for r in r_cm], t_pts, Dc)
    num = np.array([[C[t, j] for j in jdx] for t in t_pts])
    err = np.abs(num - ana)
    print("V2 mass analytic (T frozen, D=1e-8 const): Bi_m=%.4f | max err %.3e | err@3h %.3e"
          % (Bi_m, err.max(), err[-1].max()), flush=True)
    return err.max(), err[-1].max()


# ================= V4: grid convergence =================
def v4_convergence():
    print("== V4 grid convergence (full 3 days, kirchhoff) ==", flush=True)
    rows = []
    for dr, dt in [(0.001, 1.0), (0.0005, 0.25), (0.00025, 0.0625)]:
        nsteps = int(round(T_END / dt))
        T_inf, C_inf = Q.bc_arrays(t_data, T_data, C_data, nsteps, dt)
        T, C, cour, bal = Q.solve(T_inf, C_inf, T_END, dr, dt, flux_scheme="kirchhoff")
        i = int(round(T_END / dt))
        rows.append(dict(dr=dr, dt=dt,
                         Tc=T[i, 0] - 273.15, Ts=T[i, -1] - 273.15,
                         Cc=C[i, 0], Cs=C[i, -1],
                         t_hit=endpoint(C, dt), bal_rel=bal["rel"],
                         cour=float(cour.max())))
        print("  dr=%.5f m dt=%.3f s: Cc(72h)=%.6f Cs(72h)=%.6f T(72h)=%.4f"
              " | t_hit=%.2f h | bal_rel=%.2e | Cour=%.3f"
              % (dr, dt, C[i, 0], C[i, -1], T[i, 0] - 273.15,
                 rows[-1]["t_hit"], bal["rel"], cour.max()), flush=True)
    return rows


# ================= V5: asymptotic consistency =================
def v5_asymptotic():
    print("== V5 asymptotic (const BC from t=0, 5 days) ==", flush=True)
    t_end5 = 432000.0
    dt = 1.0
    nsteps = int(round(t_end5 / dt))
    T_inf = np.full(nsteps + 1, 323.315)
    C_inf = np.full(nsteps + 1, 0.04986)
    T, C, cour, bal = Q.solve(T_inf, C_inf, t_end5, 0.001, dt, flux_scheme="kirchhoff")
    dT = float(np.abs(T[-1] - 323.315).max())
    dC = float(np.abs(C[-1] - 0.04986).max())
    mono = bool(np.all(np.diff(C[:, 0]) <= 0))
    no_overshoot = bool((T <= 323.315 + 1e-9).all())
    print("  dT_end=%.3e K | dC_end=%.3e | center C mono dec=%s | T no overshoot=%s"
          % (dT, dC, mono, no_overshoot), flush=True)
    for th in (72.0, 120.0):
        i = int(th * 3600 / dt)
        print("  t=%3.0fh: center C=%.5f surface C=%.5f" % (th, C[i, 0], C[i, -1]),
              flush=True)
    return dict(dT=dT, dC=dC, mono=mono, no_overshoot=no_overshoot,
                Cc72=float(C[129600, 0]), Cc120=float(C[216000, 0]))


# ================= V6: flux scheme comparison =================
def v6_schemes():
    print("== V6 flux schemes (full 3 days, dr=1mm dt=1s) ==", flush=True)
    rows = {}
    for scheme in ("kirchhoff", "harmonic", "arithmetic"):
        nsteps = int(round(T_END / 1.0))
        T_inf, C_inf = Q.bc_arrays(t_data, T_data, C_data, nsteps, 1.0)
        T, C, cour, bal = Q.solve(T_inf, C_inf, T_END, 0.001, 1.0,
                                  flux_scheme=scheme)
        i = int(round(T_END))
        rows[scheme] = dict(Cc=C[i, 0], Cs=C[i, -1], t_hit=endpoint(C, 1.0),
                            bal_rel=bal["rel"])
        print("  %s: Cc(72h)=%.6f Cs(72h)=%.6f t_hit=%s h bal_rel=%.2e"
              % (scheme, C[i, 0], C[i, -1],
                 "%.2f" % rows[scheme]["t_hit"] if rows[scheme]["t_hit"] else "None",
                 bal["rel"]),
              flush=True)
    return rows


# ================= R1: parameter perturbation =================
def r1_params():
    print("== R1 parameter perturbation +-10% (dt=1 s, full 3 days) ==", flush=True)
    knobs = [
        ("D0", ["D0"]), ("A_C", ["A_C"]), ("A_T", ["A_T"]), ("HM", ["HM"]),
        ("H", ["H"]), ("cp", ["cp0", "cp1"]), ("k", ["k0", "k1"]),
    ]
    nsteps = int(round(T_END / 1.0))
    T_inf, C_inf = Q.bc_arrays(t_data, T_data, C_data, nsteps, 1.0)
    baseT, baseC, _, _, base_hit, base72 = full_run(T_inf, C_inf, 0.001, 1.0)
    saved = dict(Q.P)
    print("  base(dt=1): t_hit=%.2f h | Cc72=%.6f Cs72=%.6f | Cc3h=%.4f Cs3h=%.4f Ts0.5h=%.4f"
          % (base_hit, base72[0], base72[1], baseC[10800, 0], baseC[10800, -1],
             baseT[1800, -1] - 273.15), flush=True)
    rows = []
    for name, keys in knobs:
        for f in (0.9, 1.1):
            for key in keys:
                Q.P[key] = saved[key] * f
            try:
                T, C, _, _, hit, v72 = full_run(T_inf, C_inf, 0.001, 1.0)
                rows.append(dict(knob=name, f=f, t_hit=hit,
                                 dCc72=v72[0] - base72[0], dCs72=v72[1] - base72[1],
                                 dCc3h=C[10800, 0] - baseC[10800, 0],
                                 dCs3h=C[10800, -1] - baseC[10800, -1],
                                 dTs=((T[1800, -1] - 273.15)
                                      - (baseT[1800, -1] - 273.15))))
                print("  %s x%.1f: t_hit=%s h | dCc72=%+.5f dCs72=%+.5f | dCc3h=%+.5f dCs3h=%+.5f dTs=%+.4f"
                      % (name, f, "%.2f" % hit if hit else "None",
                         rows[-1]["dCc72"], rows[-1]["dCs72"],
                         rows[-1]["dCc3h"], rows[-1]["dCs3h"], rows[-1]["dTs"]),
                      flush=True)
            finally:
                for key in keys:
                    Q.P[key] = saved[key]
    return rows


# ================= R2: smoothed BC =================
def r2_smooth():
    print("== R2 smoothed BC (window 5, edge reflect, dt=1 s) ==", flush=True)
    w = 5
    Tp = np.pad(T_data, (w // 2, w // 2), mode="edge")
    Cp = np.pad(C_data, (w // 2, w // 2), mode="edge")
    Ts = np.convolve(Tp, np.ones(w) / w, mode="valid")
    Cs = np.convolve(Cp, np.ones(w) / w, mode="valid")
    nsteps = int(round(T_END / 1.0))
    T_inf, C_inf = Q.bc_arrays(t_data, Ts, Cs, nsteps, 1.0)
    T_inf_b, C_inf_b = Q.bc_arrays(t_data, T_data, C_data, nsteps, 1.0)
    baseT, baseC, _, _, base_hit, base72 = full_run(T_inf_b, C_inf_b, 0.001, 1.0)
    T, C, _, _, hit, v72 = full_run(T_inf, C_inf, 0.001, 1.0)
    print("  t_hit=%s h | dCc72=%+.6f dCs72=%+.6f | dCc3h=%+.6f dCs3h=%+.6f"
          % ("%.2f" % hit if hit else "None", v72[0] - base72[0], v72[1] - base72[1],
             C[10800, 0] - baseC[10800, 0], C[10800, -1] - baseC[10800, -1]),
          flush=True)
    return dict(t_hit=hit, dCc72=v72[0] - base72[0], dCs72=v72[1] - base72[1],
                dCc3h=C[10800, 0] - baseC[10800, 0],
                dCs3h=C[10800, -1] - baseC[10800, -1])


# ================= R3: attachment perturbation =================
def r3_perturb():
    print("== R3 attachment perturbation (dt=1 s) ==", flush=True)
    cases = {
        "T_inf +0.5C": (T_data + 0.5, C_data),
        "T_inf -0.5C": (T_data - 0.5, C_data),
        "C_inf +5%": (T_data, C_data * 1.05),
        "C_inf -5%": (T_data, C_data * 0.95),
    }
    nsteps = int(round(T_END / 1.0))
    T_inf_b, C_inf_b = Q.bc_arrays(t_data, T_data, C_data, nsteps, 1.0)
    baseT, baseC, _, _, base_hit, base72 = full_run(T_inf_b, C_inf_b, 0.001, 1.0)
    rows = []
    for name, (Td, Cd) in cases.items():
        T_inf, C_inf = Q.bc_arrays(t_data, Td, Cd, nsteps, 1.0)
        T, C, _, _, hit, v72 = full_run(T_inf, C_inf, 0.001, 1.0)
        rows.append(dict(case=name, t_hit=hit, dCc72=v72[0] - base72[0]))
        print("  %s: t_hit=%s h | dCc72=%+.5f" %
              (name, "%.2f" % hit if hit else "None", rows[-1]["dCc72"]), flush=True)
    return rows


# ================= R4: D01 asymptote cross-check =================
def r4_asymptote():
    print("== R4 asymptote BC cross-check (50.15 C / 0.0507, dt=1 s) ==", flush=True)
    nsteps = int(round(T_END / 1.0))
    T_inf, C_inf = Q.bc_arrays(t_data, T_data, C_data, nsteps, 1.0,
                               T_tail=Q.T_END_ASY, C_tail=Q.C_END_ASY)
    T_inf_b, C_inf_b = Q.bc_arrays(t_data, T_data, C_data, nsteps, 1.0)
    baseT, baseC, _, _, base_hit, base72 = full_run(T_inf_b, C_inf_b, 0.001, 1.0)
    T, C, _, _, hit, v72 = full_run(T_inf, C_inf, 0.001, 1.0)
    print("  t_hit=%s h (base %.2f h) | dCc72=%+.5f dCs72=%+.5f"
          % ("%.2f" % hit if hit else "None", base_hit, v72[0] - base72[0],
             v72[1] - base72[1]), flush=True)
    return dict(t_hit=hit, base_hit=base_hit, dCc72=v72[0] - base72[0],
                dCs72=v72[1] - base72[1])


if __name__ == "__main__":
    results = {}
    results["v1_c255"] = v1_heat_analytic(2.55, "2.55")
    results["v1_c015"] = v1_heat_analytic(0.15, "0.15")
    results["v2"] = v2_mass_analytic()
    results["v4"] = v4_convergence()
    results["v5"] = v5_asymptotic()
    results["v6"] = v6_schemes()
    results["r1"] = r1_params()
    results["r2"] = r2_smooth()
    results["r3"] = r3_perturb()
    results["r4"] = r4_asymptote()
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1, default=str)
    print("saved:", OUT, flush=True)
    print("ALL STAGE-6 CHECKS DONE", flush=True)
