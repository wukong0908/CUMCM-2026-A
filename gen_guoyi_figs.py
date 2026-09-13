# -*- coding: utf-8 -*-
"""国一论文全部可视化 — 汇总绘图脚本(2026-09-13)。

一张脚本绘全文 15 张数据图,输出到 A题 国一/figs/:
  Q1: fig_q1_results.png(图3 复刻 4 联)/ fig_q1_sens.png(图6.1 复刻)
  Q2: fig_q2_results.png(图6.2 复刻 4 联)/ fig_q2_sens.png(图6.3 复刻)
      fig_q2_process.png(全程三阶段)
  Q3: q3_scheme_compare / q3_endpoint / q3_stab_probe / q3_profiles / q3_sensitivity
  Q4: q4_R_shrink / q4_endpoint / q4_profiles / q4_shrink_effect / q4_sensitivity

数据源:
  - Q1/Q1灵敏度/Q2灵敏度:现场重跑(秒级),参数 patch 自 solve_q1/solve_q2;
  - Q2/Q3 全程、Q4:结果/q2_arrays.npz、q3_compare_arrays.npz、q4_arrays.npz、
    q4_frozen_arrays.npz(与国一 Q3/Q4 正文数值同源)。
注:Q1/Q2 重跑曲线与国一表1–4 尾位差(图上不可见);Q3/Q4 数值完全一致。
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))      # A题 国一
CODE = os.path.join(HERE, "code")                      # A题 国一/code(本地副本)
RES = os.path.join(HERE, "结果")                        # A题 国一/结果(本地副本)
FIG = os.path.join(HERE, "figs")                       # A题 国一/figs
os.makedirs(FIG, exist_ok=True)
sys.path.insert(0, CODE)

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from matplotlib.patches import Patch

import solve_q1 as S
import solve_q2 as Q
import solve_q4 as Q4

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

RES2 = os.path.join(HERE, "..", "结果")               # math/结果 只读缓存回退


def load_or_make(name, make):
    """取数顺序:本地结果目录 -> math/结果只读回退 -> 现场重算并存本地。"""
    for d in (RES, RES2):
        p = os.path.join(d, name)
        if os.path.exists(p):
            print("load npz:", p)
            return np.load(p)
    print("MISSING %s -> 现场重算(约几十分钟)..." % name)
    z = make()
    np.savez_compressed(os.path.join(RES, name), **z)
    print("saved npz:", os.path.join(RES, name))
    return z


def make_q2():
    tn, Tn, Cn = S.read_attach1(S.ATT1)
    T_inf, C_inf = Q.bc_arrays(tn, Tn, Cn, 259200, 1.0)
    T, C, _, _ = Q.solve(T_inf, C_inf, 259200.0, 0.001, 1.0, flux_scheme="kirchhoff")
    return dict(T=T, C=C)


def make_q3cmp():
    tn, Tn, Cn = S.read_attach1(S.ATT1)
    T_inf, C_inf = Q.bc_arrays(tn, Tn, Cn, 259200, 1.0)
    Th, Ch, _, _ = Q.solve(T_inf, C_inf, 259200.0, 0.001, 1.0, flux_scheme="harmonic")
    Ta, Ca, _, _ = Q.solve(T_inf, C_inf, 259200.0, 0.001, 1.0, flux_scheme="arithmetic")
    return dict(Ch=Ch, Ca=Ca)


def make_q4():
    tn, Tn, Cn = S.read_attach1(S.ATT1)
    n = int(round(Q4.T_END / 1.0))
    T_inf, C_inf = Q4.bc_arrays(tn, Tn, Cn, n, 1.0)
    T, C, Rarr, Rdot, xmax, led, cmax = Q4.solve_A(T_inf, C_inf, Q4.T_END, 1.0)
    return dict(T=T, C=C, Rarr=Rarr, Rdot=Rdot, xmax=xmax, led=led)


def make_q4frozen():
    tn, Tn, Cn = S.read_attach1(S.ATT1)
    n = int(round(Q4.T_END / 1.0))
    T_inf, C_inf = Q4.bc_arrays(tn, Tn, Cn, n, 1.0)
    T, C, Rarr, Rdot, xmax, led, cmax = Q4.solve_A(T_inf, C_inf, Q4.T_END, 1.0,
                                                   freeze_R=True)
    return dict(C=C)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, name), dpi=200)
    plt.close(fig)
    print("saved", name)


# ============================================================
# Q1:图3 复刻(4 联:时间序列 5 曲线 + 横截面 7 时刻)
# ============================================================
t_data, Td, Cd = S.read_attach1(S.ATT1)
Tf = S.make_interp(t_data, Td)
Cf = S.make_interp(t_data, Cd)
t_out, T1, C1, _, _ = S.solve(Tf, Cf, 1800.0, 0.001, 1.0)
r_cm = np.arange(21) * 0.1
j_sel = [0, 5, 10, 15, 20]
t_pts = [100, 300, 600, 900, 1200, 1500, 1800]
cmap = plt.cm.viridis(np.linspace(0, 0.85, 7))

fig, ax = plt.subplots(2, 2, figsize=(11, 7.6))
for j in j_sel:
    ax[0, 0].plot(t_out, T1[:, j] - 273.15, lw=1.5, label="r = %.1f cm" % r_cm[j])
for j in j_sel:
    ax[0, 1].plot(t_out, C1[:, j], lw=1.5, label="r = %.1f cm" % r_cm[j])
for k, tp in enumerate(t_pts):
    ax[1, 0].plot(r_cm, T1[tp, :] - 273.15, color=cmap[k], lw=1.5, label="%d s" % tp)
for k, tp in enumerate(t_pts):
    ax[1, 1].plot(r_cm, C1[tp, :], color=cmap[k], lw=1.5, label="%d s" % tp)
ax[0, 0].set(xlabel="时间 t / s", ylabel="温度 T / °C", title="(a) 温度随时间变化")
ax[0, 1].set(xlabel="时间 t / s", ylabel="水分浓度 C / (kg/kg)", title="(b) 水分浓度随时间变化")
ax[1, 0].set(xlabel="到药材中心的距离 r / cm", ylabel="温度 T / °C", title="(c) 横截面温度分布")
ax[1, 1].set(xlabel="到药材中心的距离 r / cm", ylabel="水分浓度 C / (kg/kg)", title="(d) 横截面水分浓度分布")
for a in ax.flat:
    a.grid(alpha=0.3)
    a.legend(fontsize=7.5, ncol=2)
save(fig, "fig_q1_results.png")

# ============================================================
# Q1:图6.1 复刻(参数 ±10% 灵敏度剖面,Δ 曲线)
# ============================================================
P0 = dict(RHO=S.RHO, CP=S.CP, K=S.K, H=S.H, HM=S.HM)


def q1_run(fac, key, mass=False):
    o = dict(P0)
    if key == "k":
        o["K"] *= fac
    elif key == "h":
        o["H"] *= fac
    elif key == "rcp":
        o["RHO"] *= fac
        o["CP"] *= fac
    elif key == "hm":
        o["HM"] *= fac
    S.RHO, S.CP, S.K, S.H, S.HM = o["RHO"], o["CP"], o["K"], o["H"], o["HM"]
    Df = (lambda C: 7e-9 * fac * np.exp(-0.89 / C)) if key == "d0" else S.D_nonlin
    t, T, C, _, _ = S.solve(Tf, Cf, 1800.0, 0.001, 1.0, D_func=Df)
    S.RHO, S.CP, S.K, S.H, S.HM = P0["RHO"], P0["CP"], P0["K"], P0["H"], P0["HM"]
    return T, C


fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
for key, lab in (("k", "k ±10%"), ("h", "h ±10%"), ("rcp", "ρcp ±10%")):
    for fac, ls, c in ((1.1, "--", "C1"), (0.9, "-.", "C0")):
        T, _ = q1_run(fac, key)
        ax[0].plot(r_cm, T[1800, :] - T1[1800, :],
                   lw=1.3, ls=ls, color=c, label="%s %+d%%" % (lab.split()[0], int(round((fac - 1) * 100))))
ax[0].axhline(0, ls=":", color="gray", lw=1.0)
for key, lab in (("hm", "h_m ±10%"), ("d0", "D0 ±10%")):
    for fac, ls, c in ((1.1, "--", "C1"), (0.9, "-.", "C0")):
        _, C = q1_run(fac, key, mass=True)
        ax[1].plot(r_cm, C[1800, :] - C1[1800, :], lw=1.3, ls=ls, color=c,
                   label="%s %+d%%" % (lab.split()[0], int(round((fac - 1) * 100))))
ax[1].axhline(0, ls=":", color="gray", lw=1.0)
ax[0].set(xlabel="到药材中心的距离 r / cm", ylabel="ΔT / °C", title="(a) 热学参数 ±10% 对温度剖面(t=1800 s)")
ax[1].set(xlabel="到药材中心的距离 r / cm", ylabel="ΔC / (kg/kg)", title="(b) 质学参数 ±10% 对水分剖面(t=1800 s)")
for a in ax:
    a.grid(alpha=0.3)
    a.legend(fontsize=8)
save(fig, "fig_q1_sens.png")

# ============================================================
# Q2:图6.2 复刻(4 联,0–3 h)+ 图6.3 复刻(+10% 剖面影响)
# ============================================================
Z = load_or_make("q2_arrays.npz", make_q2)
T2, C2 = Z["T"], Z["C"]
t2_pts = [1800, 3600, 5400, 7200, 9000, 10800]
labs2 = ["0.5 h", "1 h", "1.5 h", "2 h", "2.5 h", "3 h"]
cmap2 = plt.cm.viridis(np.linspace(0, 0.85, 6))
t3h = np.arange(10801) / 3600.0

fig, ax = plt.subplots(2, 2, figsize=(11, 7.6))
for j in j_sel:
    ax[0, 0].plot(t3h, T2[:10801, j] - 273.15, lw=1.5, label="r = %.1f cm" % r_cm[j])
for j in j_sel:
    ax[0, 1].plot(t3h, C2[:10801, j], lw=1.5, label="r = %.1f cm" % r_cm[j])
for k, tp in enumerate(t2_pts):
    ax[1, 0].plot(r_cm, T2[tp, :] - 273.15, color=cmap2[k], lw=1.5, label=labs2[k])
for k, tp in enumerate(t2_pts):
    ax[1, 1].plot(r_cm, C2[tp, :], color=cmap2[k], lw=1.5, label=labs2[k])
ax[0, 0].set(xlabel="时间 t / h", ylabel="温度 T / °C", title="(a) 温度随时间变化")
ax[0, 1].set(xlabel="时间 t / h", ylabel="水分浓度 C / (kg/kg)", title="(b) 水分浓度随时间变化")
ax[1, 0].set(xlabel="到药材中心的距离 r / cm", ylabel="温度 T / °C", title="(c) 3 h 横截面温度分布")
ax[1, 1].set(xlabel="到药材中心的距离 r / cm", ylabel="水分浓度 C / (kg/kg)", title="(d) 3 h 横截面水分浓度分布")
for a in ax.flat:
    a.grid(alpha=0.3)
    a.legend(fontsize=7.5, ncol=2)
save(fig, "fig_q2_results.png")

# ---- 图6.3 复刻:关键参数 +10% 对 3 h 剖面的影响 ----
Pq0 = dict(Q.P)
n2 = 10800
t_n, T_n, C_n = S.read_attach1(S.ATT1)
T_inf3, C_inf3 = Q.bc_arrays(t_n, T_n, C_n, n2, 1.0)
Tb, Cb, _, _ = Q.solve(T_inf3, C_inf3, 10800.0, 0.001, 1.0, flux_scheme="kirchhoff")


def q2_run(fac, key):
    Q.P.update(Pq0)
    if key in ("cp", "k"):
        Q.P[key + "0"] *= fac
        Q.P[key + "1"] *= fac
    else:
        Q.P[key] *= fac
    T, C, _, _ = Q.solve(T_inf3, C_inf3, 10800.0, 0.001, 1.0, flux_scheme="kirchhoff")
    Q.P.update(Pq0)
    return T, C


fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
for key, lab in (("D0", "D0"), ("A_C", "A_C"), ("A_T", "A_T"), ("HM", "h_m")):
    _, C = q2_run(1.1, key)
    ax[0].plot(r_cm, C[10800, :] - Cb[10800, :], lw=1.4, label=lab + " +10%")
for key, lab in (("H", "h"), ("cp", "c_p"), ("k", "k")):
    T, _ = q2_run(1.1, key)
    ax[1].plot(r_cm, T[10800, :] - Tb[10800, :], lw=1.4, label=lab + " +10%")
ax[0].axhline(0, ls=":", color="gray", lw=1.0)
ax[1].axhline(0, ls=":", color="gray", lw=1.0)
ax[0].set(xlabel="到药材中心的距离 r / cm", ylabel="ΔC / (kg/kg)", title="(a) 扩散系数参数 +10% 对 3 h 水分剖面")
ax[1].set(xlabel="到药材中心的距离 r / cm", ylabel="ΔT / °C", title="(b) 换热/热物性参数 +10% 对 3 h 温度剖面")
for a in ax:
    a.grid(alpha=0.3)
    a.legend(fontsize=8)
save(fig, "fig_q2_sens.png")

# ---- 全程三阶段(中心/表面 温度水分) ----
nsteps = 259200
t_h = np.arange(nsteps + 1) / 3600.0
T_inf, C_inf = Q.bc_arrays(t_n, T_n, C_n, nsteps, 1.0)
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
ax[0].plot(t_h, C2[:, 0], lw=1.6, label="中心 C(0,t)")
ax[0].plot(t_h, C2[:, -1], lw=1.6, label="表面 C(R,t)")
ax[0].plot(t_h, C_inf, lw=1.2, ls="--", color="gray", label="烘房 C∞(t)")
ax[0].axhline(0.15, ls=":", color="red", lw=1.2)
ax[0].axvline(57.21, ls=":", color="red", lw=1.2)
ax[0].annotate("C = 0.15", xy=(66, 0.155), color="red", fontsize=9)
ax[0].annotate("t = 57.21 h", xy=(43, 2.25), color="red", fontsize=9)
ax[0].set(xlabel="时间 t / h", ylabel="水分浓度 C / (kg/kg)", title="(a) 水分浓度全程")
ax[1].plot(t_h, T2[:, 0] - 273.15, lw=1.6, label="中心 T(0,t)")
ax[1].plot(t_h, T2[:, -1] - 273.15, lw=1.6, label="表面 T(R,t)")
ax[1].plot(t_h, T_inf - 273.15, lw=1.2, ls="--", color="gray", label="烘房 T∞(t)")
ax[1].set(xlabel="时间 t / h", ylabel="温度 T / °C", title="(b) 温度全程")
for a in ax:
    a.grid(alpha=0.3)
    a.legend(fontsize=8)
save(fig, "fig_q2_process.png")

# ============================================================
# Q3:五张(方案对比 / 判据首达 / 稳定性探针 / 表5剖面 / tornado)
# ============================================================
cmax = C2.max(axis=1)
hit = np.where(cmax <= 0.15)[0]
n_hit = hit[0]
Zc = load_or_make("q3_compare_arrays.npz", make_q3cmp)
Ch, Ca = Zc["Ch"], Zc["Ca"]

fig, ax = plt.subplots(figsize=(6.8, 4.2))
ax.plot(t_h, C2[:, 0], lw=1.8, label="Kirchhoff 势(本文)")
ax.plot(t_h, Ch[:, 0], lw=1.6, ls="--", label="调和平均")
ax.plot(t_h, Ca[:, 0], lw=1.6, ls="-.", label="算术平均")
ax.axhline(0.15, ls=":", color="red", lw=1.2)
ax.annotate("假停滞:3 天内无法完成", xy=(46, 0.275), fontsize=9)
ax.annotate("t = 57.21 h", xy=(48.5, 0.095), color="red", fontsize=9)
ax.annotate("t = 56.18 h(偏早)", xy=(49.5, 0.175), fontsize=9, color="C2")
ax.set(xlabel="时间 t / h", ylabel="中心水分浓度 C(0,t) / (kg/kg)",
       title="界面通量取法对干燥终点的影响", xlim=(0, 72), ylim=(0, 2.7))
ax.grid(alpha=0.3)
ax.legend(fontsize=9)
save(fig, "q3_scheme_compare.png")

fig, ax = plt.subplots(figsize=(6.8, 4.2))
ax.plot(t_h, cmax, lw=1.6, color="C0")
ax.axhline(0.15, ls=":", color="red", lw=1.2)
ax.axvline(n_hit / 3600.0, ls=":", color="red", lw=1.2)
ax.annotate("C = 0.15", xy=(64, 0.156), color="red", fontsize=9)
ax.annotate("首达 t = 205952 s = 57.21 h", xy=(n_hit / 3600.0 + 0.3, 0.9),
            color="red", fontsize=9)
ax.set(xlabel="时间 t / h", ylabel="全场最大水分浓度 max C / (kg/kg)",
       title="烘干终点判据的首达时刻", xlim=(0, 72), ylim=(0, 2.7))
ax.grid(alpha=0.3)
i0 = int(round(n_hit - 0.45 * 3600))
i1 = int(round(n_hit + 0.15 * 3600))
axin = inset_axes(ax, width="44%", height="44%", loc="lower right", borderpad=1.4)
axin.plot(t_h[i0:i1], cmax[i0:i1], lw=1.4, color="C0")
axin.axhline(0.15, ls=":", color="red", lw=1.0)
axin.axvline(n_hit / 3600.0, ls=":", color="red", lw=1.0)
axin.plot((n_hit - 60) / 3600.0, cmax[n_hit - 60], "o", ms=4, color="C3")
axin.annotate("前 60 s:%.4f" % cmax[n_hit - 60],
              xy=((n_hit - 60) / 3600.0 + 0.015, cmax[n_hit - 60] + 0.0005),
              fontsize=8, color="C3")
axin.tick_params(labelsize=7)
axin.grid(alpha=0.3)
mark_inset(ax, axin, loc1=2, loc2=4, fc="none", ec="0.6")
save(fig, "q3_endpoint.png")


def probe(dt, n=1000):
    T_inf_c = np.full(n + 1, 323.315)
    C_inf_c = np.full(n + 1, 0.04986)
    T0 = 323.315 + 1e-6 * np.where(np.arange(21) % 2 == 0, 1.0, -1.0)
    T, C, _, _ = Q.solve(T_inf_c, C_inf_c, n * dt, 0.001, dt,
                         freeze_C=True, C_init=0.15, T_init=T0)
    return np.abs(T - 323.315).max(axis=1)


d1 = probe(1.0)
d2 = probe(2.0)
d1s = np.maximum(d1, 1e-30)
d2s = np.clip(np.nan_to_num(d2, nan=1e30, posinf=1e30), None, 1e30)
fig, ax = plt.subplots(figsize=(6.8, 4.2))
ax.semilogy(np.arange(len(d1s)), d1s, lw=1.6, label="Δt = 1 s(x = 0.215)")
ax.semilogy(np.arange(len(d2s)), d2s, lw=1.6, ls="--", label="Δt = 2 s(x = 0.429)")
ax.axhline(1e-6, ls=":", color="gray", lw=1.0)
ax.annotate("初始扰动 $10^{-6}$ K", xy=(20, 1.3e-6), fontsize=8, color="gray")
ax.annotate("立即衰减", xy=(30, 1e-10), fontsize=9, color="C0")
ax.annotate("指数发散", xy=(420, 1e10), fontsize=9, color="C1")
ax.set(xlabel="步数 n", ylabel="全场最大偏差 max |T − T∞| / K",
       title="极轴稳定性探针(冻结 C = 0.15,锯齿扰动)")
ax.grid(alpha=0.3, which="both")
ax.legend(fontsize=9)
save(fig, "q3_stab_probe.png")

t_sel_h = [6, 12, 18, 24, 30, 36, 42, 48, 54]
cmap = plt.cm.viridis(np.linspace(0, 0.9, 10))
fig, ax = plt.subplots(figsize=(6.8, 4.4))
for k, th in enumerate(t_sel_h):
    ax.plot(r_cm, C2[int(th * 3600), :], color=cmap[k], lw=1.5, label="%d h" % th)
ax.plot(r_cm, C2[n_hit, :], color=cmap[9], lw=2.0, ls="--", label="57.21 h(终点)")
ax.axhline(0.15, ls=":", color="red", lw=1.2)
ax.annotate("C = 0.15", xy=(1.55, 0.153), color="red", fontsize=9)
ax.set(xlabel="到药材中心的距离 r / cm", ylabel="水分浓度 C / (kg/kg)",
       title="不同时刻的水分浓度剖面(表 5)")
ax.grid(alpha=0.3)
ax.legend(fontsize=7.5, ncol=2)
save(fig, "q3_profiles.png")


def tornado(fname, title, nominal, params, lo, hi, lo_clip, hi_clip, xlim):
    y = np.arange(len(params))[::-1]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for yi, p, l, h, lc, hc in zip(y, params, lo, hi, lo_clip, hi_clip):
        if l < nominal:
            ax.barh(yi, l - nominal, left=nominal, color="C0", alpha=0.75, height=0.55)
        if h > nominal:
            ax.barh(yi, h - nominal, left=nominal, color="C1", alpha=0.75, height=0.55)
        if lc and l < nominal:
            ax.annotate("%.1f" % l, xy=(l - 0.4, yi), ha="right", va="center",
                        fontsize=8, color="C0")
        if hc and h > nominal:
            lab = ">72" if (h >= 72) else "%.1f" % h
            ax.annotate(lab, xy=(h + 0.4, yi), ha="left", va="center",
                        fontsize=8, color="C1")
        if not hc and not lc:
            ax.annotate("%.2f" % h, xy=(h + 0.2, yi), ha="left", va="center",
                        fontsize=8, color="C1")
    ax.axvline(nominal, ls=":", color="red", lw=1.2)
    ax.annotate("名义 %.2f h" % nominal, xy=(nominal, len(params) - 0.25),
                color="red", fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels(params, fontsize=9)
    ax.set_xlabel("干燥终点 t_dry / h")
    ax.set_title(title)
    ax.set_xlim(xlim)
    ax.grid(alpha=0.3, axis="x")
    ax.legend(handles=[Patch(color="C0", alpha=0.75, label="−10%"),
                       Patch(color="C1", alpha=0.75, label="+10%")],
              fontsize=8, loc="lower right")
    save(fig, fname)


tornado("q3_sensitivity.png", "问题三:参数 ±10% 对烘干终点的影响", 57.21,
        ["$k$", "$c_p$", "$h$", "$h_m$", "$D_0$", "$A_C$", "$A_T$"],
        [57.21, 57.20, 57.22, 57.95, 62.84, 45.38, 23.13],
        [57.21, 57.22, 57.20, 56.63, 52.63, 72.0, 72.0],
        [False, False, False, False, True, True, True],
        [False, False, False, False, True, True, True], (20, 76))

# ============================================================
# Q4:五张(半径收缩 / 判据首达 / 表6剖面 / 消融对比 / tornado)
# ============================================================
Z4 = load_or_make("q4_arrays.npz", make_q4)
T4, C4, Rarr, Rdot, xmax4, led4 = (Z4["T"], Z4["C"], Z4["Rarr"], Z4["Rdot"],
                                   Z4["xmax"], Z4["led"])
cmax4 = C4.max(axis=1)
hit4 = np.where(cmax4 <= 0.15)[0]
n_hit4 = hit4[0]
Zf = load_or_make("q4_frozen_arrays.npz", make_q4frozen)
Cf4 = Zf["C"]

t2, R2 = Q4.read_attach2(Q4.ATT2)
t2h = t2 / 3600.0
fig, ax = plt.subplots(figsize=(6.5, 4))
ax.plot(t2h, R2 * 100, lw=1.8, color="C0")
ax.axhline(1.198, ls=":", color="gray", lw=1.0)
ax.annotate("21.5 h 起进入平台(≈1.207 cm)", xy=(24, 1.31), fontsize=9)
ax.annotate("72 h:1.198 cm", xy=(56, 1.215), fontsize=9, color="gray")
ax.set(xlabel="时间 t / h", ylabel="药材半径 R / cm",
       title="附件 2:烘干过程的半径收缩", xlim=(0, 72), ylim=(1.1, 2.1))
ax.grid(alpha=0.3)
save(fig, "q4_R_shrink.png")

fig, ax = plt.subplots(figsize=(6.8, 4.2))
ax.plot(t_h, cmax4, lw=1.6, color="C0")
ax.axhline(0.15, ls=":", color="red", lw=1.2)
ax.axvline(n_hit4 / 3600.0, ls=":", color="red", lw=1.2)
ax.annotate("C = 0.15", xy=(64, 0.156), color="red", fontsize=9)
ax.annotate("首达 t = 188717 s = 52.42 h", xy=(n_hit4 / 3600.0 + 0.3, 0.9),
            color="red", fontsize=9)
ax.set(xlabel="时间 t / h", ylabel="全场最大水分浓度 max C / (kg/kg)",
       title="烘干终点判据的首达时刻(动边界模型)", xlim=(0, 72), ylim=(0, 2.7))
ax.grid(alpha=0.3)
i0 = int(round(n_hit4 - 0.45 * 3600))
i1 = int(round(n_hit4 + 0.15 * 3600))
axin = inset_axes(ax, width="44%", height="44%", loc="lower right", borderpad=1.4)
axin.plot(t_h[i0:i1], cmax4[i0:i1], lw=1.4, color="C0")
axin.axhline(0.15, ls=":", color="red", lw=1.0)
axin.axvline(n_hit4 / 3600.0, ls=":", color="red", lw=1.0)
axin.plot((n_hit4 - 60) / 3600.0, cmax4[n_hit4 - 60], "o", ms=4, color="C3")
axin.annotate("前 60 s:%.4f" % cmax4[n_hit4 - 60],
              xy=((n_hit4 - 60) / 3600.0 + 0.015, cmax4[n_hit4 - 60] + 0.0005),
              fontsize=8, color="C3")
axin.tick_params(labelsize=7)
axin.grid(alpha=0.3)
mark_inset(ax, axin, loc1=2, loc2=4, fc="none", ec="0.6")
save(fig, "q4_endpoint.png")

xi = np.arange(Q4.N + 1) * Q4.DXI
t_sel4 = [6, 12, 18, 24, 30, 36, 42, 48]
cmap = plt.cm.viridis(np.linspace(0, 0.9, 9))
fig, ax = plt.subplots(figsize=(6.8, 4.4))
for k, th in enumerate(t_sel4):
    i = int(th * 3600)
    ax.plot(xi * Rarr[i] * 100, C4[i], color=cmap[k], lw=1.5, label="%d h" % th)
ax.plot(xi * Rarr[n_hit4] * 100, C4[n_hit4], color=cmap[8], lw=2.0, ls="--",
        label="52.42 h(终点)")
ax.axhline(0.15, ls=":", color="red", lw=1.2)
ax.annotate("C = 0.15", xy=(1.35, 0.153), color="red", fontsize=9)
ax.set(xlabel="到药材中心的距离 r / cm", ylabel="水分浓度 C / (kg/kg)",
       title="不同时刻的水分浓度剖面(表 6,收缩网格)", xlim=(0, 2.1))
ax.grid(alpha=0.3)
ax.legend(fontsize=7.5, ncol=2)
save(fig, "q4_profiles.png")

cmf4 = Cf4.max(axis=1)
hf = np.where(cmf4 <= 0.15)[0]
t_dry_f = hf[0] / 3600.0 if hf.size else None
fig, ax = plt.subplots(figsize=(6.8, 4.2))
ax.plot(t_h, C4[:, 0], lw=1.8, label="动边界(附件 2 收缩)")
ax.plot(t_h, Cf4[:, 0], lw=1.6, ls="--", label="冻结几何(半径 2 cm 恒定)")
ax.axhline(0.15, ls=":", color="red", lw=1.2)
ax.axvline(n_hit4 / 3600.0, ls=":", color="red", lw=1.0)
if t_dry_f:
    ax.axvline(t_dry_f, ls=":", color="C1", lw=1.0)
    ax.annotate("冻结:%.2f h" % t_dry_f, xy=(t_dry_f + 0.4, 0.5), color="C1", fontsize=9)
else:
    ax.annotate("冻结:3 天未达(72 h 中心 0.2238)", xy=(28, 0.62), color="C1", fontsize=9)
ax.annotate("收缩:52.42 h", xy=(n_hit4 / 3600.0 + 0.4, 0.25), color="red", fontsize=9)
ax.set(xlabel="时间 t / h", ylabel="中心水分浓度 C(0,t) / (kg/kg)",
       title="收缩对干燥进程的净影响(消融对比)", xlim=(0, 72), ylim=(0, 2.7))
ax.grid(alpha=0.3)
ax.legend(fontsize=9)
save(fig, "q4_shrink_effect.png")

tornado("q4_sensitivity.png", "问题四:参数 ±10% 对烘干终点的影响", 52.42,
        ["$k$", "$c_p$", "$h$", "$h_m$", "$D_0$", "$A_C$", "$A_T$"],
        [52.42, 52.41, 52.43, 52.73, 57.47, 45.57, 21.15],
        [52.42, 52.43, 52.42, 52.18, 48.30, 60.69, 72.0],
        [False, False, False, False, True, True, True],
        [False, False, False, False, True, True, True], (18, 76))

print("ALL DONE")
