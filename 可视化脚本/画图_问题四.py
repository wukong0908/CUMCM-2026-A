# -*- coding: utf-8 -*-
"""问题四可视化:q4_R_shrink(附件2 半径)+ q4_endpoint(动边界首达)
+ q4_profiles(表6 剖面)+ q4_shrink_effect(消融对比)+ q4_sensitivity(tornado)。"""
from plot_common import *

Z4 = load_or_make("q4_pchip_arrays.npz", make_q4)
C4, Rarr = Z4["C"], Z4["Rarr"]
nsteps = C4.shape[0] - 1
t_h = np.arange(nsteps + 1) / 3600.0
cmax4 = C4.max(axis=1)
hit4 = np.where(cmax4 <= 0.15)[0]
n_hit4 = hit4[0]
Zf = load_or_make("q4_pchip_frozen.npz", make_q4frozen)
Cf4 = Zf["C"]

# ---- 附件2 半径收缩 ----
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

# ---- 动边界判据首达 ----
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

# ---- 表6 剖面(收缩网格) ----
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

# ---- 消融对比 ----
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

# ---- tornado(公共模块) ----
tornado("q4_sensitivity.png", "问题四:参数 ±10% 对烘干终点的影响", 52.42,
        ["$k$", "$c_p$", "$h$", "$h_m$", "$D_0$", "$A_C$", "$A_T$"],
        [52.42, 52.41, 52.43, 52.73, 57.47, 45.57, 21.15],
        [52.42, 52.43, 52.42, 52.18, 48.30, 60.69, 72.0],
        [False, False, False, False, True, True, True],
        [False, False, False, False, True, True, True], (18, 76))
print("问题四 5 图完成")
