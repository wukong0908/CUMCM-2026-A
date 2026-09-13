import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import (
    interp1d, PchipInterpolator, CubicSpline,
    Akima1DInterpolator, UnivariateSpline
)

BASE = os.path.dirname(os.path.abspath(__file__))          # 脚本所在目录 code/
DATA = os.path.join(BASE, "..", "附件", "附件1.xlsx")        # 输入数据(相对脚本定位)

# ========== 字体设置 ==========
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Heiti TC', 'Hiragino Sans GB']
plt.rcParams['axes.unicode_minus'] = False

# ========== 1. 读取数据 ==========
df = pd.read_excel(DATA, sheet_name='Sheet1')
print("列名：", df.columns.tolist())

t_original = df['时间'].values.astype(float)
T_original = df['温度'].values.astype(float)
C_original = df['水分浓度'].values.astype(float)

print(f"原始数据点数：{len(t_original)}")

# ========== 2. 构造每秒时间轴 ==========
t_new = np.arange(t_original.min(), t_original.max() + 1, 1.0)
print(f"插值后点数：{len(t_new)}")

# ========== 3. 定义 5 种插值方法 ==========
methods = {
    '线性': {
        'T': interp1d(t_original, T_original, kind='linear', fill_value='extrapolate'),
        'C': interp1d(t_original, C_original, kind='linear', fill_value='extrapolate'),
        'color_T': 'tab:blue',
        'color_C': 'tab:orange',
    },
    '三次样条': {
        'T': CubicSpline(t_original, T_original),
        'C': CubicSpline(t_original, C_original),
        'color_T': 'tab:red',
        'color_C': 'tab:red',
    },
    'PCHIP': {
        'T': PchipInterpolator(t_original, T_original),
        'C': PchipInterpolator(t_original, C_original),
        'color_T': 'tab:green',
        'color_C': 'tab:red',
    },
    'Akima': {
        'T': Akima1DInterpolator(t_original, T_original),
        'C': Akima1DInterpolator(t_original, C_original),
        'color_T': 'tab:purple',
        'color_C': 'tab:purple',
    },
    '样条+平滑': {
        'T': UnivariateSpline(t_original, T_original, k=3, s=len(t_original) * 0.01),
        'C': UnivariateSpline(t_original, C_original, k=3, s=len(t_original) * 0.01),
        'color_T': 'tab:brown',
        'color_C': 'tab:brown',
    },
}

# ========== 4. 逐方法绘图 ==========
for name, m in methods.items():
    T_new = m['T'](t_new)
    C_new = m['C'](t_new)

    # ---- 温度图 ----
    plt.figure(figsize=(12, 5))
    plt.scatter(t_original, T_original, color='black', s=5,
                label='原始数据点', zorder=2)
    plt.plot(t_new, T_new, '-', color=m['color_T'], linewidth=1.5,
             label=f'{name} 插值（每秒）')
    plt.xlabel('时间 (s)', fontsize=12)
    plt.ylabel('温度 (°C)', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'温度_{name}插值.png', dpi=300)
    plt.show()

    # ---- 水分浓度图 ----
    plt.figure(figsize=(12, 5))
    plt.scatter(t_original, C_original, color='black', s=5,
                label='原始数据点', zorder=2)
    plt.plot(t_new, C_new, '-', color=m['color_C'], linewidth=1.5,
             label=f'{name} 插值（每秒）')
    plt.xlabel('时间 (s)', fontsize=12)
    plt.ylabel('水分浓度 (kg/kg)', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'水分浓度_{name}插值.png', dpi=300)
    plt.show()

    # ---- 导出该方法的插值结果 ----
    df_interp = pd.DataFrame({
        '时间(s)': t_new,
        '温度(°C)': T_new,
        '水分浓度(kg/kg)': C_new,
    })
    df_interp.to_excel(os.path.join(BASE, '..', '..', '结果', f'插值结果_{name}.xlsx'), index=False)

print("\n全部完成：共 10 张图，5 个 Excel 文件。")