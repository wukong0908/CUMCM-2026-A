import os
import pandas as pd
import numpy as np
from scipy.interpolate import PchipInterpolator, interp1d, CubicSpline, UnivariateSpline

BASE = os.path.dirname(os.path.abspath(__file__))          # 脚本所在目录 code/
DATA = os.path.join(BASE, "..", "附件", "附件1.xlsx")        # 输入数据(相对脚本定位)

# ========== 1. 读取原始数据 ==========
df = pd.read_excel(DATA, sheet_name='Sheet1')
t_original = df['时间'].values.astype(float)
T_original = df['温度'].values.astype(float)
C_original = df['水分浓度'].values.astype(float)

# 插值到每秒
t_new = np.arange(t_original.min(), t_original.max() + 1, 1.0)

# ========== 2. 定义插值方法 ==========
def make_interpolators(t, y):
    methods = {}
    methods['线性'] = interp1d(t, y, kind='linear', fill_value='extrapolate')
    methods['PCHIP'] = PchipInterpolator(t, y, extrapolate=True)
    methods['三次样条'] = CubicSpline(t, y, extrapolate=True)
    methods['平滑样条'] = UnivariateSpline(t, y, k=3, s=len(t) * 0.01)
    return methods

# ========== 3. CV-RMSE（留一交叉验证） ==========
def loocv_rmse(t, y, method_name):
    n = len(t)
    errors = []
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        t_train, y_train = t[mask], y[mask]
        t_test, y_test = t[i], y[i]
        methods = make_interpolators(t_train, y_train)
        f = methods[method_name]
        y_pred = f(t_test)
        errors.append((y_test - y_pred) ** 2)
    return np.sqrt(np.mean(errors))

# ========== 4. 计算所有指标 ==========
def evaluate_all(t_original, y_original, label):
    print(f"\n{'='*70}")
    print(f"评价对象：{label}")
    print(f"{'='*70}")

    # 原始数据统计量
    y_mean = np.mean(y_original)
    y_std = np.std(y_original, ddof=1)
    y_range = np.max(y_original) - np.min(y_original)

    print(f"原始数据均值：{y_mean:.6f}")
    print(f"原始数据标准差：{y_std:.6f}")
    print(f"原始数据范围：{y_range:.6f}")

    # 密集网格
    t_dense = np.arange(t_original.min(), t_original.max() + 1, 1.0)

    method_names = ['线性', 'PCHIP', '三次样条', '平滑样条']
    methods_full = make_interpolators(t_original, y_original)

    results = []
    for name in method_names:
        f = methods_full[name]
        y_dense = f(t_dense)

        # 一阶差分、二阶差分
        dy = np.diff(y_dense)
        d2y = np.diff(y_dense, 2)

        s1 = np.sum(dy ** 2)          # 一阶差分平方和
        s2 = np.sum(d2y ** 2)         # 二阶差分平方和
        s2_avg = s2 / len(d2y)        # 平均二阶差分
        s2_ratio = s2 / s1 if s1 > 0 else np.nan   # 二阶/一阶

        # CV-RMSE
        cv_rmse = loocv_rmse(t_original, y_original, name)

        # 相对指标
        cv_ratio_std = cv_rmse / y_std if y_std > 0 else np.nan
        cv_ratio_range = cv_rmse / y_range if y_range > 0 else np.nan

        results.append({
            '方法': name,
            'CV-RMSE': cv_rmse,
            '数据标准差': y_std,
            'CV-RMSE/标准差': cv_ratio_std,
            'CV-RMSE/数据范围': cv_ratio_range,
            '一阶差分平方和': s1,
            '二阶差分平方和': s2,
            '二阶/一阶': s2_ratio,
            '平均二阶差分': s2_avg,
            '数据范围': y_range,
            '平均二阶差分/数据范围': s2_avg / y_range if y_range > 0 else np.nan,
        })

    df_results = pd.DataFrame(results)

    # 格式化输出
    pd.set_option('display.float_format', lambda x: f'{x:.6e}' if abs(x) < 1e-3 else f'{x:.4f}')
    print("\n各方法指标对比：")
    print(df_results.to_string(index=False))

    return df_results, y_std, y_range

# ========== 5. 执行评价 ==========
results_T, std_T, range_T = evaluate_all(t_original, T_original, '温度')
results_C, std_C, range_C = evaluate_all(t_original, C_original, '水分浓度')

# ========== 6. 单独输出 PCHIP 的判断结果 ==========
def judge_pchip(results, std, range_, label):
    print(f"\n{'='*70}")
    print(f"PCHIP 判断结果：{label}")
    print(f"{'='*70}")
    row = results[results['方法'] == 'PCHIP'].iloc[0]

    print(f"CV-RMSE = {row['CV-RMSE']:.6e}")
    print(f"数据标准差 = {std:.6f}")
    print(f"CV-RMSE / 标准差 = {row['CV-RMSE/标准差']*100:.4f}%")
    if row['CV-RMSE/标准差'] < 0.10:
        print("  → 优秀（< 10%）")
    elif row['CV-RMSE/标准差'] < 0.20:
        print("  → 可接受（< 20%）")
    else:
        print("  → 不合格（> 20%）")

    print(f"\nCV-RMSE / 数据范围 = {row['CV-RMSE/数据范围']*100:.4f}%")
    if row['CV-RMSE/数据范围'] < 0.05:
        print("  → 优秀（< 5%）")
    else:
        print("  → 不合格（> 5%）")

    print(f"\n一阶差分平方和 = {row['一阶差分平方和']:.6e}")
    print(f"二阶差分平方和 = {row['二阶差分平方和']:.6e}")
    print(f"二阶 / 一阶 = {row['二阶/一阶']*100:.4f}%")
    if row['二阶/一阶'] < 0.01:
        print("  → 光滑（< 1%）")
    else:
        print("  → 不够光滑（> 1%）")

    print(f"\n平均二阶差分 = {row['平均二阶差分']:.6e}")
    print(f"数据范围 = {range_:.6f}")
    print(f"平均二阶差分 / 数据范围 = {row['平均二阶差分/数据范围']:.6e}")
    if row['平均二阶差分/数据范围'] < 1e-4:
        print("  → 非常光滑（< 1e-4）")
    else:
        print("  → 光滑性一般（> 1e-4）")

judge_pchip(results_T, std_T, range_T, '温度')
judge_pchip(results_C, std_C, range_C, '水分浓度')

# ========== 7. 保存到 Excel ==========
with pd.ExcelWriter('插值指标计算.xlsx') as writer:
    results_T.to_excel(writer, sheet_name='温度', index=False)
    results_C.to_excel(writer, sheet_name='水分浓度', index=False)
print("\n结果已保存到 插值指标计算.xlsx")