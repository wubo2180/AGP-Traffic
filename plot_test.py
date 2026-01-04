import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

# 数据准备
models = ['DCRNN', 'MTGNN', 'STID', 'STAEformer', 'GWNet']

# PEMS04 数据
baseline_mae = [19.63, 19.17, 18.35, 18.22, 18.74]
std_mae_mae = [18.65, 18.72, 17.93, 17.92, 17.80]
gemflow_mae = [18.01, 18.12, 17.03, 17.12, 16.73]

baseline_rmse = [31.24, 31.70, 29.86, 30.18, 30.32]
std_mae_rmse = [30.09, 31.03, 29.43, 29.37, 29.25]
gemflow_rmse = [29.13, 30.33, 28.29, 28.45, 28.39]

baseline_mape = [13.52, 13.37, 12.50, 11.98, 13.10]
std_mae_mape = [13.07, 12.72, 12.11, 12.11, 12.97]
gemflow_mape = [12.46, 11.96, 11.20, 11.32, 11.10]

# PEMS08 数据
pems08_mae_baseline = [15.21, 15.18, 14.21, 13.46, 14.55]
pems08_mae_std = [14.50, 14.84, 13.53, 13.30, 13.44]
pems08_mae_gemflow = [13.82, 14.12, 12.12, 12.21, 12.97]

# 创建1×4的子图
fig, axes = plt.subplots(1, 4, figsize=(20, 5))
fig.suptitle('Performance Comparison: GEMFlow vs Baselines', fontsize=16, fontweight='bold')

# 设置位置和宽度
x = np.arange(len(models))
width = 0.25

# 1. MAE对比 (PEMS04)
bars1 = axes[0].bar(x - width, baseline_mae, width, label='Baseline', alpha=0.8, color='lightblue')
bars2 = axes[0].bar(x, std_mae_mae, width, label='STD-MAE', alpha=0.8, color='lightgreen')
bars3 = axes[0].bar(x + width, gemflow_mae, width, label='GEMFlow', alpha=0.8, color='lightcoral')

axes[0].set_title('(a) MAE Comparison (PEMS04)')
# axes[0].set_xlabel('Models')
axes[0].set_ylabel('MAE')
axes[0].set_xticks(x)
axes[0].set_xticklabels(models, rotation=45)
axes[0].legend()
axes[0].grid(True, alpha=0.3, axis='y')

# 在柱子上添加数值标签
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=8)

# 2. RMSE对比 (PEMS04)
bars1 = axes[1].bar(x - width, baseline_rmse, width, label='Baseline', alpha=0.8, color='lightblue')
bars2 = axes[1].bar(x, std_mae_rmse, width, label='STD-MAE', alpha=0.8, color='lightgreen')
bars3 = axes[1].bar(x + width, gemflow_rmse, width, label='GEMFlow', alpha=0.8, color='lightcoral')

axes[1].set_title('(b) RMSE Comparison (PEMS04)')
# axes[1].set_xlabel('Models')
axes[1].set_ylabel('RMSE')
axes[1].set_xticks(x)
axes[1].set_xticklabels(models, rotation=45)
axes[1].legend()
axes[1].grid(True, alpha=0.3, axis='y')

# 3. MAPE对比 (PEMS04)
bars1 = axes[2].bar(x - width, baseline_mape, width, label='Baseline', alpha=0.8, color='lightblue')
bars2 = axes[2].bar(x, std_mae_mape, width, label='STD-MAE', alpha=0.8, color='lightgreen')
bars3 = axes[2].bar(x + width, gemflow_mape, width, label='GEMFlow', alpha=0.8, color='lightcoral')

axes[2].set_title('(c) MAPE Comparison (PEMS04)')
# axes[2].set_xlabel('Models')
axes[2].set_ylabel('MAPE (%)')
axes[2].set_xticks(x)
axes[2].set_xticklabels(models, rotation=45)
axes[2].legend()
axes[2].grid(True, alpha=0.3, axis='y')

# 4. MAE对比 (PEMS08)
bars1 = axes[3].bar(x - width, pems08_mae_baseline, width, label='Baseline', alpha=0.8, color='lightblue')
bars2 = axes[3].bar(x, pems08_mae_std, width, label='STD-MAE', alpha=0.8, color='lightgreen')
bars3 = axes[3].bar(x + width, pems08_mae_gemflow, width, label='GEMFlow', alpha=0.8, color='lightcoral')

axes[3].set_title('(d) MAE Comparison (PEMS08)')
# axes[3].set_xlabel('Models')
axes[3].set_ylabel('MAE')
axes[3].set_xticks(x)
axes[3].set_xticklabels(models, rotation=45)
axes[3].legend()
axes[3].grid(True, alpha=0.3, axis='y')

# 在柱子上添加数值标签
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        axes[3].text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.subplots_adjust(top=0.9)
output_path = "figure/ablation_study.pdf"
plt.savefig(output_path, format="pdf", dpi=300, bbox_inches='tight')
plt.show()

# 可选：性能提升百分比图
fig, ax = plt.subplots(1, 1, figsize=(12, 6))

# 计算GEMFlow相对于基线的提升百分比
mae_improvement = [(base - gem) / base * 100 for base, gem in zip(baseline_mae, gemflow_mae)]
rmse_improvement = [(base - gem) / base * 100 for base, gem in zip(baseline_rmse, gemflow_rmse)]
mape_improvement = [(base - gem) / base * 100 for base, gem in zip(baseline_mape, gemflow_mape)]

x_imp = np.arange(len(models))
width_imp = 0.25

bars1 = ax.bar(x_imp - width_imp, mae_improvement, width_imp, label='MAE Improvement', alpha=0.8)
bars2 = ax.bar(x_imp, rmse_improvement, width_imp, label='RMSE Improvement', alpha=0.8)
bars3 = ax.bar(x_imp + width_imp, mape_improvement, width_imp, label='MAPE Improvement', alpha=0.8)

ax.set_title('Performance Improvement of GEMFlow Over Baseline (%)')
ax.set_xlabel('Models')
ax.set_ylabel('Improvement (%)')
ax.set_xticks(x_imp)
ax.set_xticklabels(models, rotation=45)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# 添加数值标签
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.1f}%', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.show()