import pandas as pd
import numpy as np
from xgboost import XGBRegressor
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import os

excel_path = r'D:\dataset.xlsx'
data = pd.read_excel(excel_path, header=0)
print(f"Successfully read the data {data.shape}")

X = data.iloc[:, :-1]
y = data.iloc[:, -1]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=21
)

xgbc = XGBRegressor(
    booster='gbtree',
    n_estimators=244,
    max_depth=3,
    min_child_weight=7,
    gamma=0,
    subsample=0.5,
    colsample_bytree=0.6,
    learning_rate=0.4,
    random_state=40
)

print("\nTraining XGBoost model...")
xgbc.fit(X_train, y_train)

sample_idx = 12
if sample_idx >= len(X_test):
    raise ValueError(f"Test set index {sample_idx} out of range, maximum is {len(X_test)-1}")
print(f"\n📋 Original feature data for test set index {sample_idx}:")
for col, val in zip(X_test.columns, X_test.iloc[sample_idx].values):
    print(f"   {col}: {val}")

sample_row = X_test.iloc[sample_idx]
sample_true = y_test.iloc[sample_idx]
sample_pred = xgbc.predict([sample_row.values])[0]
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 17
explainer = lime.lime_tabular.LimeTabularExplainer(
    X_train.values,
    feature_names=X_train.columns.tolist(),
    mode='regression'
)
predict_fn_xgb = lambda x: xgbc.predict(np.array(x)).astype(float)

print(f"\n🔬 Performing LIME local explanation analysis...")
exp_sample = explainer.explain_instance(
    sample_row.values,
    predict_fn_xgb,
    num_features=15
)
lime_results = exp_sample.as_list()
from lime import submodular_pick

lime_results_sorted = sorted(lime_results, key=lambda x: abs(x[1]), reverse=True)
print(f"\n📊 LIME feature importance ranking (sorted by contribution magnitude):")
print("-" * 80)
for j, (feature, value) in enumerate(lime_results_sorted[:15]):
    direction = "promoting" if value > 0 else "inhibiting"
    impact_level = "strong" if abs(value) > 0.1 else "moderate" if abs(value) > 0.05 else "weak"
    print(f"  {j+1:2d}. {feature:20s}: {value:+.4f} ({impact_level} {direction} denitrification)")
feature_contributions = {}
for feature, value in lime_results_sorted:
    feature_upper = feature.strip().upper()
    if 'C ' in feature or feature_upper == 'C' or feature.startswith('C '):
        if 'CLAY' not in feature_upper:
            feature_contributions['C'] = value
    elif 'PH' in feature_upper:
        feature_contributions['pH'] = value
    elif 'CLAY' in feature_upper:
        feature_contributions['CLAY'] = value
    elif 'SOM' in feature_upper:
        feature_contributions['SOM'] = value
    elif 'SIZE' in feature_upper:
        feature_contributions['SIZE'] = value
    elif 'ZP' in feature_upper:
        feature_contributions['ZP'] = value
    elif 'ED' in feature_upper:
        feature_contributions['ED'] = value
    elif 'TEMP' in feature_upper:
        feature_contributions['TEMP'] = value

print(f"\n Key feature LIME contribution extraction:")
for feat, contrib in feature_contributions.items():
    direction = "promoting" if contrib > 0 else "inhibiting"
    print(f"   {feat}: {contrib:+.4f} ({direction} denitrification)")
features = [item[0] for item in lime_results_sorted[:10]]
values = [item[1] for item in lime_results_sorted[:10]]
light_colors = ['#B7B7EB', '#EAB883', '#F09BA0']
dark_colors = ['#7575D9', '#D49942', '#E85961']
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 12), constrained_layout=True)
bars = []
for j, (feature, coeff) in enumerate(zip(features, values)):
    abs_coeff = abs(coeff)
    max_abs_coeff = max([abs(v) for v in values]) if values else 1
    intensity = abs_coeff / max_abs_coeff if max_abs_coeff > 0 else 0
    if coeff >= 0:
        base_color = light_colors[0]
        end_color = dark_colors[0]
    else:
        base_color = light_colors[2]
        end_color = dark_colors[2]
    r1, g1, b1 = int(base_color[1:3], 16), int(base_color[3:5], 16), int(base_color[5:7], 16)
    r2, g2, b2 = int(end_color[1:3], 16), int(end_color[3:5], 16), int(end_color[5:7], 16)
    r = int(r1 + (r2 - r1) * intensity)
    g = int(g1 + (g2 - g1) * intensity)
    b = int(b1 + (b2 - b1) * intensity)
    color = f'#{r:02x}{g:02x}{b:02x}'
    y_pos = len(features) - 1 - j
    bar = ax1.barh(y_pos, coeff, color=color, alpha=0.8, edgecolor='white', linewidth=0.5)
    bars.extend(bar)
ax1.set_yticks(range(len(features)))
ax1.set_yticklabels(features, fontsize=23, fontfamily='Times New Roman', fontweight='bold')
ax1.set_xlabel('LIME Value', fontsize=22, fontfamily='Times New Roman')
ax1.set_xlim(-0.15, 0.25)
ax1.set_title('Sample 11 Denitrification LIME Feature Importance', fontsize=24, fontweight='bold', fontfamily='Times New Roman', pad=25)
ax1.axvline(x=0, color='black', linestyle='-', alpha=0.3)
for j, (feature, value) in enumerate(zip(features, values)):
    y_pos = len(features) - 1 - j
    ax1.text(value + (0.01 if value >= 0 else -0.01), y_pos, f'{value:.3f}', ha='left' if value >= 0 else 'right', va='center', fontsize=19, fontweight='bold')

# 机制解释图
ax2.axis('off')
ax2.text(0.5, 0.95, 'Sample 11 Denitrification Mechanism Analysis', ha='center', va='top', fontsize=19, fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.7))
ax2.text(0.1, 0.85, 'Key Features:', fontsize=17, fontweight='bold')
for i, (feat, contrib) in enumerate(feature_contributions.items()):
    direction = "promoting" if contrib > 0 else "inhibiting"
    ax2.text(0.15, 0.80 - i*0.05, f'• {feat}: {contrib:+.3f} ({direction} denitrification)', fontsize=15)
ax2.text(0.1, 0.35, 'Actual Denitrification Rate:', fontsize=17, fontweight='bold')
ax2.text(0.15, 0.30, f'{sample_true:.4f}', fontsize=16, fontweight='bold', color='darkblue')
ax2.text(0.1, 0.20, 'Predicted Denitrification Rate:', fontsize=17, fontweight='bold')
ax2.text(0.15, 0.15, f'{sample_pred:.4f}', fontsize=16, fontweight='bold', color='darkgreen')
plt.tight_layout()
plt.subplots_adjust(left=0.22, right=0.98, wspace=0.25)
png_path = 'detailed_mechanism_analysis_Sample11_denitrification.png'
fig.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.show()
print(f"💾 Detailed mechanism analysis figure saved: {png_path}")

lime_sample_idx = 12
print(f"\n=== Performing detailed LIME analysis for sample {lime_sample_idx}")
sample_true_12 = y_test.iloc[lime_sample_idx]
sample_pred_12 = xgbc.predict(X_test.iloc[lime_sample_idx:lime_sample_idx+1])[0]
sample_error_12 = abs(sample_true_12 - sample_pred_12)
sample_relative_error_12 = sample_error_12 / abs(sample_true_12) * 100

print(f"True value: {sample_true_12:.6f}")
print(f"Predicted value: {sample_pred_12:.6f}")
print(f"Absolute error: {sample_error_12:.6f}")
print(f"Relative error: {sample_relative_error_12:.2f}%")

print(f"\nSample feature values:")
sample_features_12 = X_test.iloc[lime_sample_idx]
for feature_name, feature_value in sample_features_12.items():
    print(f"  {feature_name}: {feature_value:.6f}")

exp_12 = explainer.explain_instance(X_test.iloc[lime_sample_idx].values, predict_fn_xgb, num_features=15)
lime_list = sorted(exp_12.as_list(), key=lambda x: abs(x[1]), reverse=True)
print(f"\nLIME explanation results (Test Index {lime_sample_idx}):")
for feature, value in lime_list:
    print(f"  {feature}: {value:.4f}")
lime_list_sorted = sorted(lime_list, key=lambda x: abs(x[1]), reverse=True)[:10]
features_plot = [f for f, v in lime_list_sorted]
values_plot = [v for f, v in lime_list_sorted]
colors_plot = ['#7575D9' if v > 0 else '#E85961' for v in values_plot]

plt.figure(figsize=(12, 7))
bars = plt.barh(range(len(features_plot)), values_plot, color=colors_plot, alpha=0.8)
plt.yticks(range(len(features_plot)), features_plot, fontsize=14, fontfamily='Times New Roman', fontweight='bold')
plt.xlim(-0.15, 0.25)
from matplotlib.ticker import FormatStrFormatter
plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
plt.xlabel('LIME Value', fontsize=16, fontfamily='Times New Roman', fontweight='bold')
plt.title(f'LIME Explanation (Test Index {lime_sample_idx})', fontsize=18, fontweight='bold', fontfamily='Times New Roman', pad=30)
plt.axvline(x=0, color='black', linestyle='-', alpha=0.7)
plt.gca().invert_yaxis()
plt.tight_layout()
for i, bar in enumerate(bars):
    plt.text(values_plot[i] + (0.01 if values_plot[i] >= 0 else -0.01),
             bar.get_y() + bar.get_height()/2,
             f'{values_plot[i]:.3f}',
             va='center', ha='left' if values_plot[i] >= 0 else 'right', fontsize=13, fontweight='bold')
lime_bar_png = f'lime_bar_test_idx{lime_sample_idx}.png'
light_colors = ['#B7B7EB', '#EAB883', '#F09BA0']
dark_colors = ['#7575D9', '#D49942', '#E85961']
plt.figure(figsize=(12, 7))
bars = []
for j, (feature, coeff) in enumerate(zip(features_plot, values_plot)):
    abs_coeff = abs(coeff)
    max_abs_coeff = max([abs(v) for v in values_plot]) if values_plot else 1
    intensity = abs_coeff / max_abs_coeff if max_abs_coeff > 0 else 0
    if coeff >= 0:
        base_color = light_colors[0]
        end_color = dark_colors[0]
    else:
        base_color = light_colors[2]
        end_color = dark_colors[2]
    r1, g1, b1 = int(base_color[1:3], 16), int(base_color[3:5], 16), int(base_color[5:7], 16)
    r2, g2, b2 = int(end_color[1:3], 16), int(end_color[3:5], 16), int(end_color[5:7], 16)
    r = int(r1 + (r2 - r1) * intensity)
    g = int(g1 + (g2 - g1) * intensity)
    b = int(b1 + (b2 - b1) * intensity)
    color = f'#{r:02x}{g:02x}{b:02x}'
    bar = plt.barh(j, coeff, color=color, alpha=0.8, edgecolor='white', linewidth=0.5)
    bars.extend(bar)
plt.yticks(range(len(features_plot)), features_plot, fontsize=17, fontfamily='Times New Roman', fontweight='bold')
plt.xlim(-0.15, 0.25)
from matplotlib.ticker import FormatStrFormatter
plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
plt.xlabel('LIME Value', fontsize=16, fontfamily='Times New Roman', fontweight='bold')
plt.title(f'LIME Explanation (Test Index {lime_sample_idx})', fontsize=18, fontweight='bold', fontfamily='Times New Roman', pad=30)
plt.axvline(x=0, color='black', linestyle='-', alpha=0.7)
plt.gca().invert_yaxis()
plt.tight_layout()
for i, bar in enumerate(bars):
    plt.text(values_plot[i] + (0.01 if values_plot[i] >= 0 else -0.01),
             bar.get_y() + bar.get_height()/2,
             f'{values_plot[i]:.3f}',
             va='center', ha='left' if values_plot[i] >= 0 else 'right', fontsize=13, fontweight='bold')
plt.savefig(lime_bar_png, dpi=300, bbox_inches='tight')
plt.show()
print(f"saved: {lime_bar_png}")

