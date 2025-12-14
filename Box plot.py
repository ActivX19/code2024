import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from time import time

# Set style for better visualization
sns.set_style("white")
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['font.family'] = 'Times New Roman'

time0 = time()

# Read three datasets
data_xgb = pd.read_excel(r'D:\XT_XGB.xlsx', header=0)
data_svr = pd.read_excel(r'D:\XT_SVR.xlsx', header=0)
data_rf = pd.read_excel(r'D:\XT_RF.xlsx', header=0)

print("Dataset shapes:")
print(f"XGB dataset: {data_xgb.shape}")
print(f"SVR dataset: {data_svr.shape}")
print(f"RF dataset: {data_rf.shape}")

def extract_metric_data(df, metric_col, value_col):
    """Extract specific metric data from dataframe"""

    mask = df[metric_col] == 1
    return df.loc[mask, value_col].values
metrics = ['MSE', 'RMSE', 'MAE', 'R2']

# XGB 
xgb_mse = extract_metric_data(data_xgb, 'MSE', 'XGB')
xgb_rmse = extract_metric_data(data_xgb, 'RMSE', 'XGB')
xgb_mae = extract_metric_data(data_xgb, 'MAE', 'XGB')
xgb_r2 = extract_metric_data(data_xgb, 'R2', 'XGB')

# SVR 
svr_mse = extract_metric_data(data_svr, 'MSE', 'SVR')
svr_rmse = extract_metric_data(data_svr, 'RMSE', 'SVR')
svr_mae = extract_metric_data(data_svr, 'MAE', 'SVR')
svr_r2 = extract_metric_data(data_svr, 'R2', 'SVR')

# RF 
rf_mse = extract_metric_data(data_rf, 'MSE', 'RF')
rf_rmse = extract_metric_data(data_rf, 'RMSE', 'RF')
rf_mae = extract_metric_data(data_rf, 'MAE', 'RF')
rf_r2 = extract_metric_data(data_rf, 'R2', 'RF')

print(f"\nExtracted data statistics:")
print(f"XGB - MSE: {len(xgb_mse)} values, RMSE: {len(xgb_rmse)} values, MAE: {len(xgb_mae)} values, R2: {len(xgb_r2)} values")
print(f"SVR - MSE: {len(svr_mse)} values, RMSE: {len(svr_rmse)} values, MAE: {len(svr_mae)} values, R2: {len(svr_r2)} values")
print(f"RF - MSE: {len(rf_mse)} values, RMSE: {len(rf_rmse)} values, MAE: {len(rf_mae)} values, R2: {len(rf_r2)} values")

fig, axes = plt.subplots(4, 1, figsize=(8, 16))
fig.suptitle('Machine Learning Model Performance Metrics Comparison\n(Box plots with Individual Data Points)', 
             fontsize=16, fontweight='bold', y=0.998)
plot_data = [
    ([xgb_mse, svr_mse, rf_mse], 'MSE', 'Mean Squared Error (MSE)'),
    ([xgb_rmse, svr_rmse, rf_rmse], 'RMSE', 'Root Mean Squared Error (RMSE)'),
    ([xgb_mae, svr_mae, rf_mae], 'MAE', 'Mean Absolute Error (MAE)'),
    ([xgb_r2, svr_r2, rf_r2], 'R²', 'R-squared (R²)')
]

model_labels = ['XGB', 'SVR', 'RF']
colors = ['#B7B7EB', '#EAB883', '#F09BA0']
scatter_colors = ['#7575D9', '#D49942', '#E85961']

for i, (data_list, metric, title) in enumerate(plot_data):
    ax = axes[i]  
    valid_data = []
    valid_labels = []
    valid_colors = []
    valid_scatter_colors = []
    
    for j, (data, label) in enumerate(zip(data_list, model_labels)):
        if len(data) > 0:
            valid_data.append(data)
            valid_labels.append(label)
            valid_colors.append(colors[j])
            valid_scatter_colors.append(scatter_colors[j])
    
    if valid_data:
        # Create boxplot
        bp = ax.boxplot(valid_data, tick_labels=valid_labels, patch_artist=True,
                       boxprops=dict(linewidth=1.5, alpha=0.8),
                       medianprops=dict(color='black', linewidth=2.5),
                       whiskerprops=dict(color='black', linewidth=1.5),
                       capprops=dict(color='black', linewidth=1.5),
                       flierprops=dict(marker='o', color='red', alpha=0.5, markersize=4))
        
        # Set box colors and remove borders
        for patch, color in zip(bp['boxes'], valid_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.8)
            patch.set_edgecolor('none')  # Remove box borders
        
        # Add scatter plot overlay with better visibility (no borders)
        for j, (data, color) in enumerate(zip(valid_data, valid_scatter_colors)):
            np.random.seed(42)  # For reproducible results
            x_positions = np.random.normal(j+1, 0.05, len(data))
            scatter = ax.scatter(x_positions, data, alpha=0.8, s=40, c=color, 
                               zorder=3, edgecolors='none')  # Remove scatter borders
        ax.set_ylabel(f'{metric} Value', fontsize=11, fontweight='bold')
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color('#CCCCCC')  # Lighter gray
            spine.set_linewidth(0.8)
        
        # R²
        if metric == 'R²':
            ax.set_ylim(-0.1, 1.0)
    
        ax.tick_params(axis='x', labelsize=10)
        ax.tick_params(axis='y', labelsize=10)

        for label in ax.get_xticklabels():
            label.set_fontweight('bold')
        
        print(f"\n{title} Statistics:")
        for label, data in zip(valid_labels, valid_data):
            if len(data) > 0:
                print(f"{label}: Mean={np.mean(data):.6f}, Std={np.std(data):.6f}, Median={np.median(data):.6f}")
    
    else:
        ax.text(0.5, 0.5, f'No {metric} data found', 
               ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_title(title, fontsize=13, fontweight='bold')

# Add legend
legend_elements = [
    plt.Rectangle((0,0),1,1, facecolor=colors[0], alpha=0.8, label='XGB'),
    plt.Rectangle((0,0),1,1, facecolor=colors[1], alpha=0.8, label='SVR'),
    plt.Rectangle((0,0),1,1, facecolor=colors[2], alpha=0.8, label='RF'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='darkred', 
               markersize=8, alpha=0.8, label='Individual Data Points')
]

fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.01), 
          ncol=4, frameon=True, fancybox=True, shadow=True, fontsize=11)

plt.tight_layout()
plt.subplots_adjust(top=0.95, bottom=0.05)
plt.savefig('model_performance_boxplots_with_scatter.png', dpi=300, bbox_inches='tight', 
           facecolor='white', edgecolor='none')
plt.show()
print("\n" + "="*80)
print("Model Performance Summary Statistics")
print("="*80)

all_data = [
    (xgb_mse, svr_mse, rf_mse, 'MSE'),
    (xgb_rmse, svr_rmse, rf_rmse, 'RMSE'),
    (xgb_mae, svr_mae, rf_mae, 'MAE'),
    (xgb_r2, svr_r2, rf_r2, 'R²')
]

for xgb_data, svr_data, rf_data, metric in all_data:
    print(f"\n{metric} Metric:")
    print("-" * 60)
    
    data_dict = {'XGB': xgb_data, 'SVR': svr_data, 'RF': rf_data}
    
    stats_list = []
    for model, data in data_dict.items():
        if len(data) > 0:
            stats_list.append({
                'Model': model,
                'Mean': np.mean(data),
                'Std': np.std(data),
                'Median': np.median(data),
                'Min': np.min(data),
                'Max': np.max(data),
                'Count': len(data)
            })
    
    if stats_list:
        stats_df = pd.DataFrame(stats_list)
        print(stats_df.round(6))
    else:
        print("No data available")

print(f"\nTotal runtime: {time() - time0:.2f} seconds")
print("\n" + "="*80)
print("Model Performance Ranking (Best to Worst)")
print("="*80)

ranking_metrics = [
    (xgb_mse, svr_mse, rf_mse, 'MSE', 'lower_better'),
    (xgb_rmse, svr_rmse, rf_rmse, 'RMSE', 'lower_better'),
    (xgb_mae, svr_mae, rf_mae, 'MAE', 'lower_better'),
    (xgb_r2, svr_r2, rf_r2, 'R²', 'higher_better')
]

for xgb_data, svr_data, rf_data, metric, direction in ranking_metrics:
    print(f"\n{metric} Ranking:")
    data_dict = {'XGB': np.mean(xgb_data), 'SVR': np.mean(svr_data), 'RF': np.mean(rf_data)}
    
    if direction == 'lower_better':
        sorted_models = sorted(data_dict.items(), key=lambda x: x[1])
    else:
        sorted_models = sorted(data_dict.items(), key=lambda x: x[1], reverse=True)
    
    for rank, (model, value) in enumerate(sorted_models, 1):
        print(f"  {rank}. {model}: {value:.6f}")
