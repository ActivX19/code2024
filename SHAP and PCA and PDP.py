import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor as XGBR
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os

def create_gradient_barh(ax, y_positions, values, labels, max_val=None, cmap_name='Blues', alpha_range=(0.35, 1)):
    
    if max_val is None:
        max_val = max(values) if len(values) > 0 else 1.0
    
    fig = ax.get_figure()
    fig.patch.set_facecolor('white')
    ax.set_facecolor('#f5f5f5')
    ax.grid(axis='x', linestyle='--', color='#e0e0e0', linewidth=1)
    
    cmap = getattr(plt.cm, cmap_name)
    bar_height = 0.7
    
    for i, (y_pos, val, label) in enumerate(zip(y_positions, values, labels)):
        y0 = y_pos - bar_height / 2
        y1 = y_pos + bar_height / 2
        
        darker_gradient = np.linspace(alpha_range[0], alpha_range[1], 256).reshape(1, -1)
        im = ax.imshow(darker_gradient, extent=(0, max_val, y0, y1), 
                      aspect='auto', cmap=cmap, vmin=0, vmax=1, zorder=2)
        
        rect = patches.Rectangle((0, y0), val, bar_height, transform=ax.transData)
        im.set_clip_path(rect)
    
    ax.margins(y=0.12)
    
    ax.grid(axis='x', linestyle='--', alpha=0.6)
    ax.set_xlim(0, max_val * 1.05)
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    return max_val

SHOW_PLOTS = True

import matplotlib
matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['font.sans-serif'] = ['Times New Roman']
matplotlib.rcParams['axes.unicode_minus'] = False

data_list = pd.read_excel(r'D:\dataset.xlsx', header=0)
X = data_list.iloc[:, :-1]
y = data_list.iloc[:, -1]

type_cols = [c for c in X.columns if str(c).startswith('type_')]
print('\nDetected type columns:', type_cols)
if len(type_cols) > 0:

    X[type_cols] = X[type_cols].fillna(0)
    try:
        X[type_cols] = (X[type_cols] != 0).astype(int)
    except Exception:

        X[type_cols] = X[type_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
        X[type_cols] = (X[type_cols] != 0).astype(int)

    print('\nPer-column value counts for type_* columns:')
    for c in type_cols:
        print(f"  {c}:\n", X[c].value_counts().to_string())

    row_sum = X[type_cols].sum(axis=1)
    print('\nRow label-count distribution (how many type_* true per row):')
    print(row_sum.value_counts().sort_index())

    def combine_types(row):
        present = [c.replace('type_', '') for c in type_cols if row.get(c, 0) == 1]
        if not present:
            return 'none'
        if len(present) == 1:
            return present[0]
        return '&'.join(sorted(present))

    X['type_multi'] = X.apply(combine_types, axis=1)

    def first_priority(row):
        for c in type_cols:
            if row.get(c, 0) == 1:
                return c.replace('type_', '')
        return 'none'
    X['TYPE'] = X.apply(first_priority, axis=1)
    print('\nCombined `TYPE` value counts (single label by priority):')
    print(X['TYPE'].value_counts())
else:
    print('\nNo type_* columns detected; skipping consolidation.')

if 'TYPE' in X.columns:

    X['type_label'] = X['TYPE']

    X['TYPE'] = pd.Categorical(X['TYPE']).codes

if 'type_multi' in X.columns:
    X = X.drop(columns=['type_multi'])

X_model = X.copy()
obj_cols = X_model.select_dtypes(include=['object']).columns.tolist()
if len(obj_cols) > 0:
    print('\nDropping non-numeric columns before modeling:', obj_cols)
    X_model = X_model.drop(columns=obj_cols)

X_train, X_test, y_train, y_test = train_test_split(
    X_model, y, test_size=0.3, random_state=21
)

model = XGBR(
    n_estimators=306,
    max_depth=8,
    min_child_weight=4,
    gamma=0,
    subsample=0.5,
    colsample_bytree=0.2,
    learning_rate=0.2,
    random_state=40
)

model.fit(X_train, y_train)

X_model_for_interaction = X_model.copy()
if 'TYPE' in X_model_for_interaction.columns:
    X_model_for_interaction = X_model_for_interaction.drop(columns=['TYPE'])

X_train_interaction, X_test_interaction, y_train_interaction, y_test_interaction = train_test_split(
    X_model_for_interaction, y, test_size=0.3, random_state=21
)

model_for_interaction = XGBR(
    n_estimators=306,
    max_depth=8,
    min_child_weight=4,
    gamma=0,
    subsample=0.5,
    colsample_bytree=0.2,
    learning_rate=0.2,
    random_state=40
)
model_for_interaction.fit(X_train_interaction, y_train_interaction)

explainer = shap.Explainer(model)
shap_values_raw = explainer(X_test)

type_cols = [c for c in X_test.columns if str(c).startswith('type_')]
raw_features = list(X_test.columns)
sv_raw = shap_values_raw.values if hasattr(shap_values_raw, 'values') else np.array(shap_values_raw)

merged_features = []
merged_sv = []
type_indices = [raw_features.index(c) for c in type_cols] if type_cols else []
non_type_indices = [i for i, c in enumerate(raw_features) if c not in type_cols]

if type_cols:

    if 'TYPE' in raw_features:

        merged_features = [c for c in raw_features if c not in type_cols and c != 'TYPE'] + ['TYPE']
        non_type_no_type_idx = [i for i, c in enumerate(raw_features) if c not in type_cols and c != 'TYPE']

        merged_sv = np.concatenate([
            sv_raw[:, non_type_no_type_idx],
            sv_raw[:, type_indices].sum(axis=1, keepdims=True)
        ], axis=1)
    else:

        merged_features = [c for c in raw_features if c not in type_cols] + ['TYPE']
        merged_sv = np.concatenate([
            sv_raw[:, non_type_indices],
            sv_raw[:, type_indices].sum(axis=1, keepdims=True)
        ], axis=1)
else:
    merged_features = raw_features[:]
    merged_sv = sv_raw.copy()

print('\nMerged features for all downstream plots:', merged_features)
print('Merged SHAP value matrix shape:', merged_sv.shape)

out_dir = os.path.join(os.path.dirname(__file__), 'outputs')
os.makedirs(out_dir, exist_ok=True)

import re
def strip_units(name):
    s = str(name)

    s = s.replace('（', '(').replace('）', ')')

    s = re.sub(r"\s*\([^)]*\)", "", s)
    s = re.sub(r"\s*\[[^\]]*\]", "", s)

    s = re.sub(r"(?i)\b(mV|V|mA|A|%|°C|nm|µm|um|kΩ|Ω|mg/L|g/L)\b", "", s)

    s = re.sub(r"\s+", " ", s).strip()
    return s

display_X_test = X_test.copy()
display_X_test.columns = [strip_units(c) for c in display_X_test.columns]

display_merged = pd.DataFrame(index=X_test.index)

for f in merged_features:
    if f == 'TYPE':
        if any(str(c).startswith('type_') for c in X_test.columns):
            type_cols_local = [c for c in X_test.columns if str(c).startswith('type_')]
            display_merged['TYPE'] = X_test[type_cols_local].sum(axis=1)
        elif 'TYPE' in X_test.columns:
            display_merged['TYPE'] = X_test['TYPE']
        else:
            display_merged['TYPE'] = 0
    else:
        if f in X_test.columns:
            display_merged[f] = X_test[f]
        else:
            display_merged[f] = 0

display_merged_clean = display_merged.copy()
display_merged_clean.columns = [strip_units(c) for c in display_merged_clean.columns]

print("\n" + "="*50)
print("Feature importance ranking")
print("="*50)

sv = merged_sv
orig_features = merged_features[:]

importance_list = []
feature_list = []
for i, feat in enumerate(orig_features):

    feature_list.append(strip_units(feat))
    importance_list.append(np.abs(sv[:, i]).mean())

feature_importance = pd.DataFrame({
    'feature': feature_list,
    'importance': importance_list
}).sort_values('importance', ascending=False)

plt.figure(figsize=(10, max(6, 0.35 * min(20, feature_importance.shape[0]))), dpi=250)
import matplotlib.patches as patches
ax = plt.gca()
fig = plt.gcf()

fig.patch.set_facecolor('white')
ax.set_facecolor('#f5f5f5')
ax.grid(axis='x', linestyle='--', color='#e0e0e0', linewidth=1)

fi = pd.DataFrame({
    'feature': feature_importance['feature'].values,
    'importance': feature_importance['importance'].values
})

top_n = min(20, fi.shape[0])
fi_top = fi.head(top_n).iloc[::-1]

import matplotlib.patches as patches
max_val = fi['importance'].max() if fi.shape[0] > 0 else 1.0
cmap = plt.cm.Blues
gradient = np.linspace(0, 1, 256).reshape(1, -1)
bar_height = 0.7

for i, (feat, val) in enumerate(zip(fi_top['feature'], fi_top['importance'])):
    y0 = i - bar_height / 2
    y1 = i + bar_height / 2

    darker_gradient = np.linspace(0.35, 1, 256).reshape(1, -1)
    im = ax.imshow(darker_gradient, extent=(0, max_val, y0, y1), aspect='auto', cmap=cmap, vmin=0, vmax=1, zorder=2)
    rect = patches.Rectangle((0, y0), val, bar_height, transform=ax.transData)
    im.set_clip_path(rect)

ax.margins(y=0.12) 

ax.set_yticks(np.arange(len(fi_top)))
ax.set_yticklabels(fi_top['feature'], fontsize=20, fontname='Times New Roman')
ax.set_xlabel('Mean |SHAP value|', fontsize=14, fontname='Times New Roman')
ax.set_ylabel('Feature', fontsize=14, fontname='Times New Roman')
ax.set_title(f'Feature importance ranking - SHAP - Top {top_n}', fontsize=16, pad=20, fontname='Times New Roman')
ax.grid(axis='x', linestyle='--', alpha=0.6)
ax.set_xlim(0, max_val * 1.05)
for spine in ax.spines.values():
    spine.set_visible(False)

for i, val in enumerate(fi_top['importance']):
    ax.text(val + (max_val * 0.01), i, f'{val:.3f}', va='center', fontsize=12, 
            fontname='Times New Roman', fontfamily='serif', zorder=3)

plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'feature_importance.png'), bbox_inches='tight', dpi=250)
if SHOW_PLOTS:
    plt.show()
plt.close()

print("\nFeature importance ranking:")
print(feature_importance.to_string(index=False))

print("\n" + "="*50)
print("SHAP Interaction Analysis - Computing interaction values")
print("="*50)

try:

    print(f"Features for interaction analysis (using X_test_interaction): {list(X_test_interaction.columns)}")
    print(f"Number of features: {X_test_interaction.shape[1]}")
    
    type_cols_for_interaction = [c for c in X_test_interaction.columns if str(c).startswith('type_')]
    print(f"Type features: {type_cols_for_interaction}")
    
    shap_interaction_values = None
    
    try:
        print("Computing SHAP interaction values using TreeExplainer...")
        tree_explainer_interaction = shap.TreeExplainer(model_for_interaction)
        shap_interaction_values = tree_explainer_interaction.shap_interaction_values(X_test_interaction)
        print(f"Successfully computed interaction values with shape: {np.array(shap_interaction_values).shape}")
    except Exception as _e:
        print('TreeExplainer.shap_interaction_values failed:', _e)

        try:
            explainer_interaction = shap.Explainer(model_for_interaction)
            shap_interaction_values = explainer_interaction.shap_interaction_values(X_test_interaction)
        except Exception as _e2:
            print('explainer.shap_interaction_values also failed:', _e2)

    if shap_interaction_values is not None and len(shap_interaction_values) > 0:

        print("\n" + "="*50)
        print("SHAP Interaction Heatmap Analysis")
        print("="*50)
        
        try:

            print("Generating SHAP interaction heatmap for all features...")
            
            interaction_matrix_mean = np.mean(np.abs(shap_interaction_values), axis=0)
            
            features_for_interaction = list(X_test_interaction.columns)
            feature_names_clean = [strip_units(f) for f in features_for_interaction]
            
            merged_feature_importance_dict = {}
            for i, feat in enumerate(merged_features):
                cleaned_feat = strip_units(feat)
                merged_feature_importance_dict[cleaned_feat] = np.abs(merged_sv[:, i]).mean()
            
            original_feature_importances = {}
            type_cols_local = [c for c in features_for_interaction if str(c).startswith('type_')]
            
            for feat in features_for_interaction:
                if str(feat).startswith('type_'):

                    if 'TYPE' in merged_feature_importance_dict:
                        original_feature_importances[feat] = merged_feature_importance_dict['TYPE']
                    else:
                        original_feature_importances[feat] = 0
                else:

                    cleaned_feat = strip_units(feat)
                    if cleaned_feat in merged_feature_importance_dict:
                        original_feature_importances[feat] = merged_feature_importance_dict[cleaned_feat]
                    else:
                        original_feature_importances[feat] = 0
            
            sorted_features = sorted(features_for_interaction, 
                                   key=lambda x: original_feature_importances[x], reverse=True)
            sorted_indices = [features_for_interaction.index(f) for f in sorted_features]
            sorted_feature_names_clean = [strip_units(f) for f in sorted_features]
            
            sorted_interaction_matrix = interaction_matrix_mean[np.ix_(sorted_indices, sorted_indices)]
            
            globals()['sorted_features'] = sorted_features
            globals()['sorted_interaction_matrix'] = sorted_interaction_matrix
            
            plt.figure(figsize=(max(12, len(features_for_interaction) * 0.6), 
                               max(10, len(features_for_interaction) * 0.6)))
            
            import seaborn as sns
            
            mask = np.triu(np.ones_like(sorted_interaction_matrix, dtype=bool), k=0)
            
            annot_matrix = np.full_like(sorted_interaction_matrix, '', dtype=object)
            
            valid_positions = []
            for i in range(sorted_interaction_matrix.shape[0]):
                for j in range(sorted_interaction_matrix.shape[1]):
                    if not mask[i, j]:
                        valid_positions.append((i, j, sorted_interaction_matrix[i, j]))
            
            for i, j, value in valid_positions:
                annot_matrix[i, j] = f'{value:.3f}'
            
            ax = sns.heatmap(sorted_interaction_matrix, 
                           annot=annot_matrix,
                           fmt='',
                           cmap='YlOrRd',
                           square=True,
                           linewidths=0.5,
                           cbar_kws={"shrink": 0.8},
                           xticklabels=sorted_feature_names_clean,
                           yticklabels=sorted_feature_names_clean,
                           mask=mask,
                           annot_kws={'fontsize': 14, 'fontname': 'Times New Roman'})
            
            cbar = ax.collections[0].colorbar
            cbar.ax.tick_params(labelsize=18)
            
            plt.xticks(rotation=45, ha='right', fontsize=22, fontname='Times New Roman')
            plt.yticks(rotation=0, fontsize=22, fontname='Times New Roman')
            
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, 'shap_interaction_heatmap_all_features_sorted.png'), 
                       bbox_inches='tight', dpi=250)
            if SHOW_PLOTS:
                plt.show()
            plt.close()
            print("SHAP interaction heatmap for all features (sorted by importance) generated successfully!")
            
            print("Generating SHAP interaction heatmap for merged TYPE features...")
            
            type_cols_local = [c for c in features_for_interaction if str(c).startswith('type_')]
            non_type_cols_local = [c for c in features_for_interaction if not str(c).startswith('type_')]
            
            if len(type_cols_local) > 0:

                merged_features_for_heatmap = non_type_cols_local + ['TYPE']
                n_merged = len(merged_features_for_heatmap)
                merged_interaction_matrix = np.zeros((n_merged, n_merged))
                
                type_indices_local = [features_for_interaction.index(c) for c in type_cols_local]
                non_type_indices_local = [features_for_interaction.index(c) for c in non_type_cols_local]
                
                for i, idx_i in enumerate(non_type_indices_local):
                    for j, idx_j in enumerate(non_type_indices_local):
                        merged_interaction_matrix[i, j] = interaction_matrix_mean[idx_i, idx_j]
                
                for i, idx_i in enumerate(non_type_indices_local):

                    type_interaction_sum = np.sum([interaction_matrix_mean[idx_i, type_idx] 
                                                 for type_idx in type_indices_local])
                    merged_interaction_matrix[i, -1] = type_interaction_sum
                    merged_interaction_matrix[-1, i] = type_interaction_sum
                
                type_internal_interaction = np.sum([interaction_matrix_mean[i, j] 
                                                  for i in type_indices_local 
                                                  for j in type_indices_local if i != j]) / 2
                merged_interaction_matrix[-1, -1] = type_internal_interaction
                
                merged_features_for_heatmap = non_type_cols_local + ['TYPE']
                
                merged_feature_importances_for_sorting = {}
                
                for feat in non_type_cols_local:
                    if feat in merged_features:
                        feat_idx = merged_features.index(feat)
                        merged_feature_importances_for_sorting[feat] = np.abs(merged_sv[:, feat_idx]).mean()
                    else:
                        merged_feature_importances_for_sorting[feat] = 0
                
                if 'TYPE' in merged_features:
                    type_idx = merged_features.index('TYPE')
                    merged_feature_importances_for_sorting['TYPE'] = np.abs(merged_sv[:, type_idx]).mean()
                else:
                    merged_feature_importances_for_sorting['TYPE'] = 0
                
                sorted_merged_features = sorted(merged_features_for_heatmap, 
                                              key=lambda x: merged_feature_importances_for_sorting[x], reverse=True)
                sorted_merged_indices = [merged_features_for_heatmap.index(f) for f in sorted_merged_features]
                sorted_merged_feature_names_clean = [strip_units(f) for f in sorted_merged_features]
                
                sorted_merged_interaction_matrix = merged_interaction_matrix[np.ix_(sorted_merged_indices, sorted_merged_indices)]
                
                plt.figure(figsize=(max(10, len(sorted_merged_features) * 0.8), 
                                   max(8, len(sorted_merged_features) * 0.8)))
                
                mask_merged = np.triu(np.ones_like(sorted_merged_interaction_matrix, dtype=bool), k=0)
                
                annot_matrix_merged = np.full_like(sorted_merged_interaction_matrix, '', dtype=object)
                
                valid_positions_merged = []
                for i in range(sorted_merged_interaction_matrix.shape[0]):
                    for j in range(sorted_merged_interaction_matrix.shape[1]):
                        if not mask_merged[i, j]:
                            valid_positions_merged.append((i, j, sorted_merged_interaction_matrix[i, j]))
                
                for i, j, value in valid_positions_merged:
                    annot_matrix_merged[i, j] = f'{value:.3f}'
                
                ax = sns.heatmap(sorted_merged_interaction_matrix, 
                               annot=annot_matrix_merged,
                               fmt='',
                               cmap='RdYlBu_r',
                               square=True,
                               linewidths=0.5,
                               cbar_kws={"shrink": 0.8},
                               xticklabels=sorted_merged_feature_names_clean,
                               yticklabels=sorted_merged_feature_names_clean,
                               mask=mask_merged,
                               annot_kws={'fontsize': 16, 'fontname': 'Times New Roman'})
                
                cbar = ax.collections[0].colorbar
                cbar.ax.tick_params(labelsize=18)
                
                plt.xticks(rotation=45, ha='right', fontsize=22, fontname='Times New Roman')
                plt.yticks(rotation=0, fontsize=22, fontname='Times New Roman')
                
                plt.tight_layout()
                plt.savefig(os.path.join(out_dir, 'shap_interaction_heatmap_type_merged_sorted.png'), 
                           bbox_inches='tight', dpi=250)
                if SHOW_PLOTS:
                    plt.show()
                plt.close()
                print("TYPE merged SHAP interaction heatmap (sorted by importance) generated successfully!")
                
                interaction_df_all = pd.DataFrame(sorted_interaction_matrix, 
                                                index=sorted_feature_names_clean, 
                                                columns=sorted_feature_names_clean)
                interaction_df_all.to_csv(os.path.join(out_dir, 'shap_interaction_matrix_all_features_sorted.csv'))
                
                interaction_df_merged = pd.DataFrame(sorted_merged_interaction_matrix, 
                                                   index=sorted_merged_feature_names_clean, 
                                                   columns=sorted_merged_feature_names_clean)
                interaction_df_merged.to_csv(os.path.join(out_dir, 'shap_interaction_matrix_type_merged_sorted.csv'))
                
                print("Interaction matrix values saved to CSV files (sorted by importance)!")
                
                print("\n" + "="*50)
                print("SHAP Interaction Summary Plots - Two Types")
                print("="*50)
                
                print("Generating interaction summary plot for all features (without merging type) (sorted by feature importance)...")
                
                interaction_features = list(X_test_interaction.columns)
                print(f"Number of interaction features: {len(interaction_features)}")
                print(f"Interaction features: {interaction_features[:10]}...")
                
                print(f"Using computed feature importance order: {list(feature_importance['feature'])}")
                
                importance_dict = dict(zip(feature_importance['feature'], feature_importance['importance']))
                print(f"Feature importance dictionary: {importance_dict}")
                
                interaction_feature_importance = {}
                for feat in interaction_features:
                    cleaned_feat = strip_units(feat)
                    if cleaned_feat in importance_dict:
                        interaction_feature_importance[feat] = importance_dict[cleaned_feat]
                    elif feat.startswith('type_') and 'TYPE' in importance_dict:

                        interaction_feature_importance[feat] = importance_dict['TYPE']
                    else:
                        interaction_feature_importance[feat] = 0
                        print(f"Warning: Feature {feat} (cleaned: {cleaned_feat}) not found in importance dictionary")
                
                sorted_features_all = sorted(interaction_features, 
                                           key=lambda x: interaction_feature_importance[x], reverse=True)
                sorted_indices_all = [interaction_features.index(f) for f in sorted_features_all]
                
                non_type_features_sorted = [f for f in sorted_features_all if not f.startswith('type_')]
                
                print(f"Sorted features (all): {sorted_features_all[:15]}")
                print(f"Top 11 non-TYPE features (sorted by importance): {non_type_features_sorted[:11]}")
                print(f"Corresponding importance values: {[interaction_feature_importance[f] for f in non_type_features_sorted[:11]]}")
                
                sorted_feature_names_clean = []
                for i, f in enumerate(sorted_features_all):
                    cleaned_name = strip_units(f)

                    if not cleaned_name or cleaned_name.strip() == '':
                        cleaned_name = str(f)
                        print(f"Warning: Feature '{f}' cleaned name is empty, using original name")
                    sorted_feature_names_clean.append(cleaned_name)
                
                print(f"Corresponding cleaned feature names: {sorted_feature_names_clean[:15]}")
                
                empty_names = [i for i, name in enumerate(sorted_feature_names_clean) if not name or name.strip() == '']
                if empty_names:
                    print(f"Empty feature names found at indices: {empty_names}")
                    for idx in empty_names:
                        sorted_feature_names_clean[idx] = f"Feature_{idx}"
                        print(f"Setting feature name at index {idx} to: {sorted_feature_names_clean[idx]}")
                
                sorted_interaction_values = shap_interaction_values[:, sorted_indices_all, :][:, :, sorted_indices_all]
                sorted_X_test_interaction = X_test_interaction[sorted_features_all].copy()
                
                sorted_X_test_interaction.columns = sorted_feature_names_clean
                
                print(f"Column names of the reordered data matrix: {list(sorted_X_test_interaction.columns)}")
                print(f"Interaction values matrix shape: {sorted_interaction_values.shape}")
                
                print("\nGenerating interaction summary plot for top 11 most important non-TYPE features...")
                top_11_non_type_features = non_type_features_sorted[:11]
                
                top_11_indices_in_sorted = [sorted_features_all.index(f) for f in top_11_non_type_features]
                top_11_indices_in_original = [interaction_features.index(f) for f in top_11_non_type_features]
                
                top_11_interaction_values = shap_interaction_values[:, top_11_indices_in_original, :][:, :, top_11_indices_in_original]
                top_11_X_test_interaction = X_test_interaction[top_11_non_type_features].copy()
                
                top_11_feature_names_clean = [strip_units(f) for f in top_11_non_type_features]
                top_11_X_test_interaction.columns = top_11_feature_names_clean
                
                print(f"Top 11 non-TYPE features: {top_11_non_type_features}")
                print(f"Cleaned feature names: {top_11_feature_names_clean}")
                
                feature_importance_top11 = feature_importance.head(11)['feature'].tolist()
                feature_importance_non_type = [f for f in feature_importance_top11 if f != 'TYPE'][:11]
                print(f"Top 11 features in importance plot (excluding TYPE): {feature_importance_non_type}")
                print(f"Top 11 non-TYPE features in interaction plot: {top_11_feature_names_clean}")
                
                plt.figure(figsize=(16, max(11, len(top_11_non_type_features) * 0.8)))
                shap.summary_plot(top_11_interaction_values, top_11_X_test_interaction, 
                                feature_names=top_11_feature_names_clean,
                                show=SHOW_PLOTS, max_display=len(top_11_non_type_features), sort=False)
                
                ax = plt.gca()
                ax.tick_params(axis='y', labelsize=28)
                ax.tick_params(axis='x', labelsize=8)
                for label in ax.get_yticklabels():
                    label.set_fontname('Times New Roman')
                for label in ax.get_xticklabels():
                    label.set_fontname('Times New Roman')
                
                plt.subplots_adjust(top=0.92)
                plt.title("SHAP Interaction Summary Plot - Top 11 Non-TYPE Features (Sorted by Importance)", 
                         fontsize=16, fontname='Times New Roman', pad=20)
                
                plt.tight_layout()
                plt.savefig(os.path.join(out_dir, 'shap_interaction_summary_top11_non_type_features.png'), bbox_inches='tight', dpi=250)
                if SHOW_PLOTS:
                    plt.show()
                plt.close()
                print("Interaction summary plot for top 11 most important non-TYPE features generated successfully!")
                print("\nGenerating interaction summary plot for remaining features...")
                
                remaining_features = [f for f in sorted_features_all if f not in top_11_non_type_features]
                
                if len(remaining_features) > 0:

                    remaining_indices_in_original = [interaction_features.index(f) for f in remaining_features]
                    
                    remaining_interaction_values = shap_interaction_values[:, remaining_indices_in_original, :][:, :, remaining_indices_in_original]
                    remaining_X_test_interaction = X_test_interaction[remaining_features].copy()
                    
                    remaining_feature_names_clean = [strip_units(f) for f in remaining_features]
                    remaining_X_test_interaction.columns = remaining_feature_names_clean
                    
                    print(f"Remaining features: {remaining_features}")
                    print(f"Cleaned names of remaining features: {remaining_feature_names_clean}")
                    
                    plt.figure(figsize=(16, max(8, len(remaining_features) * 0.6)))
                    shap.summary_plot(remaining_interaction_values, remaining_X_test_interaction, 
                                    feature_names=remaining_feature_names_clean,
                                    show=SHOW_PLOTS, max_display=len(remaining_features), sort=False)
                    
                    ax = plt.gca()
                    ax.tick_params(axis='y', labelsize=28)
                    ax.tick_params(axis='x', labelsize=8)
                    for label in ax.get_yticklabels():
                        label.set_fontname('Times New Roman')
                    for label in ax.get_xticklabels():
                        label.set_fontname('Times New Roman')
                    
                    plt.subplots_adjust(top=0.92)
                    plt.title(f"SHAP Interaction Summary Plot - Remaining {len(remaining_features)} Features (Sorted by Importance)", 
                             fontsize=16, fontname='Times New Roman', pad=20)
                    
                    plt.tight_layout()
                    plt.savefig(os.path.join(out_dir, 'shap_interaction_summary_remaining_features.png'), bbox_inches='tight', dpi=250)
                    if SHOW_PLOTS:
                        plt.show()
                    plt.close()
                    print(f"Interaction summary plot for remaining {len(remaining_features)} features generated successfully!")
                else:
                    print("No remaining features to display")
                
                print("\nGenerating complete interaction summary plot (all features)...")
                
                all_feature_names = list(sorted_X_test_interaction.columns)
                print(f"Complete list of feature names: {all_feature_names}")
                
                plt.figure(figsize=(18, max(12, len(sorted_features_all) * 0.6)))
                shap.summary_plot(sorted_interaction_values, sorted_X_test_interaction, 
                                feature_names=all_feature_names,
                                show=SHOW_PLOTS, max_display=len(sorted_features_all), sort=False)
                
                ax = plt.gca()
                ax.tick_params(axis='y', labelsize=28)
                ax.tick_params(axis='x', labelsize=8)
                for label in ax.get_yticklabels():
                    label.set_fontname('Times New Roman')
                for label in ax.get_xticklabels():
                    label.set_fontname('Times New Roman')
                
                plt.subplots_adjust(top=0.92)
                plt.title("SHAP Interaction Summary Plot - All Features (Complete Version)", 
                         fontsize=16, fontname='Times New Roman', pad=20)
                
                plt.tight_layout()
                plt.savefig(os.path.join(out_dir, 'shap_interaction_summary_all_features_complete.png'), bbox_inches='tight', dpi=250)
                if SHOW_PLOTS:
                    plt.show()
                plt.close()
                print("Complete interaction summary plot (all features) generated successfully!")
                
                print("Generating merged type (TYPE) interaction summary plot...")
                
                if len(type_cols_for_interaction) > 0:

                    non_type_cols_for_summary = [c for c in X_test_interaction.columns if not str(c).startswith('type_')]
                    
                    X_merged_for_summary = X_test_interaction[non_type_cols_for_summary].copy()
                    
                    if 'type_label' in X_test.columns:
                        X_merged_for_summary['TYPE'] = X_test['type_label'].copy()
                    else:

                        type_priority = ['Ag', 'Ag2S', 'Al2O3', 'Cu', 'CuO', 'Fe2O3', 'FeO', 'TiO2', 'ZnO']
                        type_labels = []
                        for idx in X_test_interaction.index:
                            found_type = None
                            for type_name in type_priority:
                                type_col = f'type_{type_name}'
                                if type_col in X_test_interaction.columns and X_test_interaction.loc[idx, type_col] == 1:
                                    found_type = type_name
                                    break
                            type_labels.append(found_type if found_type else 'Unknown')
                        X_merged_for_summary['TYPE'] = type_labels
                    
                    n_samples = shap_interaction_values.shape[0]
                    n_merged_features = len(non_type_cols_for_summary) + 1
                    
                    merged_interaction_values = np.zeros((n_samples, n_merged_features, n_merged_features))
                    
                    for i, feat1 in enumerate(non_type_cols_for_summary):
                        feat1_idx = X_test_interaction.columns.get_loc(feat1)
                        for j, feat2 in enumerate(non_type_cols_for_summary):
                            feat2_idx = X_test_interaction.columns.get_loc(feat2)
                            merged_interaction_values[:, i, j] = shap_interaction_values[:, feat1_idx, feat2_idx]
                    
                    type_idx = len(non_type_cols_for_summary)
                    for i, feat in enumerate(non_type_cols_for_summary):
                        feat_idx = X_test_interaction.columns.get_loc(feat)

                        type_interaction_sum = np.zeros(n_samples)
                        for type_col in type_cols_for_interaction:
                            type_col_idx = X_test_interaction.columns.get_loc(type_col)
                            type_interaction_sum += shap_interaction_values[:, feat_idx, type_col_idx]
                        merged_interaction_values[:, i, type_idx] = type_interaction_sum
                        merged_interaction_values[:, type_idx, i] = type_interaction_sum
                    
                    type_self_interaction = np.zeros(n_samples)
                    for type_col1 in type_cols_for_interaction:
                        type_col1_idx = X_test_interaction.columns.get_loc(type_col1)
                        for type_col2 in type_cols_for_interaction:
                            type_col2_idx = X_test_interaction.columns.get_loc(type_col2)
                            type_self_interaction += shap_interaction_values[:, type_col1_idx, type_col2_idx]
                    merged_interaction_values[:, type_idx, type_idx] = type_self_interaction
                    
                    merged_features_for_heatmap = non_type_cols_for_summary + ['TYPE']
                    print(f"Merged feature list: {merged_features_for_heatmap}")
                    
                    temp_sv = merged_sv
                    temp_features = merged_features[:]
                    
                    temp_importance_list = []
                    temp_feature_list = []
                    for i, feat in enumerate(temp_features):
                        temp_feature_list.append(strip_units(feat))
                        temp_importance_list.append(np.abs(temp_sv[:, i]).mean())
                    
                    temp_feature_importance = pd.DataFrame({
                        'feature': temp_feature_list,
                        'importance': temp_importance_list
                    }).sort_values('importance', ascending=False)
                    
                    importance_dict = dict(zip(temp_feature_importance['feature'], temp_feature_importance['importance']))
                    print(f"Merged feature importance dictionary: {importance_dict}")
                    
                    merged_feature_importances = {}
                    for feat in merged_features_for_heatmap:
                        cleaned_feat = strip_units(feat)
                        if cleaned_feat in importance_dict:
                            merged_feature_importances[feat] = importance_dict[cleaned_feat]
                        else:
                            merged_feature_importances[feat] = 0
                    
                    print(f"Merged feature importances: {merged_feature_importances}")
                    
                    sorted_merged_features = sorted(merged_features_for_heatmap, 
                                                  key=lambda x: merged_feature_importances[x], reverse=True)
                    sorted_merged_indices = [merged_features_for_heatmap.index(f) for f in sorted_merged_features]
                    
                    print(f"Merged features sorted by importance: {sorted_merged_features}")
                    
                    sorted_merged_feature_names_clean = [strip_units(f) for f in sorted_merged_features]
                    print(f"Cleaned names of merged features: {sorted_merged_feature_names_clean}")
                    
                    sorted_merged_interaction_values = merged_interaction_values[:, sorted_merged_indices, :][:, :, sorted_merged_indices]
                    sorted_X_merged_for_summary = X_merged_for_summary[sorted_merged_features].copy()
                    
                    sorted_X_merged_for_summary.columns = sorted_merged_feature_names_clean
                    
                    print(f"Column names of reordered merged data matrix: {list(sorted_X_merged_for_summary.columns)}")
                    print(f"Shape of merged interaction values matrix: {sorted_merged_interaction_values.shape}")
                    
                    plt.figure(figsize=(16, max(10, len(sorted_merged_features) * 0.6)))
                    shap.summary_plot(sorted_merged_interaction_values, sorted_X_merged_for_summary, 
                                    show=SHOW_PLOTS, max_display=len(sorted_merged_features), sort=False)
                    
                    ax = plt.gca()
                    ax.tick_params(axis='y', labelsize=28)
                    ax.tick_params(axis='x', labelsize=8)
                    for label in ax.get_yticklabels():
                        label.set_fontname('Times New Roman')
                    for label in ax.get_xticklabels():
                        label.set_fontname('Times New Roman')
                    
                    plt.subplots_adjust(top=0.92)
                    plt.title("SHAP Interaction Summary Plot - TYPE Merged Features (Sorted by Importance)", 
                             fontsize=16, fontname='Times New Roman', pad=20)
                    
                    plt.tight_layout()
                    plt.savefig(os.path.join(out_dir, 'shap_interaction_summary_type_merged_sorted.png'), bbox_inches='tight', dpi=250)
                    if SHOW_PLOTS:
                        plt.show()
                    plt.close()
                    print("Merged TYPE features interaction summary plot (sorted by importance) generated successfully!")
                else:
                    print("No type features found, skipping merged interaction summary plot")
                
            else:
                print("No type features found, skipping TYPE merged heatmap")
                
        except Exception as _e5:
            print(f'Error generating SHAP interaction heatmap: {_e5}')
            import traceback
            traceback.print_exc()
            
    else:
        print("SHAP interaction values could not be computed. Skipping interaction summary plots.")
        
except Exception as _e:
    print('SHAP interaction analysis failed:', _e)

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12
plt.rcParams['figure.figsize'] = (12, 8)

AXIS_FONT_SIZE = 12
AXIS_FONT = {'fontsize': AXIS_FONT_SIZE, 'fontname': 'Times New Roman'}
TICK_KW = {'fontsize': AXIS_FONT_SIZE}

print("\n" + "="*50)
print("SHAP Beeswarm Plot - All Features (No Merging)")
print("="*50)

try:

    all_features = list(X_test.columns)
    print(f"Generating beeswarm plot for all {len(all_features)} features...")
    
    display_all_features = X_test.copy()
    display_all_features.columns = [strip_units(c) for c in display_all_features.columns]
    
    sv_all = sv_raw if 'sv_raw' in globals() else (shap_values_raw.values if hasattr(shap_values_raw, 'values') else np.array(shap_values_raw))
    
    plt.figure(figsize=(14, max(10, len(all_features) * 0.4)))
    shap.summary_plot(sv_all, display_all_features, show=SHOW_PLOTS, max_display=None)
    plt.title("SHAP Beeswarm Plot - All Features (No Merging)", fontsize=16, fontname='Times New Roman')
    
    ax = plt.gca()
    ax.tick_params(axis='y', labelsize=24)
    for label in ax.get_yticklabels():
        label.set_fontname('Times New Roman')
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'shap_beeswarm_all_features_no_merging.png'), bbox_inches='tight', dpi=250)
    if SHOW_PLOTS:
        plt.show()
    plt.close()
    print("All features beeswarm plot (no merging) generated successfully!")
    
    plt.figure(figsize=(14, max(10, len(all_features) * 0.4)))
    shap.summary_plot(sv_all, X_test, show=SHOW_PLOTS, max_display=None)
    plt.title("SHAP Beeswarm Plot - All Original Features", fontsize=16, fontname='Times New Roman')
    
    ax = plt.gca()
    ax.tick_params(axis='y', labelsize=24)
    for label in ax.get_yticklabels():
        label.set_fontname('Times New Roman')
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'shap_beeswarm_all_original_features.png'), bbox_inches='tight', dpi=250)
    if SHOW_PLOTS:
        plt.show()
    plt.close()
    print("All features beeswarm plot (with all original feature names) generated successfully!")
    
except Exception as e:
    print(f'Error generating beeswarm plot for all features: {e}')

print("\n" + "="*50)
print("SHAP Beeswarm Plot - All Features with Type Features Combined in One Row")
print("="*50)

try:
    type_cols = [c for c in X_test.columns if str(c).startswith('type_')]
    non_type_cols = [c for c in X_test.columns if not str(c).startswith('type_') and c != 'TYPE']
    
    print(f"Type features: {type_cols}")
    print(f"Non-Type features: {non_type_cols}")
    
    if len(type_cols) > 0 and len(non_type_cols) > 0:

        non_type_indices = [X_test.columns.get_loc(c) for c in non_type_cols]
        sv_non_type = sv_all[:, non_type_indices]
        display_non_type = X_test[non_type_cols].copy()
        display_non_type.columns = [strip_units(c) for c in display_non_type.columns]
        
        plt.figure(figsize=(14, max(8, len(non_type_cols) * 0.5 + 2)))
        shap.summary_plot(sv_non_type, display_non_type, show=False, max_display=None)
        
        ax = plt.gca()
        
        ax.tick_params(axis='y', labelsize=24)
        for label in ax.get_yticklabels():
            label.set_fontname('Times New Roman')
        
        type_indices = [X_test.columns.get_loc(c) for c in type_cols]
        
        if 'type_label' in X.columns:
            type_labels = X.loc[X_test.index, 'type_label'].astype(str).values
        else:

            type_labels = []
            for idx in X_test.index:
                sample_types = []
                for col in type_cols:
                    if X_test.loc[idx, col] == 1:
                        sample_types.append(col.replace('type_', ''))
                if sample_types:
                    type_labels.append(sample_types[0])
                else:
                    type_labels.append('none')
            type_labels = np.array(type_labels)
        
        unique_types = ['Ag', 'CuO', 'Fe2O3', 'ZnO', 'TiO2', 'Ag2S', 'Al2O3', 'Cu', 'FeO']

        color_palette = ["#0683f9", "#f9ce60", "#ff0d0d", "#00FF00", "#c28ae9", "#6a6b62", "#00ffff", "#FF00FF", "#ff8c00"]
        color_map = {typ: color_palette[i % len(color_palette)] for i, typ in enumerate(unique_types)}
        
        yticks = ax.get_yticks()
        y_bottom = yticks.min() - 1.0
        
        legend_elements = []
        
        for i, type_col in enumerate(type_cols):
            type_name = type_col.replace('type_', '')
            type_idx = X_test.columns.get_loc(type_col)
            type_shap_values = sv_all[:, type_idx]
            
            mask = (X_test[type_col] == 1).values
            if np.any(mask):
                x_values = type_shap_values[mask]

                y_jitter = y_bottom + np.random.uniform(-0.15, 0.15, size=len(x_values))
                
                color = color_map.get(type_name, '#333333')

                scatter = ax.scatter(x_values, y_jitter, c=color, marker='o', 
                          edgecolors='none', linewidths=0, s=30, alpha=1.0)
                
                from matplotlib.patches import Patch
                legend_elements.append(Patch(facecolor=color, edgecolor='none', label=type_name))
        
        y_min, y_max = ax.get_ylim()
        ax.set_ylim(y_bottom - 0.5, y_max)
        
        if legend_elements:
            legend = ax.legend(handles=legend_elements, loc='upper center', 
                             bbox_to_anchor=(0.5, -0.12), ncol=len(legend_elements), 
                             fontsize=10, title='Type Categories', title_fontsize=11,
                             frameon=False, handlelength=1, handletextpad=0.5, columnspacing=1)
        
        plt.title("SHAP Beeswarm Plot - All Features (Type Features in One Row)", 
                 fontsize=16, fontname='Times New Roman')
        plt.xlabel('SHAP value (impact on model output)', fontsize=14, fontname='Times New Roman')
        plt.ylabel('Features', fontsize=14, fontname='Times New Roman')
        
        plt.subplots_adjust(bottom=0.2)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'shap_beeswarm_all_features_type_combined.png'), 
                   bbox_inches='tight', dpi=250)
        if SHOW_PLOTS:
            plt.show()
        plt.close()
        print("All features beeswarm plot (type features combined in one row) generated successfully!")
        
    else:
        print("Not enough type or non-type features found, skipping custom beeswarm plot")
        
except Exception as e:
    print(f'Error generating custom beeswarm plot: {e}')
    import traceback
    traceback.print_exc()

print("\n" + "="*50)
print("SHAP Beeswarm Plot - feature distribution and impact (merged version)")
print("="*50)

plt.figure()
try:
    type_cols = [c for c in X_test.columns if str(c).startswith('type_')]

    sv = merged_sv

    display_no_type = display_merged_clean.drop(columns=['TYPE'], errors='ignore')
    sv_no_type = sv[:, [i for i, f in enumerate(merged_features) if f != 'TYPE']]
    shap.summary_plot(sv_no_type, display_no_type, feature_names=display_no_type.columns, max_display=15, show=False)

    ax = plt.gca()
    ax.tick_params(axis='y', labelsize=24)
    for label in ax.get_yticklabels():
        label.set_fontname('Times New Roman')

    if len(type_cols) > 0:

        type_indices_raw = [X_test.columns.get_loc(c) for c in type_cols if c in X_test.columns]
        type_sum = sv_raw[:, type_indices_raw].sum(axis=1)

        if 'type_label' in X.columns:
            type_labels = X.loc[X_test.index, 'type_label'].astype(str).values
        else:
            type_labels = ['unknown'] * len(type_sum)

        ordered = ['Ag', 'CuO', 'Fe2O3', 'ZnO', 'TiO2']
        present_labels = [lab for lab in ordered if lab in set(type_labels)]
        color_palette = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B3']
        color_map = {lab: color_palette[i] for i, lab in enumerate(ordered)}

        yticks = ax.get_yticks()
        y_bottom = yticks.min() - 0.8

        n = len(type_sum)
        y_jitter = y_bottom + np.random.uniform(-0.12, 0.12, size=n)
        colors = [color_map.get(l, '#333333') for l in type_labels]
        ax.scatter(type_sum, y_jitter, c=colors, marker='o', edgecolors='none', linewidths=0, s=16, alpha=0.9)

        from matplotlib.patches import Patch
        legend_patches = [Patch(color=color_map[lab], label=lab) for lab in ordered if lab in present_labels]
        if legend_patches:
            ax.legend(handles=legend_patches, loc='upper right', fontsize=10)

        plt.title("SHAP Beeswarm Plot (type merged at bottom)", fontsize=16, fontname='Times New Roman')
    else:

        shap.summary_plot(merged_sv, display_merged_clean, feature_names=display_merged_clean.columns, max_display=15, show=False)

        ax = plt.gca()
        ax.tick_params(axis='y', labelsize=24)
        for label in ax.get_yticklabels():
            label.set_fontname('Times New Roman')

    plt.gcf().tight_layout()
    plt.savefig(os.path.join(out_dir, 'shap_beeswarm_type_bottom.png'), bbox_inches='tight', dpi=250)
    if SHOW_PLOTS:
        plt.show()
    plt.close()
except Exception as e:
    print('Error while preparing merged beeswarm overlay:', e)
    plt.figure()

    shap.summary_plot(merged_sv, display_merged_clean, feature_names=display_merged_clean.columns, max_display=15, show=False)

    ax = plt.gca()
    ax.tick_params(axis='y', labelsize=24)
    for label in ax.get_yticklabels():
        label.set_fontname('Times New Roman')
    plt.gcf().tight_layout()
    plt.savefig(os.path.join(out_dir, 'shap_beeswarm_fallback.png'), bbox_inches='tight', dpi=250)
    if SHOW_PLOTS:
        plt.show()
    plt.close()

print("\n" + "="*50)
print("Additional Beeswarm Plots - Non-type only & Type-only")
print("="*50)

try:
    type_cols = [c for c in X_test.columns if str(c).startswith('type_')]
    
    print("Generating beeswarm plot for non-type features...")
    non_type_cols = [c for c in X_test.columns if not str(c).startswith('type_') and c != 'TYPE']
    if len(non_type_cols) > 0:

        non_type_indices = [X_test.columns.get_loc(c) for c in non_type_cols]

        sv_non_type = sv_raw[:, non_type_indices]

        display_non_type = X_test[non_type_cols].copy()
        display_non_type.columns = [strip_units(c) for c in display_non_type.columns]
        
        plt.figure(figsize=(12, 8))
        shap.summary_plot(sv_non_type, display_non_type, show=False, max_display=20)
        plt.title('SHAP Beeswarm Plot - Non-Type Features Only', fontsize=16, fontname='Times New Roman')
        
        ax = plt.gca()
        ax.tick_params(axis='y', labelsize=24)
        for label in ax.get_yticklabels():
            label.set_fontname('Times New Roman')
        
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'shap_beeswarm_non_type_only.png'), bbox_inches='tight', dpi=250)
        if SHOW_PLOTS:
            plt.show()
        plt.close()
        print("Non-type features beeswarm plot generated successfully!")
    else:
        print("No non-type features found, skipping non-type features beeswarm plot")
    
    print("Generating beeswarm plot for type features only...")
    if len(type_cols) > 0:

        type_indices = [X_test.columns.get_loc(c) for c in type_cols]

        sv_type_only = sv_raw[:, type_indices]

        display_type_only = X_test[type_cols].copy()
        display_type_only.columns = [c.replace('type_', '') for c in type_cols]
        
        plt.figure(figsize=(12, 8))
        shap.summary_plot(sv_type_only, display_type_only, show=False, max_display=len(type_cols))
        plt.title('SHAP Beeswarm Plot - Type Features Only', fontsize=16, fontname='Times New Roman')
        
        ax = plt.gca()
        ax.tick_params(axis='y', labelsize=24)
        for label in ax.get_yticklabels():
            label.set_fontname('Times New Roman')
        
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'shap_beeswarm_type_only.png'), bbox_inches='tight', dpi=250)
        if SHOW_PLOTS:
            plt.show()
        plt.close()
        print("Type features only beeswarm plot generated successfully!")
    else:
        print("No type features found, skipping type features only beeswarm plot")
        
except Exception as e:
    print('Error generating additional beeswarm plots:', e)

print("\n" + "="*50)
print("Feature importance ranking")
print("="*50)

sv = merged_sv
orig_features = merged_features[:]

importance_list = []
feature_list = []
for i, feat in enumerate(orig_features):

    feature_list.append(strip_units(feat))
    importance_list.append(np.abs(sv[:, i]).mean())

feature_importance = pd.DataFrame({
    'feature': feature_list,
    'importance': importance_list
}).sort_values('importance', ascending=False)

plt.figure(figsize=(10, max(6, 0.35 * min(20, feature_importance.shape[0]))), dpi=250)
import matplotlib.patches as patches
ax = plt.gca()
fig = plt.gcf()

fig.patch.set_facecolor('white')
ax.set_facecolor('#f5f5f5')
ax.grid(axis='x', linestyle='--', color='#e0e0e0', linewidth=1)

fi = pd.DataFrame({
    'feature': feature_importance['feature'].values,
    'importance': feature_importance['importance'].values
})

top_n = min(20, fi.shape[0])
fi_top = fi.head(top_n).iloc[::-1]

import matplotlib.patches as patches
max_val = fi['importance'].max() if fi.shape[0] > 0 else 1.0
cmap = plt.cm.Blues
gradient = np.linspace(0, 1, 256).reshape(1, -1)
bar_height = 0.7

for i, (feat, val) in enumerate(zip(fi_top['feature'], fi_top['importance'])):
    y0 = i - bar_height / 2
    y1 = i + bar_height / 2

    darker_gradient = np.linspace(0.35, 1, 256).reshape(1, -1)
    im = ax.imshow(darker_gradient, extent=(0, max_val, y0, y1), aspect='auto', cmap=cmap, vmin=0, vmax=1, zorder=2)
    rect = patches.Rectangle((0, y0), val, bar_height, transform=ax.transData)
    im.set_clip_path(rect)

ax.margins(y=0.12)

ax.set_yticks(np.arange(len(fi_top)))
ax.set_yticklabels(fi_top['feature'], fontsize=21, fontname='Times New Roman')
ax.set_xlabel('Mean |SHAP value|', fontsize=14, fontname='Times New Roman')
ax.set_ylabel('Feature', fontsize=14, fontname='Times New Roman')
ax.set_title(f'Feature importance ranking - SHAP - Top {top_n}', fontsize=16, pad=20, fontname='Times New Roman')
ax.grid(axis='x', linestyle='--', alpha=0.6)
ax.set_xlim(0, max_val * 1.05)
for spine in ax.spines.values():
    spine.set_visible(False)

for i, val in enumerate(fi_top['importance']):
    ax.text(val + (max_val * 0.01), i, f'{val:.3f}', va='center', fontsize=12, 
            fontname='Times New Roman', fontfamily='serif', zorder=3)

plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'feature_importance.png'), bbox_inches='tight', dpi=250)
if SHOW_PLOTS:
    plt.show()
plt.close()

print("\nFeature importance ranking:")
print(feature_importance.to_string(index=False))

try:
    type_cols = [c for c in X_test.columns if str(c).startswith('type_')]
    if len(type_cols) > 0:

        raw_features = list(X_test.columns)
        type_idxs = [raw_features.index(c) for c in type_cols]

        sv_matrix = sv_raw if 'sv_raw' in globals() else (shap_values_raw.values if hasattr(shap_values_raw, 'values') else np.array(shap_values_raw))

        type_importances = np.abs(sv_matrix[:, type_idxs]).mean(axis=0)
        df_type_imp = pd.DataFrame({
            'type': [c.replace('type_', '') for c in type_cols],
            'importance': type_importances
        }).sort_values('importance', ascending=False)

        plt.figure(figsize=(8, max(2, 0.5 * df_type_imp.shape[0])), dpi=250)
        ax = plt.gca()
        
        y_positions = range(len(df_type_imp))
        values = df_type_imp['importance'].values
        labels = df_type_imp['type'].values
        max_val = create_gradient_barh(ax, y_positions, values, labels, cmap_name='Blues')
        
        ax.set_yticks(np.arange(len(df_type_imp)))
        ax.set_yticklabels(labels, fontsize=21, fontname='Times New Roman')
        ax.set_xlabel('Mean |SHAP value|', fontsize=14, fontname='Times New Roman')
        ax.set_ylabel('Type Features', fontsize=14, fontname='Times New Roman')
        ax.set_title('Per-type feature importance (decomposed type_*)', fontsize=16, pad=20, fontname='Times New Roman')
        ax.invert_yaxis()
        
        for i, v in enumerate(values):
            ax.text(v + (max_val * 0.01), i, f'{v:.3f}', va='center', fontsize=14, 
                    fontname='Times New Roman', fontfamily='serif', zorder=3)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'type_features_importance.png'), bbox_inches='tight', dpi=250)
        if SHOW_PLOTS:
            plt.show()
        plt.close()

        try:
            plt.figure()
            shap.summary_plot(sv_matrix[:, type_idxs], X_test[type_cols], feature_names=[c.replace('type_', '') for c in type_cols], show=False)
            plt.title('SHAP Beeswarm Plot for type_* features (decomposed)', fontsize=16, fontname='Times New Roman')
            
            ax = plt.gca()
            ax.tick_params(axis='y', labelsize=24)
            for label in ax.get_yticklabels():
                label.set_fontname('Times New Roman')
            
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, 'shap_type_beeswarm_decomposed.png'), bbox_inches='tight', dpi=250)
            if SHOW_PLOTS:
                plt.show()
            plt.close()
        except Exception as e:
            print('Warning: failed to render type_* beeswarm plot:', e)
    else:
        print('No type_* columns detected; skipping per-type decomposed plots.')
except Exception as e:
    print('Error while creating per-type decomposed SHAP plots:', e)

print("\n" + "="*50)
print("Feature interaction analysis")
print("="*50)

if feature_importance.shape[0] < 2:
    print('Not enough features in feature_importance to run interaction analysis.')
else:
    top_display = feature_importance.iloc[0]['feature']
    second_display = feature_importance.iloc[1]['feature']

    display_to_merged = {strip_units(f): f for f in merged_features}

    def extract_unit(orig_name):
        """Return unit string from original column name if in parentheses, else empty string."""
        s = str(orig_name)
        m = re.search(r"\(([^)]*)\)", s)
        return f"({m.group(1)})" if m else ''

    top_feat = display_to_merged.get(top_display, top_display)
    second_feat = display_to_merged.get(second_display, second_display)

    try:
        plt.figure()
        shap.dependence_plot(
            top_display, merged_sv, display_merged_clean, 
            interaction_index=second_display, show=False
        )
        plt.title(f"Feature interaction: {top_display} vs {second_display}", fontsize=16, fontname='Times New Roman')
        plt.xlabel(top_display + ' ' + extract_unit(top_feat), fontsize=14, fontname='Times New Roman')
        plt.ylabel('SHAP value', fontsize=14, fontname='Times New Roman')
        plt.xticks(fontsize=14, fontname='Times New Roman')
        plt.yticks(fontsize=14, fontname='Times New Roman')
        plt.gcf().subplots_adjust(left=0.15, right=0.95, top=0.93, bottom=0.15)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'shap_dependence_top_vs_second.png'), bbox_inches='tight', dpi=250)
        if SHOW_PLOTS:
            plt.show()
        plt.close()
    except Exception as e:
        print('Warning: method 1 dependence plot failed:', e)

    try:
        if 'C' in display_merged_clean.columns:
            print("Generating SHAP dependence plot for C feature (0-200 range)...")
            plt.figure(figsize=(10, 6))
            
            shap.dependence_plot(
                'C', merged_sv, display_merged_clean, 
                interaction_index='pH',
                show=False
            )
            
            plt.xlim(0, 200)
            plt.title("SHAP Dependence Plot: C (0-200 range) colored by pH", fontsize=16, fontname='Times New Roman')
            plt.xlabel('C (Concentration)', fontsize=14, fontname='Times New Roman')
            plt.ylabel('SHAP value for C', fontsize=14, fontname='Times New Roman')
            plt.xticks(fontsize=14, fontname='Times New Roman')
            plt.yticks(fontsize=14, fontname='Times New Roman')
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, 'shap_dependence_C_0_200_range.png'), bbox_inches='tight', dpi=250)
            if SHOW_PLOTS:
                plt.show()
            plt.close()
            print("SHAP dependence plot for C feature (0-200 range) generated successfully!")
        else:
            print("C feature not found in data")
    except Exception as e:
        print(f'Error generating SHAP dependence plot for C feature: {e}')
        import traceback
        traceback.print_exc()

    shap_interaction = None
    try:
        print('Computing SHAP interaction values (TreeExplainer)...')
        shap_interaction = shap.TreeExplainer(model).shap_interaction_values(X_test)
    except Exception as e:
        print('TreeExplainer.shap_interaction_values failed, attempting Explainer(...).interaction_values if available:', e)
        try:
            shap_interaction = explainer.shap_interaction_values(X_test)
        except Exception as e2:
            print('Explainer.shap_interaction_values also failed:', e2)

    if 'shap_interaction_values' in locals() and shap_interaction_values is not None:
        try:
            plt.figure(figsize=(12, 8))
            shap.summary_plot(shap_interaction_values, display_X_test, show=False)
            plt.title("SHAP Interaction Summary Plot", fontsize=16, fontname='Times New Roman')
            
            plt.xticks(fontsize=8, fontname='Times New Roman')
            plt.yticks(fontsize=10, fontname='Times New Roman')
            
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, 'shap_interaction_summary_alt.png'), bbox_inches='tight', dpi=250)
            if SHOW_PLOTS:
                plt.show()
            plt.close()
        except Exception:
            pass
    if shap_interaction is not None:
        print('SHAP interaction values computed successfully for enhanced analysis')
    else:
        print('No SHAP interaction values available for enhanced interaction analysis')

print("\n" + "="*50)
print("Supervised dimensionality reduction of SHAP values")
print("="*50)
n_samples = min(1000, merged_sv.shape[0])
sample_indices = np.arange(n_samples)
sample_y = y_test.iloc[sample_indices].values
norm = mcolors.Normalize(vmin=min(sample_y), vmax=max(sample_y))
cmap = plt.get_cmap('viridis')

print(f"Running PCA on first {n_samples} samples...")
shap_pca = PCA(n_components=12).fit_transform(merged_sv[:n_samples, :])

plt.figure(figsize=(12, 10))
plt.scatter(shap_pca[:, 0], shap_pca[:, 1], c=sample_y, cmap=cmap, alpha=0.7, edgecolors='w', s=50)
cb = plt.colorbar(label='Target value')
cb.ax.tick_params(labelsize=AXIS_FONT_SIZE)
plt.title('PCA of SHAP values (first 2 components)', fontsize=16, fontname='Times New Roman')
plt.xlabel('Principal Component 1', **AXIS_FONT)
plt.ylabel('Principal Component 2', **AXIS_FONT)
plt.xticks(**TICK_KW)
plt.yticks(**TICK_KW)
plt.grid(alpha=0.2)
plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'pca_shap_scatter.png'), bbox_inches='tight', dpi=250)
if SHOW_PLOTS:
    plt.show()
plt.close()

print(f"Running t-SNE on first {n_samples} samples...")
tsne = TSNE(n_components=2, perplexity=50, random_state=42)
shap_embedded = tsne.fit_transform(merged_sv[:n_samples, :])

plt.figure(figsize=(12, 10))
scatter = plt.scatter(shap_embedded[:, 0], shap_embedded[:, 1], c=sample_y, cmap=cmap, alpha=0.7, edgecolors='w', s=50)
cb = plt.colorbar(scatter, label='Target value')
cb.ax.tick_params(labelsize=AXIS_FONT_SIZE)
plt.title('t-SNE of SHAP values (perplexity=50)', fontsize=16, fontname='Times New Roman')
plt.xlabel('t-SNE dimension 1', **AXIS_FONT)
plt.ylabel('t-SNE dimension 2', **AXIS_FONT)
plt.xticks(**TICK_KW)
plt.yticks(**TICK_KW)
plt.grid(alpha=0.2)
plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'tsne_shap_scatter.png'), bbox_inches='tight', dpi=250)
if SHOW_PLOTS:
    plt.show()
plt.close()

print('Creating 3D visualization...')
fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection='3d')
shap_pca3d = PCA(n_components=3).fit_transform(merged_sv[:n_samples, :])
scatter = ax.scatter(shap_pca3d[:, 0], shap_pca3d[:, 1], shap_pca3d[:, 2], c=sample_y, cmap=cmap, s=40, alpha=0.8)
ax.set_title('3D PCA of SHAP values', fontsize=16, fontname='Times New Roman')
ax.set_xlabel('Principal Component 1', **AXIS_FONT)
ax.set_ylabel('Principal Component 2', **AXIS_FONT)
ax.set_zlabel('Principal Component 3', **AXIS_FONT)
cb = fig.colorbar(scatter, ax=ax, pad=0.1)
cb.ax.tick_params(labelsize=AXIS_FONT_SIZE)
cb.set_label('Target value', fontsize=AXIS_FONT_SIZE, fontname='Times New Roman')
plt.tight_layout()
fig.savefig(os.path.join(out_dir, 'pca_shap_3d.png'), bbox_inches='tight', dpi=250)
if SHOW_PLOTS:
    plt.show()
plt.close()

print('Analyzing embeddings by top features...')

top_features = feature_importance.head(3)['feature'].tolist()
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for i, feature in enumerate(top_features):
    ax = axes[i]

    feature_values = display_merged_clean[feature].iloc[:n_samples].values
    quantiles = np.percentile(feature_values, [0, 25, 50, 75, 100])
    colors = ['blue', 'green', 'orange', 'red']
    labels = ['Q1 (low)', 'Q2', 'Q3', 'Q4 (high)']
    color_labels = []
    for j in range(len(feature_values)):
        if feature_values[j] <= quantiles[1]:
            color_labels.append(0)
        elif feature_values[j] <= quantiles[2]:
            color_labels.append(1)
        elif feature_values[j] <= quantiles[3]:
            color_labels.append(2)
        else:
            color_labels.append(3)
    
    for color_idx, (color, label) in enumerate(zip(colors, labels)):
        idx = [j for j, cl in enumerate(color_labels) if cl == color_idx]
        if len(idx) > 0:
            ax.scatter(shap_embedded[idx, 0], shap_embedded[idx, 1], color=color, label=label, alpha=0.7)
    ax.set_title(f't-SNE grouped by feature: {feature}', fontsize=14, fontname='Times New Roman')
    ax.set_xlabel('t-SNE dimension 1', **AXIS_FONT)
    ax.set_ylabel('t-SNE dimension 2', **AXIS_FONT)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.2)

plt.suptitle('t-SNE embeddings vs top features', fontsize=16, fontname='Times New Roman')
plt.tight_layout()
plt.subplots_adjust(top=0.85)
plt.savefig(os.path.join(out_dir, 'tsne_by_top_features.png'), bbox_inches='tight', dpi=250)
if SHOW_PLOTS:
    plt.show()
plt.close()

print("\n" + "="*50)
print("SHAP Interaction-based Pairwise Feature Analysis and PDP")
print("="*50)

try:
    from sklearn.inspection import PartialDependenceDisplay
    import itertools
    
    print("Calculating pairwise feature interaction importance based on SHAP interaction values...")
    
    if 'shap_interaction_values' in locals() and shap_interaction_values is not None:
        print("Using precomputed SHAP interaction values for analysis...")
        interaction_matrix = shap_interaction_values
        features_for_interaction = list(X_test_interaction.columns)
        
        n_features = len(features_for_interaction)
        feature_pairs = []
        interaction_importances = []
        
        print(f"Analyzing pairwise interactions among {n_features} features...")
        
        for i in range(n_features):
            for j in range(i + 1, n_features):

                interaction_values = interaction_matrix[:, i, j]
                
                interaction_importance = np.mean(np.abs(interaction_values))
                
                if not np.isnan(interaction_importance):
                    feature_pairs.append((features_for_interaction[i], features_for_interaction[j]))
                    interaction_importances.append(interaction_importance)
        
        paired_data = list(zip(feature_pairs, interaction_importances))
        paired_data.sort(key=lambda x: x[1], reverse=True)
        
        print(f"Successfully calculated SHAP interaction values for {len(paired_data)} feature pairs")
        
    else:
        print("SHAP interaction values not available, falling back to feature value product method...")

        numeric_features = []
        for col in X_test.columns:
            if not str(col).startswith('type_') and col != 'TYPE':
                numeric_features.append(col)
        
        print(f"Numeric features used for product analysis: {len(numeric_features)}")
        X_numeric = X_test[numeric_features].values
        
        feature_pairs = []
        product_correlations = []
        
        for i, j in itertools.combinations(range(len(numeric_features)), 2):

            product = X_numeric[:, i] * X_numeric[:, j]
            
            correlation = np.abs(np.corrcoef(product, y_test)[0, 1])
            
            if not np.isnan(correlation):
                feature_pairs.append((numeric_features[i], numeric_features[j]))
                product_correlations.append(correlation)
        
        paired_data = list(zip(feature_pairs, product_correlations))
        paired_data.sort(key=lambda x: x[1], reverse=True)
        interaction_importances = product_correlations
    
    if 'shap_interaction_values' in locals() and shap_interaction_values is not None:

        interaction_ranking = pd.DataFrame([
            {
                'feature_pair': f"{strip_units(pair[0])} × {strip_units(pair[1])}",
                'original_pair': f"{pair[0]} × {pair[1]}",
                'interaction_importance': importance,
                'method': 'SHAP_Interaction'
            }
            for (pair, importance) in paired_data
        ])
        
        print("\nPairwise Feature Interaction Importance Ranking Based on SHAP Interaction Values (Top 10):")
        print("Note: The importance here is the mean absolute SHAP interaction value, representing the true feature interaction effect")
        print(interaction_ranking.head(10)[['feature_pair', 'interaction_importance']].to_string(index=False))
        print("\nMathematical Explanation:")
        print("Φᵢ,ⱼ = ½ * (SHAP₍ᵢ,ⱼ₎ - SHAP₍ᵢ\\ⱼ₎ - SHAP₍ⱼ\\ᵢ₎ + SHAP₍\\ᵢ,ⱼ₎)")
        print(" mean Φᵢ,ⱼ ")
        print("This value measures: the joint effect when two features are present together - the sum of their individual independent effects")
        
        value_col = 'interaction_importance'
        
    else:

        interaction_ranking = pd.DataFrame([
            {
                'feature_pair': f"{strip_units(pair[0])} × {strip_units(pair[1])}",
                'original_pair': f"{pair[0]} × {pair[1]}",
                'correlation': corr,
                'method': 'Product_Correlation'
            }
            for (pair, corr) in paired_data
        ])
        
        print("\nPairwise Feature Product Importance Ranking (Top 10):")
        print("Note: Since SHAP interaction values are not available, feature product correlation with the target variable is used as a substitute")
        print(interaction_ranking.head(10)[['feature_pair', 'correlation']].to_string(index=False))
        
        value_col = 'correlation'
    
    if 'shap_interaction_values' in locals() and shap_interaction_values is not None:
        interaction_ranking.to_csv(os.path.join(out_dir, 'shap_interaction_based_ranking.csv'), index=False)
        print(f"\nComplete SHAP interaction value ranking results saved to: shap_interaction_based_ranking.csv")
    else:
        interaction_ranking.to_csv(os.path.join(out_dir, 'feature_pairwise_products_ranking.csv'), index=False)
        print(f"\nComplete feature product ranking results saved to: feature_pairwise_products_ranking.csv")
    
    print("\nGenerating feature interaction importance ranking plot...")
    
    top_interactions = interaction_ranking.head(15)
    
    plt.figure(figsize=(12, max(6, 0.4 * len(top_interactions))))
    ax = plt.gca()
    fig = plt.gcf()
    
    fig.patch.set_facecolor('white')
    ax.set_facecolor('#f5f5f5')
    ax.grid(axis='x', linestyle='--', color='#e0e0e0', linewidth=1)
    
    y_positions = range(len(top_interactions))
    values = top_interactions[value_col].values
    labels = top_interactions['feature_pair'].values
    max_val = values.max()
    
    if 'shap_interaction_values' in locals() and shap_interaction_values is not None:
        cmap = plt.cm.Purples
        title_suffix = "Based on SHAP Interaction Values"
    else:
        cmap = plt.cm.Oranges
        title_suffix = "Based on Feature Product Correlation"
    
    bar_height = 0.7
    
    for i, (y_pos, val, label) in enumerate(zip(y_positions, values, labels)):
        y0 = y_pos - bar_height / 2
        y1 = y_pos + bar_height / 2
        
        darker_gradient = np.linspace(0.35, 1, 256).reshape(1, -1)
        im = ax.imshow(darker_gradient, extent=(0, max_val, y0, y1), 
                      aspect='auto', cmap=cmap, vmin=0, vmax=1, zorder=2)
        
        rect = patches.Rectangle((0, y0), val, bar_height, transform=ax.transData)
        im.set_clip_path(rect)
    
    ax.margins(y=0.12)
    ax.set_yticks(np.arange(len(top_interactions)))
    ax.set_yticklabels(labels, fontsize=11, fontname='Times New Roman')
    
    if 'shap_interaction_values' in locals() and shap_interaction_values is not None:
        ax.set_xlabel('Mean |SHAP Interaction Value|', fontsize=12, fontname='Times New Roman')
        ax.set_title(f'Feature Pairwise Interaction Ranking - {title_suffix}', fontsize=16, pad=20, fontname='Times New Roman')
        filename = 'shap_interaction_based_ranking.png'
    else:
        ax.set_xlabel('Correlation with Target', fontsize=12, fontname='Times New Roman')
        ax.set_title(f'Feature Pairwise Products Ranking - {title_suffix}', fontsize=16, pad=20, fontname='Times New Roman')
        filename = 'feature_pairwise_products_ranking.png'
    
    ax.set_ylabel('Feature Pairs', fontsize=12, fontname='Times New Roman')
    ax.set_xlim(0, max_val * 1.05)
    ax.invert_yaxis()
    
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    for i, val in enumerate(values):
        ax.text(val + (max_val * 0.01), i, f'{val:.3f}', va='center', fontsize=10, 
                fontname='Times New Roman', zorder=3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, filename), bbox_inches='tight', dpi=250)
    if SHOW_PLOTS:
        plt.show()
    plt.close()
    
    print("\nPerforming partial dependence analysis (PDP) on all feature interaction pairs ranked by importance...")
    
    max_pairs_to_analyze = min(12, len(paired_data))
    top_pairs_for_pdp = paired_data[:max_pairs_to_analyze]
    
    print(f"Analyzing the top {max_pairs_to_analyze} most important feature interaction pairs, divided into 3 groups for display, with 4 pairs per group")
    
    group_size = 4
    num_groups = (max_pairs_to_analyze + group_size - 1) // group_size
    
    for group_idx in range(num_groups):
        start_idx = group_idx * group_size
        end_idx = min(start_idx + group_size, max_pairs_to_analyze)
        group_pairs = top_pairs_for_pdp[start_idx:end_idx]
        
        if len(group_pairs) == 0:
            break
            
        print(f"\nGenerating PDP plots for group {group_idx + 1} (Feature pairs {start_idx + 1}-{end_idx})...")
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        for idx, ((feat1, feat2), importance_val) in enumerate(group_pairs):
            if idx >= len(axes):
                break
                
            ax = axes[idx]
            global_idx = start_idx + idx
            
            try:

                if 'shap_interaction_values' in locals() and shap_interaction_values is not None:

                    if feat1 in X_train_interaction.columns and feat2 in X_train_interaction.columns:
                        feat1_idx = list(X_train_interaction.columns).index(feat1)
                        feat2_idx = list(X_train_interaction.columns).index(feat2)
                        train_data = X_train_interaction
                        model_to_use = model_for_interaction
                    else:
                        raise ValueError(f"Feature {feat1} or {feat2} not found in interaction analysis feature set")
                else:

                    if feat1 in X_train.columns and feat2 in X_train.columns:
                        feat1_idx = list(X_train.columns).index(feat1)
                        feat2_idx = list(X_train.columns).index(feat2)
                        train_data = X_train
                        model_to_use = model
                    else:
                        raise ValueError(f"Feature {feat1} or {feat2} not found in main feature set")
                
                pdp_display = PartialDependenceDisplay.from_estimator(
                    model_to_use, train_data, features=[(feat1_idx, feat2_idx)],
                    ax=ax, grid_resolution=20
                )
                
                if 'shap_interaction_values' in locals() and shap_interaction_values is not None:
                    title = f'#{global_idx+1}: {strip_units(feat1)} × {strip_units(feat2)}\nSHAP Interaction: {importance_val:.4f}'
                else:
                    title = f'#{global_idx+1}: {strip_units(feat1)} × {strip_units(feat2)}\nCorrelation: {importance_val:.3f}'
                
                ax.set_title(title, fontsize=12, fontname='Times New Roman')
                ax.set_xlabel(strip_units(feat1), fontsize=10, fontname='Times New Roman')
                ax.set_ylabel(strip_units(feat2), fontsize=10, fontname='Times New Roman')
                
            except Exception as e:
                ax.text(0.5, 0.5, f'PDP generation failed\n{strip_units(feat1)} × {strip_units(feat2)}\n{str(e)[:50]}...', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=10)
                ax.set_title(f'#{global_idx+1}: {strip_units(feat1)} × {strip_units(feat2)}', fontsize=12)
        
        for idx in range(len(group_pairs), len(axes)):
            axes[idx].set_visible(False)
        
        if 'shap_interaction_values' in locals() and shap_interaction_values is not None:
            suptitle = f'Partial Dependence Plots - Group {group_idx + 1} (SHAP Interaction Pairs #{start_idx+1}-#{end_idx})'
            filename = f'pdp_group_{group_idx + 1}_shap_interaction_pairs.png'
        else:
            suptitle = f'Partial Dependence Plots - Group {group_idx + 1} (Feature Product Pairs #{start_idx+1}-#{end_idx})'
            filename = f'pdp_group_{group_idx + 1}_feature_pairs.png'
        
        plt.suptitle(suptitle, fontsize=16, fontname='Times New Roman')
        plt.tight_layout()
        plt.subplots_adjust(top=0.90)
        plt.savefig(os.path.join(out_dir, filename), bbox_inches='tight', dpi=250)
        if SHOW_PLOTS:
            plt.show()
        plt.close()
        print(f"PDP plots for group {group_idx + 1} generated successfully!")
    
    print("\nGenerating univariate PDP plots for important features (ranked by importance)...")
    
    feature_importance_from_interactions = {}
    for (feat1, feat2), importance_val in top_pairs_for_pdp:

        if feat1 not in feature_importance_from_interactions:
            feature_importance_from_interactions[feat1] = 0
        if feat2 not in feature_importance_from_interactions:
            feature_importance_from_interactions[feat2] = 0
        
        feature_importance_from_interactions[feat1] += importance_val
        feature_importance_from_interactions[feat2] += importance_val
    
    sorted_features_by_interaction = sorted(feature_importance_from_interactions.items(), 
                                          key=lambda x: x[1], reverse=True)
    
    top_features_for_individual_pdp = [feat for feat, _ in sorted_features_by_interaction[:8]]
    
    print(f"Top 8 features ranked by interaction importance: {[strip_units(f) for f in top_features_for_individual_pdp]}")
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    for idx, feat in enumerate(top_features_for_individual_pdp):
        if idx >= len(axes):
            break
            
        ax = axes[idx]
        
        try:

            if 'shap_interaction_values' in locals() and shap_interaction_values is not None:
                if feat in X_train_interaction.columns:
                    feat_idx = list(X_train_interaction.columns).index(feat)
                    train_data = X_train_interaction
                    model_to_use = model_for_interaction
                else:
                    raise ValueError(f"Feature {feat} not found in interaction analysis feature set")
            else:
                if feat in X_train.columns:
                    feat_idx = list(X_train.columns).index(feat)
                    train_data = X_train
                    model_to_use = model
                else:
                    raise ValueError(f"Feature {feat} not found in main feature set")
            
            pdp_display = PartialDependenceDisplay.from_estimator(
                model_to_use, train_data, features=[feat_idx],
                ax=ax, grid_resolution=50
            )
            
            cumulative_importance = feature_importance_from_interactions[feat]
            ax.set_title(f'#{idx+1}: {strip_units(feat)}\nCumulative Interaction Score: {cumulative_importance:.4f}', 
                        fontsize=11, fontname='Times New Roman')
            ax.set_xlabel(strip_units(feat), fontsize=10, fontname='Times New Roman')
            ax.set_ylabel('Partial Dependence', fontsize=10, fontname='Times New Roman')
            ax.grid(alpha=0.3)
            
        except Exception as e:
            ax.text(0.5, 0.5, f'PDP generation failed\n{strip_units(feat)}\n{str(e)[:50]}...', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=10)
            ax.set_title(f'#{idx+1}: {strip_units(feat)}', fontsize=11)
    
    for idx in range(len(top_features_for_individual_pdp), len(axes)):
        axes[idx].set_visible(False)
    
    if 'shap_interaction_values' in locals() and shap_interaction_values is not None:
        suptitle = 'Partial Dependence Plots - Top Individual Features (Ordered by Interaction Importance)'
        filename = 'pdp_top_individual_features_by_interaction_importance.png'
    else:
        suptitle = 'Partial Dependence Plots - Top Individual Features (Ordered by Product Correlation)'
        filename = 'pdp_top_individual_features_by_correlation.png'
    
    plt.suptitle(suptitle, fontsize=16, fontname='Times New Roman')
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    plt.savefig(os.path.join(out_dir, filename), bbox_inches='tight', dpi=250)
    if SHOW_PLOTS:
        plt.show()
    plt.close()
    print(f"Univariate PDP plots for the top {len(top_features_for_individual_pdp)} important features generated successfully!")
    
    if 'shap_interaction_values' in locals() and shap_interaction_values is not None:
        print("\nGenerating detailed SHAP interaction analysis...")
        
        interaction_matrix_2d = np.mean(np.abs(shap_interaction_values), axis=0)
        
        strongest_interactions = []
        for i in range(len(features_for_interaction)):
            for j in range(i + 1, len(features_for_interaction)):
                interaction_strength = interaction_matrix_2d[i, j]
                strongest_interactions.append({
                    'feature_i': features_for_interaction[i],
                    'feature_j': features_for_interaction[j],
                    'interaction_strength': interaction_strength,
                    'feature_i_clean': strip_units(features_for_interaction[i]),
                    'feature_j_clean': strip_units(features_for_interaction[j])
                })
        
        strongest_interactions.sort(key=lambda x: x['interaction_strength'], reverse=True)
        
        detailed_df = pd.DataFrame(strongest_interactions)
        detailed_df.to_csv(os.path.join(out_dir, 'detailed_shap_interactions_analysis.csv'), index=False)
        
        print("Detailed SHAP interaction analysis saved to: detailed_shap_interactions_analysis.csv")
        print(f"Top 5 strongest feature interaction pairs:")
        for i, interaction in enumerate(strongest_interactions[:5]):
            print(f"{i+1}. {interaction['feature_i_clean']} × {interaction['feature_j_clean']}: {interaction['interaction_strength']:.4f}")
    
    print(f"\nFeature interaction analysis completed!")
    if 'shap_interaction_values' in locals() and shap_interaction_values is not None:
        print(f"- SHAP interaction value-based feature interaction ranking results saved")
        print(f"- Using true SHAP interaction values, mathematical formula: Φᵢ,ⱼ = ½ * (SHAP₍ᵢ,ⱼ₎ - SHAP₍ᵢ\\ⱼ₎ - SHAP₍ⱼ\\ᵢ₎ + SHAP₍\\ᵢ,ⱼ₎)")
        print(f"- 2D PDP plots for the top 6 most important interaction pairs generated")
        print(f"- 1D PDP plots for important features generated")
    else:
        print(f"- Due to unavailability of SHAP interaction values, feature product correlation was used as an alternative")
        print(f"- PDP plots for the top 6 feature pairs generated")
        print(f"- PDP plots for important individual features generated")
    
except Exception as e:
    print(f'Feature interaction analysis error: {e}')
    import traceback
    traceback.print_exc()

print("\n" + "="*50)
print("Generating feature interaction summary plots ordered by importance")
print("="*50)

try:

    if 'feature_importance' in locals() and len(feature_importance) > 0:

        importance_ordered_features = feature_importance['feature'].tolist()
        
        print(f"Features ordered by importance: {importance_ordered_features}")
        
        clean_to_original = {}
        for orig_col in X_test.columns:
            clean_name = strip_units(orig_col)
            clean_to_original[clean_name] = orig_col
        
        type_cols_in_data = [c for c in X_test.columns if str(c).startswith('type_')]
        
        final_features_by_importance = []
        for clean_feat in importance_ordered_features:
            if clean_feat == 'TYPE':

                final_features_by_importance.extend(type_cols_in_data)
            elif clean_feat in clean_to_original:
                final_features_by_importance.append(clean_to_original[clean_feat])
            else:
                print(f"Warning: Clean feature name '{clean_feat}' could not be mapped to original feature")
        
        print(f"Mapped features ordered by importance: {final_features_by_importance}")
        
        available_importance_features = [f for f in final_features_by_importance if f in X_test.columns]
        print(f"Available features ordered by importance: {available_importance_features}")
        
        if len(available_importance_features) >= 10:

            print("\nGenerating SHAP interaction summary plot for the top 10 most important features...")
            top_10_features = available_importance_features[:10]
            
            top_10_indices = [X_test.columns.get_loc(f) for f in top_10_features]
            
            sv_top10 = sv_raw[:, top_10_indices]
            X_test_top10 = X_test[top_10_features].copy()
            
            top_10_clean_names = [strip_units(f) for f in top_10_features]
            X_test_top10_display = X_test_top10.copy()
            X_test_top10_display.columns = top_10_clean_names
            
            plt.figure(figsize=(14, max(8, len(top_10_features) * 0.6)))
            shap.summary_plot(sv_top10, X_test_top10_display, show=SHOW_PLOTS, max_display=None, sort=False)
            
            ax = plt.gca()
            ax.tick_params(axis='y', labelsize=28)
            for label in ax.get_yticklabels():
                label.set_fontname('Times New Roman')
            
            plt.subplots_adjust(top=0.92)
            plt.title("SHAP Interaction Summary Plot - Top 10 Most Important Features", 
                     fontsize=16, fontname='Times New Roman', pad=20)
            
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, 'shap_interaction_summary_top10_by_importance.png'), 
                       bbox_inches='tight', dpi=250)
            if SHOW_PLOTS:
                plt.show()
            plt.close()
            print("SHAP interaction summary plot for the top 10 most important features generated successfully!")
            
            if len(available_importance_features) > 10:
                print("\nGenerating SHAP interaction summary plot for the remaining features...")
                remaining_features = available_importance_features[10:]
                
                remaining_indices = [X_test.columns.get_loc(f) for f in remaining_features]
                
                sv_remaining = sv_raw[:, remaining_indices]
                X_test_remaining = X_test[remaining_features].copy()
                
                remaining_clean_names = [strip_units(f) for f in remaining_features]
                X_test_remaining_display = X_test_remaining.copy()
                X_test_remaining_display.columns = remaining_clean_names
                
                plt.figure(figsize=(14, max(6, len(remaining_features) * 0.6)))
                shap.summary_plot(sv_remaining, X_test_remaining_display, 
                                show=SHOW_PLOTS, max_display=None, sort=False)
                
                ax = plt.gca()
                ax.tick_params(axis='y', labelsize=28)
                for label in ax.get_yticklabels():
                    label.set_fontname('Times New Roman')
                
                plt.subplots_adjust(top=0.92)
                plt.title(f"SHAP Interaction Summary Plot - Remaining {len(remaining_features)} Features (By Importance)", 
                         fontsize=16, fontname='Times New Roman', pad=20)
                
                plt.tight_layout()
                plt.savefig(os.path.join(out_dir, 'shap_interaction_summary_remaining_by_importance.png'), 
                           bbox_inches='tight', dpi=250)
                if SHOW_PLOTS:
                    plt.show()
                plt.close()
                print(f"SHAP interaction summary plot for the remaining {len(remaining_features)} features generated successfully!")
            
            importance_ranking_df = pd.DataFrame({
                'rank': range(1, len(available_importance_features) + 1),
                'feature': available_importance_features,
                'feature_clean': [strip_units(f) for f in available_importance_features],
                'importance': [
                    feature_importance[feature_importance['feature'] == strip_units(f)]['importance'].iloc[0] 
                    if len(feature_importance[feature_importance['feature'] == strip_units(f)]) > 0
                    else (feature_importance[feature_importance['feature'] == 'TYPE']['importance'].iloc[0] 
                          if f.startswith('type_') and len(feature_importance[feature_importance['feature'] == 'TYPE']) > 0
                          else 0)
                    for f in available_importance_features
                ]
            })
            importance_ranking_df.to_csv(os.path.join(out_dir, 'features_ranking_by_importance.csv'), index=False)
            
            print(f"\nFeature interaction summary plots ordered by importance generated successfully!")
            print(f"- SHAP interaction summary plot for the top 10 most important features: shap_interaction_summary_top10_by_importance.png")
            if len(available_importance_features) > 10:
                print(f"- SHAP interaction summary plot for the remaining {len(available_importance_features)-10} features: shap_interaction_summary_remaining_by_importance.png")
            print(f"- Feature importance ranking table: features_ranking_by_importance.csv")
            
        else:
            print(f"Available features ({len(available_importance_features)}) are fewer than 10, unable to generate top 10 feature plots")
            
    else:
        print("Feature importance data not available, cannot generate ordered summary plots")
        
except Exception as e:
    print(f'Error generating feature interaction summary plots ordered by importance: {e}')
    import traceback
    traceback.print_exc()

print("\n" + "="*50)
print("Generating heatmap ordered by specified sequence")
print("="*50)

try:

    specified_order = ['C', 'type_Ag', 'type_CuO', 'type_Fe2O3', 'type_ZnO', 'type_TiO2', 'type_Ag2S', 'type_Al2O3', 'type_Cu', 'type_FeO', 
                      'SIZE', 'ZP', 'SAND', 'CLAY', 'SILT', 'SOM', 'WHC', 'pH', 'TN', 'AM', 'TEMP', 'ED']
    
    print(f"Specified feature order: {specified_order}")
    
    available_features = list(X_test.columns)
    print(f"Available features in data: {available_features}")
    
    ordered_features_present = []
    for feat in specified_order:
        if feat in available_features:
            ordered_features_present.append(feat)
        else:
            print(f"Warning: Specified feature '{feat}' not found in data")
    
    remaining_features = [feat for feat in available_features if feat not in specified_order]
    
    final_ordered_features = ordered_features_present + remaining_features
    
    print(f"Final feature order (total {len(final_ordered_features)}): {final_ordered_features}")
    print(f"Features in specified order: {len(ordered_features_present)}")
    print(f"Other features: {len(remaining_features)}")
    
    if len(final_ordered_features) > 0:

        ordered_indices = [available_features.index(feat) for feat in final_ordered_features]
        sv_ordered = sv_raw[:, ordered_indices]
        
        X_test_ordered = X_test[final_ordered_features].copy()
        
        display_features_ordered = [strip_units(feat) for feat in final_ordered_features]
        X_test_ordered_display = X_test_ordered.copy()
        X_test_ordered_display.columns = display_features_ordered
        
        print("\nGenerating SHAP beeswarm plot ordered by specified sequence...")
        plt.figure(figsize=(14, max(10, len(final_ordered_features) * 0.4)))
        shap.summary_plot(sv_ordered, X_test_ordered_display, show=SHOW_PLOTS, max_display=None, sort=False)
        plt.title("SHAP Beeswarm Plot - Custom Ordered Features\n(C, type classes, SIZE, ZP, SAND, CLAY, SILT, SOM, WHC, TN, AM, PH, TEMP, ED)", 
                 fontsize=16, fontname='Times New Roman')
        
        ax = plt.gca()
        ax.tick_params(axis='y', labelsize=28)
        for label in ax.get_yticklabels():
            label.set_fontname('Times New Roman')
        
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'shap_beeswarm_custom_ordered.png'), bbox_inches='tight', dpi=250)
        if SHOW_PLOTS:
            plt.show()
        plt.close()
        print("SHAP beeswarm plot ordered by specified sequence generated successfully!")
        
        print("\nGenerating feature importance plot ordered by specified sequence...")
        
        ordered_importance_list = []
        ordered_feature_list = []
        for i, feat in enumerate(final_ordered_features):
            ordered_feature_list.append(strip_units(feat))
            ordered_importance_list.append(np.abs(sv_ordered[:, i]).mean())
        
        ordered_feature_importance = pd.DataFrame({
            'feature': ordered_feature_list,
            'importance': ordered_importance_list
        })
        
        plt.figure(figsize=(10, max(6, 0.35 * len(ordered_feature_importance))), dpi=250)
        ax = plt.gca()
        fig = plt.gcf()
        
        fig.patch.set_facecolor('white')
        ax.set_facecolor('#f5f5f5')
        ax.grid(axis='x', linestyle='--', color='#e0e0e0', linewidth=1)
        
        fi_ordered = ordered_feature_importance.iloc[::-1]
        
        max_val = ordered_feature_importance['importance'].max() if len(ordered_feature_importance) > 0 else 1.0
        cmap = plt.cm.Greens
        bar_height = 0.7
        
        for i, (feat, val) in enumerate(zip(fi_ordered['feature'], fi_ordered['importance'])):
            y0 = i - bar_height / 2
            y1 = i + bar_height / 2

            darker_gradient = np.linspace(0.35, 1, 256).reshape(1, -1)
            im = ax.imshow(darker_gradient, extent=(0, max_val, y0, y1), aspect='auto', cmap=cmap, vmin=0, vmax=1, zorder=2)
            rect = patches.Rectangle((0, y0), val, bar_height, transform=ax.transData)
            im.set_clip_path(rect)
        
        ax.margins(y=0.12)
        
        ax.set_yticks(np.arange(len(fi_ordered)))
        ax.set_yticklabels(fi_ordered['feature'], fontsize=21, fontname='Times New Roman')
        ax.set_xlabel('Mean |SHAP value|', fontsize=14, fontname='Times New Roman')
        ax.set_ylabel('Feature', fontsize=14, fontname='Times New Roman')
        ax.set_title('Feature Importance - Custom Ordered\n(C, type classes, SIZE, ZP, SAND, CLAY, SILT, SOM, WHC, TN, AM, PH, TEMP, ED)', 
                    fontsize=16, pad=20, fontname='Times New Roman')
        ax.grid(axis='x', linestyle='--', alpha=0.6)
        ax.set_xlim(0, max_val * 1.05)
        for spine in ax.spines.values():
            spine.set_visible(False)
        
        for i, val in enumerate(fi_ordered['importance']):
            ax.text(val + (max_val * 0.01), i, f'{val:.3f}', va='center', fontsize=12, 
                    fontname='Times New Roman', fontfamily='serif', zorder=3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'feature_importance_custom_ordered.png'), bbox_inches='tight', dpi=250)
        if SHOW_PLOTS:
            plt.show()
        plt.close()
        print("Feature importance plot ordered by specified sequence generated successfully!")
        
        print("\nGenerating SHAP interaction heatmap ordered by specified sequence...")
        
        if 'shap_interaction_values' in locals() and shap_interaction_values is not None:
            try:

                interaction_features_original = list(X_test_interaction.columns)
                
                ordered_interaction_features = []
                for feat in final_ordered_features:
                    if feat in interaction_features_original:
                        ordered_interaction_features.append(feat)
                
                if len(ordered_interaction_features) > 1:

                    ordered_interaction_indices = [interaction_features_original.index(feat) for feat in ordered_interaction_features]
                    
                    custom_interaction_matrix = np.mean(np.abs(shap_interaction_values), axis=0)
                    custom_ordered_interaction_matrix = custom_interaction_matrix[np.ix_(ordered_interaction_indices, ordered_interaction_indices)]
                    
                    custom_ordered_feature_names = [strip_units(feat) for feat in ordered_interaction_features]
                    
                    plt.figure(figsize=(max(12, len(ordered_interaction_features) * 0.6), 
                                       max(10, len(ordered_interaction_features) * 0.6)))
                    
                    import seaborn as sns
                    
                    mask = np.triu(np.ones_like(custom_ordered_interaction_matrix, dtype=bool), k=1)
                    
                    ax = sns.heatmap(custom_ordered_interaction_matrix, 
                                   annot=True,
                                   fmt='.3f',
                                   cmap='YlOrRd',
                                   square=True,
                                   linewidths=0.5,
                                   cbar_kws={"shrink": 0.8},
                                   xticklabels=custom_ordered_feature_names,
                                   yticklabels=custom_ordered_feature_names,
                                   mask=mask,
                                   annot_kws={'fontsize': 8, 'fontname': 'Times New Roman'})
                    
                    plt.title('SHAP Interaction Heatmap - Custom Ordered\n(C, type classes, SIZE, ZP, SAND, CLAY, SILT, SOM, WHC, TN, AM, PH, TEMP, ED)', 
                             fontsize=16, fontname='Times New Roman', pad=20)
                    plt.xlabel('Features (Custom Order)', fontsize=14, fontname='Times New Roman')
                    plt.ylabel('Features (Custom Order)', fontsize=14, fontname='Times New Roman')
                    
                    plt.xticks(rotation=45, ha='right', fontsize=14, fontname='Times New Roman')
                    plt.yticks(rotation=0, fontsize=14, fontname='Times New Roman')
                    
                    plt.tight_layout()
                    plt.savefig(os.path.join(out_dir, 'shap_interaction_heatmap_custom_ordered.png'), 
                               bbox_inches='tight', dpi=250)
                    if SHOW_PLOTS:
                        plt.show()
                    plt.close()
                    print("SHAP interaction heatmap ordered by specified sequence generated successfully!")
                    
                    custom_interaction_df = pd.DataFrame(custom_ordered_interaction_matrix, 
                                                       index=custom_ordered_feature_names, 
                                                       columns=custom_ordered_feature_names)
                    custom_interaction_df.to_csv(os.path.join(out_dir, 'shap_interaction_matrix_custom_ordered.csv'))
                else:
                    print("Insufficient specified features available for interaction analysis, skipping SHAP interaction heatmap")
                    
            except Exception as e:
                print(f"Error generating SHAP interaction heatmap: {e}")
        else:
            print("SHAP interaction values not available, skipping SHAP interaction heatmap")
        
        print("\nGenerating SHAP values heatmap ordered by specified sequence...")
        
        n_samples_heatmap = min(50, sv_ordered.shape[0])
        sv_sample = sv_ordered[:n_samples_heatmap, :]
        
        plt.figure(figsize=(max(12, len(final_ordered_features) * 0.8), 
                           max(8, n_samples_heatmap * 0.3)))
        
        ax = sns.heatmap(sv_sample.T,
                       cmap='YlOrRd',
                       center=0,
                       linewidths=0.5,
                       cbar_kws={"shrink": 0.8, "label": "SHAP Value"},
                       yticklabels=display_features_ordered,
                       xticklabels=False,
                       annot_kws={'fontsize': 8, 'fontname': 'Times New Roman'})
        
        plt.title(f'SHAP Values Heatmap - Custom Ordered (First {n_samples_heatmap} samples)\n(C, type classes, SIZE, ZP, SAND, CLAY, SILT, SOM, WHC, TN, AM, PH, TEMP, ED)', 
                 fontsize=16, fontname='Times New Roman', pad=20)
        plt.xlabel(f'Samples (1-{n_samples_heatmap})', fontsize=12, fontname='Times New Roman')
        plt.ylabel('Features (Custom Order)', fontsize=12, fontname='Times New Roman')
        
        plt.yticks(rotation=0, fontsize=10, fontname='Times New Roman')
        
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'shap_values_heatmap_custom_ordered.png'), 
                   bbox_inches='tight', dpi=250)
        if SHOW_PLOTS:
            plt.show()
        plt.close()
        print("SHAP values heatmap ordered by specified sequence generated successfully!")
        
        ordered_feature_importance.to_csv(os.path.join(out_dir, 'feature_importance_custom_ordered.csv'), index=False)
        
        print(f"\nCustom ordered heatmaps generated successfully!")
        print(f"- SHAP beeswarm plot ordered by specified sequence")
        print(f"- Feature importance plot ordered by specified sequence")
        print(f"- SHAP interaction heatmap ordered by specified sequence (if available)") 
        print(f"- SHAP values heatmap ordered by specified sequence")
        print(f"- Data files saved to outputs folder")
        print(f"Feature order: C → type_Ag → type_CuO → type_Fe2O3 → type_ZnO → type_TiO2 → type_Ag2S → type_Al2O3 → type_Cu → type_FeO → SIZE → ZP → SAND → CLAY → SILT → SOM → WHC → pH → TN → AM → TEMP → ED")
        
    else:
        print("No features available")
        
except Exception as e:
    print(f'Error generating custom ordered heatmaps: {e}')
    import traceback
    traceback.print_exc()

print('All plots displayed!')
print(f'All outputs saved to: {out_dir}')

print('\n' + '='*50)
print('Generating shrunk-axis Partial Dependence Plots (user-specified ranges)')
print('='*50)

try:

    shrunk_pairs = [
        (('C', 'pH'), (0, 1000), None),
        (('C', 'CLAY'), (0, 1000), None),
        (('C', 'ED'), (0, 1000), None),
        (('C', 'SOM'), (0, 1000), None),
        (('C', 'SIZE'), (0, 1000), None),
        (('SIZE', 'CLAY'), None, None),
        (('C', 'SILT'), (0, 1000), None),
        (('SOM', 'pH'), None, None),
        (('C', 'SAND'), (0, 1000), None),
        (('SIZE', 'ED'), None, None),
        (('ED', 'SILT'), None, None),
        (('TN', 'pH'), None, None)
    ]

    for pair_idx, (pair, x_range, y_range) in enumerate(shrunk_pairs, start=1):
        feat_x, feat_y = pair
        display_x = strip_units(feat_x)
        display_y = strip_units(feat_y)

        print(f"\nProcessing shrunk PDP #{pair_idx}: {feat_x} × {feat_y} -> x_range={x_range}, y_range={y_range}")

        train_data = None
        model_to_use = None
        try:
            if 'shap_interaction_values' in locals() and shap_interaction_values is not None:
                if feat_x in X_train_interaction.columns and feat_y in X_train_interaction.columns:
                    train_data = X_train_interaction
                    model_to_use = model_for_interaction
            if model_to_use is None:
                if feat_x in X_train.columns and feat_y in X_train.columns:
                    train_data = X_train
                    model_to_use = model

            if model_to_use is None or train_data is None:
                print(f"Warning: Features {feat_x} and/or {feat_y} not found in training data, skipping this pair")
                continue

            feat1_idx = list(train_data.columns).index(feat_x)
            feat2_idx = list(train_data.columns).index(feat_y)

            fig = plt.figure(figsize=(10, 8))
            ax = plt.gca()
            
            fig.patch.set_facecolor('white')
            ax.set_facecolor('#f8f9fa')

            try:

                from sklearn.inspection import partial_dependence
                pdp_results = partial_dependence(
                    model_to_use, train_data, features=[(feat1_idx, feat2_idx)],
                    grid_resolution=50
                )
                
                pdp_values = pdp_results['average'][0]
                feature1_grid = pdp_results['grid'][0]
                feature2_grid = pdp_results['grid'][1]
                
                X_grid, Y_grid = np.meshgrid(feature1_grid, feature2_grid)
                
                im = ax.contourf(X_grid, Y_grid, pdp_values.T, levels=20, cmap='viridis', alpha=0.8)
                
                contour_lines = ax.contour(X_grid, Y_grid, pdp_values.T, levels=20, colors='white', 
                                         linewidths=0.3, alpha=0.4)
                
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.1)
                cbar = plt.colorbar(im, cax=cax)

                cbar.ax.tick_params(labelsize=21)
                
            except Exception as e:

                print(f"Warning: Custom PDP failed for {feat_x} × {feat_y}: {e}")
                continue

            ax.set_xlabel(display_x, fontsize=21, fontname='Times New Roman', fontweight='bold')
            ax.set_ylabel(display_y, fontsize=21, fontname='Times New Roman', fontweight='bold')
            
            ax.tick_params(axis='both', which='major', labelsize=18,
                        colors='#2c3e50', width=1.5, length=6)
            ax.tick_params(axis='both', which='minor', labelsize=15,
                        colors='#34495e', width=1, length=4)
            
            for spine in ax.spines.values():
                spine.set_color('#2c3e50')
                spine.set_linewidth(1.5)
            
            ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8, color='#7f8c8d')
            ax.set_axisbelow(True)
            
            if x_range is not None:
                ax.set_xlim(x_range[0], x_range[1])
                ax.set_autoscalex_on(False)
            if y_range is not None:
                ax.set_ylim(y_range[0], y_range[1])
                ax.set_autoscaley_on(False)

            plt.tight_layout()

            fname = f'pdp_shrunk_{feat_x}_x_{feat_y}.png'
            plt.savefig(os.path.join(out_dir, fname), bbox_inches='tight', dpi=300, 
                       facecolor='white', edgecolor='none')
            if SHOW_PLOTS:
                plt.show()
            plt.close()
            print(f"Saved shrunk-axis PDP: {fname}")

        except Exception as e:
            print(f"Error generating shrunk PDP for {feat_x} × {feat_y}: {e}")
            import traceback
            traceback.print_exc()

    print('\nFinished generating shrunk-axis PDP plots (saved to outputs).')

except Exception as e:
    print(f'Error in shrunk-axis PDP generation: {e}')
    import traceback
    traceback.print_exc()

print('\n' + '='*60)
print('saved successfully!')
print('='*60)
print(f'Saved location: {out_dir}')
print('\nSaved images include:')
print('1. Feature importance plot: feature_importance.png')
print('2. SHAP interaction heatmaps:')
print('   - shap_interaction_heatmap_all_features_sorted.png')
print('   - shap_interaction_heatmap_type_merged_sorted.png')
print('3. SHAP interaction summary plots:')
print('   - shap_interaction_summary_top11_non_type_features.png')
print('   - shap_interaction_summary_remaining_features.png')
print('   - shap_interaction_summary_all_features_complete.png')
print('   - shap_interaction_summary_type_merged_sorted.png')
print('4. SHAP beeswarm plots:')
print('   - shap_beeswarm_all_features_no_merging.png')
print('   - shap_beeswarm_all_original_features.png')
print('5. Feature ranking plots:')
print('   - shap_interaction_based_ranking.png')
print('   - feature_pairwise_products_ranking.png')
print('6. PDP group plots: pdp_group_*.png')
print('7. PDP individual feature plots:')
print('   - pdp_top_individual_features_by_interaction_importance.png')
print('   - pdp_top_individual_features_by_correlation.png')
print('8. Shrunk-axis PDP plots from section 5.6: pdp_shrunk_*.png')
print('\nAnd CSV data files:')
print('   - shap_interaction_matrix_all_features_sorted.csv')
print('   - shap_interaction_matrix_type_merged_sorted.csv')

print("\n" + "="*50)
print("Generating feature interaction summary plots ordered by heatmap interaction values")
print("="*50)

try:

    if 'shap_interaction_values' in locals() and shap_interaction_values is not None and 'sorted_interaction_matrix' in locals():
        print("Using precomputed SHAP interaction values and interaction matrix...")
        
        if 'sorted_features' in locals() and 'sorted_interaction_matrix' in locals():

            feature_interaction_totals = {}
            n_features = len(sorted_features)
            
            for i, feat in enumerate(sorted_features):

                total_interaction = 0
                for j in range(n_features):
                    if i != j:
                        total_interaction += sorted_interaction_matrix[i, j]
                feature_interaction_totals[feat] = total_interaction
            
            sorted_by_interaction = sorted(feature_interaction_totals.items(), key=lambda x: x[1], reverse=True)
            top_8_interaction_features = [feat for feat, _ in sorted_by_interaction[:8]]
            top_8_interaction_values = [val for _, val in sorted_by_interaction[:8]]
            
            print(f"Top 8 strongest interaction features based on heatmap interaction values:")
            for i, (feat, val) in enumerate(zip(top_8_interaction_features, top_8_interaction_values), 1):
                clean_name = strip_units(feat)
                print(f"{i}. {clean_name} (Total interaction strength: {val:.4f})")
            
            available_top8_features = [feat for feat in top_8_interaction_features if feat in X_test_interaction.columns]
            
            if len(available_top8_features) >= 6:
                print(f"\nGenerating interaction summary plots for the top {len(available_top8_features)} features by heatmap interaction values...")
                
                top8_indices_in_interaction = [X_test_interaction.columns.get_loc(feat) for feat in available_top8_features]
                
                top8_interaction_values_matrix = shap_interaction_values[:, top8_indices_in_interaction, :][:, :, top8_indices_in_interaction]
                top8_X_test_interaction = X_test_interaction[available_top8_features].copy()
                
                top8_feature_names_clean = [strip_units(feat) for feat in available_top8_features]
                top8_X_test_interaction.columns = top8_feature_names_clean
                
                print(f"Top {len(available_top8_features)} strongest features by heatmap interaction values: {top8_feature_names_clean}")
                
                plt.figure(figsize=(16, max(10, len(available_top8_features) * 0.8)))
                shap.summary_plot(top8_interaction_values_matrix, top8_X_test_interaction, 
                                feature_names=top8_feature_names_clean,
                                show=SHOW_PLOTS, max_display=len(available_top8_features), sort=False)
                
                ax = plt.gca()
                ax.tick_params(axis='y', labelsize=28)
                ax.tick_params(axis='x', labelsize=8)
                for label in ax.get_yticklabels():
                    label.set_fontname('Times New Roman')
                for label in ax.get_xticklabels():
                    label.set_fontname('Times New Roman')
                
                plt.subplots_adjust(top=0.92)
                plt.title(f"SHAP Interaction Summary Plot - Top {len(available_top8_features)} Features by Heatmap Interaction Values", 
                         fontsize=16, fontname='Times New Roman', pad=20)
                
                plt.tight_layout()
                plt.savefig(os.path.join(out_dir, f'shap_interaction_summary_top{len(available_top8_features)}_heatmap_based.png'), bbox_inches='tight', dpi=250)
                if SHOW_PLOTS:
                    plt.show()
                plt.close()
                
                print(f"Interaction summary plot for the top {len(available_top8_features)} features by heatmap interaction values generated successfully!")
                print(f"Filename: shap_interaction_summary_top{len(available_top8_features)}_heatmap_based.png")
                
                top8_details = pd.DataFrame({
                    'rank': range(1, len(available_top8_features) + 1),
                    'feature_original': available_top8_features,
                    'feature_clean': top8_feature_names_clean,
                    'total_interaction_strength': [feature_interaction_totals[feat] for feat in available_top8_features],
                    'selection_method': 'heatmap_interaction_values'
                })
                top8_details.to_csv(os.path.join(out_dir, 'top8_heatmap_interaction_features.csv'), index=False)
                print("Detailed information for the top 8 strongest features by heatmap interaction values saved to: top8_heatmap_interaction_features.csv")
                
                if len(available_top8_features) == 8:

                    print("\nGenerating interaction summary plots for the top 4 features by heatmap interaction values...")
                    top4_features = available_top8_features[:4]
                    top4_indices = [X_test_interaction.columns.get_loc(feat) for feat in top4_features]
                    top4_interaction_values = shap_interaction_values[:, top4_indices, :][:, :, top4_indices]
                    top4_X_test = X_test_interaction[top4_features].copy()
                    top4_clean_names = [strip_units(feat) for feat in top4_features]
                    top4_X_test.columns = top4_clean_names
                    
                    plt.figure(figsize=(14, max(6, len(top4_features) * 0.8)))
                    shap.summary_plot(top4_interaction_values, top4_X_test, 
                                    feature_names=top4_clean_names, show=SHOW_PLOTS, 
                                    max_display=len(top4_features), sort=False)
                    
                    ax = plt.gca()
                    ax.tick_params(axis='y', labelsize=28)
                    for label in ax.get_yticklabels():
                        label.set_fontname('Times New Roman')
                    
                    plt.subplots_adjust(top=0.92)
                    plt.title("SHAP Interaction Summary Plot - Top 4 Features by Heatmap Interaction Values", 
                             fontsize=16, fontname='Times New Roman', pad=20)
                    plt.tight_layout()
                    plt.savefig(os.path.join(out_dir, 'shap_interaction_summary_top4_heatmap_based.png'), 
                               bbox_inches='tight', dpi=250)
                    if SHOW_PLOTS:
                        plt.show()
                    plt.close()
                    
                    print("Generating interaction summary plots for the bottom 4 of the top 8 features by heatmap interaction values...")
                    bottom4_features = available_top8_features[4:8]
                    bottom4_indices = [X_test_interaction.columns.get_loc(feat) for feat in bottom4_features]
                    bottom4_interaction_values = shap_interaction_values[:, bottom4_indices, :][:, :, bottom4_indices]
                    bottom4_X_test = X_test_interaction[bottom4_features].copy()
                    bottom4_clean_names = [strip_units(feat) for feat in bottom4_features]
                    bottom4_X_test.columns = bottom4_clean_names
                    
                    plt.figure(figsize=(14, max(6, len(bottom4_features) * 0.8)))
                    shap.summary_plot(bottom4_interaction_values, bottom4_X_test, 
                                    feature_names=bottom4_clean_names, show=SHOW_PLOTS, 
                                    max_display=len(bottom4_features), sort=False)
                    
                    ax = plt.gca()
                    ax.tick_params(axis='y', labelsize=28)
                    for label in ax.get_yticklabels():
                        label.set_fontname('Times New Roman')
                    
                    plt.subplots_adjust(top=0.92)
                    plt.title("SHAP Interaction Summary Plot - Bottom 4 of Top 8 Features by Heatmap Interaction Values", 
                             fontsize=16, fontname='Times New Roman', pad=20)
                    plt.tight_layout()
                    plt.savefig(os.path.join(out_dir, 'shap_interaction_summary_bottom4_of_top8_heatmap_based.png'), 
                               bbox_inches='tight', dpi=250)
                    if SHOW_PLOTS:
                        plt.show()
                    plt.close()
                    
                    print("Interaction summary plots for the top 4 and bottom 4 of the top 8 features by heatmap interaction values generated successfully!")
                
            else:
                print(f"Insufficient number of strongest features by heatmap interaction values available (only {len(available_top8_features)}), at least 6 features are required")
                
        else:
            print("Sorted features or sorted interaction matrix not found, unable to generate summary plots based on heatmap interaction values")
            
    else:
        print("SHAP interaction values or interaction matrix not available, unable to generate summary plots based on heatmap interaction values")
        
except Exception as e:
    print(f'Error generating interaction summary plots based on heatmap interaction values: {e}')
    import traceback
    traceback.print_exc()

print('='*60)
print(f'Total save directory: {out_dir}')
print(f'All images and data files have been successfully saved to the outputs folder!')
print(' New feature: Interaction summary plots for the top 8 features by heatmap interaction values have been added!')
print('='*60)
