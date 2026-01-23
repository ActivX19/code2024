import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Times New Roman'
plt.rcParams['mathtext.it'] = 'Times New Roman:italic'
plt.rcParams['mathtext.bf'] = 'Times New Roman:bold'
plt.rcParams['font.size'] = 23  
plt.rcParams['axes.unicode_minus'] = False  
sns.set(style="ticks", font='Times New Roman', font_scale=2.0)


def calculate_cochran_q_and_i2(data, predictor_col, outcome_col):
   
    valid_data = data[[predictor_col, outcome_col]].dropna()
    
    if len(valid_data) < 3:
        return {
            'Variable': predictor_col,
            'N': len(valid_data),
            'N_groups': 0,
            'Q': np.nan,
            'df': np.nan,
            'Q_pvalue': np.nan,
            'I2': np.nan,
            'I2_interpretation': 'Insufficient data',
            'tau2': np.nan,
            'Heterogeneity_level': 'Cannot determine'
        }
    
    unique_vals = valid_data[predictor_col].nunique()
    
    if unique_vals <= 1:
        return {
            'Variable': predictor_col,
            'N': len(valid_data),
            'N_groups': unique_vals,
            'Q': np.nan,
            'df': np.nan,
            'Q_pvalue': np.nan,
            'I2': np.nan,
            'I2_interpretation': 'No variability',
            'tau2': np.nan,
            'Heterogeneity_level': 'Cannot determine'
        }
    
    
    if unique_vals <= 10:
        groups = valid_data.groupby(predictor_col)[outcome_col]
    else:
        valid_data = valid_data.copy()
        try:
            valid_data['quartile'] = pd.qcut(valid_data[predictor_col], q=4, duplicates='drop')
            groups = valid_data.groupby('quartile', observed=False)[outcome_col]
        except ValueError:
            try:
                valid_data['quartile'] = pd.qcut(valid_data[predictor_col], q=3, duplicates='drop')
                groups = valid_data.groupby('quartile', observed=False)[outcome_col]
            except ValueError:
                median_val = valid_data[predictor_col].median()
                valid_data['quartile'] = (valid_data[predictor_col] > median_val).astype(int)
                groups = valid_data.groupby('quartile')[outcome_col]
    
    group_stats = []
    for name, group in groups:
        if len(group) > 0:
            n_i = len(group)
            mean_i = group.mean()
            var_i = group.var(ddof=1) if n_i > 1 else 0
            se_i = np.sqrt(var_i / n_i) if n_i > 1 else np.inf
           
            w_i = 1.0 / (se_i ** 2) if se_i > 0 and np.isfinite(se_i) else 0
            
            group_stats.append({
                'name': name,
                'n': n_i,
                'mean': mean_i,
                'var': var_i,
                'se': se_i,
                'weight': w_i
            })
    
    k = len(group_stats)
    if k < 2:
        return {
            'Variable': predictor_col,
            'N': len(valid_data),
            'N_groups': k,
            'Q': np.nan,
            'df': np.nan,
            'Q_pvalue': np.nan,
            'I2': np.nan,
            'I2_interpretation': 'Insufficient groups',
            'tau2': np.nan,
            'Heterogeneity_level': 'Cannot determine'
        }
    

    means = np.array([g['mean'] for g in group_stats])
    weights = np.array([g['weight'] for g in group_stats])
    
    if np.sum(weights) == 0 or not np.all(np.isfinite(weights)):
        weights = np.ones(k)
    sum_weights = np.sum(weights)
    pooled_effect = np.sum(weights * means) / sum_weights
    Q = np.sum(weights * (means - pooled_effect) ** 2)
    df = k - 1
    Q_pvalue = 1 - chi2.cdf(Q, df) if df > 0 and np.isfinite(Q) else np.nan
    if Q > df and df > 0:
        I2 = ((Q - df) / Q) * 100
    else:
        I2 = 0

    if df > 0 and np.isfinite(Q):
        C = sum_weights - np.sum(weights ** 2) / sum_weights
        if C > 0:
            tau2 = max(0, (Q - df) / C)
        else:
            tau2 = 0
    else:
        tau2 = np.nan

    if I2 <= 40:
        i2_interp = 'Low heterogeneity (0-40%)'
        hetero_level = 'Low'
    elif I2 <= 60:
        i2_interp = 'Moderate heterogeneity (40-60%)'
        hetero_level = 'Moderate'
    elif I2 <= 75:
        i2_interp = 'Substantial heterogeneity (60-75%)'
        hetero_level = 'Substantial'
    else:
        i2_interp = 'Considerable heterogeneity (>75%)'
        hetero_level = 'Considerable'
    
    return {
        'Variable': predictor_col,
        'N': len(valid_data),
        'N_groups': k,
        'Q': Q,
        'df': df,
        'Q_pvalue': Q_pvalue,
        'I2': I2,
        'I2_interpretation': i2_interp,
        'tau2': tau2,
        'Heterogeneity_level': hetero_level
    }


def perform_heterogeneity_analysis_all_predictors(df, outcome_col):
    print(f"\n分析目标变量: {outcome_col}")
    print(f"数据集包含 {len(df)} 个观测值")
    predictor_cols = [col for col in df.columns if col != outcome_col]
    print(f"预测变量数量: {len(predictor_cols)}")
    
    results = []
    
    for i, pred_col in enumerate(predictor_cols, 1):
        print(f"  处理变量 {i}/{len(predictor_cols)}: {pred_col}", end='\r')
        result = calculate_cochran_q_and_i2(df, pred_col, outcome_col)
        results.append(result)
    
    print("\n" + " " * 80)  
    
    results_df = pd.DataFrame(results)
    
    results_df = results_df.sort_values('I2', ascending=False)
    
    return results_df


def plot_heterogeneity_summary(heterogeneity_df, output_path='heterogeneity_summary.png'):
    valid_df = heterogeneity_df.dropna(subset=['I2', 'Q_pvalue'])
    
    if valid_df.empty:
        print("non_data。")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    colors = []
    for i2 in valid_df['I2']:
        if i2 <= 40:
            colors.append('#2ECC71')  
        elif i2 <= 60:
            colors.append('#F39C12')  
        elif i2 <= 75:
            colors.append('#E67E22')  
        else:
            colors.append('#E74C3C')  
    
    y_pos = np.arange(len(valid_df))
    ax1.barh(y_pos, valid_df['I2'], color=colors, alpha=0.7)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(valid_df['Variable'], fontsize=15)
    ax1.set_xlabel('I² (%)', fontsize=19)
    ax1.set_title('I² Statistics for All Predictors', fontsize=21, fontweight='bold', y=1.02)
    ax1.axvline(40, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='40% threshold')
    ax1.axvline(60, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='60% threshold')
    ax1.axvline(75, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='75% threshold')
    ax1.legend(frameon=False, fontsize=15, loc='lower right', handlelength=0.5)
    ax1.invert_yaxis()
    valid_df_copy = valid_df.copy()
    min_pvalue = np.finfo(float).tiny  
    valid_df_copy['Q_pvalue_adj'] = valid_df_copy['Q_pvalue'].replace(0, min_pvalue)
    valid_df_copy['neg_log_p'] = -np.log10(valid_df_copy['Q_pvalue_adj'])

    max_display = 50  
    valid_df_copy.loc[valid_df_copy['Q_pvalue'] == 0, 'neg_log_p'] = max_display

    p_colors = ['#E74C3C' if p < 0.10 else '#3498DB' for p in valid_df['Q_pvalue']]
    
    y_pos2 = np.arange(len(valid_df_copy))
    ax2.barh(y_pos2, valid_df_copy['neg_log_p'], color=p_colors, alpha=0.7)
    ax2.set_yticks(y_pos2)
    ax2.set_yticklabels(valid_df_copy['Variable'], fontsize=15)
    ax2.set_xlabel('-log₁₀(P-value)', fontsize=19)
    ax2.set_title('Cochran\'s Q Test Significance', fontsize=21, fontweight='bold', y=1.02)
    ax2.axvline(-np.log10(0.10), color='red', linestyle='--', linewidth=2, label='p = 0.10 threshold')
    ax2.axvline(-np.log10(0.05), color='darkred', linestyle='--', linewidth=2, label='p = 0.05 threshold')
    ax2.legend(frameon=False, fontsize=15, handlelength=0.5)
    ax2.invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=1000, bbox_inches='tight')
    print(f"save: {output_path}")
    plt.close()


def plot_heterogeneity_scatter(heterogeneity_df, output_path='heterogeneity_scatter.png'):
    valid_df = heterogeneity_df.dropna(subset=['I2', 'Q_pvalue'])
    
    if valid_df.empty:
        print("non_data。")
        return
    
    plt.figure(figsize=(10, 8))
    color_map = {
        'Low': '#2ECC71',
        'Moderate': '#F39C12',
        'Substantial': '#E67E22',
        'Considerable': '#E74C3C'
    }
    
    colors = [color_map.get(level, 'gray') for level in valid_df['Heterogeneity_level']]
    min_pvalue = np.finfo(float).tiny
    valid_df_copy = valid_df.copy()
    valid_df_copy['Q_pvalue_adj'] = valid_df_copy['Q_pvalue'].replace(0, min_pvalue)
    valid_df_copy['neg_log_p'] = -np.log10(valid_df_copy['Q_pvalue_adj'])
    max_display = 50
    valid_df_copy.loc[valid_df_copy['Q_pvalue'] == 0, 'neg_log_p'] = max_display
    
    plt.scatter(valid_df_copy['I2'], valid_df_copy['neg_log_p'], 
                c=colors, s=100, alpha=0.6, edgecolors='black', linewidth=1)
    for idx, row in valid_df_copy.iterrows():
        original_row = valid_df.loc[idx]
        if original_row['I2'] > 60 or original_row['Q_pvalue'] < 0.10:
            plt.annotate(row['Variable'], 
                        (row['I2'], row['neg_log_p']),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=14, alpha=0.8)
    
    plt.xlabel('I² (%)', fontsize=19)
    plt.ylabel('-log₁₀(Q test P-value)', fontsize=19)
    plt.title('Heterogeneity Analysis: I² vs Q Test Significance', fontsize=21, fontweight='bold')
    plt.axhline(-np.log10(0.10), color='red', linestyle='--', linewidth=1, alpha=0.5, label='p = 0.10')
    plt.axvline(60, color='orange', linestyle='--', linewidth=1, alpha=0.5, label='I² = 60%')
    plt.axvline(75, color='red', linestyle='--', linewidth=1, alpha=0.5, label='I² = 75%')
    
    plt.legend(frameon=False, fontsize=15, handlelength=0.5)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=1000, bbox_inches='tight')
    print(f"save: {output_path}")
    plt.close()


def plot_i2_distribution(heterogeneity_df, output_path='i2_distribution.png'):
    valid_df = heterogeneity_df.dropna(subset=['I2'])
    
    if valid_df.empty:
        print("non_data。")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.hist(valid_df['I2'], bins=20, color='skyblue', edgecolor='black', alpha=0.7)
    ax1.axvline(valid_df['I2'].mean(), color='red', linestyle='--', linewidth=2, 
                label=f'Mean: {valid_df["I2"].mean():.1f}%')
    ax1.axvline(valid_df['I2'].median(), color='green', linestyle='--', linewidth=2, 
                label=f'Median: {valid_df["I2"].median():.1f}%')
    ax1.set_xlabel('I² (%)', fontsize=19)
    ax1.set_ylabel('Frequency', fontsize=19)
    ax1.set_title('Distribution of I² Statistics', fontsize=21, fontweight='bold')
    ax1.legend(frameon=False, handlelength=0.5)
    ax1.grid(True, alpha=0.3)
    level_counts = valid_df['Heterogeneity_level'].value_counts()
    colors_pie = ['#2ECC71', '#F39C12', '#E67E22', '#E74C3C']
    ax2.pie(level_counts.values, labels=level_counts.index, autopct='%1.1f%%',
            colors=colors_pie[:len(level_counts)], startangle=90, textprops={'fontsize': 17})
    ax2.set_title('Heterogeneity Level Distribution', fontsize=21, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=1000, bbox_inches='tight')
    print(f"save: {output_path}")
    plt.close()


def generate_report(heterogeneity_df, output_file='heterogeneity_report.txt'):
    valid_df = heterogeneity_df.dropna(subset=['I2', 'Q_pvalue'])
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("异质性分析报告 - Cochran's Q and I²\n")
        f.write("=" * 80 + "\n\n")
        
        # 1. 总体摘要
        f.write("1. 总体摘要\n")
        f.write("-" * 80 + "\n")
        f.write(f"分析变量总数: {len(heterogeneity_df)}\n")
        f.write(f"有效分析数: {len(valid_df)}\n")
        f.write(f"平均I²值: {valid_df['I2'].mean():.2f}%\n")
        f.write(f"I²中位数: {valid_df['I2'].median():.2f}%\n")
        f.write(f"I²范围: {valid_df['I2'].min():.2f}% - {valid_df['I2'].max():.2f}%\n\n")
        
        # 2. 异质性水平分布
        f.write("2. 异质性水平分布\n")
        f.write("-" * 80 + "\n")
        level_counts = valid_df['Heterogeneity_level'].value_counts()
        for level, count in level_counts.items():
            pct = count / len(valid_df) * 100
            f.write(f"  {level}: {count} ({pct:.1f}%)\n")
        f.write("\n")
        
        # 3. Q检验结果
        f.write("3. Cochran's Q Test Results\n")
        f.write("-" * 80 + "\n")
        significant_q = valid_df[valid_df['Q_pvalue'] < 0.10]
        f.write(f"Number of variables with significant Q test (p < 0.10): {len(significant_q)} / {len(valid_df)}\n")
        f.write(f"Proportion: {len(significant_q) / len(valid_df) * 100:.1f}%\n\n")
        
        if len(significant_q) > 0:
            f.write("List of variables with significant Q test:\n")
            for idx, row in significant_q.iterrows():
                f.write(f"  - {row['Variable']}: p = {row['Q_pvalue']:.4f}, I² = {row['I2']:.1f}%\n")
        else:
            f.write("No variables with significant Q test (p < 0.10)\n")
        f.write("\n")

        f.write("4. High Heterogeneity Variables (I² > 60%)\n")
        f.write("-" * 80 + "\n")
        high_hetero = valid_df[valid_df['I2'] > 60]
        if len(high_hetero) > 0:
            f.write(f"Detected {len(high_hetero)} high heterogeneity variables:\n\n")
            for idx, row in high_hetero.iterrows():
                f.write(f"Variable: {row['Variable']}\n")
                f.write(f"  I²: {row['I2']:.1f}%\n")
                f.write(f"  Q: {row['Q']:.2f}\n")
                f.write(f"  P-value: {row['Q_pvalue']:.4f}\n")
                f.write(f"  tau²: {row['tau2']:.4f}\n")
                f.write(f"  Heterogeneity Level: {row['Heterogeneity_level']}\n\n")
        else:
            f.write("No variables with I² greater than 60%\n\n")

        f.write("5. Detailed Results for All Variables\n")
        f.write("-" * 80 + "\n")
        f.write(valid_df[['Variable', 'N', 'N_groups', 'Q', 'df', 'Q_pvalue', 'I2', 'Heterogeneity_level']].to_string(index=False))
        f.write("\n\n")

        f.write("6. Interpretation of Results\n")
        f.write("-" * 80 + "\n")
        f.write("Cochran's Q Test:\n")
        f.write("  - Null hypothesis: All studies have the same true effect\n")
        f.write("  - P < 0.10: Reject null hypothesis, significant heterogeneity present\n")
        f.write("  - P ≥ 0.10: Do not reject null hypothesis, no significant heterogeneity\n\n")
        f.write("I² Statistic Interpretation:\n")
        f.write("  - 0-40%: Heterogeneity may not be important\n")
        f.write("  - 40-60%: Moderate heterogeneity\n")
        f.write("  - 60-75%: Substantial heterogeneity\n")
        f.write("  - >75%: Considerable heterogeneity\n\n")
        
        f.write("=" * 80 + "\n")
    
    print(f"✓ Heterogeneity analysis report saved to: {output_file}")


def main():
    """
    Main function: Execute the complete heterogeneity analysis workflow
    """
    file_path = r'D:\data.xlsx'
    
    print("=" * 80)
    print("Heterogeneity Analysis - Cochran's Q Test and I² Statistic")
    print("=" * 80)
    try:
        df = pd.read_excel(file_path)
        print(f"✓ Data loaded successfully!")
        print(f"  - Data dimensions: {df.shape}")
        print(f"  - Columns: {', '.join(df.columns.tolist())}")
    except Exception as e:
        print(f"✗ Failed to read data file: {e}")
        return
    type_cols = ['type_Ag', 'type_CuO', 'type_Fe2O3', 'type_ZnO', 'type_TiO2']
    existing_type_cols = [col for col in type_cols if col in df.columns]
    
    if existing_type_cols:
        print("\n✓ Found type columns, proceeding to merge into 'TYPE' column.")
        def get_material_type(row):
            for col in existing_type_cols:
                if row[col] == 1:
                    return col.replace('type_', '')
            return 'Other'
        
        df['TYPE'] = df.apply(get_material_type, axis=1)
        print(f"✓ 'TYPE' column created successfully, categories: {df['TYPE'].unique().tolist()}")
        
        # Remove original one-hot encoded columns
        df = df.drop(columns=existing_type_cols)
        print(f"✓ Removed original type columns: {', '.join(existing_type_cols)}")
    else:
        print("\n⚠ No type columns found, skipping merge step")
    outcome_col = 'Deni'
    
    if outcome_col not in df.columns:
        print(f"✗ Error: Outcome variable '{outcome_col}' not found in data")
        return

    print("\n" + "=" * 80)
    print("Starting heterogeneity analysis...")
    print("=" * 80)
    
    heterogeneity_results = perform_heterogeneity_analysis_all_predictors(df, outcome_col)

    output_csv = 'heterogeneity_analysis_results.csv'
    heterogeneity_results.to_csv(output_csv, index=False)
    print(f"\n✓ Heterogeneity analysis results saved to: {output_csv}")

    print("\n" + "=" * 80)
    print("Heterogeneity Analysis Summary")
    print("=" * 80)
    
    valid_results = heterogeneity_results.dropna(subset=['I2'])
    print(f"\nTotal variables: {len(heterogeneity_results)}")
    print(f"Valid analyses: {len(valid_results)}")
    if len(valid_results) > 0:
        print(f"Average I²: {valid_results['I2'].mean():.2f}%")
        print(f"Median I²: {valid_results['I2'].median():.2f}%")
        print(f"Average number of groups: {valid_results['N_groups'].mean():.1f}")

    print("\nHeterogeneity level distribution:")
    level_counts = heterogeneity_results['Heterogeneity_level'].value_counts()
    for level, count in level_counts.items():
        print(f"  {level}: {count}")

    significant_q = heterogeneity_results[heterogeneity_results['Q_pvalue'] < 0.10]
    print(f"\nVariables with significant Q test (p < 0.10): {len(significant_q)} / {len(heterogeneity_results)}")
    
    if len(significant_q) > 0:
        print("\nVariables with significant Q test:")
        print(significant_q[['Variable', 'Q_pvalue', 'I2', 'Heterogeneity_level']].to_string(index=False))
    high_hetero = heterogeneity_results[heterogeneity_results['I2'] > 60]
    if len(high_hetero) > 0:
        print(f"\nHigh heterogeneity variables (I² > 60%): {len(high_hetero)}")
        print(high_hetero[['Variable', 'I2', 'Q_pvalue', 'Heterogeneity_level']].to_string(index=False))
    print("\n" + "=" * 80)
    print("Generating visualizations...")
    print("=" * 80)
    
    plot_heterogeneity_summary(heterogeneity_results, output_path='heterogeneity_summary.png')
    plot_heterogeneity_scatter(heterogeneity_results, output_path='heterogeneity_scatter.png')
    plot_i2_distribution(heterogeneity_results, output_path='i2_distribution.png')
    print("\n" + "=" * 80)
    print("Generating report...")
    print("=" * 80)
    generate_report(heterogeneity_results, output_file='heterogeneity_report.txt')
    print("\n" + "=" * 80)
    print("Heterogeneity analysis completed!")
    print("=" * 80)
    print("\nGenerated files:")
    print("  1. heterogeneity_analysis_results.csv - Detailed analysis results")
    print("  2. heterogeneity_summary.png - I² and Q test summary plot")
    print("  3. heterogeneity_scatter.png - I² vs P-value scatter plot")
    print("  4. i2_distribution.png - I² distribution plot")
    print("  5. heterogeneity_report.txt - Text report")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
