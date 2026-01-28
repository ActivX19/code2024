import numpy as np 
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import ShuffleSplit, train_test_split, RandomizedSearchCV, cross_val_score, cross_validate
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import time

start_time = time.time()
data_list=pd.read_excel(r'D:\Desk\A\0731_de_data.xlsx', header=0)
X=data_list.iloc[:,:-1]                                                                                                                                                                                                                                                                        
y =data_list.iloc[:,-1] 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=65)
cv_split = ShuffleSplit(n_splits=5, random_state=40)

param_grid = dict(
    n_estimators=[i for i in range(10,1000,1)],
    max_depth=[i for i in range(3,8,1)],
    max_features=[i for i in range(3,8,1)],
    min_samples_leaf=[i for i in range(1,5,1)],
    min_samples_split=[i for i in range(2,5,1)],)

def count_space(param):
    no_option =1
    for i in param_grid:
        no_option*=len(param_grid[i])
    print(no_option)
count_space(param_grid)

rfc=RandomForestRegressor(random_state=90)
model= RandomizedSearchCV(estimator=rfc,
                            param_distributions=param_grid,
                            n_iter=200,
                            scoring='neg_mean_squared_error',
                            cv=cv_split,
                            random_state=90)
model.fit(X_train,y_train)
best_model=model.best_estimator_

# 输出五次交叉验证的具体数值
print("\n=== 五次交叉验证详细结果 ===")
cv_results = model.cv_results_
best_idx = model.best_index_

# 获取最佳参数对应的交叉验证分数
best_cv_scores = []
for i in range(5):  # 5折交叉验证
    score_key = f'split{i}_test_score'
    if score_key in cv_results:
        score = cv_results[score_key][best_idx]
        # 转换为正值（因为使用的是neg_mean_squared_error）
        mse = -score
        rmse = np.sqrt(mse)
        best_cv_scores.append(mse)
        print(f'第{i+1}折交叉验证 - MSE: {mse:.6f}, RMSE: {rmse:.6f}')

# 计算交叉验证的统计信息
cv_mse_mean = np.mean(best_cv_scores)
cv_mse_std = np.std(best_cv_scores)
cv_rmse_mean = np.sqrt(cv_mse_mean)
cv_rmse_std = np.sqrt(cv_mse_std)

print(f'\n交叉验证MSE - 平均值: {cv_mse_mean:.6f} ± {cv_mse_std:.6f}')
print(f'交叉验证RMSE - 平均值: {cv_rmse_mean:.6f} ± {cv_rmse_std:.6f}')

# 最佳交叉验证分数
print(f'最佳交叉验证分数 (负MSE): {model.best_score_:.6f}')
print(f'最佳交叉验证MSE: {-model.best_score_:.6f}')
print(f'最佳交叉验证RMSE: {np.sqrt(-model.best_score_):.6f}')

# 计算每折的R²值
print("\n=== 五次交叉验证R²详细结果 ===")
best_params = model.best_params_
best_rf = RandomForestRegressor(**best_params, random_state=90)

# 使用cross_val_score计算R²
r2_scores = cross_val_score(best_rf, X_train, y_train, cv=cv_split, scoring='r2')

for i, r2 in enumerate(r2_scores):
    print(f'第{i+1}折交叉验证 - R²: {r2:.6f}')

# 计算R²的统计信息
r2_mean = np.mean(r2_scores)
r2_std = np.std(r2_scores)
print(f'\n交叉验证R² - 平均值: {r2_mean:.6f} ± {r2_std:.6f}')

# 使用cross_val_score计算MAE
print("\n=== 五次交叉验证MAE详细结果 ===")
mae_scores = cross_val_score(best_rf, X_train, y_train, cv=cv_split, scoring='neg_mean_absolute_error')

for i, mae in enumerate(mae_scores):
    mae_positive = -mae  # 转换为正值
    print(f'第{i+1}折交叉验证 - MAE: {mae_positive:.6f}')

# 计算MAE的统计信息
mae_mean = np.mean(-mae_scores)  # 转换为正值
mae_std = np.std(-mae_scores)
print(f'\n交叉验证MAE - 平均值: {mae_mean:.6f} ± {mae_std:.6f}')

print("\n=== 最终模型结果 ===")

y_test_predict = best_model.predict(X_test)
y_train_predict = best_model.predict(X_train)
print('best model：', model.best_params_)
print('prediction_y', y_test_predict )
print('trainMSE:', mean_squared_error(y_train, y_train_predict))
print('trainRMSE:', np.sqrt(mean_squared_error(y_train, y_train_predict)))
print('trainMAE:', mean_absolute_error(y_train, y_train_predict))
print('train-r2', r2_score(y_train, y_train_predict))
print('testMSE:', mean_squared_error(y_test, y_test_predict))
print('testRMSE:', np.sqrt(mean_squared_error(y_test, y_test_predict)))
print('testMAE:', mean_absolute_error(y_test, y_test_predict))
print('test-r2', r2_score(y_test, y_test_predict))


end_time = time.time()
run_time = end_time - start_time
print('Run time:', run_time)
