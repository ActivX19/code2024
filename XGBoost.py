import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split, ShuffleSplit, cross_val_score
from xgboost import XGBRegressor as XGBR
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import RandomizedSearchCV
from time import time

time0=time()
data_list=pd.read_excel(r'D:\data.xlsx', header=0)
data_list.head()
X=data_list.iloc[:,:-1]                                                                                                                                                                                                                                                                        
y =data_list.iloc[:,-1]
 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=21)

cv_split = ShuffleSplit(n_splits=5, random_state=40)
reg = XGBR(random_state=40)
params = dict(
    n_estimators=[i for i in range(1,300,1)], 
    max_depth=[i for i in range(1,9,1)],     
    min_child_weight=[i for i in range(1,9,1)], 
    gamma=[i/10.0 for i in range(1,9,1)], 
    subsample=[i/10.0 for i in range(1,9,1)],      
    colsample_bytree=[i/10.0  for i in range(1,9,1)],
    learning_rate=[i/10.0 for i in range(1,9,1)],)  

def count_space(param):
    no_option =1
    for i in params:
        no_option*=len(params[i])
    print(no_option)
count_space(params)

model= RandomizedSearchCV(estimator=reg,
                            param_distributions=params,
                            n_iter=200,
                            scoring='neg_mean_squared_error',
                            cv=cv_split,
                            random_state=40)
model.fit(X_train, y_train)

print("\n=== Detailed results of five-fold cross-validation ===")
cv_results = model.cv_results_
best_idx = model.best_index_

best_cv_scores = []
for i in range(5):  
    score_key = f'split{i}_test_score'
    if score_key in cv_results:
        score = cv_results[score_key][best_idx]
        mse = -score
        rmse = np.sqrt(mse)
        best_cv_scores.append(mse)
        print(f'{i+1} - MSE: {mse:.6f}, RMSE: {rmse:.6f}')

cv_mse_mean = np.mean(best_cv_scores)
cv_mse_std = np.std(best_cv_scores)
cv_rmse_mean = np.sqrt(cv_mse_mean)
cv_rmse_std = np.sqrt(cv_mse_std)

print(f'\nCross-validation MSE - Mean: {cv_mse_mean:.6f} ± {cv_mse_std:.6f}')
print(f'Cross-validation RMSE - Mean: {cv_rmse_mean:.6f} ± {cv_rmse_std:.6f}')

print(f'Best cross-validation score (negative MSE): {model.best_score_:.6f}')
print(f'Best cross-validation MSE: {-model.best_score_:.6f}')
print(f'Best cross-validation RMSE: {np.sqrt(-model.best_score_):.6f}')

print("\n=== Final results ===")
best_params = model.best_params_
best_xgb = XGBR(**best_params, random_state=40)

r2_scores = cross_val_score(best_xgb, X_train, y_train, cv=cv_split, scoring='r2')

for i, r2 in enumerate(r2_scores):
    print(f'{i+1} - R²: {r2:.6f}')

r2_mean = np.mean(r2_scores)
r2_std = np.std(r2_scores)
print(f'\nCross-validation R² - Mean: {r2_mean:.6f} ± {r2_std:.6f}')

print("\n=== Five-fold cross-validation MAE detailed results ===")
mae_scores = cross_val_score(best_xgb, X_train, y_train, cv=cv_split, scoring='neg_mean_absolute_error')

for i, mae in enumerate(mae_scores):
    mae_positive = -mae  
    print(f'{i+1} - MAE: {mae_positive:.6f}')
mae_mean = np.mean(-mae_scores)  
mae_std = np.std(-mae_scores)
print(f'\nCross-validation MAE - Mean: {mae_mean:.6f} ± {mae_std:.6f}')

best_model=model.best_estimator_
best_score=model.best_score_
y_test_predict = best_model.predict(X_test)
y_train_predict = best_model.predict(X_train)

print('\n=== Final Results ===')
print('best_model:', best_model)
print('best_params:', model.best_params_)
print('trainMSE:', mean_squared_error(y_train, y_train_predict))
print('trainRMSE:', np.sqrt(mean_squared_error(y_train, y_train_predict)))
print('trainMAE:', mean_absolute_error(y_train, y_train_predict))
print('testMSE:', mean_squared_error(y_test, y_test_predict))
print('testRMSE:', np.sqrt(mean_squared_error(y_test, y_test_predict)))
print('testMAE:', mean_absolute_error(y_test, y_test_predict))
print('train-r2', r2_score(y_train, y_train_predict))
print('test-r2', r2_score(y_test, y_test_predict))
print('time:',time()-time0)





