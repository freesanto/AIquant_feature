#wrapper
'''
 

'''
import pandas as pd
import numpy as np
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.metrics import accuracy_score
from joblib import Parallel, delayed

data = pd.read_csv(r"\\108-pc\INDEXM\DOC\data1\Train_101_0_1.csv", header=0)
   

feature_names = [i for i in data.columns if i not in ["id", "order", "o_date", "o_time", "idx1", "profit_loss", "END_date", "end_time", "wgt", "target",  "buy_clr"]]

 

# Impute missing values with mean
#for column in feature_names:
#    mean_value = data[column].mean()
#    data[column].fillna(mean_value, inplace=True)


# 提取"id", "date", "mi_time"三个特征，并按照它们排序（从小到大）
sort_columns = ["id", "o_date", "o_time"]
sorted_data = data.sort_values(by=sort_columns)

# 取前30个数据的平均值
mean_values = sorted_data.head(30)[feature_names].mean()

# 使用平均值填充缺失值
for column in feature_names:
    data[column].fillna(mean_values[column], inplace=True)

   
data.dropna(subset=['profit_loss'], how='any', inplace=True)


data['target'] = (data['profit_loss'] > 0).astype(int)

feature_names_with_profit_loss = feature_names + ['target']






def process_iteration(X_train, y_train, X_test,y_test, num_features_to_select):
    model = RandomForestClassifier()
    rfe = RFE(model, n_features_to_select=num_features_to_select)

    rfe.fit(X_train, y_train)
    least_important_index = min(range(len(rfe.support_)), key=lambda i: rfe.support_[i])
    removed_feature = X_train.columns[least_important_index]
    X_train = X_train.drop(removed_feature, axis=1)
    X_test = X_test.drop(removed_feature, axis=1)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    return X_train, X_test, removed_feature, accuracy

def feature_selection_with_rfe_parallel(data, target_column, num_features_to_select):
    # 选择特征和目标
   
    X = data[feature_names]
    
    X = X.replace(np.nan, 0)
    
    
    y = data["target"]
    
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # 划分训练集和测试集
 
    # 开始逐步迭代
    for i in range(X_train.shape[1], num_features_to_select, -1):  # 从初始特征数到选择的特征数量
        # 并行处理每个迭代步骤
        results = Parallel(n_jobs=48)(delayed(process_iteration)(X_train.copy(), y_train.copy(), X_test.copy(),y_test.copy(), num_features_to_select) for _ in range(10))

        # 选择具有最佳性能的结果
        best_result = max(results, key=lambda x: x[3])

        # 解包结果
        X_train, X_test, removed_feature, accuracy = best_result

        # 输出当前迭代的选择的特征和模型性能
        selected_features = X.columns[X_train.columns]
        print(f"Iteration {X_train.shape[1]}: Selected Features - {selected_features}, Removed Feature - {removed_feature}, Accuracy - {accuracy}")

    final_selected_features = X.columns[X_train.columns]
    print("Final Selected Features:", final_selected_features)

    return final_selected_features

# 示例用法
 
num_features_to_select = 305  # 请替换为实际的最终选择的特征数量

final_selected_features = feature_selection_with_rfe_parallel(data, "target", num_features_to_select)

    