# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 14:18:03 2024

@author: model
#wrapper


"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import RFE

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



def forward_feature_selection(data, target_column, num_features_to_select):
    # 选择特征和目标
    X = data[feature_names]
    
    X = X.replace(np.nan, 0)
    
    
    y = data["target"]
    
    # 1)信息价值
     
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # 划分训练集和测试集
 
 
    # 使用随机森林作为基模型
    model = RandomForestClassifier()

    # 初始化 RFE
    rfe = RFE(model, n_features_to_select=num_features_to_select)  # 最终选择的特征数量

    # 开始逐步添加特征
    selected_features = []
    for i in range(num_features_to_select):
        # 训练模型并计算特征的重要性
        rfe.fit(X_train, y_train)

        # 获取当前迭代中最重要的特征的索引
        most_important_index = max(range(len(rfe.support_)), key=lambda i: rfe.support_[i])

        # 添加最重要的特征
        added_feature = X_train.columns[most_important_index]
        selected_features.append(added_feature)
        X_train = X_train.drop(added_feature, axis=1)
        X_test = X_test.drop(added_feature, axis=1)

        # 输出当前迭代的选择的特征和模型性能
        print(f"Iteration {i + 1}: Added Feature - {added_feature}")

        # 重新拟合模型
        model.fit(X_train, y_train)

        # 输出模型的特征重要性（可选）
        print(f"Feature Importance: {model.feature_importances_}")

        # 预测
        y_pred = model.predict(X_test)

        # 评估模型性能
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Iteration {i + 1}: Accuracy - {accuracy}")

    print("Final Selected Features:", selected_features)

    return selected_features

# 示例用法

target_column = "target"  # 请替换为实际的目标列名
num_features_to_select = 200  # 请替换为实际的最终选择的特征数量

final_selected_features = forward_feature_selection(data, target_column, num_features_to_select)
