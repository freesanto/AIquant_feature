# -*- coding: utf-8 -*-
"""
Created on Sun Dec 17 20:55:04 2023

@author: model
"""
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score, accuracy_score
import numpy as np
import matplotlib.pyplot as plt

 
from sklearn.model_selection import train_test_split
from boruta import BorutaPy  
from sklearn.ensemble import RandomForestClassifier
 
import os
import joblib
# Load data
      
data = pd.read_csv(r"\\108-pc\INDEXM\DOC\data2\Adata_101_1_1.csv", header=0)
 
    

feature_names = [i for i in data.columns if i not in ["id", "o_date", "o_time", "idx1", "profit_loss", "END_date", "end_time", "wgt", "GBIE", "order", "buy_clr"]]

 

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
 

feature_names_with_profit_loss = feature_names + ['profit_loss',"id", "o_date", "o_time"]

X = data[feature_names_with_profit_loss]

y = data["target"]
 
 
X.o_date = pd.to_datetime(X.o_date, format='%Y.%m.%d')
 
 

F1 ='N6903'
X1 = X[['profit_loss',"id", "o_date", "o_time", F1]]

# 先处理特征F1的缺失值，随机放在第5和第6的位置
df = X1
 

# 按照日期和特征F1进行排序
#df.sort_values(by=['o_date', F1], inplace=True)
# 按照日期和特征F1进行排序（特征排序从高到低）
df.sort_values(by=['o_date', F1], inplace=True, ascending=[True, False])

# 计算每天特征F1的排名
df['F1_rank'] = df.groupby('o_date')[F1].rank(method='first')
#df['F1_rank'] = df.groupby('o_date')[F1].rank(method='dense', ascending=False) + np.random.uniform(low=0, high=0.01, size=len(df)) * 0.1


# 划分每天数据为10等份
df['F1_group'] = df.groupby('o_date')['F1_rank'].transform(lambda x: pd.qcut(x, q=10, labels=False, precision=6))

# 提取第1和第2等份的数据
df_selected_group = df[df['F1_group'].isin([0])]

# 按照o_date每天汇总profit_loss
daily_profit_loss = df_selected_group.groupby('o_date')['profit_loss'].sum().reset_index()

# 计算每天第1和第2等份的累积profit_loss
daily_profit_loss['cum_profit_loss'] = daily_profit_loss['profit_loss'].cumsum()

# 作图
plt.figure(figsize=(10, 6))
plt.plot(daily_profit_loss['o_date'], daily_profit_loss['cum_profit_loss'], marker='o', linestyle='-', markersize=3)

plt.xlabel('Date')
plt.ylabel('Cumulative Profit Loss')
plt.title('Cumulative Profit Loss for the 1st and 2nd Groups')
plt.show()
