#标准的过滤特征的方法
'''
Boruta方法

'''
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

# Load data
data = pd.read_csv(r"\\108-pc\INDEXM\DOC\data1\Train_101_0_1.csv", header=0)

# Select features and target
feature_names = [i for i in data.columns if i not in ["id", "o_date", "o_time", "idx1", "profit_loss", "END_date", "end_time", "wgt", "GBIE", "order", "buy_clr"]]

 
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


X = data[feature_names]

y = data["target"]

y1 = data["profit_loss"]

missing_ratio = X.isnull().sum() / len(X)
to_drop = missing_ratio[missing_ratio > 0.00].index
X.drop(to_drop, axis=1, inplace=True)   

# Split the data into train, test, and validation sets while preserving indices
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

feat_selector = BorutaPy(RandomForestClassifier())

X_train = X_train.values
y_train = y_train.values
 
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(
   n_estimators=100,
   max_depth=5,
   n_jobs=-1
)

feat_selector = BorutaPy(
   estimator=rf,
   max_iter=3  
)
 
feat_selector.fit(X_train, y_train)
 

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)


selected = feat_selector.support_weak_
cols = X.columns
selected_cols = cols[selected]

feature_index = ["id", "o_date", "o_time", "idx1", "profit_loss", "END_date", "end_time", "wgt", "GBIE", "order", "buy_clr",'target']
 

feature_names_all = selected_cols.union(feature_index)

 
data_train = data[feature_names_all]

data_train.to_csv("H:\\indexm\\CTAHOT\\data\\filter_data_train_201.csv", index=False)


