#标准的过滤特征的方法
'''
二、	变量筛选方法
1.	变量筛选指标
•	信息价值(IV)
•	信息增益(IG)
•	卡方(CHI-SQ)
•	单变量显著性
•	偏相关分析
2.	变量筛选过程
1)	计算每个候选变量的信息价值，并找出信息价值最大的300个变量及后续500个变量。
2)	计算每个候选变量的信息增益，并找出信息增益最大的300个变量及后续500个变量。
3)	计算每个候选变量的卡方分布值，并找出卡方值最大的300个变量及后续500个变量。
4)	计算每个候选变量的单变量显著性，并找出显著性水平最大的300个变量及后续500个变量。
5)	计算每个候选变量的偏相关系数，并找出偏相关系数绝对值最大的300个变量及后续500个变量。
6)	将5类指标分别选出的300个变量合并起来，取其并集。
7)	将5类指标分别选出的后续500个变量合并起来，取其交集。
8)	将6）、7）形成的变量集合并起来，并对其进行偏相关分析。
9)	根据偏相关分析的结果，删除明显不具有贡献的变量，剩下的变量集合即为模型最终候选变量集。

'''
import pandas as pd
from sklearn.feature_selection import chi2, mutual_info_classif, f_classif, SelectKBest
from sklearn.preprocessing import LabelEncoder
from scipy import stats
import numpy as np
import concurrent.futures
from sklearn.feature_selection import mutual_info_classif, f_classif
from scipy.stats import chi2_contingency, pearsonr

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


X = data[feature_names]

X = X.replace(np.nan, 0)


y = data["target"]

# 1)信息价值
 
 
# 定义函数计算信息价值(IV)
def calculate_iv(data, feature, target):
    iv_values = mutual_info_classif(data[feature].values.reshape(-1, 1), target)
    return feature, iv_values[0]

# 定义函数计算信息增益(IG)
def calculate_ig(data, feature, target):
    ig_values = f_classif(data[feature].values.reshape(-1, 1), target)
    return feature, ig_values[0][0]

# 定义函数计算卡方值(CHI-SQ)
def calculate_chi_sq(data, feature, target):
    contingency_table = pd.crosstab(data[feature], target)
    chi2, _, _, _ = chi2_contingency(contingency_table)
    return feature, chi2

# 定义函数计算单变量显著性
def calculate_univariate_significance(data, feature, target):
    _, p_value = f_classif(data[feature].values.reshape(-1, 1), target)
    return feature, p_value[0]

# 定义函数计算偏相关系数
def calculate_partial_correlation(data, feature, target):
    partial_corr, _ = pearsonr(data[feature],target)
    return feature, abs(partial_corr)



# 并行计算信息价值
with concurrent.futures.ThreadPoolExecutor() as executor:
    iv_scores = list(executor.map(lambda f: calculate_iv(X, f, y), feature_names))

# 并行计算信息增益
with concurrent.futures.ThreadPoolExecutor() as executor:
    ig_scores = list(executor.map(lambda f: calculate_ig(X, f, y), feature_names))

# 并行计算卡方值
with concurrent.futures.ThreadPoolExecutor() as executor:
    chi_sq_scores = list(executor.map(lambda f: calculate_chi_sq(X, f, y), feature_names))

# 并行计算单变量显著性
with concurrent.futures.ThreadPoolExecutor() as executor:
    significance_scores = list(executor.map(lambda f: calculate_univariate_significance(X, f, y), feature_names))

# 并行计算偏相关系数
with concurrent.futures.ThreadPoolExecutor() as executor:
    partial_corr_scores = list(executor.map(lambda f: calculate_partial_correlation(X, f, y), feature_names))

# 按分数降序排序
iv_scores.sort(key=lambda x: x[1], reverse=True)
ig_scores.sort(key=lambda x: x[1], reverse=True)
chi_sq_scores.sort(key=lambda x: x[1], reverse=True)
significance_scores.sort(key=lambda x: x[1], reverse=True)
partial_corr_scores.sort(key=lambda x: x[1], reverse=True)

firstnum=100
nextnum=100

# 取前300个特征
top_iv_features = [feature for feature, _ in iv_scores[:firstnum]]
top_ig_features = [feature for feature, _ in ig_scores[:firstnum]]
top_chi_sq_features = [feature for feature, _ in chi_sq_scores[:firstnum]]
top_significance_features = [feature for feature, _ in significance_scores[:firstnum]]
top_partial_corr_features = [feature for feature, _ in partial_corr_scores[:firstnum]]

# 取后续500个特征
next_iv_features = [feature for feature, _ in iv_scores[firstnum:nextnum+firstnum]]
next_ig_features = [feature for feature, _ in ig_scores[firstnum:nextnum+firstnum]]
next_chi_sq_features = [feature for feature, _ in chi_sq_scores[firstnum:nextnum+firstnum]]
next_significance_features = [feature for feature, _ in significance_scores[firstnum:nextnum+firstnum]]
next_partial_corr_features = [feature for feature, _ in partial_corr_scores[firstnum:nextnum+firstnum]]

# 取并集和交集
selected_features_union = set(top_iv_features) | set(top_ig_features) | set(top_chi_sq_features) | set(top_significance_features) | set(top_partial_corr_features)
selected_features_intersection = set(next_iv_features) & set(next_ig_features) & set(next_chi_sq_features) & set(next_significance_features) & set(next_partial_corr_features)

# 偏相关分析
final_selected_features = selected_features_union | selected_features_intersection
final_data = data[list(final_selected_features)]

# 输出最终的候选变量集
print(final_data.head())












 




