import tensorflow as tf
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
#数据预处理阶段
missing_values = ["n/a", "na", "--"] #指定空数据类型,将"n/a", "na", "--"指定为NaN类型
x_data = pd.read_excel('相关性分析数据.xlsx', na_values = missing_values)  # 加载数据
x_data.drop(['序号', '县','市'], axis=1, inplace=True)   # 去掉无用的列字段
print (x_data.index[np.where(x_data.isnull())[0]])      #打印出NaN的行数
print (x_data.columns[np.where(x_data.isnull())[1]])    #打印出NaN出现的指标,这两句话可以查找出现NaN的具体位置方便更改。
#x_data.dropna(axis=0, inplace=True)                 #删除包含空字段的行，axis：默认为 0，表示逢空值剔除整行，如果设置参数 axis＝1 表示逢空值去掉整列。 inplace：如果设置 True，将计算得到的值直接覆盖之前的值并返回 None，修改的是源数据。
# 把NaN替换为0    df.fillna(0, inplace=True)
# 检查数据集中是否存在缺失数据
print(x_data.isnull().sum())  #检查缺失数据
x_data.fillna(0, inplace=True) #用零填充缺失数据
x_data.to_excel('0填充数据.xls')
x_data = x_data.apply(lambda x: ((x - np.min(x)) / (np.max(x) - np.min(x))))  #数据归一化
x_data.to_excel('归一化值1.xls')

result1 = x_data.corr()
result1.to_excel('指标相关性.xls')
#维度的相关关系图
#pd.plotting.scatter_matrix(x_data, figsize=(12,12),range_padding=0.5)
#plt.show()
#热力图
#figure, ax = plt.subplots(figsize=(12, 12))
#sns.heatmap(result1, square=True, annot=True, ax=ax)
#plt.show()