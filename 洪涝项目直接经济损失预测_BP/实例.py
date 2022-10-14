import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model

#数据预处理阶段
missing_values = ["n/a", "na", "--"] #指定空数据类型,将"n/a", "na", "--"指定为NaN类型
x_data = pd.read_excel('20市潜在特征第一组.xlsx', na_values = missing_values)  # 加载数据
x_data.drop(['县代码（灾情上报系统）', '县代码（普查系统）', '区域代码', '区域'], axis=1, inplace=True)   # 去掉无用的列字段
print (x_data.index[np.where(x_data.isnull())[0]])      #打印出NaN的行数
print (x_data.columns[np.where(x_data.isnull())[1]])    #打印出NaN出现的指标,这两句话可以查找出现NaN的具体位置方便更改。
x_data.dropna(axis=0, inplace=True)                 #删除包含空字段的行，axis：默认为 0，表示逢空值剔除整行，如果设置参数 axis＝1 表示逢空值去掉整列。 inplace：如果设置 True，将计算得到的值直接覆盖之前的值并返回 None，修改的是源数据。
# 把NaN替换为0    df.fillna(0, inplace=True)
# 检查数据集中是否存在缺失数据
print(x_data.isnull().sum())  #检查缺失数据
x_data.fillna(0, inplace=True) #用零填充缺失数据

#加载模型（在test中有示例）
new_model = load_model('model_BP01.h5')
print(new_model.summary()) #显示模型结构
pred = new_model.predict(x_data)
print(pred)
pred = pd.DataFrame(pred)
pred.to_excel('20市经济损失预测值第一组.xls')