# -*- coding:utf-8 -*-
#加载模块
import pandas as pd
import numpy as np
import altair as alt
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras import Model
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
alt.renderers.enable('altair_viewer')   #让altair模块适用于pycharm
import copy

# 10指标（100） 降维  2列


missing_values = ["n/a", "na", "--"] #指定空数据类型,将"n/a", "na", "--"指定为NaN类型
# 加载数据
data = pd.read_excel('综合数据1.xlsx', na_values = missing_values)
print (data.index[np.where(data.isnull())[0]])      #打印出NaN的行数
print (data.columns[np.where(data.isnull())[1]])    #打印出NaN出现的指标,这两句话可以查找出现NaN的具体位置方便更改。
data.drop(['行政区划（管理区划）','行政区划（标准区划）','行政区划（2021普查）','县', '市','序号','县', '市'], axis=1, inplace=True)   # 去掉无用的列字段
# data.dropna(axis=0, inplace=True)                 #删除包含空字段的行，axis：默认为 0，表示逢空值剔除整行，如果设置参数 axis＝1 表示逢空值去掉整列。 inplace：如果设置 True，将计算得到的值直接覆盖之前的值并返回 None，修改的是源数据。
# 把NaN替换为0    df.fillna(0, inplace=True)
# 检查数据集中是否存在缺失数据
print(data.isnull().sum())  #检查缺失数据
data.fillna(0, inplace=True) #用零填充缺失数据
data1 = copy.deepcopy(data)
#将所有特征的值都被重新调整到 [0, 1] 的范围
numeric_columns = data.columns.values.tolist()  #columns按照纵列归一化
scaler = MinMaxScaler()
data[numeric_columns] = scaler.fit_transform(data[numeric_columns])
#print(data.head())  #显示前五行，已经完成归一化
#data.to_excel('excel_data1.xls')  #导出为excel

#自编码器网络构建
input_df = Input( shape = (10, ))   #10个指标输入层为10层
x = Dense(10, activation = 'relu', name='input_layer_1')(input_df)
x = Dense(20, activation = 'relu', kernel_initializer='glorot_uniform')(x)
x = Dense(40, activation = 'relu', kernel_initializer='glorot_uniform')(x)
x = Dense(20, activation = 'relu', kernel_initializer='glorot_uniform')(x)
x = Dense(5, activation = 'relu', kernel_initializer='glorot_uniform')(x)

encoded = Dense(2, activation = 'relu', kernel_initializer='glorot_uniform', name='potential_layer')(x)

x = Dense(5, activation = 'relu', kernel_initializer='glorot_uniform')(encoded)
x = Dense(10, activation = 'relu', kernel_initializer='glorot_uniform')(x)
x = Dense(40, activation = 'relu', kernel_initializer='glorot_uniform')(x)
x = Dense(20, activation = 'relu', kernel_initializer='glorot_uniform')(x)
x = Dense(20, activation = 'relu', kernel_initializer='glorot_uniform')(x)
decoded = Dense(10, kernel_initializer='glorot_uniform', name='output_layer_1')(x)

autoencoder = Model(input_df, decoded)
encoder = Model(input_df, encoded)

autoencoder.compile(optimizer = 'adam', loss = 'mean_squared_error')   #优化器选择adam，误差为均方误差
autoencoder.fit(data, data, batch_size=32, epochs = 20000, verbose = 1)  #  verbose = 1 为输出进度条记录，batch_size= 120一次放入数据120条，例如若1200条数据要分10次放入， epochs = 50为迭代次数50次，次数影响计算准确度

pred = autoencoder.predict(data)
pred = pd.DataFrame(pred)
pred.to_excel('excel_output1.xls')#打印自编码器重构的X的值
potential = encoder.predict(data)
potential = pd.DataFrame(potential)
potential.to_excel('excel_output2.xls')
# 保存模型
autoencoder.save('model_zbm02.h5')  #保存autoencoder的参数

#加载模型（在test中有示例）
#new_model = load_model('model.h5')
#pred = new_model.predict(data)
#print(pred)
#print(new_model.summary()) #显示模型结构








