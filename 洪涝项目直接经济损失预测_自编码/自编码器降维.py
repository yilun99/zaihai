# -*- coding:utf-8 -*-
#加载模块
import pandas as pd
import numpy as np
import altair as alt
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras import Model
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler  
from sklearn.preprocessing import StandardScaler
alt.renderers.enable('altair_viewer')   #让altair模块适用于pycharm
import copy

#  13指标（100） 降维  7列
#  降雨量（毫米）          "倒塌+严损房屋间数（间）"	倒塌+严损房屋户数（户）	万人倒塌严损房屋率
#  因灾死亡+失踪人数（人）	万人死亡和失踪率	        万人转移安置率	         农作物受灾面积（公顷）
#  农林牧渔业损失（万元）	基础设施损失（万元）	"工矿商贸业损失（万元）"	     "公共服务损失（万元）"
#  "道路损失（千米）"
# y直接经济损失

missing_values = ["n/a", "na", "--"] #指定空数据类型,将"n/a", "na", "--"指定为NaN类型
# 加载数据
data = pd.read_excel('数据（1）.xlsx', na_values = missing_values)
print (data.index[np.where(data.isnull())[0]])      #打印出NaN的行数
print (data.columns[np.where(data.isnull())[1]])    #打印出NaN出现的指标,这两句话可以查找出现NaN的具体位置方便更改。
data.drop(['序号','县代码（灾情上报系统）', '县代码（普查系统）', '区域代码', '区域'], axis=1, inplace=True)   # 去掉无用的列字段
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
data.to_excel('excel_data1.xls')  #导出为excel

#转换为张量tensor
# x_data = tf.convert_to_tensor(data, tf.float32)# 将数组转化为tensor
# print(x_data.shape)
#训练集和测试集确定
# x_train = data[:-100]
# x_test = data[-100:]
# x_train, x_test = train_test_split(data, test_size=0.25,random_state=1)


#自编码器网络构建
input_df = Input( shape = (13, ))   #13个指标输入层为13层
x = Dense(13, activation = 'relu', name='input_layer_1')(input_df)
x = Dense(26, activation = 'relu', kernel_initializer='glorot_uniform')(x)
x = Dense(52, activation = 'relu', kernel_initializer='glorot_uniform')(x)
x = Dense(26, activation = 'relu', kernel_initializer='glorot_uniform')(x)
x = Dense(13, activation = 'relu', kernel_initializer='glorot_uniform')(x)


encoded = Dense(7, activation = 'relu', kernel_initializer='glorot_uniform', name='potential_layer')(x)


x = Dense(13, activation = 'relu', kernel_initializer='glorot_uniform')(encoded)
x = Dense(26, activation = 'relu', kernel_initializer='glorot_uniform')(x)
x = Dense(52, activation = 'relu', kernel_initializer='glorot_uniform')(x)
x = Dense(26, activation = 'relu', kernel_initializer='glorot_uniform')(x)
decoded = Dense(13, kernel_initializer='glorot_uniform', name='output_layer_1')(x)

autoencoder = Model(input_df, decoded)
encoder = Model(input_df, encoded)
autoencoder.compile(optimizer = 'adam', loss = 'mean_squared_error')   #优化器选择adam，误差为均方误差
#  verbose = 1 为输出进度条记录，batch_size= 120一次放入数据120条，例如若1200条数据要分10次放入， epochs = 50为迭代次数50次，次数影响计算准确度

history = autoencoder.fit(data, data,
                          batch_size=64, epochs = 1000,
                          verbose = 1)
# autoencoder.summary()

pred = autoencoder.predict(data)
pred = pd.DataFrame(pred)
pred.to_excel('excel_output1.xls')#打印自编码器重构的X的值
potential = encoder.predict(data)
potential = pd.DataFrame(potential)
potential.to_excel('excel_output2.xls')

# 保存模型
autoencoder.save('model_zbm01.h5')  #保存autoencoder的参数



# #赋值
# loss = history.history['loss']
# val_loss = history.history['val_loss']
# epochs = range(len(loss))
# #处理loss数据
# data1_loss = pd.DataFrame(loss)
# data1_loss.columns = ["loss"] #加表头
# data1_loss['序号'] = range(1, len(data1_loss)+1)  #加序号
# data1_loss = data1_loss[['序号', 'loss']]
# #处理val_loss数据
# data2_loss = pd.DataFrame(val_loss)
# data2_loss.columns = ["val_loss"]
# data2_loss['序号']  = range(1, len(data2_loss)+1)
# data2_loss = data2_loss[['序号', 'val_loss']]
#
# #导出为excel
# data1_loss.to_excel('zbm_loss.xlsx')
# data2_loss.to_excel('zbm_val_loss.xlsx')
#
# fig = plt.figure(figsize=(7,5))       #figsize是图片的大小
# ax1 = fig.add_subplot(1, 1, 1)        # ax1是子图的名字
# plt.ylim((0, 0.02))  #坐标轴范围
# plt.plot(epochs, loss, 'r', label='Training loss')
# plt.plot(epochs, val_loss, 'b', label='validation loss')
# plt.xlabel('epoch')
# plt.ylabel('loss')
# plt.title('Training and Validation Loss')
# plt.legend()  #画出图例
# plt.show()



#加载模型（在test中有示例）
# new_model = load_model('model_zbm01.h5')
# pred = new_model.predict(data)
# print(pred)
# print(new_model.summary()) #显示模型结构









