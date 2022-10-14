import tensorflow as tf
import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

#  13指标（100）
#  降雨量（毫米）          "倒塌+严损房屋间数（间）"	倒塌+严损房屋户数（户）	万人倒塌严损房屋率
#  因灾死亡+失踪人数（人）	万人死亡和失踪率	        万人转移安置率	         农作物受灾面积（公顷）
#  农林牧渔业损失（万元）	基础设施损失（万元）	"工矿商贸业损失（万元）"	     "公共服务损失（万元）"
#  "道路损失（千米）"
# y直接经济损失

#数据预处理阶段
missing_values = ["n/a", "na", "--"] #指定空数据类型,将"n/a", "na", "--"指定为NaN类型
x_data = pd.read_excel('数据（1）.xlsx', na_values = missing_values)  # 加载数据
x_data.drop(['序号', '县代码（灾情上报系统）', '县代码（普查系统）', '区域代码', '区域'], axis=1, inplace=True)   # 去掉无用的列字段
print (x_data.index[np.where(x_data.isnull())[0]])      #打印出NaN的行数
print (x_data.columns[np.where(x_data.isnull())[1]])    #打印出NaN出现的指标,这两句话可以查找出现NaN的具体位置方便更改。
x_data.dropna(axis=0, inplace=True)                 #删除包含空字段的行，axis：默认为 0，表示逢空值剔除整行，如果设置参数 axis＝1 表示逢空值去掉整列。 inplace：如果设置 True，将计算得到的值直接覆盖之前的值并返回 None，修改的是源数据。
# 把NaN替换为0    df.fillna(0, inplace=True)
# 检查数据集中是否存在缺失数据
print(x_data.isnull().sum())  #检查缺失数据
x_data.fillna(0, inplace=True) #用零填充缺失数据
x_data = x_data.apply(lambda x: ((x - np.min(x)) / (np.max(x) - np.min(x))))  #数据归一化
x_data.to_excel('归一化值1.xls')
''
#训练集和测试集确定
x_data = x_data.values  #将dataframe转换为数组
y_data = pd.read_excel('y归一化（1）.xlsx', na_values = missing_values)  # 加载数据
y_data = y_data.values  #将dataframe转换为数组

#数据集乱序
#np.random.seed(11) #使用相同的seed，使输入特征和标签一一对应
#np.random.shuffle(x_data)
#np.random.seed(11)
#np.random.shuffle(y_data)
#tf.random.set_seed(11)


#转换为张量tensor
x_data = tf.convert_to_tensor(x_data, tf.float32)# 将数组转化为tensor
print(x_data.shape)
y_data = tf.convert_to_tensor(y_data, tf.float32)# 将数组转化为tensor
print(y_data.shape)
#数据集分出永不相见的训练集合测试集
x_train = x_data[:]   #元组的副本，而不是对元组本身的引用
y_train = y_data[:]
x_test = x_data[-20:]  #取后面20
y_test = y_data[-20:]

#搭建网络结构
model = tf.keras.models.Sequential()
#第1层2个神经元# 现在模型就会以尺寸为 (*, 2) 的数组作为输入，# 在第一层之后，就不再需要指定输入的尺寸了：
model.add(tf.keras.layers.Dense(13, activation='relu', input_shape=(13,)))
model.add(tf.keras.layers.Dense(14, activation='relu'))
model.add(tf.keras.layers.Dense(14, activation='relu'))
model.add(tf.keras.layers.Dense(14, activation='relu'))
model.add(tf.keras.layers.Dense(14, activation='relu'))
model.add(tf.keras.layers.Dense(14, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

#配置训练方法
model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.005),   #优化器adam
              loss=tf.keras.losses.MeanSquaredError()       #损失函数选择均方误差mse
              )

#fit执行训练过程
history = model.fit(x_train, y_train,
                    batch_size=16,
                    epochs=1000,
                    validation_data=(x_test, y_test),
                    validation_freq=1)
#一次喂入数据大小batch_size，轮数epochs，validation_data规定了测试集，
# validation_split从训练集划分一定比例到测试集,validation_freq多少epoch测试一次
#打印出网络结构和参数统计
model.summary()

#参数提取
#print(model.trainable_variables) #返回模型中可训练参数
file = open('./weights.txt', 'w')  #存入文本
for v in model.trainable_variables:
    file.write(str(v.name) + '\n')
    file.write(str(v.shape) + '\n')
    file.write(str(v.numpy()) + '\n')
file.close()

model.save('model_BP02.h5')  #保存model的参数

pred = model.predict(x_test)
print(pred)
pred = pd.DataFrame(pred)
pred.to_excel('20市经济损失预测值第2组.xlsx')

#赋值
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(loss))
#处理loss数据
data1_loss = pd.DataFrame(loss)
data1_loss.columns = ["loss"] #加表头
data1_loss['序号'] = range(1, len(data1_loss)+1)  #加序号
data1_loss = data1_loss[['序号', 'loss']]
#处理val_loss数据
data2_loss = pd.DataFrame(val_loss)
data2_loss.columns = ["val_loss"]
data2_loss['序号']  = range(1, len(data2_loss)+1)
data2_loss = data2_loss[['序号', 'val_loss']]

#导出为excel
data1_loss.to_excel('loss.xlsx')
data2_loss.to_excel('val_loss.xlsx')

fig = plt.figure(figsize=(7,5))       #figsize是图片的大小
ax1 = fig.add_subplot(1, 1, 1)        # ax1是子图的名字
plt.ylim((0, 0.01))  #坐标轴范围
plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='validation loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.title('Training and Validation Loss')
plt.legend()  #画出图例
plt.show()
