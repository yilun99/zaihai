# -*- coding:utf-8 -*-
# 加载模块
import pandas as pd
import numpy as np
import altair as alt
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA  # 加载PCA算法包

alt.renderers.enable('altair_viewer')  # 让altair模块适用于pycharm

missing_values = ["n/a", "na", "--"]  # 指定空数据类型,将"n/a", "na", "--"指定为NaN类型
# 加载数据
df = pd.read_excel('特征6.xlsx', na_values=missing_values)
print(df.info())  # info() 方法返回表格的一些基本信息
print(df.head())  # head( n ) 方法用于读取前面的 n 行，如果不填参数 n ，默认返回 5 行
# print (df.tail())   # tail( n ) 方法用于读取尾部的 n 行，如果不填参数 n ，默认返回 5 行，空行各个字段的值返回 NaN
df.isnull()  # 判断各单元是否为空NaN，空则返回True
print(df.index[np.where(df.isnull())[0]])  # 打印出NaN的行数
print(df.columns[np.where(df.isnull())[1]])  # 打印出NaN出现的指标,这两句话可以查找出现NaN的具体位置方便更改。
df.drop(['序号'], axis=1, inplace=True)  # 去掉无用的列字段
df.dropna(axis=0,
          inplace=True)  # 删除包含空字段的行，axis：默认为 0，表示逢空值剔除整行，如果设置参数 axis＝1 表示逢空值去掉整列。 inplace：如果设置 True，将计算得到的值直接覆盖之前的值并返回 None，修改的是源数据。
# 把NaN替换为0    df.fillna(0, inplace=True)

# print (df.head())
# df.convert_objects(convert_numeric=True)  #将object格式转float64格式
# df.to_excel('excel_output.xls')  #导出为excel
# print(df.columns)

# 导入机器学习自带模块
from sklearn.cluster import KMeans

# 确定分成几类为最优解
sse = []  # sse是误差平方和  肘部法则SSE（误差平方和）
for k in range(1, 11):  # 确定分成1类到10类sse的变化
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(df[['特征A',
                   '特征B',
                   ]])
    sse.append(kmeans.inertia_)
sse
plt.plot(range(1, 11), sse)  # 将误差平方和可视化，观察手肘位置。
# 1.手肘法思想随着聚类数k的增大，样本划分会更加精细，每个簇的聚合程度会逐渐提高，那么误差平方和SSE自然会逐渐减小。
# 1.手肘法当k小于真实聚类数时，由于k的增大会大幅增加每个簇的聚合程度，故SSE的下降幅度会很大，而当K到达真实聚类数时，再增加k所得到的聚合程度回报会迅速变小，所以SSE的下降幅度会骤减，然后随着k值的继续增大而趋于平缓，也就是说SSE和K的关系图是一个手肘的形状，而这个肘部对应的k值就是数据的真实聚类数。
plt.show()
# 由图选k=3

# 设置k的值
kmeans = KMeans(n_clusters=3)

# fit all columns   #未做归一化的KMeans
kmeans.fit(df[['特征A',  # 把需要聚类的属性值放入，不包含字符串类。
               '特征B',
               ]])
# extract the labels
df['label'] = kmeans.labels_ + 1  # kmeans.labels_将类别标记，得出label列，并+1让label从1开始
print(df)
df.groupby('label').size()  # Groupby可以将资料「分组」，之后在分组的资料上做运算，然后再将运算的结果组合起来,GroupBy的size方法，它可以返回分组的数量大小
print(df.groupby('label').size())
df.to_excel('聚类结果6.xls')  # 导出为excel

# 作散点图1
chart = alt.Chart(df).mark_circle().encode(
    x='特征A',
    y='特征B',
    # size='Displacement:Q',   #设置形状
    # color=alt.Color('区域'),
    color=alt.Color('label:N', scale=alt.Scale(scheme='dark2')),  # 按照类别来显示不同颜色
    # tooltip=list(df.columns)
).interactive()
chart.show()

'''
#归一化的KMeans
#数据预处理，数据标准化，如果某个特征的方差远大于其他特征的方差，那么它将会在算法学习职工占据主导位置，导致我们的学习器不能像我们期望的那样去学习其他特征，
这个将导致最后的模型收敛速度慢甚至不收敛，因此我们需要对这样的特征数据进行标准化/归一化
#标准化数据通过减去均值然后除以方差（或标准差），这种数据标准化方法经过处理后数据符合标准正态分布，即均值为0，标准差为1，转化函数为： x=（x-均值）/方差
from sklearn.preprocessing import StandardScaler
#fit all columns
df_scaled = StandardScaler().fit_transform(df[['降雨量（毫米）',
                                                '倒塌+严损房屋间数（模型评估）',
                                                '万人倒塌严损房屋率(模型评估)',
                                                '因灾死亡+失踪人口', '万人死亡和失踪率',
                                                '万人转移安置率',
                                                '工矿商贸业损失（万元）',
                                                '公共服务损失（万元）',
                                                '道路损失（模型评估）（千米）']])
#extract the labels
kmeans.fit(df_scaled)
df['label_standarder']=kmeans.labels_+1  #kmeans.labels_将类别标记，并在df后加一列名为label_standarder，并+1让label从1开始
print(df)
df.groupby('label_standarder').size()     #  Groupby可以将资料「分组」，之后在分组的资料上做运算，然后再将运算的结果组合起来,GroupBy的size方法，它可以返回分组的数量大小
print(df.groupby('label_standarder').size())
df.to_excel('excel_output1.xls')  #导出为excel

#作散点图2
chart=alt.Chart(df).mark_circle().encode(
    x='降雨量（毫米）',
    y='因灾死亡+失踪人口',
    #size='Displacement:Q',   #设置形状
    #color=alt.Color('区域'),
    color=alt.Color('label_standarder:N', scale=alt.Scale(scheme='dark2')),  #按照类别来显示不同颜色
    #tooltip=list(df.columns)
).interactive()
chart.show()
'''
