import pandas as pd
import numpy as np
# 读取数据
df = pd.read_csv('Dataset/ecommerce_consume.csv')
#1.处理年龄缺失值，按性别分组填充中位数
df['年龄']=df.groupby('性别')['年龄'].transform(lambda x:x.fillna(x.median()))
#2.处理消费金额异常值
Q1=df['消费金额'].quantile(0.25)
Q3=df['消费金额'].quantile(0.75)
IQR=Q3-Q1

upper_bound=Q3+1.5*IQR
lower_bound=Q1-1.5*IQR
#截断异常值
df['消费金额']=np.where(df['消费金额']>upper_bound,upper_bound,df['消费金额'])
df['消费金额']=np.where(df['消费金额']<lower_bound,lower_bound,df['消费金额'])
