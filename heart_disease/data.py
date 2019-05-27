import pandas as pd
import numpy as np

#import data
df = pd.read_csv('heart.csv')

#randomize data
df = df.sample(frac = 1)

#split data
df_train = df.head(round(len(df)/1.5))
print(len(df_train))


df_test = df.tail(len(df)-202)

x_train = np.array(df_train.drop(['target'], axis = 1))
y_train = np.array(df_train.target)
x_test = np.array(df_test.drop(['target'], axis = 1))
y_test = np.array(df_test.target)
print(x_test.shape)
