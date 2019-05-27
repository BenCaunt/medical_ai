#numpy for matrice math and resizing arrays
import numpy as np
import random
#importing/handling data
import pandas as pd

seed = random.randint(1,10000)
np.random.seed(seed)
import tensorflow as tf

from tensorflow.keras.layers import Dense
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

#prints out the datas end
print(df.tail)



#creating our neural network with tensorflow and keras
print(f"random seed is {seed}")
tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None)

model = tf.keras.models.Sequential()
model.add(Dense(13, kernel_initializer='random_normal',bias_initializer='random_uniform', activation='elu'))
model.add(Dense(10, kernel_initializer='random_normal',bias_initializer='random_uniform', activation='relu'))

model.add(Dense(2, kernel_initializer='random_normal',bias_initializer='random_uniform', activation='sigmoid'))
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
model.fit(x_train, y_train, epochs = 1000)

model.evaluate(x_test,y_test)

print(model.predict_classes(x_test))

model.summary()

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")
