import tensorflow as tf
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

#prints out the datas end
print(df.tail)

#splits training data

#gets rid of the target variable for training
x = df.drop(['target'], axis =1)
x_train = x
#the target variable that is used to validate a training session
y = df.target
y_train = y
#turn our data into numpy arrays to be reshaoed
x_train = np.array(x_train)
y_train = np.array(y_train)



#creating our neural network with tensorflow and keras

#initializes our model as sequential
model = tf.keras.models.Sequential()

#adds our input layer with 13 nodes as there is 13 dimensions to train on
model.add(tf.keras.layers.Dense(13, activation = 'elu'))

#adds our hidden layer
model.add(tf.keras.layers.Dense(7, activation = 'elu'))

#our output layer with two nodes as our data has two classes
model.add(tf.keras.layers.Dense(2, activation = 'elu'))

#compiling our model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs = 15)

model.summary()



model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")
