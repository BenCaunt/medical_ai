import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import math

'''
beutiful resources:
https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/

'''


#logistic sigmoid
def sigmoid(x):
    x = np.float64(x)
    return 1/(1+np.exp(-x))

#sigmoid derivative
def sigmoid_p(x):
    return sigmoid(x) * (1-sigmoid(x))


#stepper function
def step(x):
    z = None
    if x > 0:
        z = 1
    else:
        z = 0
    return z

#actual neural network prediction
def nn(x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13):
    layer_1 = sigmoid((x1*layer_1_weights[0])+(x2*layer_1_weights[1])+(x3*layer_1_weights[2])+(x4*layer_1_weights[3])+(x5*layer_1_weights[4])+(x6*layer_1_weights[5])+(x7*layer_1_weights[6])+(x8*layer_1_weights[7])+(x9*layer_1_weights[8])+(x10*layer_1_weights[9])+(x11*layer_1_weights[10])+(x12*layer_1_weights[11])+(x13*layer_1_weights[12])+b)
    
    
    
    return layer_1

#neural network derivative thing
def nn_p(x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13):
    prediction = sigmoid_p((x1*m1)+(x2*m2)+(x3*m3)+(x4*m4)+(x5*m5)+(x6*m6)+(x7*m7)+(x8*m8)+(x9*m9)+(x10*m10)+(x11*m11)+(x12*m12)+(x13*m13)+b)
    return prediction
#my probably awful loss function:
def loss(x, y):
    d = x-y
    return 0-d

def error(target,prediction):
    return (1/2)*(target-prediction)**2


print(loss(12,1))

#import data
df = pd.read_csv('heart.csv')

#randomize data
df = df.sample(frac = 1)

#split data
df_train = df.head(round(len(df)/2))
df_test = df.tail(len(df)-152)

x_train = np.array(df_train.drop(['target'], axis = 1))
y_train = np.array(df_train.target)
x_test = np.array(df_test.drop(['target'], axis = 1))
y_test = np.array(df_test.target)
print(x_test.shape)


layer_1_weights = []
layer_2_weights = []
#CREATES WEIGHTS FOR OUR FIRST LAYER (INPUT)
for i in range(0,12):
    layer_1_weights.append(np.random.uniform(-1,1))

#CREATES OUR HIDDEN LAYER WEIGHTS
for i in range(6):
    layer_2.append(np.random.uniform(-1,1))

#assign bias

b = np.random.uniform(-1,1)

learning_rate = 0.01


train_len = len(df_test)
'''
________________________________________________________________________________________________________________________________________________________________
*********************** training loop start ****************************** training loop start ************************* training loop start ********************
________________________________________________________________________________________________________________________________________________________________
'''
for epochs in range(1,10000):
    print("epoch: {0}".format(epochs))
    for i in range(train_len):
        prediction = nn(x_train[i][0],x_train[i][1],x_train[i][2],x_train[i][3],x_train[i][4],x_train[i][5],x_train[i][6],x_train[i][7],x_train[i][8],x_train[i][9],x_train[i][10],x_train[i][11],x_train[i][12])
        actual = y_train[i]
        calc_error = error(actual,prediction)
        print("prediction is: {0};  correct value is {1}".format(round(prediction), actual))


        acc_list = []

        if round(prediction) != actual:
            dcost_dpred = 2 * (prediction - actual)

            #derivative prediction thing
            dpred_dz = nn_p(x_train[i][0],x_train[i][1],x_train[i][2],x_train[i][3],x_train[i][4],x_train[i][5],x_train[i][6],x_train[i][7],x_train[i][8],x_train[i][9],x_train[i][10],x_train[i][11],x_train[i][12])

            dcost_dz = dcost_dpred * dpred_dz

            dz_dw1 = x_train[i][0]
            dz_dw2 = x_train[i][1]
            dz_dw3 = x_train[i][2]
            dz_dw4 = x_train[i][3]
            dz_dw5 = x_train[i][4]
            dz_dw6 = x_train[i][5]
            dz_dw7 = x_train[i][6]
            dz_dw8 = x_train[i][7]
            dz_dw9 = x_train[i][8]
            dz_dw10 = x_train[i][9]
            dz_dw11 = x_train[i][10]
            dz_dw12 = x_train[i][11]
            dz_dw13 = [i][12]x_train
            dz_db = 1

            dcost_dw1 = dcost_dz * dz_dw1
            dcost_dw2 = dcost_dz * dz_dw2
            dcost_dw3 = dcost_dz * dz_dw3
            dcost_dw4 = dcost_dz * dz_dw4
            dcost_dw5 = dcost_dz * dz_dw5
            dcost_dw6 = dcost_dz * dz_dw6
            dcost_dw7 = dcost_dz * dz_dw7
            dcost_dw8 = dcost_dz * dz_dw8
            dcost_dw9 = dcost_dz * dz_dw9
            dcost_dw10 = dcost_dz * dz_dw10
            dcost_dw11 = dcost_dz * dz_dw11
            dcost_dw12 = dcost_dz * dz_dw12
            dcost_dw13 = dcost_dz * dz_dw13
            dcost_db = dcost_dz * dz_db


            m1 = m1 - learning_rate * dcost_dw1
            m2 = m2 - learning_rate * dcost_dw2
            m3 = m3 - learning_rate * dcost_dw3
            m4 = m4 - learning_rate * dcost_dw4
            m5 = m5 - learning_rate * dcost_dw5
            m6 = m6 - learning_rate * dcost_dw6
            m7 = m7 - learning_rate * dcost_dw7
            m8 = m8 - learning_rate * dcost_dw8
            m9 = m9 - learning_rate * dcost_dw9
            m10 = m10 - learning_rate * dcost_dw10
            m11 = m11 - learning_rate * dcost_dw11
            m12 = m12 - learning_rate * dcost_dw12
            m13 = m13 - learning_rate * dcost_dw13
            b = b - learning_rate * dcost_db

            print('wrong')
            acc_list.append(0)


        else:
            print('correct')
            acc_list.append(1)

print(np.random.seed())


print(m1)
print(m2)
print(m3)
print(m4)
print(m5)
print(m6)
print(m7)
print(m8)
print(m9)
print(m10)
print(m11)
print(m12)
print(m13)
print(b)

print(m1)

acc_list_len = len(acc_list)

acc_list = sum(acc_list)

print('accuracy: {0}'.format(acc_list/acc_list_len))
