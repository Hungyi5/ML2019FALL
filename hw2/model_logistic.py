#!/usr/bin/env python
# coding: utf-8

# ### Logistic regression
# 
# **To Do**
# 
#     1. regularization
#     
#     2. kernel (feature engineering)

# In[1]:


import numpy as np
import pandas as pd
import sys


# We only use one-hot-encoding feature here
# 
# [Shell script usage](https://hackmd.io/@NeYbO-fDS5-UW6DQTmpVBA/HJIiFdZur?fbclid=IwAR0zGWEENLKgk3pmyng7CzUloZsHD0DDtYsNzumXzI2DIPZ9aoaluq-5WDA)

# In[2]:


def load_data(path_x_train, path_y_train, path_test):
    x_train = pd.read_csv(path_x_train)
    x_test = pd.read_csv(path_test)

    x_train = x_train.values
    x_test = x_test.values

    y_train = pd.read_csv(path_y_train, header = None)
    y_train = y_train.values
    y_train = y_train.reshape(-1)

    return x_train, y_train, x_test

# Use np.clip to prevent overflow
def sigmoid(z):
    res = 1 / (1.0 + np.exp(-z))
    return np.clip(res, 1e-6, 1-1e-6)


# Feature normalize, only on continues variable

# In[3]:


def normalize(x_train, x_test):
    
    x_all = np.concatenate((x_train, x_test), axis = 0)
    mean = np.mean(x_all, axis = 0)
    std = np.std(x_all, axis = 0)

    index = [0, 1, 3, 4, 5]
    mean_vec = np.zeros(x_all.shape[1])
    std_vec = np.ones(x_all.shape[1])
    mean_vec[index] = mean[index]
    std_vec[index] = std[index]

    x_all_nor = (x_all - mean_vec) / std_vec

    x_train_nor = x_all_nor[0:x_train.shape[0]]
    x_test_nor = x_all_nor[x_train.shape[0]:]

    return x_train_nor, x_test_nor


# Gradient descent using adagrad

# In[4]:


def train(x_train, y_train):
    b = 0.0
    w = np.zeros(x_train.shape[1])
    lr = 0.05
    epoch = 1000
    b_lr = 0
    w_lr = np.ones(x_train.shape[1])
    
    for e in range(epoch):
        z = np.dot(x_train, w) + b
        pred = sigmoid(z)
        loss = y_train - pred

        b_grad = -1*np.sum(loss)
        w_grad = -1*np.dot(loss, x_train)

        b_lr += b_grad**2
        w_lr += w_grad**2


        b = b-lr/np.sqrt(b_lr)*b_grad
        w = w-lr/np.sqrt(w_lr)*w_grad

        if(e+1)%500 == 0:
            loss = -1*np.mean(y_train*np.log(pred+1e-100) + (1-y_train)*np.log(1-pred+1e-100))
            print('epoch:{}\nloss:{}\n'.format(e+1,loss))
    return w, b


# 

# In[5]:


if __name__ == '__main__':

    if sys.argv[5] == '--test':
        x_train, y_train, x_test = load_data(sys.argv[1], sys.argv[2], sys.argv[3])
        wb = np.load('log_wb.npy')
        w = wb[0:-1]
        b = wb[-1]
        y_score = np.dot(x_test, w) + b
        pred_y = sigmoid(y_score)
        pred_y = (pred_y > 0.5).astype(int)

        output = zip(list(range(1,len(x_test)+1)), pred_y.ravel())
        output = pd.DataFrame(output, columns = ['id','label'])
        output_name = sys.argv[4]
        output.to_csv(output_name, index=False)
        
    elif sys.argv[5] == '--train':
        path_x_train= './data/X_train'
        path_y_train='./data/Y_train'
        path_test='./data/X_test'
        x_train, y_train, x_test = load_data(path_x_train, path_y_train, path_test)

        x_train, x_test = normalize(x_train, x_test)
        # perform worse
#         for idx, val in enumerate(x_train.shape):
#             x_train[idx] =  val + np.random.normal(0, 0.1**(1/2) ,x_train.shape[1])
        w, b = train(x_train, y_train)

        y_score = np.dot(x_test, w) + b
        pred_y = sigmoid(y_score)
        pred_y = (pred_y > 0.5).astype(int)

        wb = np.insert(w,len(w),b)
        np.save('log_wb.npy',wb)

#         output = zip(list(range(1,len(x_test)+1)), pred_y.ravel())
#         output = pd.DataFrame(output, columns = ['id','label'])
#         output_name = './output/log_tutorial_unnormalize.csv'
#         output.to_csv(output_name, index=False)


# In[ ]:





# ### Tip for math problem
# [p1](https://people.eecs.berkeley.edu/~jrs/189/exam/mids14.pdf)  
# [p2&3](https://people.eecs.berkeley.edu/~jordan/courses/260-spring10/other-readings/chapter13.pdf)  
# [p3](https://stats.stackexchange.com/questions/351549/maximum-likelihood-estimators-multivariate-gaussian)
