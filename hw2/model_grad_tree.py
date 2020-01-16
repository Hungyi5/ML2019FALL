#!/usr/bin/env python
# coding: utf-8

# **To Do list**
# 
#     1.Data Cleaning
#         (NA)
#         
#     2.Feature engineering
# 
#     3.DNN

# In[1]:


import numpy as np
import pandas as pd
import sys

from sklearn.ensemble import GradientBoostingClassifier


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


# In[4]:


def normalize(x_train, x_test):
    
    x_all = np.concatenate((x_train, x_test), axis = 0)
    mean = np.mean(x_all, axis = 0)
    std = np.std(x_all, axis = 0)

    index = [0, 1, 3, 4, 5,106,107,108]
    mean_vec = np.zeros(x_all.shape[1])
    std_vec = np.ones(x_all.shape[1])
    mean_vec[index] = mean[index]
    std_vec[index] = std[index]

    x_all_nor = (x_all - mean_vec) / std_vec

    x_train_nor = x_all_nor[0:x_train.shape[0]]
    x_test_nor = x_all_nor[x_train.shape[0]:]

    return x_train_nor, x_test_nor


# In[5]:


if __name__ == '__main__':
    x_train, y_train, x_test = load_data(sys.argv[1], sys.argv[2], sys.argv[3])
    x_train = np.column_stack((x_train, x_train[::,5]**2, 1/x_train[::,5]))
    x_test = np.column_stack((x_test, x_test[::,5]**2, 1/x_test[::,5]))

    for i in [3,4]:
        x_train[::,i] = x_train[::,i] + 1
        x_test[::,i] = x_test[::,i] + 1
    train_capital_rate = x_train[::,i] / (x_train[::,3]+x_train[::,4])
    x_train = np.column_stack((x_train, train_capital_rate))
    test_capital_rate = x_test[::,i] / (x_test[::,3]+x_test[::,4])
    x_test = np.column_stack((x_test, test_capital_rate))
    # x_train, x_test = normalize(x_train, x_test)
    
    # Train
    clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,
    max_depth=2, random_state=0,subsample =0.8)
    clf.fit(x_train, y_train)
    pred_y = clf.predict(x_test)

    output = zip(list(range(1,len(x_test)+1)), pred_y.ravel())
    output = pd.DataFrame(output, columns = ['id','label'])
    output_name = sys.argv[4]
    output.to_csv(output_name, index=False)

