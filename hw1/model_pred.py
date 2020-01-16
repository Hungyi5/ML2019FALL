#!/usr/bin/env python
# coding: utf-8

# ## ML HW1 手把手教學

# In[1]:


import math
import numpy as np
import pandas as pd

import sys


# For Data Preprocessing, first we deal with anomaly data, basically data with wrong or invalid format.

# In[2]:


def readdata(data):
    
	# 把有些數字後面的奇怪符號刪除
	for col in list(data.columns[2:]):
		data[col] = data[col].astype(str).map(lambda x: x.rstrip('x*#A'))
	data = data.values
	
	# 刪除欄位名稱及日期
	data = np.delete(data, [0,1], 1)
	
	# 特殊值補0
	data[ data == 'NR'] = 0
	data[ data == ''] = 0
	data[ data == 'nan'] = 0
	data = data.astype(np.float)

	return data


# We flatten our data to be in such format (col: one hour/ per col, row: one feature/ per row)

# In[3]:


def extract(data):
	N = data.shape[0] // 18

	temp = data[:18, :]
    
    # Shape 會變成 (x, 18) x = 取多少hours
	for i in range(1, N):
		temp = np.hstack((temp, data[i*18: i*18+18, :]))
	return temp


# In[1]:


def parse2test(data):
	x = []
	
	total_length = data.shape[1] // 9
	for i in range(total_length):
		x_tmp = data[:,i*9:i*9+9]
		x.append(x_tmp.reshape(-1,))
	# x 會是一個(n, 18, 9)的陣列， y 則是(n, 1) 
	x = np.array(x)
	return x

def read_test(path3):
    testing_pd = pd.read_csv(path3)
    testing = readdata(testing_pd)
    testing_data = extract(testing)
    return parse2test(testing_data)


# **Combine them together!**

# In[7]:


if __name__ == "__main__":

    if sys.argv[3]== "--base":
        w = np.load('base_w.npy')
        bias = np.load('base_bias.npy')
    elif sys.argv[3] == "--best":
        w = np.load('best_w.npy')
        bias = np.load('best_bias.npy')
    
    # read testing
#     test_path = './data/testing_data.csv'
    test_path = sys.argv[1]
    test_x = read_test(test_path)
    
    pred_y = test_x @ w + bias
    id_list = [ "id_"+str(i) for i in range(len(test_x))]
    output_pd = pd.DataFrame(zip(id_list, pred_y.ravel()), columns=['id', 'value'])

#     output_name = 'base_output.csv'
    output_name = sys.argv[2]
    output_pd.to_csv(output_name, index=False)

