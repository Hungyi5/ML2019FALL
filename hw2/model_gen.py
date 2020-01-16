#!/usr/bin/env python
# coding: utf-8

# ## Can we add regularization term in generative models?
# 
# **Refers to HW1 reports**
# 
# 

# In[1]:


import numpy as np
import pandas as pd
import math

dim = 106


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


# 

# ### Probabilistic generative model
# 

# 參考 [上課投影片](https://drive.google.com/file/d/1WKjqkJVPIxYh1REbzy6HeoGfZj-mj6NJ/view) P18 and P23
# 
# 

# In[3]:


def train(x_train, y_train):
    cnt1 = 0
    cnt2 = 0
    
    mu1 = np.zeros((dim,))
    mu2 = np.zeros((dim,))
    
    for i in range(x_train.shape[0]):
        if y_train[i] == 1:
            cnt1 += 1
            mu1 += x_train[i]
        else:
            cnt2 += 1
            mu2 += x_train[i]
    mu1 /= cnt1
    mu2 /= cnt2

    sigma1 = np.zeros((dim,dim))
    sigma2 = np.zeros((dim,dim))
    for i in range(x_train.shape[0]):
        if y_train[i] == 1:
            sigma1 += np.dot(np.transpose([x_train[i] - mu1]), [(x_train[i] - mu1)])
        else:
            sigma2 += np.dot(np.transpose([x_train[i] - mu2]), [(x_train[i] - mu2)])
    sigma1 /= cnt1
    sigma2 /= cnt2

    
    share_sigma = (cnt1 / x_train.shape[0]) * sigma1 + (cnt2 / x_train.shape[0]) * sigma2
    return mu1, mu2, share_sigma, cnt1, cnt2


# In[4]:


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


# 參考 [上課投影片](https://drive.google.com/file/d/1WKjqkJVPIxYh1REbzy6HeoGfZj-mj6NJ/view) P33
# 
# 

# In[5]:


def predict(x_test, mu1, mu2, share_sigma, N1, N2):
    sigma_inverse = np.linalg.inv(share_sigma)

    w = np.dot( (mu1-mu2), sigma_inverse)
    b = (-0.5) * np.dot(np.dot(mu1.T, sigma_inverse), mu1) + (0.5) * np.dot(np.dot(mu2.T, sigma_inverse), mu2) + np.log(float(N1)/N2)

    z = np.dot(w, x_test.T) + b
    pred = sigmoid(z)
    return pred

def load_par(par):
    par = np.load(par)
    return par[0,::], par[1,::], par[2,0], par[3,0], par[4::,::]


# 

# In[6]:


if __name__ == '__main__':

    if sys.argv[5] == '--test':
        x_train, y_train, x_test = load_data(sys.argv[1], sys.argv[2], sys.argv[3])
        mu1, mu2, N1, N2, shared_sigma = load_par('gen_mu12N12sigma.npy')
        pred_y = predict(x_test, mu1, mu2, shared_sigma, N1, N2)
        pred_y = (pred_y > 0.5).astype(int)
        output = zip(list(range(1,len(x_test)+1)), pred_y.ravel())
        output = pd.DataFrame(output, columns = ['id','label'])
        output_name = sys.argv[4]
        output.to_csv(output_name, index=False)

    else sys.argv[5] == '--train':
        path_x_train= './data/X_train'
        path_y_train='./data/Y_train'
        path_test='./data/X_test'
        x_train, y_train, x_test = load_data(path_x_train, path_y_train, path_test)

        mu1, mu2, shared_sigma, N1, N2 = train(x_train, y_train)
        N1 = N1* np.ones(dim)
        N2 = N2* np.ones(dim)
        par = np.vstack((mu1,mu2,N1,N2,shared_sigma))
        np.save('gen_mu12N12sigma.npy', par)

#     y = predict(x_train, mu1, mu2, shared_sigma, N1, N2)
#     y = np.around(y)
#     result = (y_train == y)
    
#     print('Train acc = %f' % (float(result.sum()) / result.shape[0]))
    
    #predict x_test    
#     pred_y = predict(x_test, mu1, mu2, shared_sigma, N1, N2)
#     pred_y = (pred_y > 0.5).astype(int)
    
#     output = zip(list(range(1,len(x_test)+1)), pred_y.ravel())
#     output = pd.DataFrame(output, columns = ['id','label'])
#     output_name = './output/gen_tutorial_normalize.csv'
#     output.to_csv(output_name, index=False)


# ### Tip for math problem
# [p1](https://people.eecs.berkeley.edu/~jrs/189/exam/mids14.pdf)  
# [p2&3](https://people.eecs.berkeley.edu/~jordan/courses/260-spring10/other-readings/chapter13.pdf)  
# [p3](https://stats.stackexchange.com/questions/351549/maximum-likelihood-estimators-multivariate-gaussian)
