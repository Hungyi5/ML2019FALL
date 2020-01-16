#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import Dataset

import os
import cv2
import sys


# check pytorch gpu
# 
# https://stackoverflow.com/questions/48152674/how-to-check-if-pytorch-is-using-the-gpu

# In[8]:


# 3-layer CNN model [64, 128, 512] 
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1,64, kernel_size=(5,5), stride=(1,1))
        self.conv2 = nn.Conv2d(64,128, kernel_size=(5,5), stride=(1,1))
        self.conv3 = nn.Conv2d(128,512, kernel_size=(4,4), stride=(1,1))
        # Need to take data size into consideration.
        self.fc1 = nn.Linear(in_features=3*3*512, out_features=50, bias = True)
        self.fc2 = nn.Linear(in_features=50, out_features=7, bias = True)
    
    def forward(self,x):
        x = F.relu(F.max_pool2d(self.conv1(x),2))
        x = F.relu(F.max_pool2d(self.conv2(x),2))
        x = F.relu(F.max_pool2d(self.conv3(x),2))
        x = x.view(x.size(0),-1)
        x = F.relu(self.fc1(x))
        out = self.fc2(x)
        return out

# 48-5+1 = 44 = 22 -5+1 =18 = 9 -4+1 = 4 =>2
# m1: [100 x 4608], m2: [2048 x 50]


# In[ ]:


class train_hw3(Dataset):
    def __init__(self, data_dir, label):
        self.data_dir = data_dir
        self.label = label
    
    def __getitem__(self, index):
        pic_file = '{:0>5d}.jpg'.format(self.label[index][0])
        img = cv2.imread(os.path.join(self.data_dir, pic_file), cv2.IMREAD_GRAYSCALE)
        # 48*48 to 1*48*48 wheew 1 for number of channel.
        img = np.expand_dims(img,0)
        return torch.FloatTensor(img), self.label[index, 1]
    
    def __len__(self):
        return self.label.shape[0]

class test_hw3(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
    
    def __getitem__(self, index):
        pic_file = '{:0>4d}.jpg'.format(index)
        img = cv2.imread(os.path.join(self.data_dir, pic_file), cv2.IMREAD_GRAYSCALE)
        # 48*48 to 1*48*48 wheew 1 for number of channel.
        img = np.expand_dims(img,0)
        return torch.FloatTensor(img), index
    
    def __len__(self):
        return 7000


# In[3]:


if __name__ == '__main__':
    
    if sys.argv[3] == '--train':
        train_dir = sys.argv[1]
        train_csv = sys.argv[2]

        case = 'wide_resnet50_2'
        epoch = 50
        if case == 'resnet18':
            model = models.resnet18(pretrained=True)
            num_ftrs = model.fc.in_features
            model.fc= nn.Linear(in_features=num_ftrs, out_features=7)
            model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,
                                           bias=False)

        elif case == 'wide_resnet50_2':
            model = models.wide_resnet50_2(pretrained=True)
            num_ftrs = model.fc.in_features
            model.fc= nn.Linear(in_features=num_ftrs, out_features=7)
            model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,
                                       bias=False)

        elif case == 'cnn3':
            model = ConvNet()

        # Load data
        label = pd.read_csv(train_csv)
        label = label.values
        train_dataset = train_hw3(data_dir=train_dir, label=label)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=100)

        device = torch.device('cuda')
        model = model.to(device)

        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        EPOCH = 2

        for epoch in range(EPOCH):
            model.train()
            train_loss = 0
            correct = 0
            for batch_idx, (data, label) in enumerate(train_loader):
                data, label = data.to(device), label.to(device)
                # choose optimizer
                optimizer.zero_grad()
                output = model(data)
                # choose loss
                loss = F.cross_entropy(output, label)
                train_loss += F.cross_entropy(output, label).item()
                loss.backward()
                optimizer.step()

        # save
        if case == 'resnet18':
            model_name = './model/model_resnet18_{}_SGD_lr1-3_m9.pkl'.format(epoch)
        elif case == 'wide_resnet50_2':
            model_name = './model/model_wide_resnet50_2_ep{}_SGD_lr1-3_m9.pkl'.format(epoch)
        elif case == 'cnn3':
            model_name = './model/ConvNet3_ep{}_SGD_lr1-3_m9.pkl'.format(epoch)

        torch.save(model.state_dict(), model_name)
    # torch.save(model, './model/initial_model_{}.pkl'.format(epoch))
    else:
        output_name = sys.argv[2]
        test_dir = sys.argv[1]

        device = torch.device('cuda')
        # Here the size of each output sample is set to 2.
        # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
        model = models.wide_resnet50_2()
        num_ftrs = model.fc.in_features
        model.fc= nn.Linear(in_features=num_ftrs, out_features=7)
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,
                                       bias=False)
        epoch = 49
        model_name = './model/model_wide_resnet50_2_ep{}_SGD_lr1-3_m9.pkl'.format(epoch)


        model.load_state_dict(torch.load(model_name))
        model = model.to(device)
        model.eval()

        test_dataset = test_hw3(data_dir=test_dir)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1)
        prediction = []

        with torch.no_grad():
            for batch_idx, (img, index) in enumerate(test_loader):
                img = img.to(device)
                out = model(img)
                _, pred_label = torch.max(out,-1)
                prediction.append((index.item(), pred_label.item()))

        output = pd.DataFrame(prediction, columns=['id','label'])
        output.to_csv(output_name, index=False)


