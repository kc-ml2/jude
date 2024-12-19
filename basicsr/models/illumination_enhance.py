#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   illumination_enhance.py
@Time    :   2024/04/01 15:51:24
@Author  :   Tu Vo
@Version :   1.0
@Contact :   vovantu.hust@gmail.com
@License :   (C)Copyright 2020-2021, Tu Vo
@Desc    :   KC Machine Learning Lab
"""


import torch.nn as nn
from models.architecture import get_conv2d_layer


class Illumination_Alone(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = get_conv2d_layer(in_c=3, out_c=32, k=5, s=1, p=2)
        self.conv2 = get_conv2d_layer(in_c=32, out_c=32, k=5, s=1, p=2)
        self.conv3 = get_conv2d_layer(in_c=32, out_c=32, k=5, s=1, p=2)
        self.conv4 = get_conv2d_layer(in_c=32, out_c=1, k=5, s=1, p=2)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.relu = nn.ReLU()
    def forward(self, l):
        x = self.conv1(l)            
        x = self.conv2(self.leaky_relu(x))   
        x = self.conv3(self.leaky_relu(x))   
        x = self.conv4(self.leaky_relu(x))
        # x = self.relu(x) 
        return x
