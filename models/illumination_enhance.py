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
import torch
import torch.nn.functional as F


class RelightNetv2(nn.Module):
    def __init__(self, channel=64, kernel_size=3):
        super(RelightNetv2, self).__init__()

        self.relu = nn.ReLU()
        self.net2_conv0_1 = nn.Conv2d(
            3, channel, kernel_size, padding=1, padding_mode="replicate"
        )

        self.net2_conv1_1 = nn.Conv2d(
            channel, channel, kernel_size, stride=2, padding=1, padding_mode="replicate"
        )
        self.net2_conv1_2 = nn.Conv2d(
            channel, channel, kernel_size, stride=2, padding=1, padding_mode="replicate"
        )
        self.net2_conv1_3 = nn.Conv2d(
            channel, channel, kernel_size, stride=2, padding=1, padding_mode="replicate"
        )

        self.net2_deconv1_1 = nn.Conv2d(
            channel * 2, channel, kernel_size, padding=1, padding_mode="replicate"
        )
        self.net2_deconv1_2 = nn.Conv2d(
            channel * 2, channel, kernel_size, padding=1, padding_mode="replicate"
        )
        self.net2_deconv1_3 = nn.Conv2d(
            channel * 2, channel, kernel_size, padding=1, padding_mode="replicate"
        )

        self.net2_fusion = nn.Conv2d(
            channel * 3, channel, kernel_size=1, padding=1, padding_mode="replicate"
        )
        self.net2_output = nn.Conv2d(channel, 3, kernel_size=3, padding=0)

    def forward(self, input_L, input_R):
        input_img = input_R * input_L
        out0 = self.net2_conv0_1(input_img)
        out1 = self.relu(self.net2_conv1_1(out0))
        out2 = self.relu(self.net2_conv1_2(out1))
        out3 = self.relu(self.net2_conv1_3(out2))

        out3_up = F.interpolate(out3, size=(out2.size()[2], out2.size()[3]))
        deconv1 = self.relu(self.net2_deconv1_1(torch.cat((out3_up, out2), dim=1)))
        deconv1_up = F.interpolate(deconv1, size=(out1.size()[2], out1.size()[3]))
        deconv2 = self.relu(self.net2_deconv1_2(torch.cat((deconv1_up, out1), dim=1)))
        deconv2_up = F.interpolate(deconv2, size=(out0.size()[2], out0.size()[3]))
        deconv3 = self.relu(self.net2_deconv1_3(torch.cat((deconv2_up, out0), dim=1)))

        deconv1_rs = F.interpolate(deconv1, size=(input_R.size()[2], input_R.size()[3]))
        deconv2_rs = F.interpolate(deconv2, size=(input_R.size()[2], input_R.size()[3]))
        feats_all = torch.cat((deconv1_rs, deconv2_rs, deconv3), dim=1)
        feats_fus = self.net2_fusion(feats_all)
        output = self.net2_output(feats_fus)
        return output
