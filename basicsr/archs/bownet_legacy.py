#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   network_p4ip_68_patch_pad_with_e_batch_v9_1_rgb_new_kernel_prediction.py -> change kernel prediction
@Time    :   2024/03/18 14:58:38
@Author  :   Tu Vo
@Version :   1.0
@Contact :   vovantu.hust@gmail.com
@License :   (C)Copyright 2020-2021, Tu Vo
@Desc    :   KC Machine Learning Lab
'''


import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from torch.autograd import Variable

from models.ResUNet import ResUNet
from models.TwoHeadsNetwork import TwoHeadsNetwork as KernelsNetwork
from models.illumination_enhance import Illumination_Alone

from utils.utils_torch import (
    conv_fft_batch, hadamard_batch, hadamard_batches, conv_fft_batches
)
import os
from basicsr.utils.registry import ARCH_REGISTRY

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"


def p2o_4d(psf, shape):
    """
    Convert point-spread function to optical transfer function.
    otf = p2o(psf) computes the Fast Fourier Transform (FFT) of the
    point-spread function (PSF) array and creates the optical transfer
    function (OTF) array that is not influenced by the PSF off-centering.
    Args:
        psf: NxCxhxw
        shape: [H, W]
    Returns:
        otf: NxCxHxWx2
    """
    otf = torch.zeros(shape).type_as(psf)
    psf_rp = psf.unsqueeze(2).repeat_interleave(3, dim=2)
    otf[..., : psf.shape[-2], : psf.shape[-1]].copy_(psf_rp)
    for axis, axis_size in enumerate(psf_rp.shape[-2:]):
        otf = torch.roll(otf, -int(axis_size / 2), dims=axis + 4)
    otf = torch.fft.fftn(otf, dim=(-2, -1))
    # n_ops = torch.sum(torch.tensor(psf.shape).type_as(psf) * torch.log2(torch.tensor(psf.shape).type_as(psf)))
    # otf[..., 1][torch.abs(otf[..., 1]) < n_ops*2.22e-16] = torch.tensor(0).type_as(psf)
    return otf

def p2o(psf, shape):
    '''
    Args:
        psf: NxCxhxw
        shape: [H,W]

    Returns:
        otf: NxCxHxWx2
    '''
    # otf = torch.zeros(shape).type_as(psf)
    otf = torch.zeros(psf.shape[:-2] + shape).type_as(psf)
    otf[...,:psf.shape[2],:psf.shape[3]].copy_(psf)
    for axis, axis_size in enumerate(psf.shape[2:]):
        otf = torch.roll(otf, -int(axis_size / 2), dims=axis+2)
    otf = torch.fft.fftn(otf, dim=(-2, -1))
    # n_ops = torch.sum(torch.tensor(psf.shape).type_as(psf) * torch.log2(torch.tensor(psf.shape).type_as(psf)))
    # otf[...,1][torch.abs(otf[...,1])<n_ops*2.22e-16] = torch.tensor(0).type_as(psf)
    return otf

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.kaiming_normal(m.weight.data, a=0, mode="fan_in")
    elif classname.find("Linear") != -1:
        nn.init.kaiming_normal(m.weight.data, a=0, mode="fan_in")
    elif classname.find("BatchNorm") != -1:
        # nn.init.uniform(m.weight.data, 1.0, 0.02)
        m.weight.data.normal_(mean=0, std=math.sqrt(2.0 / 9.0 / 64.0)).clamp_(
            -0.025, 0.025
        )
        nn.init.constant(m.bias.data, 0.0)


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2), DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)



class InitNet(nn.Module):
    def __init__(self, n, k):
        super(InitNet, self).__init__()
        self.n = n
        self.conv_layers = nn.Sequential(
            Down(k, 4), Down(4, 8), Down(8, 16), Down(16, 16)
        )

        self.mlp = nn.Sequential(
            nn.Linear(16 * 8 * 8, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 7 * (self.n)),
            nn.Softplus(),
        )
        self.resize = nn.Upsample(size=[256, 256], mode="bilinear", align_corners=True)

    def forward(self, kernel):
        # kernel_stitch = stitch_patches(kernel.permute(2,1,0,3,4).unsqueeze(-1), self.rows, self.cols, stitch_axis=(-3, -2))
        N, C, H, W = kernel.size()
        h1, h2 = int(np.floor(0.5 * (128 - H))), int(np.ceil(0.5 * (128 - H)))
        w1, w2 = int(np.floor(0.5 * (128 - W))), int(np.ceil(0.5 * (128 - W)))
        k_pad = F.pad(kernel, (w1, w2, h1, h2), "constant", 0)
        A = torch.fft.fftn(k_pad, dim=[-2, -1])
        AtA_fft = torch.abs(A) ** 2
        x = self.conv_layers(AtA_fft.float())
        # x = torch.cat((x.view(N,1,16*8*8),  M.float().view(N,1,1)), axis=2).float()
        h = self.mlp(x.view(N, 1, 16 * 8 * 8).float()) + 1e-6

        lambda1_iters = h[:, :, 0 : self.n].view(N, 1, 1, self.n)
        lambda2_iters = h[:, :, self.n : 2 * self.n].view(
            N, 1, 1, self.n
        )
        lambda3_iters = h[:, :, 2 * self.n : 3 * self.n].view(
            N, 1, 1, self.n
        )
        lambda4_iters = h[:, :, 3 * self.n : 4 * self.n].view(
            N, 1, 1, self.n
        )
        lambda5_iters = h[:, :, 4 * self.n : 5 * self.n].view(
            N, 1, 1, self.n
        )
        lambda6_iters = h[:, :, 5 * self.n : 6 * self.n].view(
            N, 1, 1, self.n
        )
        lambda7_iters = h[:, :, 6 * self.n : 7 * self.n].view(
            N, 1, 1, self.n
        )
        return (
            lambda1_iters,
            lambda2_iters,
            lambda3_iters,
            lambda4_iters,
            lambda5_iters,
            lambda6_iters,
            lambda7_iters
        )

class update_P(nn.Module):
    """
    g2(P) + ||P - (-Q*I + rho3 + R * lambda6)/(Q*Q * lambda2 + lambda6)|| * (Q*Q * lambda2 + lambda6)
    """
    def __init__(self):
        super().__init__()
        self.denoiser = ResUNet(in_nc=3, out_nc=3)
    
    def forward(self, Q, I, lambda2, lambda6, rho3, R):
        QI = Q * I
        input_tensor = (QI + rho3 + R * lambda6) / (Q * Q * lambda2 + lambda6)
        P_ = self.denoiser(input_tensor)
        return P_

class update_R(nn.Module):
    """
    R = (lambda6*P - rho3) / (lambda6)
    """
    def __init__(self):
        super().__init__()

    def forward(self, P, lambda6, rho3):
        R_ = (P  * lambda6 - rho3) / lambda6
        return R_

class update_Q(nn.Module):
    """
    g3(Q) + ||Q - (P*I + rho4 + L * lambda7)/(P*P * lambda2 + lambda7)|| * (lambda7 + P * P * lambda2)
    """
    def __init__(self):
        super().__init__()
        self.denoiser = ResUNet(in_nc=3, out_nc=3)
    
    def forward(self, P, I, lambda2, lambda7, rho4, L):
        PI = P * I
        input_tensor = (PI + rho4 + L * lambda7) / (lambda7 + P * P * lambda2)
        Q_ = self.denoiser(input_tensor)
        return Q_

class update_L(nn.Module):
    """
    R = (Q * lambda7 - rho4) / (lambda7)
    """
    def __init__(self):
        super().__init__()

    def forward(self, Q, lambda7, rho4):
        L_ = (Q  * lambda7 - rho4) / lambda7
        return L_

class update_K(nn.Module):
    """
    g4(K) + ||K - (rho1 + lambda4 * E) /lambda4|| * lambda4
    """
    def __init__(self):
        super().__init__()
        self.denoiser = ResUNet(in_nc=3, out_nc=3)
    
    def forward(self, E, rho1, lambda4):
        input_tensor = (rho1 + lambda4 * E) / lambda4
        K_ = self.denoiser(input_tensor)
        return K_

class update_E(nn.Module):
    """
    ||E||_1*(lambda3/(lambda4+lambda1)) + 1/2||E - 1/(lambda4+lambda1)(X + lambda4 * K - rho1 - HI)||2
    """
    def __init__(self) -> None:
        super().__init__()

    def forward(self, K, H, I, X, lambda1, lambda3, lambda4, logits, rho1, pad_width, filter_halfwidth):
        I_patch = torch.nn.functional.pad(I, pad=pad_width)
        HI = conv_fft_batch(I_patch, H)
        HI = torch.real(torch.fft.ifftn(HI, dim=[-2, -1]))

        HI = HI[
            ..., filter_halfwidth:-filter_halfwidth, filter_halfwidth:-filter_halfwidth
        ]
        HI = hadamard_batches(HI, logits)
        N = (X + lambda4 * K - rho1 - HI) / (lambda4 + lambda1)
        E_ = torch.mul(torch.sign(N), nn.functional.relu(torch.abs(N) - lambda3/(lambda4 + lambda1)))
        return E_

class update_M(nn.Module):
    """
    g1(M) + lambda5/2||M - (rho2 + lambda5 * I)/lambda5||
    """
    def __init__(self):
        super().__init__()
        self.denoiser = ResUNet(in_nc=3, out_nc=3)
    
    def forward(self, I , lambda5, rho2):
        input_tensor = (rho2 + I * lambda5) / (lambda5)
        M_ = self.denoiser(input_tensor)
        return M_ 

class update_I(nn.Module):
    """
    I = F^-1{F(lambda1 * (X-E)H^T + lambda2 * PQ - rho2 + lambda5 * M) / (lambda1 * HHT + lambda2 + lambda5)}
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, X, E, P, Q, HT, HHT, logits, M, lambda1, lambda2, lambda5, rho2, pad_width, filter_halfwidth):
        X_patch = torch.nn.functional.pad(X-E, pad=pad_width)
        XHt = conv_fft_batch(X_patch, HT)
        XHt = torch.real(torch.fft.ifftn(XHt, dim=[-2, -1]))

        XHt = XHt[
            ..., filter_halfwidth:-filter_halfwidth, filter_halfwidth:-filter_halfwidth
        ]
        HHT = HHT[
            ..., filter_halfwidth:-filter_halfwidth, filter_halfwidth:-filter_halfwidth
        ]

        XHt = hadamard_batches(XHt, logits)
        inter = torch.fft.fftn(lambda1 * XHt + lambda2 * P*Q - rho2 + lambda5 * M, dim=[-2, -1]) 
        I_ = torch.real(
            torch.fft.ifftn((inter) / (lambda1 * HHT + lambda2 + lambda5), dim=[-2, -1])
        )
        return I_ 

class update_H(nn.Module):
    """
    ||R - H*I||^2
    """
    def __init__(self):
        super().__init__()

    def forward(self, P, I):
        H = torch.fft.ifftn(torch.fft.fftn(P)/torch.fft.fftn(I))
        return H

@ARCH_REGISTRY.register()
class BOWNet(nn.Module):
    """
    This version is for batches
    """

    def __init__(self, device, opt):
        super(BOWNet, self).__init__()
        self.device = device
        self.n = opt["network"]["stages"]
        self.k = 1
        self.opt = opt

        self.init = InitNet(self.n, self.k)
        self.kernel = KernelsNetwork(K=self.k)
        self.update_P = update_P()
        self.update_R = update_R()
        self.update_Q = update_Q()
        self.update_L = update_L()
        self.update_K = update_K()
        self.update_E = update_E()
        self.update_M = update_M()
        self.update_I = update_I()
        # self.update_H = update_H()
        self.illumination_enhance = Illumination_Alone()
        # self.decom = Decom()

    
    def run_iter_rgb(self, X, P, R, Q, L, K, E, M, I, H, Ht, HtH, rhos, lambdas, logits, pad_width, filter_halfwidth):
        [lambda1, lambda2, lambda3, lambda4, lambda5, lambda6, lambda7] = lambdas
        [rho1, rho2, rho3, rho4] = rhos
        ## Update P
        P_ = self.update_P(Q, I, lambda2, lambda6, rho3, R)

        ## update R
        R_ = self.update_R(P_, lambda6, rho3)

        ## update Q
        Q_ = self.update_Q(P_, I, lambda2, lambda7, rho4, L)

        ## update L
        L_ = self.update_L(Q_, lambda7, rho4)

        ## update K
        K_ = self.update_K(E, rho1, lambda4)

        ## update E
        E_ = self.update_E(K_, H, I, X, lambda1, lambda3, lambda4, logits, rho1, pad_width, filter_halfwidth)

        ## update M
        M_ = self.update_M(I, lambda5, rho2)

        ## update_I 
        I_ = self.update_I(X, E_, P_, Q_, Ht, HtH, logits, M_, lambda1, lambda2, lambda5, rho2, pad_width, filter_halfwidth)

        ## update_H
        # H_ = self.update_H(P_, I_)

        ## update auxialaries
        rho1 = rho1 + lambda4 * (E_ - K_)
        rho2 = rho4 + lambda5 * (I_ - M_)
        rho3 = rho3 + lambda6 * (R_ - P_)
        rho4 = rho4 + lambda7 * (L_ - Q_)
        rhos = [rho1, rho2, rho3, rho4]

        return P_, R_, Q_, L_, K_, E_, M_, I_, rhos

    def forward(self, X, kernels):
        P_list = []
        I_list = []
        Q_list = []
        N, _, _, _ = X.size()
        patch_size = self.opt["datasets"]["train"]["crop_size"]

        """
        Estimate kernel
        """
        # if H != patch_size or W != patch_size:
        #     top = random.randint(0, H - patch_size)
        #     left = random.randint(0, W - patch_size)

        #     # crop lq patch
        #     X_patch = X[..., top:top + patch_size, left:left + patch_size]

        #     kernels = self.kernel(X_patch)
        # else:
        # P, Q = self.decom(X)
        kernels, logits = self.kernel(X)
        # kernels = torch.zeros((N, 1, 32, 32))

        # Generate auxiliary variables for convolution
        (
            lambda1_iters,
            lambda2_iters,
            lambda3_iters,
            lambda4_iters,
            lambda5_iters,
            lambda6_iters,
            lambda7_iters
        ) = self.init(
            kernels
        )  # Hyperparameters
        M = Variable(X.data.clone()).to(self.device)
        I = Variable(M.data.clone()).to(self.device)
        E = Variable(M.data.clone()).to(self.device)
        K = Variable(M.data.clone()).to(self.device)
        P = Variable(M.data.clone()).to(self.device)
        R = Variable(M.data.clone()).to(self.device)
        Q = Variable(M.data.clone()).to(self.device)
        L = Variable(M.data.clone()).to(self.device)
        # M = torch.zeros(X.size()).to(self.device)
        # I = torch.zeros(X.size()).to(self.device)
        # E = torch.zeros(X.size()).to(self.device)
        # K = torch.zeros(X.size()).to(self.device)
        # P = torch.zeros(X.size()).to(self.device)
        # R = torch.zeros(X.size()).to(self.device)
        # Q = torch.zeros(X.size()).to(self.device)
        # L = torch.zeros(X.size()).to(self.device)
        rho1 = torch.zeros(X.size()).to(self.device)
        rho2 = torch.zeros(X.size()).to(self.device)
        rho3 = torch.zeros(X.size()).to(self.device)
        rho4 = torch.zeros(X.size()).to(self.device)

        filter_halfwidth = kernels.shape[-1] // 2
        pad_width = (
            filter_halfwidth,
            filter_halfwidth,
            filter_halfwidth,
            filter_halfwidth,
        )
        X_rp = torch.nn.functional.pad(X, pad=pad_width)

        H = p2o(kernels, X_rp.size()[-2:])
        H = H.to(self.device)
        Ht, HtH = torch.conj(H), torch.abs(H) ** 2
        rhos = [rho1, rho2, rho3, rho4]
        for n in range(self.n):
            lambda1 = lambda1_iters[:, :, :, n].view(N, 1, 1, 1)
            lambda2 = lambda2_iters[:, :, :, n].view(N, 1, 1, 1)
            lambda3 = lambda3_iters[:, :, :, n].view(N, 1, 1, 1)
            lambda4 = lambda4_iters[:, :, :, n].view(N, 1, 1, 1)
            lambda5 = lambda5_iters[:, :, :, n].view(N, 1, 1, 1)
            lambda6 = lambda6_iters[:, :, :, n].view(N, 1, 1, 1)
            lambda7 = lambda7_iters[:, :, :, n].view(N, 1, 1, 1)
            
            lambdas = [lambda1, lambda2, lambda3, lambda4, lambda5, lambda6, lambda7]
            P, R, Q, L, K, E, M, I, rhos = self.run_iter_rgb(X, P, R, Q, L, K, E, M, I, H, Ht, HtH, rhos, lambdas, logits, pad_width, filter_halfwidth)
            # H = torch.nn.functional.pad(H_, pad=pad_width)
            # Ht, HtH = torch.conj(H), torch.abs(H) ** 2

            P_list.append(P)
            Q_list.append(Q)
        L_final = self.illumination_enhance(Q_list[-1])
        output = L_final * P_list[-1]

        # dark_deblurred = Q_list[-1] * M_list[-1]
        # dark_reblurred = conv_fft_batch(torch.nn.functional.pad(dark_deblurred, pad=pad_width), H)
        # dark_reblurred = torch.real(torch.fft.ifftn(dark_reblurred, dim=[-2, -1]))

        # dark_reblurred = dark_reblurred[
        #     ..., filter_halfwidth:-filter_halfwidth, filter_halfwidth:-filter_halfwidth
        # ]
        return output

        
## still need some work here as L only has 1 channel