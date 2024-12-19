#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   model_v10.py
@Time    :   2024/06/13 11:31:47
@Author  :   Tu Vo
@Version :   1.0
@Contact :   vovantu.hust@gmail.com
@License :   (C)Copyright 2020-2021, Tu Vo
@Desc    :   KC Machine Learning Lab -> v9 + reflectance enhancement
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable


from models.ResUNet import ResUNet, IRCNN
from models.TwoHeadsNetwork import TwoHeadsNetwork_v2 as KernelsNetwork
from models.illumination_enhance import RelightNetv2

from utils.utils_torch import (
    conv_fft_batch,
    hadamard_batches,
)


def p2o(psf, shape):
    """
    Args:
        psf: NxCxhxw
        shape: [H,W]

    Returns:
        otf: NxCxHxWx2
    """
    otf = torch.zeros(psf.shape[:-2] + shape).type_as(psf)
    otf[..., : psf.shape[2], : psf.shape[3]].copy_(psf)
    for axis, axis_size in enumerate(psf.shape[2:]):
        otf = torch.roll(otf, -int(axis_size / 2), dims=axis + 2)
    otf = torch.fft.fftn(otf, dim=(-2, -1))
    return otf


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
            nn.Linear(64, 10 * (self.n)),
            nn.Softplus(),
        )
        self.resize = nn.Upsample(size=[256, 256], mode="bilinear", align_corners=True)

    def forward(self, kernel):
        N, C, H, W = kernel.size()
        h1, h2 = int(np.floor(0.5 * (128 - H))), int(np.ceil(0.5 * (128 - H)))
        w1, w2 = int(np.floor(0.5 * (128 - W))), int(np.ceil(0.5 * (128 - W)))
        k_pad = F.pad(kernel, (w1, w2, h1, h2), "constant", 0)
        A = torch.fft.fftn(k_pad, dim=[-2, -1])
        AtA_fft = torch.abs(A) ** 2
        x = self.conv_layers(AtA_fft.float())
        h = self.mlp(x.view(N, 1, 16 * 8 * 8).float()) + 1e-6

        lambda1_iters = h[:, :, 0 : self.n].view(N, 1, 1, self.n)
        lambda2_iters = h[:, :, self.n : 2 * self.n].view(N, 1, 1, self.n)
        lambda4_iters = h[:, :, 3 * self.n : 4 * self.n].view(N, 1, 1, self.n)
        lambda5_iters = h[:, :, 4 * self.n : 5 * self.n].view(N, 1, 1, self.n)
        lambda6_iters = h[:, :, 5 * self.n : 6 * self.n].view(N, 1, 1, self.n)
        gamma1_iters = h[:, :, 6 * self.n : 7 * self.n].view(N, 1, 1, self.n)
        gamma2_iters = h[:, :, 7 * self.n : 8 * self.n].view(N, 1, 1, self.n)
        gamma3_iters = h[:, :, 8 * self.n : 9 * self.n].view(N, 1, 1, self.n)
        return (
            lambda1_iters,
            lambda2_iters,
            lambda4_iters,
            lambda5_iters,
            lambda6_iters,
            gamma1_iters,
            gamma2_iters,
            gamma3_iters,
        )


class update_P(nn.Module):
    """
    g2(P) + ||P - (lambda2*Q*Z + lambda4 * R + rho2)/(Q*Q * lambda2 + lambda4 )|| * (Q*Q * lambda2 + lambda4)
    """

    def __init__(self):
        super().__init__()
        self.denoiser = ResUNet(in_nc=4, out_nc=3)

    def forward(self, Q, Z, R, lambda2, lambda4, rho2, gamma1):
        QZ = Q * Z
        input_tensor = (lambda2 * QZ + lambda4 * R + rho2) / (Q * Q * lambda2 + lambda4)
        input_tensor = torch.cat(
            (
                input_tensor,
                gamma1.repeat(1, 1, input_tensor.size(2), input_tensor.size(3)),
            ),
            dim=1,
        )
        P_ = self.denoiser(input_tensor)
        return P_


class update_R(nn.Module):
    """
    R = (lambda4*P - rho2) / (lambda4)
    """

    def __init__(self):
        super().__init__()

    def forward(self, P, lambda4, rho2):
        R_ = (lambda4 * P - rho2) / (lambda4)
        return R_


class update_Q(nn.Module):
    """
    g3(Q) + ||Q - (lambda2 * P*Z + rho3 + L * lambda5)/(P*P * lambda2 + lambda5)|| * (lambda5 + P * P * lambda2)
    """

    def __init__(self):
        super().__init__()
        self.denoiser = ResUNet(in_nc=4, out_nc=3)

    def forward(self, P, Z, L, lambda2, lambda5, rho3, gamma2):
        input_tensor = (lambda2 * P * Z + rho3 + L * lambda5) / (
            P * P * lambda2 + lambda5
        )
        input_tensor = torch.cat(
            (
                input_tensor,
                gamma2.repeat(1, 1, input_tensor.size(2), input_tensor.size(3)),
            ),
            dim=1,
        )
        Q_ = self.denoiser(input_tensor)
        return Q_


class update_L(nn.Module):
    """
    R = (Q * lambda5 - rho3) / (lambda5)
    """

    def __init__(self):
        super().__init__()

    def forward(self, Q, lambda5, rho3):
        L_ = (Q * lambda5 - rho3) / (lambda5)
        return L_


class update_U(nn.Module):
    """
    U = (lambda1 * X + lambda3 * HI + rho1) / (lambda1 + lambda3)
    """

    def __init__(self):
        super().__init__()

    def forward(self, X, HI, lambda1, lambda3, rho1):
        U_ = (lambda1 * X + lambda3 * HI + rho1) / (lambda1 + lambda3)
        return U_


class update_Z(nn.Module):
    """
    g4(Z) + ||Z - (lambda2*P*Q + lambda6 * I + rho4) /(lambda2 + lambda6)|| * (lambda2 + lambda6)
    """

    def __init__(self):
        super().__init__()
        self.denoiser = ResUNet(in_nc=4, out_nc=3)

    def forward(self, I, P, Q, rho4, lambda2, lambda6, gamma3):
        input_tensor = (lambda2 * P * Q + lambda6 * I + rho4) / (lambda2 + lambda6)
        input_tensor = torch.cat(
            (
                input_tensor,
                gamma3.repeat(1, 1, input_tensor.size(2), input_tensor.size(3)),
            ),
            dim=1,
        )
        Z_ = self.denoiser(input_tensor)
        return Z_


class update_E(nn.Module):
    """
    ||E||_1*(lambda3/(lambda4+lambda1)) + 1/2||E - 1/(lambda4+lambda1)(X + lambda4 * K - rho1 - HI)||2
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        K,
        H,
        I,
        X,
        lambda1,
        lambda3,
        lambda4,
        logits,
        rho1,
        pad_width,
        filter_halfwidth,
    ):
        I_patch = torch.nn.functional.pad(I, pad=pad_width)
        HI = conv_fft_batch(I_patch, H)
        HI = torch.real(torch.fft.ifftn(HI, dim=[-2, -1]))

        HI = HI[
            ..., filter_halfwidth:-filter_halfwidth, filter_halfwidth:-filter_halfwidth
        ]
        HI = hadamard_batches(HI, logits)
        N = (X + lambda4 * K - rho1 - HI) / (lambda4 + lambda1)
        E_ = torch.mul(
            torch.sign(N),
            nn.functional.relu(torch.abs(N) - lambda3 / (lambda4 + lambda1)),
        )
        return E_


class update_M(nn.Module):
    """
    g1(M) + lambda5/2||M - (rho2 + lambda5 * I)/lambda5||
    """

    def __init__(self):
        super().__init__()
        self.denoiser = ResUNet(in_nc=4, out_nc=3)

    def forward(self, I, lambda5, rho2, gamma4):
        input_tensor = (rho2 + I * lambda5) / (lambda5)
        input_tensor = torch.cat(
            (
                input_tensor,
                gamma4.repeat(1, 1, input_tensor.size(2), input_tensor.size(3)),
            ),
            dim=1,
        )
        M_ = self.denoiser(input_tensor)
        return M_


class update_I(nn.Module):
    """
    I = F^-1{F(lambda1 * X)H^T + lambda6 * Z - rho4) / (lambda1 * F(H)^2 + lambda6)}
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        X,
        Z,
        HT,
        HHT,
        logits,
        lambda1,
        lambda6,
        rho4,
        pad_width,
        filter_halfwidth,
    ):
        X_patch = torch.nn.functional.pad(X, pad=pad_width)
        XHt = conv_fft_batch(X_patch, HT)
        XHt = torch.real(torch.fft.ifftn(XHt, dim=[-2, -1]))

        XHt = XHt[
            ..., filter_halfwidth:-filter_halfwidth, filter_halfwidth:-filter_halfwidth
        ]
        HHT = HHT[
            ..., filter_halfwidth:-filter_halfwidth, filter_halfwidth:-filter_halfwidth
        ]

        XHt = hadamard_batches(XHt, logits)
        inter = torch.fft.fftn(lambda1 * XHt + lambda6 * Z - rho4, dim=[-2, -1])
        I_ = torch.real(
            torch.fft.ifftn((inter) / (lambda1 * HHT + lambda6), dim=[-2, -1])
        )
        return I_


class update_H(nn.Module):
    """
    ||R - H*I||^2
    """

    def __init__(self):
        super().__init__()

    def forward(self, P, I):
        H = torch.fft.ifftn(torch.fft.fftn(P) / torch.fft.fftn(I))
        return H


class JUDE(nn.Module):
    """
    This version is for batches
    """

    def __init__(self, device, opt):
        super(JUDE, self).__init__()
        self.device = device
        self.n = opt["network"]["stages"]
        self.k = 1
        self.opt = opt

        self.init = InitNet(self.n, self.k)
        self.kernel = KernelsNetwork(K=self.k, scale=1)
        self.update_P = update_P()
        self.update_R = update_R()
        self.update_Q = update_Q()
        self.update_L = update_L()
        self.update_Z = update_Z()
        self.update_I = update_I()
        self.illumination_enhance = RelightNetv2()
        self.reflectance_denoise = IRCNN(in_nc=3, out_nc=3, nc=32)

    def run_iter_rgb(
        self,
        X,
        P,
        R,
        Q,
        L,
        Z,
        I,
        H,
        Ht,
        HtH,
        rhos,
        lambdas,
        gammas,
        logits,
        pad_width,
        filter_halfwidth,
    ):
        [lambda1, lambda2, lambda4, lambda5, lambda6] = lambdas
        [rho2, rho3, rho4] = rhos
        [gamma1, gamma2, gamma3] = gammas
        ## Update P
        P_ = self.update_P(Q, Z, R, lambda2, lambda4, rho2, gamma1)

        ## update R
        R_ = self.update_R(P_, lambda4, rho2)

        ## update Q
        Q_ = self.update_Q(P_, Z, L, lambda2, lambda5, rho3, gamma2)

        ## update L
        L_ = self.update_L(Q_, lambda5, rho3)

        ## update Z
        Z_ = self.update_Z(I, P_, Q_, rho4, lambda2, lambda6, gamma3)

        ## update I
        I_ = self.update_I.forward(
            X, Z_, Ht, HtH, logits, lambda1, lambda6, rho4, pad_width, filter_halfwidth
        )

        ## update auxialaries
        rho2 = rho2 + lambda4 * (R_ - P_)
        rho3 = rho3 + lambda5 * (L_ - Q_)
        rho4 = rho4 + lambda6 * (I_ - Z_)
        rhos = [rho2, rho3, rho4]
        return P_, R_, Q_, L_, Z_, I_, rhos

    def forward(self, X, kernels):
        P_list = []
        # I_list = []
        Q_list = []
        N, _, _, _ = X.size()

        """
        Estimate kernel
        """
        kernels, logits = self.kernel(X)

        # Generate auxiliary variables for convolution
        (
            lambda1_iters,
            lambda2_iters,
            lambda4_iters,
            lambda5_iters,
            lambda6_iters,
            gamma1_iters,
            gamma2_iters,
            gamma3_iters,
        ) = self.init(
            kernels
        )  # Hyperparameters
        I = Variable(X.data.clone()).to(self.device)
        Z = Variable(I.data.clone()).to(self.device)
        P = Variable(I.data.clone()).to(self.device)
        R = Variable(I.data.clone()).to(self.device)
        Q = Variable(I.data.clone()).to(self.device)
        L = Variable(I.data.clone()).to(self.device)
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
        rhos = [rho2, rho3, rho4]
        for n in range(self.n):
            lambda1 = lambda1_iters[:, :, :, n].view(N, 1, 1, 1)
            lambda2 = lambda2_iters[:, :, :, n].view(N, 1, 1, 1)
            lambda4 = lambda4_iters[:, :, :, n].view(N, 1, 1, 1)
            lambda5 = lambda5_iters[:, :, :, n].view(N, 1, 1, 1)
            lambda6 = lambda6_iters[:, :, :, n].view(N, 1, 1, 1)

            gamma1 = gamma1_iters[:, :, :, n].view(N, 1, 1, 1)
            gamma2 = gamma2_iters[:, :, :, n].view(N, 1, 1, 1)
            gamma3 = gamma3_iters[:, :, :, n].view(N, 1, 1, 1)

            lambdas = [lambda1, lambda2, lambda4, lambda5, lambda6]
            gammas = [gamma1, gamma2, gamma3]
            P, R, Q, L, Z, I, rhos = self.run_iter_rgb(
                X,
                P,
                R,
                Q,
                L,
                Z,
                I,
                H,
                Ht,
                HtH,
                rhos,
                lambdas,
                gammas,
                logits,
                pad_width,
                filter_halfwidth,
            )

            P_list.append(P)
            Q_list.append(Q)
        L_final = self.illumination_enhance(Q_list[-1], P_list[-1])
        R_final = self.reflectance_denoise(P_list[-1])
        output = L_final * R_final
        return output
