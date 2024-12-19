#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   utils_torch.py
@Time    :   2024/03/27 15:49:22
@Author  :   Tu Vo
@Version :   1.0
@Contact :   vovantu.hust@gmail.com
@License :   (C)Copyright 2020-2021, Tu Vo
@Desc    :   KC Machine Learning Lab
'''


import numpy as np
from utils.utils_deblur import gauss_kernel, pad, crop
from numpy.fft import fft2
import torch
import torch.fft
import torch.nn as nn
from collections import OrderedDict
import os
import cv2
import torch.nn.functional as F
from torchvision import models, transforms
from basicsr.archs.vgg_arch import VGGFeatureExtractor

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"


# functionName implies a torch version of the function
def fftn(x):
    x_fft = torch.fft.fftn(x, dim=[-2, -1])
    return x_fft


def ifftn(x):
    return torch.fft.ifftn(x, dim=[-2, -1])


def ifftshift(x):
    # Copied from user vmos1 in a forum - https://github.com/locuslab/pytorch_fft/issues/9
    for dim in range(len(x.size()) - 1, 0, -1):
        x = torch.roll(x, dims=dim, shifts=x.size(dim) // 2)
    return x


def p2o(psf, shape):
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
    otf[..., : psf.shape[1], : psf.shape[2]].copy_(psf)
    for axis, axis_size in enumerate(psf.shape[1:]):
        otf = torch.roll(otf, -int(axis_size / 2), dims=axis + 1)
    otf = torch.fft.fftn(otf, dim=(-2, -1))
    # n_ops = torch.sum(torch.tensor(psf.shape).type_as(psf) * torch.log2(torch.tensor(psf.shape).type_as(psf)))
    # otf[..., 1][torch.abs(otf[..., 1]) < n_ops*2.22e-16] = torch.tensor(0).type_as(psf)
    return otf


def p2o_3d(psf, shape):
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
    otf = torch.zeros(psf.shape[:-2] + shape).type_as(psf)
    otf[..., : psf.shape[2], : psf.shape[3]].copy_(psf)
    for axis, axis_size in enumerate(psf.shape[2:]):
        otf = torch.roll(otf, -int(axis_size / 2), dims=axis + 2)
    otf = torch.fft.fftn(otf, dim=(-2, -1))
    # n_ops = torch.sum(torch.tensor(psf.shape).type_as(psf) * torch.log2(torch.tensor(psf.shape).type_as(psf)))
    # otf[..., 1][torch.abs(otf[..., 1]) < n_ops*2.22e-16] = torch.tensor(0).type_as(psf)
    return otf


def conv_fft(H, x):
    if x.ndim > 3:
        # Batched version of convolution
        Y_fft = fftn(x) * H.repeat([x.size(0), 1, 1, 1])
        y = ifftn(Y_fft)
    if x.ndim == 3:
        # Non-batched version of convolution
        Y_fft = torch.fft.fftn(x, dim=[1, 2]) * H
        y = torch.fft.ifftn(Y_fft, dim=[1, 2])
    return y.real


def conv_fft_batch(x, H):
    # Batched version of convolution
    # ims_reshape = ims.view(-1, *ims.shape[-3:])
    # fs_reshape = fs.view(-1, *fs.shape[-2:])
    Y_fft = fftn(x) * H
    # y = ifftn(Y_fft)
    return Y_fft


def conv_fft_batches(x, H):
    # Batched version of convolution
    Y_fft = 0.0
    for i in range(H.shape[1]):
        Y_fft += fftn(x) * H[:, i, :, :].unsqueeze(1)
    return Y_fft


def conv_fft_batch_48(kernels, x):
    # Batched version of convolution
    Y_ffts = []
    for i in range(kernels.shape[1]):
        kernel = kernels[:, i, :, :]
        _, H = psf_to_otf(torch.stack([kernel, kernel, kernel], dim=1), x.size())
        H = H.cuda()
        Y_fft = fftn(x) * H
        Y_ffts.append(Y_fft)
    Y_ffts = torch.stack(Y_ffts, dim=0).sum(0)
    y = ifftn(Y_ffts)
    y = torch.real(y)
    # y = y / torch.max(y)
    y = normalize_0_to_1(y)
    return y


def conv_fft_batch_68(y, kernels):
    rhs = []
    kernels = kernels.permute(2, 1, 0, 3, 4)
    filter_halfwidth = kernels.shape[-1] // 2
    num_observations = kernels.shape[-3]
    y_rp = y.repeat_interleave(12, dim=0).permute(0, 2, 3, 1)
    pad_width = (
        0,
        0,
        filter_halfwidth,
        filter_halfwidth,
        filter_halfwidth,
        filter_halfwidth,
        0,
        0,
    )
    y_rp = torch.nn.functional.pad(y_rp, pad=pad_width)
    y_patches_o = extract_patches(
        y_rp, patch_size=84, num_rows=6, num_cols=8, padding=filter_halfwidth
    )
    y_patches = y_patches_o.squeeze()
    kernels = kernels.squeeze()
    for i in range(kernels.shape[1]):
        kernel, y_patch = kernels[:, i, :, :], y_patches[:, i, :, :]
        y_patch = y_patch.reshape(-1, *y_patch.shape[1:])
        _, A = psf_to_otf_68(kernel, y_patch.size())
        # A = torch.fft.fftn(kernels.squeeze(), y_patches.size())
        A = A.cuda()

        rh = conv_fft_batch(A, y_patch)
        rhs.append(rh)

    rhs = torch.stack(rhs, dim=0).sum(0)
    Y_fft = rhs[
        :, filter_halfwidth:-filter_halfwidth, filter_halfwidth:-filter_halfwidth
    ]
    Y_fft = torch.real(torch.fft.ifftn(Y_fft, dim=[-2, -1]))
    Y_fft = Y_fft.reshape(48, -1, 84, 84)
    Y_fft = stitch_patches(Y_fft.unsqueeze(-1).unsqueeze(2), 6, 8, stitch_axis=(-3, -2))
    Y_ffts = Y_fft.sum(0)
    Y_ffts = Y_ffts.permute(0, 3, 1, 2)
    x0 = torch.clamp(Y_ffts, 0, 1)
    return x0


def conv_fft_batch_68_v2(patches, kernels):
    lhs = 0
    for i in range(kernels.shape[0]):
        k, p = (
            kernels[i, :, :],
            patches[i, :, :],
        )
        k = psf_to_otf(k, p.size())
        lh = torch.fft.fftn(p, dim=[-2, -1]) * k
        lhs += lh
    Y_fft = torch.real(torch.fft.ifftn((lhs), dim=[-2, -1]))
    Y_fft = Y_fft / Y_fft.max()
    # Y_fft = torch.clamp(Y_fft, 0.0, 1.0)
    return Y_fft


def conv_fft_batch_68_v3(y_patches, kernels, size):
    kernels = kernels.permute(2, 1, 0, 3, 4)
    kernels = kernels.squeeze()
    # filter_halfwidth = kernels.shape[-1] // 2
    A = p2o_3d(kernels, y_patches.size()[-2:])
    A = A.to("cuda:0")
    lh = conv_fft_batch(y_patches, A)

    lhs = lh.sum(1)
    Y_fft = torch.real(torch.fft.ifftn((lhs), dim=[-2, -1]))
    # Y_fft = Y_fft[
    #     :,
    #     filter_halfwidth:-filter_halfwidth,
    #     filter_halfwidth:-filter_halfwidth,
    # ]
    Y_fft = Y_fft.reshape(48, -1, size, size)
    Y_fft = stitch_patches(Y_fft.unsqueeze(-1).unsqueeze(2), 6, 8, stitch_axis=(-3, -2))
    Y_fft = Y_fft.sum(0)
    Y_fft = Y_fft.permute(0, 3, 1, 2)
    return Y_fft


def conv_fft_batch_68_v4(y_patches, kernels, size):
    kernels = kernels.permute(2, 1, 0, 3, 4)
    kernels = kernels.squeeze()
    # filter_halfwidth = kernels.shape[-1] // 2
    A = p2o_3d(kernels, y_patches.size()[-2:])
    A = A.to("cuda:1")
    lh = conv_fft_batch(y_patches, A)

    lhs = lh.sum(1)
    Y_fft = torch.real(torch.fft.ifftn((lhs), dim=[-2, -1]))
    # Y_fft = Y_fft
    output = Y_fft.reshape(48, -1, size, size)
    output = stitch_patches(
        output.unsqueeze(-1).unsqueeze(2), 6, 8, stitch_axis=(-3, -2)
    )
    output = output.sum(0)
    output = output.permute(0, 3, 1, 2)
    return output


def img_to_tens(x):
    return torch.from_numpy(np.expand_dims(np.expand_dims(x, 0), 0))


def scalar_to_tens(x):
    return torch.Tensor([x]).view(1, 1, 1, 1)


def conv_kernel(k, x, mode="cyclic"):
    _, h, w = x.size()
    h1, w1 = np.shape(k)
    k = torch.from_numpy(np.expand_dims(k, 0))
    k_pad, H = psf_to_otf(k.view(1, 1, h1, w1), [1, 1, h, w])
    H = H.view(1, h, w)
    Ax = conv_fft(H, x)

    return Ax, k_pad


def conv_kernel_symm(k, x):
    _, h, w = x.size()
    h1, w1 = np.int32(h / 2), np.int32(w / 2)
    m = nn.ReflectionPad2d((h1, h1, w1, w1))
    x_pad = m(x.view(1, 1, h, w)).view(1, h + 2 * h1, w + 2 * w1)
    k_pad = torch.from_numpy(np.expand_dims(pad(k, [h + 2 * h1, w + 2 * w1]), 0))
    H = torch.fft.fftn(k_pad, dim=[1, 2])
    Ax_pad = conv_fft(H, x_pad)
    Ax = Ax_pad[:, h1 : h + h1, w1 : w + w1]
    return Ax, k_pad


def psf_to_otf_68(ker, size):
    psf = torch.zeros(size)
    # ker = ker.reshape(-1, *ker.shape[2:])
    # circularly shift

    centre = ker.shape[-1] // 2 + 1
    psf[:, :centre, :centre] = ker[:, (centre - 1) :, (centre - 1) :]
    psf[:, :centre, -(centre - 1) :] = ker[:, (centre - 1) :, : (centre - 1)]
    psf[:, -(centre - 1) :, :centre] = ker[:, : (centre - 1), (centre - 1) :]
    psf[:, -(centre - 1) :, -(centre - 1) :] = ker[:, : (centre - 1), : (centre - 1)]
    # compute the otf
    # otf = torch.rfft(psf, 3, onesided=False)
    otf = torch.fft.fftn(psf, dim=[-2, -1])
    return psf, otf


def psf_to_otf_68_chatgpt(ker, size):
    psf = torch.zeros(size)

    # Calculate the center index of the PSF
    center = ker.shape[-1] // 2

    # Circularly shift the PSF to the center
    psf[..., :center, :center] = ker[..., -center:, -center:]
    psf[..., :center, -center:] = ker[..., -center:, :center]
    psf[..., -center:, :center] = ker[..., :center, -center:]
    psf[..., -center:, -center:] = ker[..., :center, :center]

    # Compute the OTF using FFT
    otf = torch.fft.fftn(psf, dim=(-2, -1))

    return otf


def psf_to_otf(psf, size):
    """
    Convert Point Spread Function (PSF) to Optical Transfer Function (OTF).

    Args:
        psf (torch.Tensor): The Point Spread Function.
        size (tuple): The size of the output OTF.

    Returns:
        torch.Tensor: The Optical Transfer Function (OTF).
    """
    # psf_pad = torch.nn.functional.pad(
    #     psf, pad=(0, size[0] - psf.shape[-2], 0, size[1] - psf.shape[-1])
    # )

    psf_fft = torch.fft.fftn(psf, size)
    # otf = torch.fft.fftshift(psf_fft)

    return psf_fft


def psf_to_otf_3d(ker, size):
    psf = torch.zeros(size)
    ker = ker.reshape(-1, *ker.shape[1:])
    # circularly shift

    centre = ker.shape[-1] // 2 + 1
    psf[:, :, :centre, :centre] = ker[:, :, (centre - 1) :, (centre - 1) :]
    psf[:, :, :centre, -(centre - 1) :] = ker[:, :, (centre - 1) :, : (centre - 1)]
    psf[:, :, -(centre - 1) :, :centre] = ker[:, :, : (centre - 1), (centre - 1) :]
    psf[:, :, -(centre - 1) :, -(centre - 1) :] = ker[
        :, :, : (centre - 1), : (centre - 1)
    ]
    # compute the otf
    # otf = torch.rfft(psf, 3, onesided=False)
    otf = torch.fft.fftn(psf, dim=[-2, -1])
    return otf


class MultiScaleLoss(torch.nn.Module):
    def __init__(self, scales=3, norm="L1"):
        super(MultiScaleLoss, self).__init__()
        self.scales = scales
        if norm == "L1":
            self.loss = torch.nn.L1Loss()
        if norm == "L2":
            self.loss = torch.nn.MSELoss()

        self.weights = torch.FloatTensor(
            [1 / (2**scale) for scale in range(self.scales)]
        )
        self.multiscales = [
            nn.AvgPool2d(2**scale, 2**scale) for scale in range(self.scales)
        ]

    def forward(self, output, target):
        loss = 0
        for i in range(self.scales):
            output_i, target_i = self.multiscales[i](output), self.multiscales[i](
                target
            )
            loss += self.weights[i] * self.loss(output_i, target_i)

        return loss


def rename_state_dict_keys(state_dict):
    new_state_dict = OrderedDict()
    for key, item in state_dict.items():
        new_key = key.partition(".")[2]
        new_state_dict[new_key] = item
    return new_state_dict


def extract_patches(images, patch_size, num_rows, num_cols, padding=0):
    """Divide images into image patches according to patch parameters

    Args:
      images: [..., #rows * P, #cols * P, C] height, width, #channels, P: patch size

    Returns:
      image_patches: [#rows * #cols, ..., P, P, C] The resulting image patches.
    """

    # xv, yv = torch.meshgrid(torch.arange(num_cols), torch.arange(num_rows))
    yv, xv = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols))
    yv = yv * patch_size
    xv = xv * patch_size

    patch_size_padding = patch_size + 2 * padding
    yv_size, xv_size = torch.meshgrid(
        torch.arange(patch_size_padding), torch.arange(patch_size_padding)
    )

    # yv_all = yv.reshape(-1)[..., None, None] + yv_size[None, ...]
    # xv_all = xv.reshape(-1)[..., None, None] + xv_size[None, ...]
    yv_all = yv.view(-1, 1, 1) + yv_size.view(-1, *yv_size.shape)
    xv_all = xv.view(-1, 1, 1) + xv_size.view(-1, *xv_size.shape)

    # yv_all = yv_all.view(-1)
    # xv_all = xv_all.view(-1)
    patches = images[..., yv_all, xv_all, :]
    patches = patches.permute(1, 0, 2, 3, 4)

    return patches


def stitch_patches(patches, num_rows, num_cols, stitch_axis):
    """Stitch patches according to the given dimension

    Args:
      patches: [#rows * #cols, ..., P, P, C] / [#rows * #cols, ..., F, F]
      stitch_axis: (-3, -2) / (-2, -1)

    Returns:
      [..., #rows * P, #cols * P, C]  stitched images / [..., #rows * F, #cols * F] stitched kernels
    """

    axis_row, axis_col = stitch_axis
    patches_reshape = patches.view(num_rows, num_cols, *patches.shape[1:])
    patches_reshape = patches_reshape.permute(2, 3, 0, 4, 1, 5, 6)
    new_shape = torch.tensor(patches.shape[1:])
    new_shape[axis_row] *= num_rows
    new_shape[axis_col] *= num_cols
    # images = patches_reshape.reshape(list(new_shape.numpy()))
    images = patches_reshape.contiguous().view(*list(new_shape.numpy()))

    return images


def normalize_0_to_1(x):
    if (torch.max(x) - torch.min(x)) < 1e-10:
        y = 0.5 * torch.ones_like(x)
    else:
        y = (x - torch.min(x)) / (torch.max(x) - torch.min(x))
    return y


class Dynamic_conv(nn.Module):
    def __init__(self, kernel_size):
        super(Dynamic_conv, self).__init__()

        self.reflect_pad = nn.ReflectionPad2d(kernel_size // 2)
        self.kernel_size = kernel_size

    def forward(self, x, kernel):
        out = torch.zeros_like(x)
        x = x[:, 0, :, :].unsqueeze(1)
        kernel = kernel.unsqueeze(2)
        b, c, h, w = x.size()
        x = self.reflect_pad(x)

        # kernel = F.softmax(kernel, dim=1)
        for i in range(kernel.shape[0]):
            k = kernel[i, :, :, :, :]
            im = x[i, :, :, :].unsqueeze(0)
            out_i = F.conv2d(im, k).permute(1, 0, 2, 3)
            out_i = out_i / torch.max(out_i)
            out[:, i, :, :] = out_i.squeeze()

        return out


def forward_reblur(sharp_estimated, kernels, masks, device, size="same"):
    n_kernels = kernels.size(1)
    K = kernels.size(-1)
    N = sharp_estimated.size(0)
    C = sharp_estimated.size(1)
    H = sharp_estimated.size(2)
    W = sharp_estimated.size(3)
    if size == "valid":
        H += -K + 1
        W += -K + 1
    else:
        padding = torch.nn.ReflectionPad2d(K // 2)
        sharp_estimated = padding(sharp_estimated)

    output_reblurred = torch.empty(N, n_kernels, C, H, W).to(device)
    for num in range(N):  # print('n = ',n)
        for c in range(C):
            # print('gt padded one channel shape: ', gt_n_padded_c.shape)

            conv_output = F.conv2d(
                sharp_estimated[num : num + 1, c : c + 1, :, :],
                kernels[num][:, np.newaxis, :, :],
            )

            # print('conv output shape: ', conv_output.shape)
            output_reblurred[num : num + 1, :, c, :, :] = (
                conv_output * masks[num : num + 1]
            )
            del conv_output

    # print('reblur_image shape before sum:', reblurred_images.shape)
    output_reblurred = torch.sum(output_reblurred, (1))

    output_reblurred = apply_saturation_function(output_reblurred, 0.5)

    return output_reblurred


def apply_saturation_function(img, max_value=0.5, get_derivative=False):
    """
    Implements the saturated function proposed by Whyte
    https://www.di.ens.fr/willow/research/saturation/whyte11.pdf
    :param img: input image may have values above max_value
    :param max_value: maximum value
    :return:
    """

    a = 50
    img[img > max_value + 0.5] = max_value + 0.5  # to avoid overflow in exponential

    if get_derivative == False:
        saturated_image = img - 1.0 / a * torch.log(
            1 + torch.exp(a * (img - max_value))
        )
        output_image = F.relu(saturated_image + (1 - max_value)) - (1 - max_value)

        # del saturated_image
    else:
        output_image = 1.0 / (1 + torch.exp(a * (img - max_value)))

    return output_image


def hadamard(x, kmap):
    # Compute hadamard product (pixel-wise)
    # x: input of shape (C,H,W)
    # kmap: input of shape (H,W)

    C, H, W = x.shape
    kmap = kmap.view(1, H, W)
    kmap = kmap.repeat(C, 1, 1)
    return x * kmap


def convolve_tensor(x, k):
    # Compute product convolution
    # x: input of shape (C,H,W)
    # k: input of shape (H_k,W_k)

    H_k, W_k = k.shape
    C, H, W = x.shape
    k = torch.flip(k, dims=(0, 1))
    k = k.view(1, 1, H_k, W_k).repeat(C, 1, 1, 1)
    x = x[None]
    x = torch.nn.functional.pad(
        x, (W_k // 2, W_k // 2, H_k // 2, H_k // 2), mode="circular"
    )
    o = torch.nn.functional.conv2d(x, k, groups=C, padding=0, stride=1)
    return o[0]


def cross_correlate_tensor(x, k):
    # x: input of shape (C,H,W)
    # k: input of shape (H_k,W_k)

    C, H, W = x.shape
    H_k, W_k = k.shape
    # k = torch.flip(k, dims =(0,1))
    k = k.view(1, 1, H_k, W_k).repeat(C, 1, 1, 1)
    x = x[None]
    x = torch.nn.functional.pad(
        x, (W_k // 2, W_k // 2, H_k // 2, H_k // 2), mode="circular"
    )
    o = torch.nn.functional.conv2d(x, k, groups=C, padding=0, stride=1)
    return o[0]


def o_leary(x, kmap, basis, manage_saturated_pixels=False):
    # Apply O'Leary convolution model blurry = sum(U_i H_i x)
    # x: input of shape (C,H,W)
    # kmap: input of shape (P,H,W)
    # basis: input of shape (P,H_k,W_k)

    assert len(kmap) == len(basis), str(len(kmap)) + "," + str(len(basis))
    c = 0
    for i in range(len(kmap)):
        c += hadamard(convolve_tensor(x, basis[i]), kmap[i])
    return c


def o_leary_batch(x, kmap, basis, manage_saturated_pixels=False):
    # Apply O'Leary convolution model blurry = sum(U_i H_i x)
    # x: input of shape (B,C,H,W)
    # kmap: input of shape (B,P,H,W)
    # basis: input of shape (B,P,H_k,W_k)

    assert len(x) == len(kmap) and len(kmap) == len(basis), print(
        "Batch size must be the same for all inputs"
    )

    return torch.cat([o_leary(x[i], kmap[i], basis[i])[None] for i in range(len(x))])


def o_leary_batch_v2(x, kmap, basis, manage_saturated_pixels=False):
    # Apply O'Leary convolution model blurry = sum(U_i H_i x)
    # x: input of shape (B,C,H,W)
    # kmap: input of shape (B,P,H,W)
    # basis: input of shape (B,P,H_k,W_k)

    assert len(kmap) == len(basis), print("48")

    return torch.cat([o_leary(x[i], kmap[i], basis[i])[None] for i in range(len(x))])


def o_leary_batches(x, kmap, basis, manage_saturated_pixels=False):
    # Apply O'Leary convolution model blurry = sum(U_i H_i x)
    # x: input of shape (B,C,H,W)
    # kmap: input of shape (B,P,H,W)
    # basis: input of shape (B,P,H_k,W_k)

    assert len(x) == len(kmap) and len(kmap) == len(basis), print(
        "Batch size must be the same for all inputs"
    )

    return torch.cat(
        [o_leary_batch_v2(x[i], kmap[i], basis[i])[None] for i in range(len(x))]
    )


def hadamard_batch(x, kmap):
    # Apply O'Leary convolution model blurry = sum(U_i H_i x)
    # x: input of shape (C,H,W)
    # kmap: input of shape (P,H,W)
    # basis: input of shape (P,H_k,W_k)
    c = 0
    for i in range(len(kmap)):
        c += hadamard(x, kmap[i])
    return c


def hadamard_batches(x, kmap):
    # Apply O'Leary convolution model blurry = sum(U_i H_i x)
    # x: input of shape (B,C,H,W)
    # kmap: input of shape (B,P,H,W)
    # basis: input of shape (B,P,H_k,W_k)

    return torch.cat([hadamard_batch(x[i], kmap[i])[None] for i in range(len(x))])


def hadamard_batches_4d(x, kmap):
    # Apply O'Leary convolution model blurry = sum(U_i H_i x)
    # x: input of shape (B,M,C,H,W)
    # kmap: input of shape (B,M,P,H,W)
    # basis: input of shape (B,M,P,H_k,W_k)

    return torch.cat([hadamard_batches(x[i], kmap[i])[None] for i in range(len(x))])


def transpose_o_leary(x, kmap, basis):
    # Apply the transpose of O'Leary convolution model blurry = sum(H_i^T U_i x)
    # x: input of shape (C,H,W)
    # kmap: input of shape (P,H,W)
    # basis: input of shape (P,H_k,W_k)

    assert len(kmap) == len(basis), str(len(kmap)) + "," + str(len(basis))
    c = 0
    for i in range(len(kmap)):
        c += cross_correlate_tensor(hadamard(x, kmap[i]), basis[i])
    return c


def transpose_o_leary_batch(x, kmap, basis):
    # Apply the transpose of O'Leary convolution model blurry = sum(H_i^T U_i x)
    # x: input of shape (B,C,H,W)
    # kmap: input of shape (B,P,H,W)
    # basis: input of shape (B,P,H_k,W_k)

    assert len(x) == len(kmap) and len(kmap) == len(basis), print(
        "Batch size must be the same for all inputs"
    )

    return torch.cat(
        [transpose_o_leary(x[i], kmap[i], basis[i])[None] for i in range(len(x))]
    )


def transpose_o_leary_batch_v2(x, kmap, basis):
    # Apply the transpose of O'Leary convolution model blurry = sum(H_i^T U_i x)
    # x: input of shape (B,C,H,W)
    # kmap: input of shape (B,P,H,W)
    # basis: input of shape (B,P,H_k,W_k)

    assert len(basis) == len(kmap), print("48")

    return torch.cat(
        [transpose_o_leary(x[i], kmap[i], basis[i])[None] for i in range(len(x))]
    )


def transpose_o_leary_batches(x, kmap, basis):
    # Apply the transpose of O'Leary convolution model blurry = sum(H_i^T U_i x)
    # x: input of shape (B,C,H,W)
    # kmap: input of shape (B,P,H,W)
    # basis: input of shape (B,P,H_k,W_k)

    assert len(x) == len(kmap) and len(kmap) == len(basis), print(
        "Batch size must be the same for all inputs"
    )

    return torch.cat(
        [
            transpose_o_leary_batch_v2(x[i], kmap[i], basis[i])[None]
            for i in range(len(x))
        ]
    )

def save_checkpoint(state, filename):
  torch.save(state, filename)


class VGGLoss(nn.Module):
    """Computes the VGG perceptual loss between two batches of images.

    The input and target must be 4D tensors with three channels
    ``(B, 3, H, W)`` and must have equivalent shapes. Pixel values should be
    normalized to the range 0–1.

    The VGG perceptual loss is the mean squared difference between the features
    computed for the input and target at layer :attr:`layer` (default 8, or
    ``relu2_2``) of the pretrained model specified by :attr:`model` (either
    ``'vgg16'`` (default) or ``'vgg19'``).

    If :attr:`shift` is nonzero, a random shift of at most :attr:`shift`
    pixels in both height and width will be applied to all images in the input
    and target. The shift will only be applied when the loss function is in
    training mode, and will not be applied if a precomputed feature map is
    supplied as the target.

    :attr:`reduction` can be set to ``'mean'``, ``'sum'``, or ``'none'``
    similarly to the loss functions in :mod:`torch.nn`. The default is
    ``'mean'``.

    :meth:`get_features()` may be used to precompute the features for the
    target, to speed up the case where inputs are compared against the same
    target over and over. To use the precomputed features, pass them in as
    :attr:`target` and set :attr:`target_is_features` to :code:`True`.

    Instances of :class:`VGGLoss` must be manually converted to the same
    device and dtype as their inputs.
    """

    models = {'vgg16': models.vgg16, 'vgg19': models.vgg19}

    def __init__(self, model='vgg16', layer=8, shift=0, reduction='mean'):
        super().__init__()
        self.shift = shift
        self.reduction = reduction
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])
        self.model = self.models[model](pretrained=True).features[:layer+1]
        self.model.eval()
        self.model.requires_grad_(False)

    def get_features(self, input):
        return self.model(self.normalize(input))

    def train(self, mode=True):
        self.training = mode

    def forward(self, input, target, target_is_features=False):
        if target_is_features:
            input_feats = self.get_features(input)
            target_feats = target
        else:
            sep = input.shape[0]
            batch = torch.cat([input, target])
            if self.shift and self.training:
                padded = F.pad(batch, [self.shift] * 4, mode='replicate')
                batch = transforms.RandomCrop(batch.shape[2:])(padded)
            feats = self.get_features(batch)
            input_feats, target_feats = feats[:sep], feats[sep:]
        return F.mse_loss(input_feats, target_feats, reduction=self.reduction)

class PerceptualLoss(nn.Module):
    """Perceptual loss with commonly used style loss.

    Args:
        layer_weights (dict): The weight for each layer of vgg feature.
            Here is an example: {'conv5_4': 1.}, which means the conv5_4
            feature layer (before relu5_4) will be extracted with weight
            1.0 in calculting losses.
        vgg_type (str): The type of vgg network used as feature extractor.
            Default: 'vgg19'.
        use_input_norm (bool):  If True, normalize the input image in vgg.
            Default: True.
        range_norm (bool): If True, norm images with range [-1, 1] to [0, 1].
            Default: False.
        perceptual_weight (float): If `perceptual_weight > 0`, the perceptual
            loss will be calculated and the loss will multiplied by the
            weight. Default: 1.0.
        style_weight (float): If `style_weight > 0`, the style loss will be
            calculated and the loss will multiplied by the weight.
            Default: 0.
        criterion (str): Criterion used for perceptual loss. Default: 'l1'.
    """

    def __init__(self,
                 layer_weights,
                 vgg_type='vgg19',
                 use_input_norm=True,
                 range_norm=False,
                 perceptual_weight=1.0,
                 style_weight=0.,
                 criterion='l1'):
        super(PerceptualLoss, self).__init__()
        self.perceptual_weight = perceptual_weight
        self.style_weight = style_weight
        self.layer_weights = layer_weights
        self.vgg = VGGFeatureExtractor(
            layer_name_list=list(layer_weights.keys()),
            vgg_type=vgg_type,
            use_input_norm=use_input_norm,
            range_norm=range_norm)

        self.criterion_type = criterion
        if self.criterion_type == 'l1':
            self.criterion = torch.nn.L1Loss()
        elif self.criterion_type == 'l2':
            self.criterion = torch.nn.L2loss()
        elif self.criterion_type == 'mse':
            self.criterion = torch.nn.MSELoss(reduction='mean')
        elif self.criterion_type == 'fro':
            self.criterion = None
        else:
            raise NotImplementedError(f'{criterion} criterion has not been supported.')

    def forward(self, x, gt):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).
            gt (Tensor): Ground-truth tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """
        # extract vgg features
        x_features = self.vgg(x)
        gt_features = self.vgg(gt.detach())

        # calculate perceptual loss
        if self.perceptual_weight > 0:
            percep_loss = 0
            for k in x_features.keys():
                if self.criterion_type == 'fro':
                    percep_loss += torch.norm(x_features[k] - gt_features[k], p='fro') * self.layer_weights[k]
                else:
                    percep_loss += self.criterion(x_features[k], gt_features[k]) * self.layer_weights[k]
            percep_loss *= self.perceptual_weight
        else:
            percep_loss = None

        # calculate style loss
        if self.style_weight > 0:
            style_loss = 0
            for k in x_features.keys():
                if self.criterion_type == 'fro':
                    style_loss += torch.norm(
                        self._gram_mat(x_features[k]) - self._gram_mat(gt_features[k]), p='fro') * self.layer_weights[k]
                else:
                    style_loss += self.criterion(self._gram_mat(x_features[k]), self._gram_mat(
                        gt_features[k])) * self.layer_weights[k]
            style_loss *= self.style_weight
        else:
            style_loss = None

        return percep_loss, style_loss

    def _gram_mat(self, x):
        """Calculate Gram matrix.

        Args:
            x (torch.Tensor): Tensor with shape of (n, c, h, w).

        Returns:
            torch.Tensor: Gram matrix.
        """
        n, c, h, w = x.size()
        features = x.view(n, c, w * h)
        features_t = features.transpose(1, 2)
        gram = features.bmm(features_t) / (c * h * w)
        return gram