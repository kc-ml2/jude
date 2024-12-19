

#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   dataloader.py
@Time    :   2024/03/27 15:48:58
@Author  :   Tu Vo
@Version :   1.0
@Contact :   vovantu.hust@gmail.com
@License :   (C)Copyright 2020-2021, Tu Vo
@Desc    :   KC Machine Learning Lab
'''

import sys

sys.path.insert(0, ".")
import os
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as F
import cv2
from glob import glob
import copy
from basicsr.utils import scandir
from utils.utils_image import read_img

np.random.seed(4)
torch.manual_seed(4)

from skimage.metrics import peak_signal_noise_ratio as PSNR
from skimage.metrics import structural_similarity as SSIM
import lpips
loss_fn_alex = lpips.LPIPS(net='alex').cuda()

def compute_metrics(out, gt):
    lpips_val=loss_fn_alex.forward(out.permute(2,0,1).to('cuda:0') * 2. - 1., gt.permute(2,0,1).to('cuda:0') * 2. - 1.).mean()
    out_numpy=out.squeeze().numpy().transpose(0,1,2).clip(0,1)
    gt_numpy=gt.squeeze().numpy().transpose(0,1,2).clip(0,1)
    psnr = PSNR(gt_numpy, out_numpy)
    ssim = SSIM(gt_numpy, out_numpy, channel_axis=-1, data_range=1)


    return {
        'psnr':psnr,
        'ssim':ssim,
        'lpips': lpips_val.item(),
        'out_numpy':out_numpy
    }


def parse(data):
    blur = data["lq"]
    image = data["gt"]
    kernel = data['kn']
    name = data['nm']
    return blur, image, kernel, name

def parse_test(data):
    blur = data["lq"]
    image = data["gt"]
    name = data['nm']
    return blur, image, name


def load_image(image_file, bit=16):
    """
    Read image, normalize and ensure even size
    """
    image = cv2.imread(image_file, -1)
    image = image / (2**bit-1)
    Hi, Wi, _ = image.shape
    image = image[: 2 * (Hi // 2), : 2 * (Wi // 2), :]
    return image.astype("float32")


def list_image_files(image_dir, gt=True):
    """
    List image files under "image_dir"
    """

    image_suffices = ["png", "bmp", "gif", "jpg"]
    image_files = []
    if gt:
        for suffix in image_suffices:
            image_files += sorted(glob(image_dir + suffix))
    else:
        for suffix in image_suffices:
            image_files += sorted(glob(image_dir + suffix))

    return image_files[:]
    
class BlurredImageDataset(Dataset):
    def __init__(self, data_dir, phase='test', dataset='DDDP'):
        if dataset == 'DDDP':
            if phase == "train+val":
                self.blur_image_files = sorted(list(scandir(data_dir + "/gt", suffix=('jpg', 'png'), recursive=True, full_path=True)))
                self.sharp_image_files = sorted(list(scandir(data_dir + "/low_blur", suffix=('jpg', 'png'), recursive=True, full_path=True)))

        assert len(self.blur_image_files) == len(self.sharp_image_files)
        self.num_images = len(self.sharp_image_files)

        self.train_img_list = []
        self.train_gt_list = []

        print("loading dataset...")
        for i in range(len(self.blur_image_files)):
            blurred = torch.from_numpy(read_img(self.blur_image_files[i])).permute(
            2, 0, 1)
            image = torch.from_numpy(read_img(self.sharp_image_files[i])).permute(
            2, 0, 1)
            self.train_img_list.append(blurred)
            self.train_gt_list.append(image)  

        print("loaded dataset...")

    def data_argumentation(self, img, hf, vf, rot):
        if hf:
            img = F.hflip(img)
        if vf:
            img = F.vflip(img)
        img = torch.rot90(img, rot, [1, 2])
        return img

    def __len__(self):
        return len(self.train_img_list)

    def __getitem__(self, idx):
        
        image = self.train_gt_list[idx]
        blurred = self.train_img_list[idx]
        kernel = self.kernel_list[idx]

        hf = np.random.randint(1, 2)
        vf = np.random.randint(1, 2)
        rot = np.random.randint(1, 4)

        blurred = self.data_argumentation(blurred, hf, vf, rot)
        image = self.data_argumentation(image, hf, vf, rot)

        return {
            "blurred": blurred,
            "image": image
        }
        

class RunTestPatch(Dataset):
    def __init__(self, data_dir, opt, resize=False, phase='test', model=None, dataset='RealDOF'):
        self.dataset = dataset
        self.bit = 16
        if dataset == 'RealDOF':
            self.blur_image_files = list_image_files(data_dir + "/source/*.", gt=False)
            self.sharp_image_files = list_image_files(data_dir + "/target/*.", gt=True)
            self.save_path = "results/RealDOF"
        elif dataset == 'CUHK':
            self.blur_image_files = list_image_files(data_dir + "/*.", gt=False)
            self.sharp_image_files = list_image_files(data_dir + "/*.", gt=True)
            self.save_path = "results/CUHK"
            self.bit = 8
        elif dataset == 'LFDOF':
            self.blur_image_files, self.sharp_image_files = self.load_all_lfdof_testset("/home/tuvv/poisson-deblurring/LFDOF/test_data/input", "/home/tuvv/poisson-deblurring/LFDOF/test_data/ground_truth")
            self.save_path = "results/LFDOF"
            self.bit = 8

        elif dataset == 'RTF':
            self.blur_image_files = list_image_files(data_dir + "/source/*.", gt=False)
            self.sharp_image_files = list_image_files(data_dir + "/target/*.", gt=True)
            self.save_path = "results/RTF"
            self.bit = 8
        else:    
            self.blur_image_files = list_image_files(data_dir + "/{}_c/source/*.".format(phase), gt=False)
            self.sharp_image_files = list_image_files(data_dir + "/{}_c/target/*.".format(phase), gt=True)
            self.save_path = "results/DDP"

        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        assert len(self.blur_image_files) == len(self.sharp_image_files)
        self.num_images = len(self.sharp_image_files)
        self.patch = 140
        self.rows = opt.rows
        self.cols = opt.cols
        self.rows_cut = opt.rows_cut
        self.cols_cut = opt.cols_cut
        self.crop_size = [self.rows_cut * opt.patch, self.cols_cut * opt.patch]
        self.resize = resize
        self.phase = phase

        self.train_img_list = []
        self.train_gt_list = []
        self.kernel_list = []
        print("loading dataset...")
        self.model = model

        # Load the data files path (use this when testing on all the LFDOF testing set)
    def load_all_lfdof_testset(self, img_path, gt_path):
        df_img_files_name,  gt_files_name = [], []
        fd_list = sorted(os.listdir(img_path))
        for fd in fd_list:
            last_path = os.path.join(img_path, fd)
            imgs = list(os.path.join(last_path, name) for name in os.listdir(last_path))
            df_img_files_name.extend(imgs)
            img_gt = os.path.join(gt_path, fd + '.png')
            gt_files_name.extend(img_gt for i in range(len(imgs)))

        return df_img_files_name, gt_files_name
    
    def run_gaussian_rgb(self):
        if self.dataset == 'RealDOF':
            img_height = 1536
            img_width = 2320
        elif self.dataset == 'LFDOF':
            img_height = 688
            img_width = 1008
        elif self.dataset == 'RTF':
            img_height = 360
            img_width = 360
        else:
            img_height = 1120
            img_width = 1680

        rows_cut = self.rows_cut
        cols_cut = self.cols_cut

        # Defocus Reconstruction
        # images = sorted(os.listdir(os.path.join(img_folder, 'HDR')))
        total_psnr = 0.0
        total_ssim = 0.0
        total_lpips = 0.0
        num_images = len(self.blur_image_files)
        for i in range(len(self.blur_image_files)):
            path = self.blur_image_files[i]
            if self.dataset == 'RealDOF':
                seq = path.split('/')[-1]
            else:
                seq = path.split('/')[-1]
            print(seq)

            # read blurry image 
            blurred = torch.from_numpy(load_image(self.blur_image_files[i], self.resize, 'test', bit=self.bit)).permute(
            2, 0, 1)
            image = torch.from_numpy(load_image(self.sharp_image_files[i], self.resize, 'test', bit=self.bit))
            # Storing final output image
            # Grid
            img_height, img_width, _ = image.shape
            w_grid = [0]; h_grid = [0]

            while True:
                if h_grid[-1] + self.patch * rows_cut < img_height:
                    h_grid.append(h_grid[-1] + self.patch)
                if w_grid[-1] + self.patch * rows_cut < img_width:
                    w_grid.append(w_grid[-1] + self.patch)
                else:
                    h_grid[-1] = img_height - self.patch * rows_cut
                    w_grid[-1] = img_width - self.patch * rows_cut
                    break
            w_grid = np.array(w_grid, dtype=np.uint16)
            h_grid = np.array(h_grid, dtype=np.uint16)

            # Gaussian mask 1D
            gm = np.round(matlab_style_gauss2D((1, self.patch * rows_cut), 35) * 87.5, 2)
            # Vertical split
            gm_ver = np.ones((1, self.patch * rows_cut))
            gm_ver[:, 0:self.patch] = gm[:, 0:self.patch]
            gm_ver = np.tile(gm_ver, (self.patch * rows_cut, 1))
            gm_ver = np.stack((gm_ver, gm_ver, gm_ver), axis=2)
            gm_ver_inv = np.fliplr(copy.deepcopy(gm_ver))
            gm_ver_inv[:, self.patch:self.patch * rows_cut, :] = 1 - gm_ver[:, 0:self.patch, :]
            # Horizontal Split
            gm_hor = np.ones((self.patch * rows_cut, 1))
            gm_hor[0:self.patch, :] = np.transpose(gm[:, 0:self.patch])
            gm_hor = np.tile(gm_hor, (1, self.patch * rows_cut))
            gm_hor = np.stack((gm_hor, gm_hor, gm_hor), axis=2)
            gm_hor_inv = np.flipud(copy.deepcopy(gm_hor))
            gm_hor_inv[self.patch:self.patch * rows_cut, :, :] = 1 - gm_hor[0:self.patch, :, :]
            # Gaussian mask 2D
            gm_2d = np.round(matlab_style_gauss2D((140 * 2, 140 * 2), 42.5) * 11280, 2)
            gm_2d = np.stack((gm_2d, gm_2d, gm_2d), axis=2)
            gm_2d_inv = 1 - gm_2d

            HDR = np.float32(np.zeros((img_height, img_width, 3)))
        
            # Patch reconstruction
            i = 0; j = 0
            try:
                while i < len(h_grid):
                    while j < len(w_grid):
                        h = h_grid[i]
                        w = w_grid[j]

                        # for k in range(3):

                        blurred_p = blurred[..., :3, h:h+self.patch*rows_cut, w:w+self.patch*cols_cut].unsqueeze(0)
                        # kernel_p = kernel[61 * i: 61 * (i + rows_cut), 61 * j: 61 * (j + cols_cut)]
                        # kernel_p = extract_patches(np.expand_dims(kernel_p, -1), **patch_params_crop_kernel, padding=0)[:, :, :, 0]
                        # filter_halfwidth = kernel_p.shape[-1] // 2
                        # kernel_resized = unp.rescale_blur_kernels(kernel_p, filter_halfwidth * 2 + 1, scales)
                        
                        with torch.no_grad():
                            hdr_patch = self.model(blurred_p, torch.from_numpy(np.array([0.0])).unsqueeze(0)).squeeze()
                            # hdr_patch = torch.clamp(hdr_patch, 0.0, 1.0).permute(1,2,0)
                            hdr_patch = hdr_patch.permute(1,2,0)
                            hdr_patch = hdr_patch.cpu().numpy()

                        if i == 0 and j == 0:
                            HDR[h:h+self.patch*rows_cut, w:w+self.patch*cols_cut, :] = copy.deepcopy(hdr_patch)
                        elif i == 0:
                            hdr_patch = np.multiply(hdr_patch, gm_ver)
                            HDR[h:h+self.patch*rows_cut, w-self.patch:w+self.patch, :] = np.multiply(HDR[h:h+self.patch*rows_cut, w-self.patch:w+self.patch, :], gm_ver_inv)
                            HDR[h:h+self.patch*rows_cut, w+self.patch:w+self.patch*cols_cut, :] = copy.deepcopy(hdr_patch[:, self.patch:self.patch*cols_cut])
                            HDR[h:h+self.patch*rows_cut, w:w+self.patch, :] = HDR[h:h+self.patch*rows_cut, w:w+self.patch, :] + hdr_patch[:, 0:self.patch]

                        elif j == 0:
                            hdr_patch = np.multiply(hdr_patch, gm_hor)
                            HDR[h-self.patch:h+self.patch, w:w+self.patch*cols_cut, :] = np.multiply(HDR[h-self.patch:h+self.patch, w:w+self.patch*cols_cut, :], gm_hor_inv)
                            HDR[h+self.patch:h+self.patch*rows_cut, w:w+self.patch*cols_cut, :] = copy.deepcopy(hdr_patch[self.patch:self.patch*rows_cut, :, :])
                            HDR[h:h+self.patch, w:w+self.patch*rows_cut, :] = HDR[h:h+self.patch, w:w+self.patch*cols_cut, :] + hdr_patch[0:self.patch, :, :]
                        else:
                            if i == len(h_grid) - 1:
                                HDR[h+self.patch:h+self.patch*rows_cut, w+int(self.patch//2):w+self.patch*cols_cut, :] = copy.deepcopy(hdr_patch[self.patch:self.patch*rows_cut, int(self.patch//2):self.patch*cols_cut])
                            elif j == len(w_grid) - 1:
                                HDR[h+int(self.patch//2):h+self.patch*rows_cut, w+self.patch:w+self.patch*cols_cut, :] = copy.deepcopy(hdr_patch[int(self.patch//2):self.patch*rows_cut, self.patch:self.patch*cols_cut])
                            else:
                                HDR[h+self.patch:h+self.patch*rows_cut, w+self.patch:w+self.patch*cols_cut, :] = copy.deepcopy(hdr_patch[self.patch:self.patch*rows_cut, self.patch:self.patch*cols_cut])
                            patch_2d = np.multiply(hdr_patch, gm_2d)
                            patch_2d_inv = np.multiply(HDR[h:h+self.patch*rows_cut, w:w+self.patch*cols_cut, :], gm_2d_inv)
                            HDR[h:h+self.patch*rows_cut, w:w+self.patch*cols_cut, :] = patch_2d + patch_2d_inv
                        j = j + 1
                    i = i + 1
                    j = 0
                metrics = compute_metrics(image[:,:,:3], torch.from_numpy(HDR))
                total_psnr += metrics['psnr']
                total_ssim += metrics['ssim']
                total_lpips += metrics['lpips']
                if self.bit == 8:
                    cv2.imwrite('{}/{}'.format(self.save_path, seq), np.uint8(np.clip(HDR, 0.0, 1.0)*255))
                else:
                    cv2.imwrite('{}/{}'.format(self.save_path, seq), np.uint16(np.clip(HDR, 0.0, 1.0)*65535))
                with open("{}/metrics.txt".format(self.save_path), "a") as f:
                    f.writelines(
                        f"{seq}: PSNR: {metrics['psnr']:.4f} dB, SSIM: {metrics['ssim']:.4f}, LPIPS: {metrics['lpips']:.4f}\n"
                    )
            except:
                continue
        # Average PSNR over all images
        avg_psnr = total_psnr / num_images
        avg_ssim = total_ssim / num_images
        avg_lpips = total_lpips / num_images
        # Print the result
        print(f"Average PSNR: {avg_psnr:.4f} dB")
        print(f"Average SSIM: {avg_ssim:.4f}")
        print(f"Average LPIPS: {avg_lpips:.4f}")
        with open("{}/metrics.txt".format(self.save_path), "a") as f:
            f.writelines(
                f"Average PSNR: {avg_psnr:.4f} dB\n"
            )
            f.writelines(
                f"Average SSIM: {avg_ssim:.4f}\n"
            )
            f.writelines(
                f"Average LPIPS: {avg_lpips:.4f}\n"
            )

    def run_gaussian_rgb_rtf(self):
        # Load model
        if self.dataset == 'RealDOF':
            img_height = 1536
            img_width = 2320
        elif self.dataset == 'LFDOF':
            img_height = 688
            img_width = 1008
        elif self.dataset == 'RTF':
            img_height = 360
            img_width = 360
        else:
            img_height = 1120
            img_width = 1680

        rows_cut = self.rows_cut
        cols_cut = self.cols_cut
        

        # Defocus Reconstruction
        # images = sorted(os.listdir(os.path.join(img_folder, 'HDR')))
        total_psnr = 0.0
        total_ssim = 0.0
        total_lpips = 0.0
        num_images = len(self.blur_image_files)
        for i in range(len(self.blur_image_files)):
            path = self.blur_image_files[i]
            if self.dataset == 'RealDOF':
                seq = path.split('/')[-1]
            else:
                # seq = path[:-4].split("_")[-1]
                seq = path.split('/')[-1]
            print(seq)

            # read blurry image 
            blurred = torch.from_numpy(load_image(self.blur_image_files[i], self.resize, 'test', bit=self.bit)).permute(
            2, 0, 1)
            blurred = torch.cat([blurred, blurred], dim=1)
            blurred = torch.cat([blurred, blurred], dim=2)
            image = torch.from_numpy(load_image(self.sharp_image_files[i], self.resize, 'test', bit=self.bit))
            image = torch.cat([image, image], dim=1)
            image = torch.cat([image, image], dim=0)
            # Storing final output image
            # Grid
            img_height, img_width, _ = image.shape
            w_grid = [0]; h_grid = [0]

            while True:
                if h_grid[-1] + self.patch * rows_cut < img_height:
                    h_grid.append(h_grid[-1] + self.patch)
                if w_grid[-1] + self.patch * rows_cut < img_width:
                    w_grid.append(w_grid[-1] + self.patch)
                else:
                    h_grid[-1] = img_height - self.patch * rows_cut
                    w_grid[-1] = img_width - self.patch * rows_cut
                    break
            w_grid = np.array(w_grid, dtype=np.uint16)
            h_grid = np.array(h_grid, dtype=np.uint16)

            # Gaussian mask 1D
            gm = np.round(matlab_style_gauss2D((1, self.patch * rows_cut), 35) * 87.5, 2)
            # Vertical split
            gm_ver = np.ones((1, self.patch * rows_cut))
            gm_ver[:, 0:self.patch] = gm[:, 0:self.patch]
            gm_ver = np.tile(gm_ver, (self.patch * rows_cut, 1))
            gm_ver = np.stack((gm_ver, gm_ver, gm_ver), axis=2)
            gm_ver_inv = np.fliplr(copy.deepcopy(gm_ver))
            gm_ver_inv[:, self.patch:self.patch * rows_cut, :] = 1 - gm_ver[:, 0:self.patch, :]
            # Horizontal Split
            gm_hor = np.ones((self.patch * rows_cut, 1))
            gm_hor[0:self.patch, :] = np.transpose(gm[:, 0:self.patch])
            gm_hor = np.tile(gm_hor, (1, self.patch * rows_cut))
            gm_hor = np.stack((gm_hor, gm_hor, gm_hor), axis=2)
            gm_hor_inv = np.flipud(copy.deepcopy(gm_hor))
            gm_hor_inv[self.patch:self.patch * rows_cut, :, :] = 1 - gm_hor[0:self.patch, :, :]
            # Gaussian mask 2D
            gm_2d = np.round(matlab_style_gauss2D((140 * 2, 140 * 2), 42.5) * 11280, 2)
            gm_2d = np.stack((gm_2d, gm_2d, gm_2d), axis=2)
            gm_2d_inv = 1 - gm_2d

            HDR = np.float32(np.zeros((img_height, img_width, 3)))
        
            # Patch reconstruction
            i = 0; j = 0
            try:
                while i < len(h_grid):
                    while j < len(w_grid):
                        h = h_grid[i]
                        w = w_grid[j]

                        # for k in range(3):

                        blurred_p = blurred[..., :3, h:h+self.patch*rows_cut, w:w+self.patch*cols_cut].unsqueeze(0)

                        with torch.no_grad():
                            hdr_patch = self.model(blurred_p, torch.from_numpy(np.array([0.0])).unsqueeze(0)).squeeze()
                            # hdr_patch = torch.clamp(hdr_patch, 0.0, 1.0).permute(1,2,0)
                            hdr_patch = hdr_patch.permute(1,2,0)
                            hdr_patch = hdr_patch.cpu().numpy()

                        if i == 0 and j == 0:
                            HDR[h:h+self.patch*rows_cut, w:w+self.patch*cols_cut, :] = copy.deepcopy(hdr_patch)
                        elif i == 0:
                            hdr_patch = np.multiply(hdr_patch, gm_ver)
                            HDR[h:h+self.patch*rows_cut, w-self.patch:w+self.patch, :] = np.multiply(HDR[h:h+self.patch*rows_cut, w-self.patch:w+self.patch, :], gm_ver_inv)
                            HDR[h:h+self.patch*rows_cut, w+self.patch:w+self.patch*cols_cut, :] = copy.deepcopy(hdr_patch[:, self.patch:self.patch*cols_cut])
                            HDR[h:h+self.patch*rows_cut, w:w+self.patch, :] = HDR[h:h+self.patch*rows_cut, w:w+self.patch, :] + hdr_patch[:, 0:self.patch]

                        elif j == 0:
                            hdr_patch = np.multiply(hdr_patch, gm_hor)
                            HDR[h-self.patch:h+self.patch, w:w+self.patch*cols_cut, :] = np.multiply(HDR[h-self.patch:h+self.patch, w:w+self.patch*cols_cut, :], gm_hor_inv)
                            HDR[h+self.patch:h+self.patch*rows_cut, w:w+self.patch*cols_cut, :] = copy.deepcopy(hdr_patch[self.patch:self.patch*rows_cut, :, :])
                            HDR[h:h+self.patch, w:w+self.patch*rows_cut, :] = HDR[h:h+self.patch, w:w+self.patch*cols_cut, :] + hdr_patch[0:self.patch, :, :]
                        else:
                            if i == len(h_grid) - 1:
                                HDR[h+self.patch:h+self.patch*rows_cut, w+int(self.patch//2):w+self.patch*cols_cut, :] = copy.deepcopy(hdr_patch[self.patch:self.patch*rows_cut, int(self.patch//2):self.patch*cols_cut])
                            elif j == len(w_grid) - 1:
                                HDR[h+int(self.patch//2):h+self.patch*rows_cut, w+self.patch:w+self.patch*cols_cut, :] = copy.deepcopy(hdr_patch[int(self.patch//2):self.patch*rows_cut, self.patch:self.patch*cols_cut])
                            else:
                                HDR[h+self.patch:h+self.patch*rows_cut, w+self.patch:w+self.patch*cols_cut, :] = copy.deepcopy(hdr_patch[self.patch:self.patch*rows_cut, self.patch:self.patch*cols_cut])
                            patch_2d = np.multiply(hdr_patch, gm_2d)
                            patch_2d_inv = np.multiply(HDR[h:h+self.patch*rows_cut, w:w+self.patch*cols_cut, :], gm_2d_inv)
                            HDR[h:h+self.patch*rows_cut, w:w+self.patch*cols_cut, :] = patch_2d + patch_2d_inv
                        j = j + 1
                    i = i + 1
                    j = 0
                metrics = compute_metrics(image[:360,:360,:3], torch.from_numpy(HDR)[:360,:360,:3])
                total_psnr += metrics['psnr']
                total_ssim += metrics['ssim']
                total_lpips += metrics['lpips']
                if self.bit == 8:
                    cv2.imwrite('{}/{}'.format(self.save_path, seq), np.uint8(np.clip(HDR[:360,:360,:3], 0.0, 1.0)*255))
                else:
                    cv2.imwrite('{}/{}'.format(self.save_path, seq), np.uint16(np.clip(HDR[:360,:360,:3], 0.0, 1.0)*65535))
                with open("{}/metrics.txt".format(self.save_path), "a") as f:
                    f.writelines(
                        f"{seq}: PSNR: {metrics['psnr']:.4f} dB, SSIM: {metrics['ssim']:.4f}, LPIPS: {metrics['lpips']:.4f}\n"
                    )
            except:
                continue
        # Average PSNR over all images
        avg_psnr = total_psnr / num_images
        avg_ssim = total_ssim / num_images
        avg_lpips = total_lpips / num_images
        # Print the result
        print(f"Average PSNR: {avg_psnr:.4f} dB")
        print(f"Average SSIM: {avg_ssim:.4f}")
        print(f"Average LPIPS: {avg_lpips:.4f}")
        with open("{}/metrics.txt".format(self.save_path), "a") as f:
            f.writelines(
                f"Average PSNR: {avg_psnr:.4f} dB\n"
            )
            f.writelines(
                f"Average SSIM: {avg_ssim:.4f}\n"
            )
            f.writelines(
                f"Average LPIPS: {avg_lpips:.4f}\n"
            )

def matlab_style_gauss2D(shape=(3,3),sigma=0.5):
  """
  2D gaussian mask - should give the same result as MATLAB's fspecial('gaussian',[shape],[sigma])
  Acknowledgement : https://stackoverflow.com/questions/171901869/how-to-obtain-a-gaussian-filter-in-python (Author@ali_m)
  """
  m,n = [(ss-1.)/2. for ss in shape]
  y,x = np.ogrid[-m:m+1,-n:n+1]
  h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
  h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
  sumh = h.sum()
  if sumh != 0:
    h /= sumh
  return h