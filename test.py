#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   test.py
@Time    :   2024/04/07 21:36:41
@Author  :   Tu Vo
@Version :   1.0
@Contact :   vovantu.hust@gmail.com
@License :   (C)Copyright 2020-2021, Tu Vo
@Desc    :   KC Machine Learning Lab
"""

import os
import yaml
import lpips
import torch
import numpy as np
import torch.nn.functional as F

from tqdm import tqdm
from glob import glob
from models.jude import JUDE
from basicsr.utils import tensor2img
from utils.utils_image import save_img
from pytorch_ssim import ssim as ssim_metric
from utils.dataloader import parse_test as parse
from basicsr.data.lol_image_dataset import LOLImageDataset
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type="vgg").cuda()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

with open("options/train.yml") as f:
    opt = yaml.full_load(f)

seed = opt["manual_seed"]

"""
Initiate a model, and transfer to gpu
"""
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Number of GPUS available: ", torch.cuda.device_count())


"""
Setting up training data - blur kernels and photon levels
"""
model = JUDE(device=device, opt=opt)
print(
    "number of params {}".format(
        sum(p.numel() for p in model.parameters() if p.requires_grad)
    )
)
model = torch.nn.DataParallel(model).to(device)


"""
Load checkpoint
"""
load_path = (
    opt["path"]["save"]
    + "-"
    + str(opt["network"]["stages"])
    + "-"
    + str(opt["datasets"]["train"]["crop_size"])
    + "-"
    + opt["network"]["denoiser"]["name"]
)
if opt["val"]["checkpoint"] == "last":
    path = sorted(glob(load_path + "/*.pth"))[1]
else:
    path = sorted(glob(load_path + "/*.pth"))[0]
path = "model_zoo/BOWNet_kernel_prediction_model_v10-5-512-ResUNet_mix/bownet_best.pth"
checkpoint = torch.load(path)
model.load_state_dict(checkpoint["state_dict"])
print("Loading checkpoint from {}".format(path))
model.to(device)

# create saving folder
if opt["val"]["save_img"]["state"]:
    save_path = (
        opt["val"]["save_img"]["save_folder"]
        + "_"
        + str(opt["datasets"]["train"]["crop_size"])
    )
    if not os.path.exists(save_path):
        os.makedirs(save_path)

"""
prepare data
"""

data_test = LOLImageDataset(opt["datasets"]["test"])

test_loader = torch.utils.data.DataLoader(
    data_test,
    batch_size=1,
    shuffle=False,
    drop_last=False,
    num_workers=1,
    pin_memory=True,
    worker_init_fn=lambda seed: np.random.seed(seed),
)

"""
testing
"""


with tqdm(total=len(data_test), desc=f"Testing .....", unit="its") as pbar:
    idx = 0
    total_psnr = 0.0
    total_ssim = 0.0
    total_lpips = 0.0
    for test_data in test_loader:
        idx += 1

        with torch.no_grad():
            blurred, image, name = parse(test_data)
            _, _, H, W = blurred.size()
            H = int(int(H / 128) * 128)
            W = int(int(W / 128) * 128)
            blurred = blurred[..., :H, :W]
            image = image[..., :H, :W]

            dirname1 = os.path.basename(os.path.dirname(name[0]))
            dirname2 = os.path.basename(os.path.dirname(os.path.dirname(name[0])))
            name = os.path.basename(name[0])
            dirname3 = os.path.basename(load_path)
            full_dir_path = os.path.join(save_path, dirname2, dirname3, dirname1)
            os.makedirs(full_dir_path, exist_ok=True)

            x, y, k = (
                blurred.to(device),
                image.to(device),
                torch.from_numpy(np.array([1])).to(device),
            )
            out_z = model(x, k)
            psnr = psnr = (
                10
                * torch.log10(1 / F.mse_loss(tensor2img(out_z), tensor2img(y))).item()
            )
            _, _, H, W = out_z.size()
            down_ratio = max(1, round(min(H, W) / 256))
            ssim = ssim_metric(
                F.adaptive_avg_pool2d(
                    out_z, (int(H / down_ratio), int(W / down_ratio))
                ),
                F.adaptive_avg_pool2d(y, (int(H / down_ratio), int(W / down_ratio))),
                data_range=1,
                size_average=False,
            ).item()
            lpips = lpips_metric(out_z, y).item()
            total_psnr += psnr
            total_ssim += ssim
            total_lpips += lpips
            if opt["val"]["save_img"]["state"]:
                save_img(tensor2img(out_z), os.path.join(full_dir_path, name))
                with open("{}/metrics.txt".format(save_path), "a") as f:
                    f.writelines(
                        f"{dirname2}-{dirname1}-{name}: PSNR: {psnr:.4f} dB, SSIM: {ssim:.4f}, LPIPS: {lpips:.4f}\n"
                    )
            pbar.update(1)

    """
    Average PSNR over all images
    """

    avg_psnr = total_psnr / idx
    avg_ssim = total_ssim / idx
    avg_lpips = total_lpips / idx

    """
    Print the result
    """

    print(f"Average PSNR: {avg_psnr:.4f} dB")
    print(f"Average SSIM: {avg_ssim:.4f}")
    print(f"Average LPIPS: {avg_lpips:.4f}")
    with open("{}/metrics.txt".format(save_path), "a") as f:
        f.writelines(f"Average PSNR: {avg_psnr:.4f} dB\n")
        f.writelines(f"Average SSIM: {avg_ssim:.4f}\n")
        f.writelines(f"Average LPIPS: {avg_lpips:.4f}\n")
