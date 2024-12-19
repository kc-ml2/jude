import sys
import torch
import os
import cv2
import argparse
import numpy as np
from basicsr.utils import scandir
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
import torch.nn.functional as F
from pytorch_msssim import ssim as ssim_metric
from DISTS_pytorch import *
from torch import nn

import pyiqa

lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type="vgg").cuda()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result_path",
        type=str,
        default="/home/tuvv/workspaces/BOWNet_old/comparison/FFTformer/results/fourllie-fftformer/GoPro",
    )
    parser.add_argument(
        "--gt_path",
        type=str,
        default="/home/tuvv/workspaces/BOWNet_old/comparison/FourLLIE/results/test/images/GT",
    )
    parser.add_argument(
        "--metrics", nargs="+", default=["psnr", "ssim", "lpips", "dists", "mae",]
    )
    parser.add_argument(
        "--has_aligned", action="store_true", help="Input are cropped and aligned faces"
    )

    args = parser.parse_args()

    if args.result_path.endswith("/"):  # solve when path ends with /
        args.result_path = args.result_path[:-1]
    if args.gt_path.endswith("/"):  # solve when path ends with /
        args.gt_path = args.gt_path[:-1]

    # Initialize metrics
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    (
        iqa_psnr,
        iqa_ssim,
        iqa_lpips,
        iqa_dists,
        iqa_mae,
        iqa_brisque,
        iqa_ilmae,
        iqa_pi,
        iqa_musiq,
    ) = (None, None, None, None, None, None, None, None, None)
    psnr, ssim, lpips, dists, mae, brisque, ilmae, pi, musiq = [
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]
    (
        score_psnr_all,
        score_ssim_all,
        score_lpips_all,
        score_dists_all,
        score_mae_all,
        score_brisque_all,
        score_ilmae_all,
        score_pi_all,
        score_musiq_all,
    ) = (
        [],
        [],
        [],
        [],
        [],
        [0],
        [0],
        [0],
        [0],
    )
    (
        score_psnr_forder,
        score_ssim_forder,
        score_lpips_forder,
        score_dists_folder,
        score_mae_folder,
        score_brisque_folder,
        score_ilmae_folder,
        score_pi_folder,
        score_musiq_folder,
    ) = ({}, {}, {}, {}, {}, {}, {}, {}, {})

    f_result = open(os.path.join(args.result_path, "results.csv"), "w")
    print(args.metrics)
    if "psnr" in args.metrics:
        iqa_psnr = pyiqa.create_metric("psnr").to(device)
        iqa_psnr.eval()
    if "ssim" in args.metrics:
        iqa_ssim = pyiqa.create_metric("ssim").to(device)
        iqa_ssim.eval()
    if "lpips" in args.metrics:
        # iqa_lpips = pyiqa.create_metric('lpips').to(device)
        iqa_lpips = pyiqa.create_metric("lpips-vgg").to(device)
        iqa_lpips.eval()
    if "dists" in args.metrics:
        iqa_dists = DISTS().to(device).to(device)
        iqa_dists.eval()
    if "mae" in args.metrics:
        iqa_mae = nn.L1Loss().to(device)
        iqa_mae.eval()

    img_out_paths = sorted(
        list(
            scandir(
                args.result_path, suffix=("jpg", "png"), recursive=True, full_path=True
            )
        )
    )
    total_num = len(img_out_paths)

    for i, img_out_path in enumerate(img_out_paths):
        img_name = img_out_path.replace(args.result_path + "/", "")
        cur_i = i + 1
        print(f"[{cur_i}/{total_num}] Processing: {img_name}")
        forder_name = img_name[:4]

        if not forder_name in list(score_psnr_forder.keys()):
            score_psnr_forder[forder_name] = []
            score_ssim_forder[forder_name] = []
            score_lpips_forder[forder_name] = []
            score_dists_folder[forder_name] = []
            score_mae_folder[forder_name] = []

        img_out = cv2.imread(img_out_path).astype(np.float32) / 255.0
        img_out = np.transpose(img_out, (2, 0, 1))
        img_out = torch.from_numpy(img_out).float()

        # try:
        # img_gt_path = img_out_path.replace(args.result_path, args.gt_path)
        img_gt_path = os.path.join(
            args.gt_path,
            os.path.basename(os.path.dirname(img_out_path)),
            os.path.basename(img_out_path),
        )
        img_gt = cv2.imread(img_gt_path).astype(np.float32) / 255.0
        img_gt = np.transpose(img_gt, (2, 0, 1))
        img_gt = torch.from_numpy(img_gt).float()
        with torch.no_grad():
            img_out = img_out.unsqueeze(0).to(device)
            img_gt = img_gt.unsqueeze(0).to(device)

            _, _, H, W = img_gt.size()
            H = int(int(H / 128) * 128)
            W = int(int(W / 128) * 128)
            img_gt = img_gt[..., :H, :W]
            img_out = img_out[..., :H, :W]
            if iqa_psnr is not None:
                # psnr = iqa_psnr(img_out, img_gt).item()
                psnr = 10 * torch.log10(1 / F.mse_loss(img_out, img_gt)).item()
                score_psnr_forder[forder_name].append(psnr)
                score_psnr_all.append(psnr)
            if iqa_ssim is not None:
                # ssim = iqa_ssim(img_out, img_gt).item()
                _, _, H, W = img_out.size()
                down_ratio = max(1, round(min(H, W) / 256))
                ssim_x = ssim_metric(
                    F.adaptive_avg_pool2d(
                        img_out, (int(H / down_ratio), int(W / down_ratio))
                    ),
                    F.adaptive_avg_pool2d(
                        img_gt, (int(H / down_ratio), int(W / down_ratio))
                    ),
                    data_range=1,
                    size_average=False,
                ).item()
                score_ssim_forder[forder_name].append(ssim_x)
                score_ssim_all.append(ssim_x)
            if iqa_lpips is not None:
                # lpips = iqa_lpips(img_out, img_gt).item()
                lpips = lpips_metric(img_out, img_gt).item()
                score_lpips_forder[forder_name].append(lpips)
                score_lpips_all.append(lpips)
            if iqa_dists is not None:
                dists = iqa_dists(img_out, img_gt)
                score_dists_folder[forder_name].append(dists)
                score_dists_all.append(dists)
            if iqa_mae is not None:
                mae = iqa_mae(img_out, img_gt).item()
                score_mae_folder[forder_name].append(mae)
                score_mae_all.append(mae)
            if iqa_brisque is not None:
                brisque = iqa_brisque(img_out).item()
                score_brisque_folder[forder_name].append(brisque)
                score_brisque_all.append(brisque)
            if iqa_ilmae is not None:
                ilmae = iqa_ilmae(img_out).item()
                score_ilmae_folder[forder_name].append(ilmae)
                score_ilmae_all.append(ilmae)
            if iqa_pi is not None:
                pi = iqa_pi(img_out).item()
                score_pi_folder[forder_name].append(pi)
                score_pi_all.append(pi)
            if iqa_musiq is not None:
                musiq = iqa_musiq(img_out).item()
                score_musiq_folder[forder_name].append(musiq)
                score_musiq_all.append(musiq)

            f_result.write(
                "%s,%.02f,%.03f,%.03f,%.03f,%.03f,%.03f,%.03f,%.03f,%.03f\n"
                % (
                    img_gt_path,
                    psnr,
                    ssim,
                    lpips,
                    dists,
                    mae,
                    brisque,
                    ilmae,
                    pi,
                    musiq,
                )
            )
        # except:
        #     print(f"skip: {img_name}")
        #     continue
        if (i + 1) % 2000 == 0:
            print(
                f"[{cur_i}/{total_num}] PSNR: {sum(score_psnr_all)/len(score_psnr_all)},\n \
                      SSIM: {sum(score_ssim_all)/len(score_ssim_all)},\n \
                      LPIPS: {sum(score_lpips_all)/len(score_lpips_all)},\n \
                    dists: {sum(score_dists_all)/len(score_dists_all)},\n \
                    mae: {sum(score_mae_all)/len(score_mae_all)},\n "
            )

    print("-------------------Final Scores-------------------\n")
    print(
        f"Average:\
            PSNR: {sum(score_psnr_all)/len(score_psnr_all)},\n \
            SSIM: {sum(score_ssim_all)/len(score_ssim_all)},\n \
            LPIPS: {sum(score_lpips_all)/len(score_lpips_all)},\n \
    dists: {sum(score_dists_all)/len(score_dists_all)},\n \
    mae: {sum(score_mae_all)/len(score_mae_all)},\n "
    )

    for k in list(score_psnr_forder.keys()):
        print(
            f"Folder Name: {k}\
                PSNR: {sum(score_psnr_forder[k])/len(score_psnr_forder[k])},\n \
                SSIM: {sum(score_ssim_forder[k])/len(score_ssim_forder[k])},\n \
                LPIPS: {sum(score_lpips_all)/len(score_lpips_all)},\n \
        dists: {sum(score_dists_folder[k])/len(score_dists_folder[k])},\n \
        mae: {sum(score_mae_folder[k])/len(score_mae_folder[k])},\n "
        )

    # Output test results to text file
    result_file = open(os.path.join(args.result_path, "test_result.txt"), "w")
    sys.stdout = result_file
    print("-------------------Final Scores-------------------\n")
    print(
        f"Average:\
            PSNR: {sum(score_psnr_all)/len(score_psnr_all)},\n \
            SSIM: {sum(score_ssim_all)/len(score_ssim_all)},\n \
            LPIPS: {sum(score_lpips_all)/len(score_lpips_all)},\n \
    dists: {sum(score_dists_folder[k])/len(score_dists_folder[k])},\n \
    mae: {sum(score_mae_folder[k])/len(score_mae_folder[k])},\n "
    )

    for k in list(score_psnr_forder.keys()):
        print(
            f"Folder Name: {k}\
                PSNR: {sum(score_psnr_forder[k])/len(score_psnr_forder[k])},\n \
                SSIM: {sum(score_ssim_forder[k])/len(score_ssim_forder[k])},\n \
                LPIPS: {sum(score_lpips_all)/len(score_lpips_all)},\n \
        dists: {sum(score_dists_folder[k])/len(score_dists_folder[k])},\n \
        mae: {sum(score_mae_folder[k])/len(score_mae_folder[k])},\n "
        )
    result_file.close()
    f_result.close()
