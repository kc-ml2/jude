import sys
import torch
import os
import cv2
import argparse
import os.path as osp
import numpy as np
import glob
from basicsr.utils import scandir
import torchvision.transforms as T

import pyiqa


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result_path",
        type=str,
        default="test_results/BOWNet_LEDNetData/real_data/BOWNet_kernel_prediction_model_v5-10-256-ResUNet-noisy-data/real_lolblur_video/",
    )
    parser.add_argument(
        "--gt_path",
        type=str,
        default="test_results/BOWNet_LEDNetData/real_data/BOWNet_kernel_prediction_model_v5-10-256-ResUNet-noisy-data/real_lolblur_video/",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=[
            "cnniqa",
            "clipiqa+",
            "musiq-koniq",
            "maniqa",
            "tres-koniq",
            "topiq_nr",
            "dbcnn",
            "hyperiqa",
            "liqe",
        ],
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
    iqa_psnr, iqa_ssim, iqa_lpips, iqa_nrqm, iqa_niqe, iqa_musiq = (
        None,
        None,
        None,
        None,
        None,
        None,
    )
    (
        score_psnr_all,
        score_ssim_all,
        score_lpips_all,
        score_nrqm_all,
        score_niqe_all,
        score_brisque_all,
        score_ilniqe_all,
        score_pi_all,
        score_musiq_all,
    ) = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    (
        score_psnr_forder,
        score_ssim_forder,
        score_lpips_forder,
        score_nrqm_folder,
        score_niqe_folder,
        score_brisque_folder,
        score_ilniqe_folder,
        score_pi_folder,
        score_musiq_folder,
    ) = ({}, {}, {}, {}, {}, {}, {}, {}, {})

    f_result = open(os.path.join(args.result_path, "results_v3.csv"), "w")
    print(args.metrics)
    if "cnniqa" in args.metrics:  # smaller value better image , RGB , 224
        iqa_psnr = pyiqa.create_metric("cnniqa").to(device)
        iqa_psnr.eval()
    if "clipiqa+" in args.metrics:  # 224 higher value better image
        iqa_ssim = pyiqa.create_metric("clipiqa").to(device)
        iqa_ssim.eval()
    if "musiq-koniq" in args.metrics:
        # iqa_lpips = pyiqa.create_metric('lpips').to(device)
        iqa_lpips = pyiqa.create_metric("musiq").to(device)
        iqa_lpips.eval()
    if "maniqa" in args.metrics:  # 224 higher value better image
        iqa_nrqm = pyiqa.create_metric("maniqa").to(device)
        iqa_nrqm.eval()
    if "tres-koniq" in args.metrics:  # 224 resize before anything
        iqa_niqe = pyiqa.create_metric("tres-flive").to(device)
        iqa_niqe.eval()

    ## added
    if "topiq_nr" in args.metrics:
        iqa_brisque = pyiqa.create_metric("topiq_nr").to(device)
        iqa_brisque.eval()
    if "dbcnn" in args.metrics:  # 256 256  -> higher the better
        iqa_ilniqe = pyiqa.create_metric("dbcnn").to(device)
        iqa_ilniqe.eval()
    if (
        "hyperiqa" in args.metrics
    ):  # convert to RGB, resize 224 224 -> higher the better
        iqa_pi = pyiqa.create_metric("hyperiqa").to(device)
        iqa_pi.eval()
    if "liqe" in args.metrics:  # 384 -> higher the better
        iqa_musiq = pyiqa.create_metric("liqe", as_loss=False).to(device)
        iqa_musiq.eval()

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
            score_nrqm_folder[forder_name] = []
            score_niqe_folder[forder_name] = []
            score_brisque_folder[forder_name] = []
            score_ilniqe_folder[forder_name] = []
            score_pi_folder[forder_name] = []
            score_musiq_folder[forder_name] = []

        img_out = (
            cv2.imread(img_out_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.0
        )
        img_out = cv2.cvtColor(img_out, cv2.COLOR_BGR2RGB)
        img_out = np.transpose(img_out, (2, 0, 1))
        img_out = torch.from_numpy(img_out).float()

        # try:
        img_gt_path = img_out_path.replace(args.result_path, args.gt_path)
        img_gt = (
            cv2.imread(img_gt_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.0
        )
        img_gt = cv2.cvtColor(img_gt, cv2.COLOR_BGR2RGB)
        img_gt = np.transpose(img_gt, (2, 0, 1))
        img_gt = torch.from_numpy(img_gt).float()
        _, H, W = img_out.shape

        with torch.no_grad():
            img_out = img_out.unsqueeze(0).to(device)
            img_gt = img_gt.unsqueeze(0).to(device)
            if iqa_psnr is not None:
                # img_out_rgb = (
                #     cv2.imread(img_out_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
                #     / 1.0
                # )
                # img_out_rgb = cv2.cvtColor(img_out_rgb, cv2.COLOR_BGR2RGB)
                # img_out_rgb = np.transpose(img_out_rgb, (2, 0, 1))
                # img_out_rgb = torch.from_numpy(img_out_rgb).float()
                # img_out_rgb = img_out_rgb.unsqueeze(0).to(device)
                psnr = iqa_psnr(img_out).item()
                score_psnr_forder[forder_name].append(psnr)
                score_psnr_all.append(psnr)
            if iqa_ssim is not None:
                # H_ = 224
                # W_ = 224
                # transform = T.Resize(size=(H_, W_))
                # img_out_ = transform(img_out/255.)
                ssim = iqa_ssim(img_out).item()
                score_ssim_forder[forder_name].append(ssim)
                score_ssim_all.append(ssim)
            if iqa_lpips is not None:
                lpips = iqa_lpips(img_out).item()
                score_lpips_forder[forder_name].append(lpips)
                score_lpips_all.append(lpips)
            if iqa_nrqm is not None:
                # H_ = 224
                # W_ = 224
                # transform = T.Resize(size=(H_, W_))
                # img_out_ = transform(img_out)
                nrqm = iqa_nrqm(img_out).item()
                score_nrqm_folder[forder_name].append(nrqm)
                score_nrqm_all.append(nrqm)
            if iqa_niqe is not None:
                # transforms = T.Compose(
                #     [
                #         # T.Resize((224, 224)),
                #         # T.RandomCrop(size=224),
                #         T.ToTensor(),
                #     ]
                # )
                # T.Normalize(mean=(0.485, 0.456, 0.406),
                #                                  std=(0.229, 0.224, 0.225))])
                # img_out_ = transform(img_out)
                niqe = iqa_niqe(img_out).item()
                score_niqe_folder[forder_name].append(niqe)
                score_niqe_all.append(niqe)
            if iqa_brisque is not None:
                brisque = iqa_brisque(img_out).item()
                score_brisque_folder[forder_name].append(brisque)
                score_brisque_all.append(brisque)
            if iqa_ilniqe is not None:
                # transforms = T.Compose(
                #     [
                #         T.Resize((256, 256)),
                #         # T.RandomCrop(size=224),
                #         T.ToTensor(),
                #     ]
                # )
                # img_out_rgb = (
                #     cv2.imread(img_out_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
                #     / 255.0
                # )
                # img_out_rgb = cv2.cvtColor(img_out_rgb, cv2.COLOR_BGR2RGB)
                # img_out_rgb = np.transpose(img_out_rgb, (2, 0, 1))
                # img_out_rgb = torch.from_numpy(img_out_rgb).float()
                # img_out_rgb = img_out_rgb.unsqueeze(0).to(device)
                # img_out_ = transform(img_out_rgb)
                ilniqe = iqa_ilniqe(img_out).item()
                score_ilniqe_folder[forder_name].append(ilniqe)
                score_ilniqe_all.append(ilniqe)
            if iqa_pi is not None:
                # img_out_rgb = (
                #     cv2.imread(img_out_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
                #     / 255.0
                # )
                # img_out_rgb = cv2.cvtColor(img_out_rgb, cv2.COLOR_BGR2RGB)
                # img_out_rgb = np.transpose(img_out_rgb, (2, 0, 1))
                # img_out_rgb = torch.from_numpy(img_out_rgb).float()
                # img_out_rgb = img_out_rgb.unsqueeze(0).to(device)
                # transforms = T.Compose(
                #     [
                #         # T.Resize((224, 224)),
                #         # T.RandomCrop(size=224),
                #         T.ToTensor(),
                #     ]
                # )
                # T.Normalize(mean=(0.485, 0.456, 0.406),
                #                                  std=(0.229, 0.224, 0.225))])
                # img_out_ = transform(img_out_rgb)
                pi = iqa_pi(img_out).item()
                score_pi_folder[forder_name].append(pi)
                score_pi_all.append(pi)
            if iqa_musiq is not None:
                # H_ = int((H // 224) * 224)
                # W_ = int((W // 224) * 224)
                # transform = T.Resize(size=(H_, W_))
                # img_out_ = transform(img_out/255.)
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
                    nrqm,
                    niqe,
                    brisque,
                    ilniqe,
                    pi,
                    musiq,
                    # 0.0,
                )
            )
        # except Exception as e:
        #     print(f"skip: {img_name} --- {e}")
        #     continue
        if (i + 1) % 20 == 0:
            print(
                f"[{cur_i}/{total_num}] cnniqa: {sum(score_psnr_all)/len(score_psnr_all)}, \
                      clipiqa: {sum(score_ssim_all)/len(score_ssim_all)}, \
                      musiq-koniq: {sum(score_lpips_all)/len(score_lpips_all)}, \
                      maniqa: {sum(score_nrqm_all)/len(score_nrqm_all)}, \
                      tres-koniq: {sum(score_niqe_all)/len(score_niqe_all)}, \
                      topiq_nr: {sum(score_brisque_all)/len(score_brisque_all)}, \
                      dbcnn: {sum(score_ilniqe_all)/len(score_ilniqe_all)}, \
                      hyperiqa: {sum(score_pi_all)/len(score_pi_all)}, \
                      liqe: {sum(score_musiq_all)/len(score_musiq_all)}\n"
            )
        # print(
        #     f"[{cur_i}/{total_num}] cnniqa: {sum(score_psnr_all)/len(score_psnr_all)}, \
        #       clipiqa: {sum(score_ssim_all)/len(score_ssim_all)}"
        # )
        #   print(f'[{cur_i}/{total_num}] maniqa: {sum(score_nrqm_all)/len(score_nrqm_all)}, \
        #           tres-koniq: {sum(score_niqe_all)/len(score_niqe_all)}\n')

    print("-------------------Final Scores-------------------\n")
    print(
        f"Average:\
            cnniqa: {sum(score_psnr_all)/len(score_psnr_all)}, \
            clipiqa: {sum(score_ssim_all)/len(score_ssim_all)}, \
            musiq-koniq: {sum(score_lpips_all)/len(score_lpips_all)}, \
            maniqa: {sum(score_nrqm_all)/len(score_nrqm_all)}, \
            tres-koniq: {sum(score_niqe_all)/len(score_niqe_all)}, \
            topiq_nr: {sum(score_brisque_all)/len(score_brisque_all)}, \
            dbcnn: {sum(score_ilniqe_all)/len(score_ilniqe_all)}, \
            hyperiqa: {sum(score_pi_all)/len(score_pi_all)}, \
            liqe: {sum(score_musiq_all)/len(score_musiq_all)}\n"
    )
    # print(
    #     f"Average:\
    #         cnniqa: {sum(score_psnr_all)/len(score_psnr_all)}, \
    #         clipiqa: {sum(score_ssim_all)/len(score_ssim_all)}"
    # )
    # print(f'Average:\
    #         maniqa: {sum(score_nrqm_all)/len(score_nrqm_all)}, \
    #         tres-koniq: {sum(score_niqe_all)/len(score_niqe_all)}')

    for k in list(score_psnr_forder.keys()):
        print(
            f"Folder Name: {k}\
                cnniqa: {sum(score_psnr_forder[k])/len(score_psnr_forder[k])}, \
                clipiqa: {sum(score_ssim_forder[k])/len(score_ssim_forder[k])}, \
                musiq-koniq: {sum(score_lpips_forder[k])/len(score_lpips_forder[k])}, \
                maniqa: {sum(score_nrqm_folder[k])/len(score_nrqm_folder[k])}, \
                tres-koniq: {sum(score_niqe_folder[k])/len(score_niqe_folder[k])}, \
                topiq_nr: {sum(score_brisque_folder[k])/len(score_brisque_folder[k])}, \
                dbcnn: {sum(score_ilniqe_folder[k])/len(score_ilniqe_folder[k])}, \
                hyperiqa: {sum(score_pi_folder[k])/len(score_pi_folder[k])}, \
                liqe: {sum(score_musiq_folder[k])/len(score_musiq_folder[k])}\n"
        )
    # print(
    #     f"Folder Name: {k}\
    #     cnniqa: {sum(score_psnr_forder[k])/len(score_psnr_forder[k])}, \
    #     clipiqa: {sum(score_ssim_forder[k])/len(score_ssim_forder[k])}"
    # )
    # print(f'Folder Name: {k}\
    #     maniqa: {sum(score_nrqm_folder[k])/len(score_nrqm_folder[k])}, \
    #     tres-koniq: {sum(score_niqe_folder[k])/len(score_niqe_folder[k])}')

    # Output test results to text file
    result_file = open(os.path.join(args.result_path, "test_result_v3.txt"), "w")
    sys.stdout = result_file
    print("-------------------Final Scores-------------------\n")
    print(
        f"Average:\
            cnniqa: {sum(score_psnr_all)/len(score_psnr_all)}, \
            clipiqa: {sum(score_ssim_all)/len(score_ssim_all)}, \
            musiq-koniq: {sum(score_lpips_all)/len(score_lpips_all)}, \
            maniqa: {sum(score_nrqm_all)/len(score_nrqm_all)}, \
            tres-koniq: {sum(score_niqe_all)/len(score_niqe_all)}, \
            topiq_nr: {sum(score_brisque_all)/len(score_brisque_all)}, \
            dbcnn: {sum(score_ilniqe_all)/len(score_ilniqe_all)}, \
            hyperiqa: {sum(score_pi_all)/len(score_pi_all)}, \
            liqe: {sum(score_musiq_all)/len(score_musiq_all)}\n"
    )
    # print(
    #     f"Average:\
    #         cnniqa: {sum(score_psnr_all)/len(score_psnr_all)}, \
    #         clipiqa: {sum(score_ssim_all)/len(score_ssim_all)}"
    # )
    # print(f'Average:\
    #         maniqa: {sum(score_nrqm_all)/len(score_nrqm_all)}, \
    #         tres-koniq: {sum(score_niqe_all)/len(score_niqe_all)}')

    for k in list(score_psnr_forder.keys()):
        print(
            f"Folder Name: {k}\
                cnniqa: {sum(score_psnr_forder[k])/len(score_psnr_forder[k])}, \
                clipiqa: {sum(score_ssim_forder[k])/len(score_ssim_forder[k])}, \
                musiq-koniq: {sum(score_lpips_forder[k])/len(score_lpips_forder[k])}, \
                maniqa: {sum(score_nrqm_folder[k])/len(score_nrqm_folder[k])}, \
                tres-koniq: {sum(score_niqe_folder[k])/len(score_niqe_folder[k])}, \
                topiq_nr: {sum(score_brisque_folder[k])/len(score_brisque_folder[k])}, \
                dbcnn: {sum(score_ilniqe_folder[k])/len(score_ilniqe_folder[k])}, \
                hyperiqa: {sum(score_pi_folder[k])/len(score_pi_folder[k])}, \
                liqe: {sum(score_musiq_folder[k])/len(score_musiq_folder[k])}\n"
        )
    # print(
    #     f"Folder Name: {k}\
    #     cnniqa: {sum(score_psnr_forder[k])/len(score_psnr_forder[k])}, \
    #     clipiqa: {sum(score_ssim_forder[k])/len(score_ssim_forder[k])}"
    # )
    # print(f'Folder Name: {k}\
    #     maniqa: {sum(score_nrqm_folder[k])/len(score_nrqm_folder[k])}, \
    #     tres-koniq: {sum(score_niqe_folder[k])/len(score_niqe_folder[k])}')
    result_file.close()
    f_result.close()
