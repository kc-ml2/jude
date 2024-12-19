from torchmetrics.multimodal import CLIPImageQualityAssessment
import torch
import argparse
from basicsr.utils import scandir
import cv2
import os
import numpy as np

_ = torch.manual_seed(42)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result_path",
        type=str,
        default="/home/tuvv/workspaces/BOWNet_old/test_results/BOWNet_MAE_denoiser_model_v9_512/real_data/BOWNet_kernel_prediction_model_v10-5-512-ResUNet_mix/all",
    )

    args = parser.parse_args()

    if args.result_path.endswith("/"):  # solve when path ends with /
        args.result_path = args.result_path[:-1]
    if args.gt_path.endswith("/"):  # solve when path ends with /
        args.gt_path = args.gt_path[:-1]

    img_out_paths = sorted(
        list(
            scandir(
                args.result_path, suffix=("jpg", "png"), recursive=True, full_path=True
            )
        )
    )
    total_num = len(img_out_paths)
    scores = []
    with open(os.path.join(os.path.dirname(args.im_path), "CLIPIQA.txt"), "w") as f:
        f.writelines("--- \n")

    for i, img_out_path in enumerate(img_out_paths):
        img_name = img_out_path.replace(args.result_path + "/", "")
        cur_i = i + 1
        print(f"[{cur_i}/{total_num}] Processing: {img_name}")

        img_out = (
            cv2.imread(img_out_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.0
        )
        img_out = cv2.cvtColor(img_out, cv2.COLOR_BGR2RGB)
        img_out = np.transpose(img_out, (2, 0, 1))
        img_out = torch.from_numpy(img_out).float()

        _, H, W = img_out.shape
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        with torch.no_grad():
            img_out = img_out.unsqueeze(0).to(device)
            metric = CLIPImageQualityAssessment(prompts=("Bright photo.", "Sharp photo"))
            score = metric(img_out).mean(0).item()
            scores.append(score)
            with open(os.path.join(os.path.dirname(args.im_path), "CLIPIQA.txt"), "a") as f:
                f.writelines("{} --- {}\n".format(os.path.basename(img_out_path), score))

    with open(os.path.join(os.path.dirname(args.im_path), "CLIPIQA.txt"), "a") as f:
        f.writelines("Average --- {}".format(sum(scores)/len(scores)))

