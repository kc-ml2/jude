import models.resnet_basicblock as B
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class ResUNet(nn.Module):
    def __init__(
        self,
        in_nc=4,
        out_nc=3,
        nc=[64, 128, 256, 512],
        nb=2,
        act_mode="R",
        downsample_mode="strideconv",
        upsample_mode="convtranspose",
    ):
        super(ResUNet, self).__init__()

        self.m_head = B.conv(in_nc, nc[0], bias=False, mode="C")

        # downsample
        if downsample_mode == "avgpool":
            downsample_block = B.downsample_avgpool
        elif downsample_mode == "maxpool":
            downsample_block = B.downsample_maxpool
        elif downsample_mode == "strideconv":
            downsample_block = B.downsample_strideconv
        else:
            raise NotImplementedError(
                "downsample mode [{:s}] is not found".format(downsample_mode)
            )

        self.m_down1 = B.sequential(
            *[
                B.ResBlock(nc[0], nc[0], bias=False, mode="C" + act_mode + "C")
                for _ in range(nb)
            ],
            downsample_block(nc[0], nc[1], bias=False, mode="2")
        )
        self.m_down2 = B.sequential(
            *[
                B.ResBlock(nc[1], nc[1], bias=False, mode="C" + act_mode + "C")
                for _ in range(nb)
            ],
            downsample_block(nc[1], nc[2], bias=False, mode="2")
        )
        self.m_down3 = B.sequential(
            *[
                B.ResBlock(nc[2], nc[2], bias=False, mode="C" + act_mode + "C")
                for _ in range(nb)
            ],
            downsample_block(nc[2], nc[3], bias=False, mode="2")
        )

        self.m_body = B.sequential(
            *[
                B.ResBlock(nc[3], nc[3], bias=False, mode="C" + act_mode + "C")
                for _ in range(nb)
            ]
        )

        # upsample
        if upsample_mode == "upconv":
            upsample_block = B.upsample_upconv
        elif upsample_mode == "pixelshuffle":
            upsample_block = B.upsample_pixelshuffle
        elif upsample_mode == "convtranspose":
            upsample_block = B.upsample_convtranspose
        else:
            raise NotImplementedError(
                "upsample mode [{:s}] is not found".format(upsample_mode)
            )

        self.m_up3 = B.sequential(
            upsample_block(nc[3], nc[2], bias=False, mode="2"),
            *[
                B.ResBlock(nc[2], nc[2], bias=False, mode="C" + act_mode + "C")
                for _ in range(nb)
            ]
        )
        self.m_up2 = B.sequential(
            upsample_block(nc[2], nc[1], bias=False, mode="2"),
            *[
                B.ResBlock(nc[1], nc[1], bias=False, mode="C" + act_mode + "C")
                for _ in range(nb)
            ]
        )
        self.m_up1 = B.sequential(
            upsample_block(nc[1], nc[0], bias=False, mode="2"),
            *[
                B.ResBlock(nc[0], nc[0], bias=False, mode="C" + act_mode + "C")
                for _ in range(nb)
            ]
        )

        self.m_tail = B.conv(nc[0], out_nc, bias=False, mode="C")

    def forward(self, x):

        h, w = x.size()[-2:]
        paddingBottom = int(np.ceil(h / 8) * 8 - h)
        paddingRight = int(np.ceil(w / 8) * 8 - w)
        x = nn.ReplicationPad2d((0, paddingRight, 0, paddingBottom))(x)

        x1 = self.m_head(x)
        x2 = self.m_down1(x1)
        x3 = self.m_down2(x2)
        x4 = self.m_down3(x3)
        x = self.m_body(x4)
        x = self.m_up3(x + x4)
        x = self.m_up2(x + x3)
        x = self.m_up1(x + x2)
        x = self.m_tail(x + x1)

        x = x[..., :h, :w]

        return x


class IRCNN(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nc=32):
        """
        # ------------------------------------
        denoiser of IRCNN
        in_nc: channel number of input
        out_nc: channel number of output
        nc: channel number
        nb: total number of conv layers
        act_mode: batch norm + activation function; 'BR' means BN+ReLU.
        # ------------------------------------
        Batch normalization and residual learning are
        beneficial to Gaussian denoising (especially
        for a single noise level).
        The residual of a noisy image corrupted by additive white
        Gaussian noise (AWGN) follows a constant
        Gaussian distribution which stablizes batch
        normalization during training.
        # ------------------------------------
        """
        super(IRCNN, self).__init__()
        L = []
        L.append(
            nn.Conv2d(
                in_channels=in_nc,
                out_channels=nc,
                kernel_size=3,
                stride=1,
                padding=1,
                dilation=1,
                bias=True,
            )
        )
        L.append(nn.ReLU(inplace=True))
        L.append(
            nn.Conv2d(
                in_channels=nc,
                out_channels=nc,
                kernel_size=3,
                stride=1,
                padding=2,
                dilation=2,
                bias=True,
            )
        )
        L.append(nn.ReLU(inplace=True))
        L.append(
            nn.Conv2d(
                in_channels=nc,
                out_channels=nc,
                kernel_size=3,
                stride=1,
                padding=3,
                dilation=3,
                bias=True,
            )
        )
        L.append(nn.ReLU(inplace=True))
        L.append(
            nn.Conv2d(
                in_channels=nc,
                out_channels=nc,
                kernel_size=3,
                stride=1,
                padding=4,
                dilation=4,
                bias=True,
            )
        )
        L.append(nn.ReLU(inplace=True))
        L.append(
            nn.Conv2d(
                in_channels=nc,
                out_channels=nc,
                kernel_size=3,
                stride=1,
                padding=3,
                dilation=3,
                bias=True,
            )
        )
        L.append(nn.ReLU(inplace=True))
        L.append(
            nn.Conv2d(
                in_channels=nc,
                out_channels=nc,
                kernel_size=3,
                stride=1,
                padding=2,
                dilation=2,
                bias=True,
            )
        )
        L.append(nn.ReLU(inplace=True))
        L.append(
            nn.Conv2d(
                in_channels=nc,
                out_channels=out_nc,
                kernel_size=3,
                stride=1,
                padding=1,
                dilation=1,
                bias=True,
            )
        )
        self.model = B.sequential(*L)

    def forward(self, x):
        n = self.model(x)
        return x - n
