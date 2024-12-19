import torch
import torch.nn as nn
import torch.nn.functional as F
import utils.arch_utils as arch_util
import functools

# from torch import vmap
from utils.utils_torch import stitch_patches, extract_patches, p2o_3d, conv_fft_batch

class Down(nn.Module):
    """double conv and then downscaling with maxpool"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,  kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
           # nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
            #nn.BatchNorm2d(out_channels),
        )


        self.down_sampling = nn.MaxPool2d(2)


    def forward(self, x):
        feat = self.double_conv(x)
        down_sampled = self.down_sampling(feat)
        return feat, down_sampled


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, feat_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels,  in_channels, kernel_size=2, stride=2)

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            #nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            #nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True)
        )
        self.feat = nn.Sequential(
            nn.Conv2d(feat_channels + out_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x1, x2=None):
        #print('initial x1: ', x1.shape)
        x1 = self.up(x1)
        x1 = self.double_conv(x1)
        #print('x1 after upsampling: ', x1.shape)

        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        if x2 is not None:
          
            diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
            diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

            x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

            x = torch.cat([x2, x1], dim=1)
        else:
            x = x1

        feat = self.feat(x)
        return feat

class PooledSkip(nn.Module):
    def __init__(self, output_spatial_size):
        super().__init__()

        self.output_spatial_size = output_spatial_size

    def forward(self, x):
        global_avg_pooling = x.mean((2,3), keepdim=True) #self.gap(x)
        #print('gap shape:' , global_avg_pooling.shape)
        return global_avg_pooling.repeat(1,1,self.output_spatial_size,self.output_spatial_size)

class TwoHeadsNetwork(nn.Module):
    def __init__(self, K=9, blur_kernel_size=33, bilinear=False,
                 no_softmax=False, c = 1):
        super(TwoHeadsNetwork, self).__init__()
        self.c = c
        self.no_softmax = no_softmax
        if no_softmax:
            print('Softmax is not being used')

        self.inc_rgb = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
        )

        self.inc_gray = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
        )
        self.blur_kernel_size = blur_kernel_size
        self.K=K

        self.down1 = Down(64, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        self.down4 = Down(256, 512)
        self.down5 = Down(512, 1024)
        self.feat =   nn.Sequential(
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
        )
        self.up1 = Up(1024,1024, 512, bilinear)
        self.up2 = Up(512,512, 256, bilinear)
        self.up3 = Up(256,256, 128, bilinear)
        self.up4 = Up(128,128, 64, bilinear)
        self.up5 = Up(64,64, 64, bilinear)

        self.masks_end = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(64, K*3, kernel_size=3, padding=1),
            nn.Softmax(dim=1),
        )

        self.feat5_gap = PooledSkip(2)
        self.feat4_gap = PooledSkip(4)  
        self.feat3_gap = PooledSkip(8)  
        self.feat2_gap = PooledSkip(16)  
        self.feat1_gap = PooledSkip(32) 

        self.kernel_up1 = Up(1024,1024, 512, bilinear)
        self.kernel_up2 = Up(512,512, 256, bilinear)
        self.kernel_up3 = Up(256,256, 256, bilinear)
        self.kernel_up4 = Up(256,128, 128, bilinear)
        self.kernel_up5 = Up(128,64, 64, bilinear)
        if self.blur_kernel_size>33:
            self.kernel_up6 = Up(64, 0, 64, bilinear)

        self.kernels_end = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=2, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(64, K * self.c, kernel_size=3, padding=1) # change to 4 if 280
            #nn.Conv2d(128, K*self.blur_kernel_size*self.blur_kernel_size, kernel_size=8),
        )
        self.kernel_softmax = nn.Softmax(dim=3)

    def forward(self, x):
        #Encoder
        if x.shape[1]==3:
            x1 = self.inc_rgb(x)
        else:
            x1 = self.inc_gray(x)
        x1_feat, x2 = self.down1(x1)
        x2_feat, x3 = self.down2(x2)
        x3_feat, x4 = self.down3(x3)
        x4_feat, x5 = self.down4(x4)
        x5_feat, x6 = self.down5(x5)
        x6_feat = self.feat(x6)

        #k = self.kernel_network(x3)
        feat6_gap = x6_feat.mean((2,3), keepdim=True) #self.feat6_gap(x6_feat)
        #print('x6_feat: ', x6_feat.shape,'feat6_gap: ' , feat6_gap.shape)
        feat5_gap = self.feat5_gap(x5_feat)
        #print('x5_feat: ', x5_feat.shape,'feat5_gap: ' , feat5_gap.shape)
        feat4_gap = self.feat4_gap(x4_feat)
        #print('x4_feat: ', x4_feat.shape,'feat4_gap: ' , feat4_gap.shape)
        feat3_gap = self.feat3_gap(x3_feat)
        #print('x3_feat: ', x3_feat.shape,'feat3_gap: ' , feat3_gap.shape)
        feat2_gap = self.feat2_gap(x2_feat)
        #print('x2_feat: ', x2_feat.shape,'feat2_gap: ' , feat2_gap.shape)
        feat1_gap = self.feat1_gap(x1_feat)
        #print(feat5_gap.shape, feat4_gap.shape)
        k1 = self.kernel_up1(feat6_gap, feat5_gap)
        #print('k1 shape', k1.shape)
        k2 = self.kernel_up2(k1, feat4_gap)
        #print('k2 shape', k2.shape)
        k3 = self.kernel_up3(k2, feat3_gap)
        #print('k3 shape', k3.shape)
        k4 = self.kernel_up4(k3, feat2_gap)
        #print('k4 shape', k4.shape)
        k5 = self.kernel_up5(k4, feat1_gap)

        if self.blur_kernel_size==65:
            k6 = self.kernel_up6(k5)
            k = self.kernels_end(k6)
        else:
            k = self.kernels_end(k5)
        N, f, H, W = k.shape  # H and W should be one
        k = k.view(N, self.K, -1, self.blur_kernel_size * self.blur_kernel_size) # change to 4 if 280

        if self.no_softmax:
            k = F.leaky_relu(k)
            #suma = k5.sum(2, keepdim=True)
            #k = k5 / suma
        else:
            k = self.kernel_softmax(k)

        k = k.view(N, self.K, -1, self.blur_kernel_size, self.blur_kernel_size) # change to 4 if 280

        #Decoder
        x7 = self.up1(x6_feat, x5_feat)
        x8 = self.up2(x7, x4_feat)
        x9 = self.up3(x8, x3_feat)
        x10 = self.up4(x9, x2_feat)
        x11 = self.up5(x10, x1_feat)
        logits = self.masks_end(x11)

        return  k,logits
    
class TwoHeadsNetwork16(nn.Module):
    def __init__(self, K=9, blur_kernel_size=33, bilinear=False,
                 no_softmax=False):
        super(TwoHeadsNetwork16, self).__init__()
        self.no_softmax = no_softmax
        if no_softmax:
            print('Softmax is not being used')

        self.inc_rgb = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
        )

        self.inc_gray = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
        )
        self.blur_kernel_size = blur_kernel_size
        self.K=K

        self.down1 = Down(16, 16)
        self.down2 = Down(16, 32)
        self.down3 = Down(32, 64)
        self.down4 = Down(64, 128)
        self.down5 = Down(128, 256)
        self.feat =   nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
        )
        self.up1 = Up(256,256, 128, bilinear)
        self.up2 = Up(128,128, 64, bilinear)
        self.up3 = Up(64,64, 32, bilinear)
        self.up4 = Up(32,32, 16, bilinear)
        self.up5 = Up(16,16, 16, bilinear)

        self.masks_end = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(16, K*3, kernel_size=3, padding=1),
            nn.Softmax(dim=1),
        )

        self.feat5_gap = PooledSkip(2)
        self.feat4_gap = PooledSkip(4)  
        self.feat3_gap = PooledSkip(8)  
        self.feat2_gap = PooledSkip(16)  
        self.feat1_gap = PooledSkip(16) 

        self.kernel_up1 = Up(256,256, 128, bilinear)
        self.kernel_up2 = Up(128,128, 64, bilinear)
        self.kernel_up3 = Up(64,64, 64, bilinear)
        self.kernel_up4 = Up(64,32, 32, bilinear)
        self.kernel_up5 = Up(32,16, 16, bilinear)
        if self.blur_kernel_size>33:
            self.kernel_up6 = Up(16, 0, 16, bilinear)

        self.kernels_end = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=2, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(16, self.K, kernel_size=3, padding=1) # change to 4 if 280
            #nn.Conv2d(128, K*self.blur_kernel_size*self.blur_kernel_size, kernel_size=8),
        )
        self.kernel_softmax = nn.Softmax(dim=3)

    def forward(self, x):
        #Encoder
        if x.shape[1]==3:
            x1 = self.inc_rgb(x)
        else:
            x1 = self.inc_gray(x)
        x1_feat, x2 = self.down1(x1)
        x2_feat, x3 = self.down2(x2)
        x3_feat, x4 = self.down3(x3)
        x4_feat, x5 = self.down4(x4)
        x5_feat, x6 = self.down5(x5)
        x6_feat = self.feat(x6)

        #k = self.kernel_network(x3)
        feat6_gap = x6_feat.mean((2,3), keepdim=True) #self.feat6_gap(x6_feat)
        #print('x6_feat: ', x6_feat.shape,'feat6_gap: ' , feat6_gap.shape)
        feat5_gap = self.feat5_gap(x5_feat)
        #print('x5_feat: ', x5_feat.shape,'feat5_gap: ' , feat5_gap.shape)
        feat4_gap = self.feat4_gap(x4_feat)
        #print('x4_feat: ', x4_feat.shape,'feat4_gap: ' , feat4_gap.shape)
        feat3_gap = self.feat3_gap(x3_feat)
        #print('x3_feat: ', x3_feat.shape,'feat3_gap: ' , feat3_gap.shape)
        feat2_gap = self.feat2_gap(x2_feat)
        #print('x2_feat: ', x2_feat.shape,'feat2_gap: ' , feat2_gap.shape)
        feat1_gap = self.feat1_gap(x1_feat)
        #print(feat5_gap.shape, feat4_gap.shape)
        k1 = self.kernel_up1(feat6_gap, feat5_gap)
        #print('k1 shape', k1.shape)
        k2 = self.kernel_up2(k1, feat4_gap)
        #print('k2 shape', k2.shape)
        k3 = self.kernel_up3(k2, feat3_gap)
        #print('k3 shape', k3.shape)
        k4 = self.kernel_up4(k3, feat2_gap)
        #print('k4 shape', k4.shape)
        k5 = self.kernel_up5(k4, feat1_gap)

        if self.blur_kernel_size==65:
            k6 = self.kernel_up6(k5)
            k = self.kernels_end(k6)
        else:
            k = self.kernels_end(k5)
        N, f, H, W = k.shape  # H and W should be one
        k = k.view(N, self.K, -1, self.blur_kernel_size * self.blur_kernel_size) # change to 4 if 280

        if self.no_softmax:
            k = F.leaky_relu(k)
            #suma = k5.sum(2, keepdim=True)
            #k = k5 / suma
        else:
            k = self.kernel_softmax(k)

        k = k.view(N, self.K, -1, self.blur_kernel_size, self.blur_kernel_size) # change to 4 if 280

        #Decoder
        x7 = self.up1(x6_feat, x5_feat)
        x8 = self.up2(x7, x4_feat)
        x9 = self.up3(x8, x3_feat)
        x10 = self.up4(x9, x2_feat)
        x11 = self.up5(x10, x1_feat)
        logits = self.masks_end(x11)

        return  k,logits

class CalLoss():
    def __init__(self, loss, patch, device):
        self.loss = loss
        self.patch = patch
        self.device = device

    def extract_patch(self, x, kernels, padding=0):
        kernels = kernels.permute(2, 1, 0, 3, 4)
        x_rp = x.repeat_interleave(12, dim=0).permute(0, 2, 3, 1)
        x_patches = extract_patches(
            x_rp, patch_size=self.patch, num_rows=6, num_cols=8, padding=padding
        )
        x_patches = x_patches.squeeze()

        kernels = kernels.squeeze()
        return x_patches, kernels
    
    def convolution(self, x, k):
        return vmap(F.conv2d)(x, k.unsqueeze(1))

    def convolution_batch(self, xx, kk):
        return vmap(self.convolution)(xx, kk)

    def convolution_batches(self, y, kernels, size):
        # to replace the conv_fft_batch_68_v4
        kernels = kernels.permute(2, 1, 0, 3, 4)
        filter_halfwidth = kernels.shape[-1] // 2
        pad_width = (
            # 0,
            # 0,
            filter_halfwidth,
            filter_halfwidth,
            filter_halfwidth,
            filter_halfwidth,
            # 0,
            # 0,
        )
        y_rp = torch.nn.functional.pad(y, pad=pad_width)
        y_patches_o, _ = self.extract_patch(y_rp, kernels, padding=filter_halfwidth)

        y_patches = y_patches_o.squeeze()
        kernels = kernels.squeeze()
        output = self.convolution_batch(y_patches.unsqueeze(2), kernels.unsqueeze(2))
        output = output.sum(1)
        output = output / output.max()
        output = output.reshape(48, -1, size, size)
        output = stitch_patches(
            output.unsqueeze(-1).unsqueeze(2), 6, 8, stitch_axis=(-3, -2)
        )
        output = output.sum(0)
        output = output.permute(0, 3, 1, 2)
        return output

    def convolution_fft(self, y, kernels):
        filter_halfwidth = kernels.shape[-1] // 2
        pad_width = (
            # 0,
            # 0,
            filter_halfwidth,
            filter_halfwidth,
            filter_halfwidth,
            filter_halfwidth,
            # 0,
            # 0,
        )
        y_rp = torch.nn.functional.pad(y, pad=pad_width)
        y_patches, _ = self.extract_patch(y_rp, kernels, padding=filter_halfwidth)
        kernels = kernels.squeeze()
        A = p2o_3d(kernels, y_patches.shape[-2:])
        A = A.to(self.device)
        lh = conv_fft_batch(y_patches, A)

        lhs = lh.sum(1)
        Y_fft = torch.real(torch.fft.ifftn(lhs, dim=[-2, -1]))
        Y_fft = Y_fft[
            :,
            filter_halfwidth:-filter_halfwidth,
            filter_halfwidth:-filter_halfwidth,
        ]
        output = Y_fft.reshape(48, -1, self.patch, self.patch)
        output = stitch_patches(
            output.unsqueeze(-1).unsqueeze(2), 6, 8, stitch_axis=(-3, -2)
        )
        output = output.sum(-1)
        output = output / output.max()
        return output
    
    def cal_loss(self, gt, im, ke):
        ke = ke.permute(0,2,1,3,4)
        predicted = self.convolution_fft(gt, ke)
        loss = self.loss(predicted, im)
        return loss


def flipcat(x, dim):
    return torch.cat([x,x.flip(dim).index_select(dim,torch.arange(1,x.size(dim)).cuda())],dim)
    
class KPN(nn.Module):
    def __init__(self, nf=64):
        super(KPN,self).__init__()
        
        self.conv_first = nn.Conv2d(2, nf, 3, 1, 1, bias=True)
        basic_block = functools.partial(arch_util.ResidualBlock_noBN, nf=nf)
        self.block1 = basic_block()
        self.block2 = basic_block()
        self.block3 = basic_block()
        self.out = nn.Conv2d(64, 35, 3, 1, 1, bias=True)
        self.kernel_pred = KernelConv()
        
    def forward(self, data_with_est, data):
        x = self.conv_first(data_with_est)
        x = self.block3(self.block2(self.block1(x)))
        core = self.out(x)
        
        return self.kernel_pred(data, core)


class KernelConv(nn.Module):
    """
    the class of computing prediction
    """
    def __init__(self):
        super(KernelConv, self).__init__()
    
    def _list_core(self, core_list, batch_size, N, color, height, width):
        """
        convert the sep_conv core to conv2d core
        2p --> p^2
        :param core: shape: batch*(N*2*K)*height*width
        :return:
        """
        core_out = {}
        core_out_list = []
    
        for i in range(len(core_list)):
            core = core_list[i]
            core = torch.abs(core)
            wide = core.shape[1]
            final_wide = (core.shape[1]-1)*2+1
            kernel = torch.zeros((batch_size,wide,wide,color,height, width)).cuda()
            mid = torch.Tensor([core.shape[1]-1]).cuda()
            for i in range(wide):
                for j in range(wide):
                    distance = torch.sqrt((i-mid)**2 + (j-mid)**2)
                    low = torch.floor(distance)
                    high = torch.ceil(distance)
                    if distance > mid:
                        kernel[:,i,j,0,:,:] = 0
                    elif low == high:
                        kernel[:,i,j,0,:,:] = core[:,int(low),:,:]
                    else:
                        y0 = core[:,int(low),:,:]
                        y1 = core[:,int(low)+1,:,:]
                        kernel[:,i,j,0,:,:] = (distance-low)*y1 + (high-distance)*y0
                    
            kernel = flipcat(flipcat(kernel,1), 2)   
            core_ori = kernel.view(batch_size, N, final_wide * final_wide, color, height, width)
            core_out[final_wide] = F.softmax(core_ori,dim=2)
            core_out_list.append(F.softmax(core_ori,dim=2))
        # core_out_list = torch.stack(core_out_list)
        return core_out, core_out_list

    def forward(self, frames, core):
        """
        compute the pred image according to core and frames
        :param frames: [batch_size, N, 3, height, width]
        :param core: [batch_size, N, dict(kernel), 3, height, width]
        :return:
        """
        # pred_img = [frames]
        pred_img = 0.0
        batch_size, N, height, width = frames.size()
        color = 1
        frames = frames.view(batch_size, N, color, height, width)
        
        section = [2,3,4,5,6,7,8]
        core_list = []
        core_list = torch.split(core, section, dim=1)

        core_out, core_out_list = self._list_core(core_list, batch_size, 1, color, height, width)                
        kernel_list = [3,5,7,9,11,13,15]
        
                
        for index, K in enumerate(kernel_list):
            img_stack = []
            frame_pad = F.pad(frames, [K // 2, K // 2, K // 2, K // 2])
            for i in range(K):
                for j in range(K):
                    img_stack.append(frame_pad[..., i:i + height, j:j + width])
            img_stack = torch.stack(img_stack, dim=2)
            pred = torch.sum(core_out[K].mul(img_stack), dim=2, keepdim=False)
            if batch_size == 1:
                pred = pred.squeeze().unsqueeze(0)
            else:
                pred = pred.squeeze().unsqueeze(1)
            pred_img += pred

        return pred_img, core_out_list


class KERNEL_MAP(nn.Module):
    def __init__(self, nf=64):
        super(KERNEL_MAP,self).__init__()
        
        self.conv_first = nn.Conv2d(2, nf, 3, 1, 1, bias=True)
        basic_block = functools.partial(arch_util.ResidualBlock_noBN, nf=nf)
        self.block1 = basic_block()
        self.block2 = basic_block()
        self.block3 = basic_block()
        self.out = nn.Conv2d(64, 1, 3, 1, 1, bias=True)
        
    def forward(self, x):
        x = self.conv_first(x)
        x = self.block3(self.block2(self.block1(x)))
        kernel_map = self.out(x)
        kernel_map = F.softmax(kernel_map, dim=1)
        return kernel_map
    
class Reblur_Model(nn.Module):
    def __init__(self):
        super(Reblur_Model, self).__init__()
        # self.kernel_map_gene = KERNEL_MAP()
        self.kpn = KPN()
        self.apply(self._init_weights)
    
    @staticmethod     
    def _init_weights(m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_normal_(m.weight.data)
            nn.init.constant_(m.bias.data, 0.0)
    
    # def forward(self,im_s,im_b):
    def forward(self,im_b):
        est_input = torch.cat([im_b,im_b], dim=1)
        pred_img, kernel_map = self.kpn(est_input, im_b)
        # kernel_map = self.kernel_map_gene(est_input)
        
        # map0,map1,map2,map3,map4,map5,map6,map7 = torch.chunk(kernel_map, 8, dim=1)
        # map_list = [map0,map1,map2,map3,map4,map5,map6,map7]
        # output = map0*pred_img[0]
        # for n in range(1,8):
        #     output += (pred_img[n])*(map_list[n])
        
        return pred_img, kernel_map