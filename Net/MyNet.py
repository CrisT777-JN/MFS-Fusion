import numpy as np
import torch
import torch.nn as nn
import torchvision
from matplotlib import pyplot as plt

from Net.MobileViT import mobile_vit_small
from Net.pvt import PyramidVisionTransformerV2, PyramidVisionTransformerV2_one
import torch.nn.functional as F
import os
# import onnx
from einops import rearrange
from mamba_ssm import Mamba

def custom_complex_normalization(input_tensor, dim=-1):
    real_part = input_tensor.real
    imag_part = input_tensor.imag
    norm_real = F.softmax(real_part, dim=dim)
    norm_imag = F.softmax(imag_part, dim=dim)

    normalized_tensor = torch.complex(norm_real, norm_imag)

    return normalized_tensor


os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def conv3x3(in_planes, out_planes, stride=1, has_bias=False):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=has_bias)


def conv3x3_bn_relu(in_planes, out_planes, stride=1):
    return nn.Sequential(
        conv3x3(in_planes, out_planes, stride),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(inplace=True),
    )

def conv1x1(in_planes, out_planes, stride=1, has_bias=False):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                     padding=0, bias=has_bias)


def conv1x1_bn_relu(in_planes, out_planes, stride=1):
    return nn.Sequential(
        conv1x1(in_planes, out_planes, stride),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(inplace=True),
    )



class MambaLayer(nn.Module):
    def __init__(self, dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.dim = dim
        self.nin = conv1x1(dim, dim)
        self.nin2 = conv1x1(dim, dim)
        self.norm2 = nn.BatchNorm2d(dim) # LayerNorm
        self.relu2 = nn.ReLU(inplace=True)
        self.relu3 = nn.ReLU(inplace=True)

        self.norm = nn.BatchNorm2d(dim) # LayerNorm
        self.relu = nn.ReLU(inplace=True)
        self.mamba = Mamba(
            d_model=dim,  # Model dimension d_model
            d_state=d_state,  # SSM state expansion factor
            d_conv=d_conv,  # Local convolution width
            expand=expand  # Block expansion factor
        )

    def forward(self, x):
        B, C = x.shape[:2]
        x = self.nin(x)
        x = self.norm(x)
        x = self.relu(x)
        act_x = x
        assert C == self.dim
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)

        x_flip_l = torch.flip(x_flat, dims=[2])
        x_flip_c = torch.flip(x_flat, dims=[1])
        x_flip_lc = torch.flip(x_flat, dims=[1,2])
        x_ori = self.mamba(x_flat)
        x_mamba_l = self.mamba(x_flip_l)
        x_mamba_c = self.mamba(x_flip_c)
        x_mamba_lc = self.mamba(x_flip_lc)
        x_ori_l = torch.flip(x_mamba_l, dims=[2])
        x_ori_c = torch.flip(x_mamba_c, dims=[1])
        x_ori_lc = torch.flip(x_mamba_lc, dims=[1,2])
        x_mamba = (x_ori+x_ori_l+x_ori_c+x_ori_lc)/4

        out = x_mamba.transpose(-1, -2).reshape(B, C, *img_dims)
        out += act_x
        out = self.nin2(out)
        out = self.norm2(out)
        out = self.relu2(out)
        return out



class AttenFFT(nn.Module):
    def __init__(self, dim, num_heads=2, bias=False, ):
        super(AttenFFT, self).__init__()
        self.num_heads = num_heads

        self.qkv1conv_1 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.qkv1conv_3 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.qkv1conv_5 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)

        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.weight = nn.Sequential(
            nn.Conv2d(dim, dim, 1, bias=True),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.Conv2d(dim, dim, 1, bias=True),
            nn.Sigmoid())

        self.project_out = nn.Conv2d(dim * 2, dim, kernel_size=1, bias=bias)


    def forward(self, x):
        b, c, h, w = x.shape
        q_s = self.qkv1conv_5(x)
        k_s = self.qkv1conv_3(x)
        v_s = self.qkv1conv_1(x)

        q_s = torch.fft.fft2(q_s.float())
        k_s = torch.fft.fft2(k_s.float())
        v_s = torch.fft.fft2(v_s.float())

        q_s = rearrange(q_s, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k_s = rearrange(k_s, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v_s = rearrange(v_s, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q_s = torch.nn.functional.normalize(q_s, dim=-1)
        k_s = torch.nn.functional.normalize(k_s, dim=-1)
        attn_s = (q_s @ k_s.transpose(-2, -1)) * self.temperature
        attn_s = custom_complex_normalization(attn_s, dim=-1)


        outr = torch.abs(torch.fft.ifft2(attn_s @ v_s))
        outr = rearrange(outr, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out_f_lr = torch.abs(torch.fft.ifft2(self.weight(torch.fft.fft2(x.float()).real) * torch.fft.fft2(x.float())))
        outr = self.project_out(torch.cat((outr, out_f_lr), 1))


        return outr


class ASI(nn.Module):
    def __init__(self, dim):
        super(ASI, self).__init__()
        self.dim = dim // 4
        self.conv1 = nn.Conv2d(self.dim, self.dim, kernel_size=3, stride=1, padding=1, dilation=1)
        self.conv2 = nn.Conv2d(self.dim, self.dim, kernel_size=3, stride=1, padding=2, dilation=2)
        self.conv3 = nn.Conv2d(self.dim, self.dim, kernel_size=3, stride=1, padding=3, dilation=3)
        self.conv4 = nn.Conv2d(self.dim, self.dim, kernel_size=3, stride=1, padding=4, dilation=4)

        self.fft1 = AttenFFT(self.dim)
        self.fft2 = AttenFFT(self.dim)
        self.fft3 = AttenFFT(self.dim)
        self.fft4 = AttenFFT(self.dim)

        self.project_out = nn.Conv2d(dim, dim, kernel_size=3,padding=1)


    def forward(self, x):
        x1,x2,x3,x4 = torch.chunk(x,4,dim=1)

        x1 = self.fft1(self.conv1(x1)) + x1
        x2 = self.fft2(self.conv2(x2)) + x2
        x3 = self.fft3(self.conv3(x3)) + x3
        x4 = self.fft4(self.conv4(x4)) + x4

        outs = torch.cat((x1,x2,x3,x4),1) + x

        out = self.project_out(outs)

        return out


class sa_layer(nn.Module):
    def __init__(self, channel, groups=4):
        super(sa_layer, self).__init__()
        self.groups = groups
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.cweight = nn.Parameter(torch.zeros(1, channel // (2 * groups), 1, 1))
        self.cbias = nn.Parameter(torch.ones(1, channel // (2 * groups), 1, 1))
        self.sweight = nn.Parameter(torch.zeros(1, channel // (2 * groups), 1, 1))
        self.sbias = nn.Parameter(torch.ones(1, channel // (2 * groups), 1, 1))

        self.sigmoid = nn.Sigmoid()
        self.gn = nn.GroupNorm(channel // (2 * groups), channel // (2 * groups))

    @staticmethod
    def channel_shuffle(x, groups):
        b, c, h, w = x.shape

        x = x.reshape(b, groups, -1, h, w)
        x = x.permute(0, 2, 1, 3, 4)

        x = x.reshape(b, -1, h, w)

        return x

    def forward(self, x):
        b, c, h, w = x.shape

        x = x.reshape(b * self.groups, -1, h, w)
        x_0, x_1 = x.chunk(2, dim=1)

        # channel attention
        xn = self.avg_pool(x_0)
        xn = self.cweight * xn + self.cbias
        xn = x_0 * self.sigmoid(xn)

        # spatial attention
        xs = self.gn(x_1)
        xs = self.sweight * xs + self.sbias
        xs = x_1 * self.sigmoid(xs)

        # concatenate along channel axis
        out = torch.cat([xn, xs], dim=1)
        out = out.reshape(b, -1, h, w)

        out = self.channel_shuffle(out, 2)

        return out


class MEF(nn.Module):
    def __init__(self, dim):
        super(MEF, self).__init__()

        self.spatial_attention = SpatialAttention()

        self.feir = ASI(dim)
        self.fevi = ASI(dim)

        self.cov = conv3x3_bn_relu(2 * dim, dim)

    def forward(self, ir, vi):
        mul_fuse = ir * vi
        sa = self.spatial_attention(mul_fuse)
        vi = vi * sa + vi
        ir = ir * sa + ir

        vi = self.fevi(vi) + vi
        ir = self.feir(ir) + ir

        out = self.cov(torch.cat((vi, ir), 1))

        return ir, vi, out


class MyNet(nn.Module):
    def __init__(self, in_chans=1, hidden_chans=[48, 96, 192], pool_ratio=[8, 6, 4], out_chans=1, linear=True):
        super(MyNet, self).__init__()

        self.pool_ratio = pool_ratio
        self.pre_x = nn.Conv2d(in_chans, hidden_chans[0], 3, 1, 1)
        self.pre_y = nn.Conv2d(in_chans, hidden_chans[0], 3, 1, 1)


        self.un_x = PyramidVisionTransformerV2(in_chans=48, embed_dims=hidden_chans,
                                               num_heads=[1, 2, 4], mlp_ratios=[4, 4, 4],
                                               depths=[2, 2, 2], sr_ratios=[8, 4, 2], num_stages=3, linear=linear)
        self.un_y = PyramidVisionTransformerV2(in_chans=48, embed_dims=hidden_chans,
                                               num_heads=[1, 2, 4], mlp_ratios=[4, 4, 4],
                                               depths=[2, 2, 2], sr_ratios=[8, 4, 2], num_stages=3, linear=linear)

        self.fuse0 = MEF(dim=hidden_chans[0])
        self.fuse1 = MEF(dim=hidden_chans[1])
        self.fuse2 = MEF(dim=hidden_chans[2])
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.fes = [self.fuse0,self.fuse1,self.fuse2]
        self.last = nn.Sequential(
            nn.Conv2d(in_channels=hidden_chans[0], out_channels=hidden_chans[0], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_chans[0]),
            nn.GELU(),
            # self.upsample2,
            ESI(hidden_chans[0]),
            sa_layer(hidden_chans[0]),
            nn.Conv2d(in_channels=hidden_chans[0], out_channels=out_chans, kernel_size=3, padding=1, bias=True),
        )



        self.dec = Decode(hidden_chans[0],hidden_chans[1],hidden_chans[2])

    def forward(self, x,y):
        fuses = []


        h, w = x.shape[2], x.shape[3]
        img_size = (h, w)
        x = self.pre_x(x)
        y = self.pre_y(y)
        short_x = x
        short_y = y


        B = x.shape[0]
        for i in range(self.un_x.num_stages):
            patch_embedx = getattr(self.un_x, f"patch_embed{i + 1}")
            blockx = getattr(self.un_x, f"block{i + 1}")
            normx = getattr(self.un_x, f"norm{i + 1}")

            patch_embedy = getattr(self.un_y, f"patch_embed{i + 1}")
            blocky = getattr(self.un_y, f"block{i + 1}")
            normy = getattr(self.un_y, f"norm{i + 1}")

            x, H, W = patch_embedx(x)
            for blkx in blockx:
                x = blkx(x, H, W)
            x = normx(x)
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

            y, H, W = patch_embedy(y)
            for blky in blocky:
                y = blky(y, H, W)
            y = normy(y)
            y = y.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

            x,y,out = self.fes[i](x,y)

            fuses.append(out)


        out = self.dec(fuses[0],fuses[1],fuses[2])
        out = self.last(out+short_x+short_y)


        return out



class Decode(nn.Module):
    def __init__(self, in1,in2,in3):
        super(Decode, self).__init__()

        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # self.conv_up4 = nn.Sequential(
        #     nn.Conv2d(in_channels=in4, out_channels=in3, kernel_size=1, bias=False),
        #     nn.BatchNorm2d(in3),
        #     nn.GELU(),
        #     self.upsample2
        # )
        self.conv_up3 = nn.Sequential(
            nn.Conv2d(in_channels=in3, out_channels=in2, kernel_size=1, bias=False),
            nn.BatchNorm2d(in2),
            nn.GELU(),
            self.upsample2,
            ESI(in2),
            sa_layer(in2)
        )
        self.conv_up2 = nn.Sequential(
            nn.Conv2d(in_channels=in2*2, out_channels=in1, kernel_size=1, bias=False),
            nn.BatchNorm2d(in1),
            nn.GELU(),
            self.upsample2,
            ESI(in1),
            sa_layer(in1)
        )
        self.conv_up1 = nn.Sequential(
            nn.Conv2d(in_channels=in1*2, out_channels=in1, kernel_size=1, bias=False),
            nn.BatchNorm2d(in1),
            nn.GELU(),
            self.upsample2,
            ESI(in1),
            sa_layer(in1)
        )


        self.p_1 = nn.Sequential(
            nn.Conv2d(in_channels=in1, out_channels=in1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in1),
            nn.GELU(),
            self.upsample2,
            nn.Conv2d(in_channels=in1, out_channels=in1, kernel_size=3, padding=1, bias=True),
        )



    def forward(self,x1,x2,x3):

        # up4 = self.conv_up4(x4)
        up3 = self.conv_up3(x3)
        up2 = self.conv_up2(torch.cat((up3,x2),1))
        up1 = self.conv_up1(torch.cat((up2,x1), 1))



        return up1


class ESI(nn.Module):
    def __init__(self, dim):
        super(ESI, self).__init__()
        self.dim = dim // 4
        self.conv1 = nn.Conv2d(dim, self.dim, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(dim, self.dim, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(dim, self.dim, kernel_size=5, stride=1, padding=2)
        self.conv4 = nn.Conv2d(dim, self.dim, kernel_size=7, stride=1, padding=3)

        self.mmb1 = MambaLayer(self.dim)
        self.mmb2 = MambaLayer(self.dim)
        self.mmb3 = MambaLayer(self.dim)
        self.mmb4 = MambaLayer(self.dim)

        self.project_out = nn.Conv2d(dim, dim, kernel_size=3, padding=1)

    def forward(self, x):

        x1 = self.mmb1(self.conv1(x))
        x2 = self.mmb2(self.conv2(x))
        x3 = self.mmb3(self.conv3(x))
        x4 = self.mmb4(self.conv4(x))

        outs = torch.cat((x1, x2, x3, x4), 1) + x

        out = self.project_out(outs)

        return out


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = max_out
        x = self.conv1(x)
        return self.sigmoid(x)




if __name__ == '__main__':
    import torch
    import torchvision
    from thop import profile

    model = MyNet().cuda()

    a = torch.randn(1, 1, 128, 128).cuda()
    b = torch.randn(1, 1, 128, 128).cuda()
    flops, params = profile(model, (a,b))
    print('flops: ', flops, 'params: ', params)
    print('flops: %.2f G, params: %.2f M' % (flops / 1000000000.0, params / 1000000.0))
