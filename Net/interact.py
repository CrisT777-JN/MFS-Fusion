# pvt模块的参数按照pvt_v2_b1的参数设置

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange
import numbers

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class FMultiOrderDWConv(nn.Module):
    def __init__(self,
                 embed_dims,):
        super(FMultiOrderDWConv, self).__init__()

        self.dp = nn.Sequential(
            nn.Conv2d(  # point-wise convolution
                in_channels=embed_dims,
                out_channels=embed_dims,
                kernel_size=1),
            nn.Conv2d(
                in_channels=embed_dims,
                out_channels=embed_dims,
                kernel_size=3,
                padding=(3) // 2,
                groups=embed_dims,
                stride=1),
            # nn.ReLU()
            # nn.LeakyReLU(negative_slope=0.2, inplace=True)
            nn.GELU()
        )
        # self.act = nn.GELU()

    def forward(self, x):
        x = x * self.dp(x)

        return x


class MultiOrderAvgPool(nn.Module):
    def __init__(self,
                 embed_dims,
                 channel_split=[2,4,2]):
        super(MultiOrderAvgPool, self).__init__()

        self.split_ratio = [i / sum(channel_split) for i in channel_split]
        self.embed_dims_1 = int(self.split_ratio[1] * embed_dims)
        self.embed_dims_2 = int(self.split_ratio[2] * embed_dims)
        self.embed_dims_0 = embed_dims - self.embed_dims_1 - self.embed_dims_2
        self.embed_dims = embed_dims
        assert embed_dims % sum(channel_split) == 0
        # basic DW conv
        # self.DW_conv0 = nn.Sequential(
        #     nn.Conv2d(in_channels=self.embed_dims_0, out_channels=self.embed_dims_0, kernel_size=1),
        #     nn.Conv2d(in_channels=self.embed_dims_0, out_channels=self.embed_dims_0,
        #               kernel_size=3,  groups=self.embed_dims_0),
        # )
        # # DW conv 1
        # self.DW_conv1 = nn.Sequential(
        #     nn.Conv2d(in_channels=self.embed_dims_1, out_channels=self.embed_dims_1, kernel_size=1),
        #     nn.Conv2d(in_channels=self.embed_dims_1, out_channels=self.embed_dims_1,
        #               kernel_size=5,  groups=self.embed_dims_1)
        # )
        # # DW conv 2
        # self.DW_conv2 = nn.Sequential(
        #     nn.Conv2d(in_channels=self.embed_dims_2, out_channels=self.embed_dims_2, kernel_size=1),
        #     nn.Conv2d(in_channels=self.embed_dims_2, out_channels=self.embed_dims_2,
        #               kernel_size=7,  groups=self.embed_dims_2)
        # )
        # # a channel convolution
        # self.PW_conv = nn.Sequential(
        #     nn.Conv2d(in_channels=embed_dims, out_channels=embed_dims, kernel_size=1),
        #     # nn.GELU(),
        # )
        self.DW_conv = nn.Sequential(
            nn.Conv2d(in_channels=self.embed_dims, out_channels=self.embed_dims, kernel_size=1),
            nn.Conv2d(in_channels=self.embed_dims, out_channels=self.embed_dims,
                      kernel_size=5, padding=2, groups=self.embed_dims),
            )

    def forward(self, x, x_size):
        # x_0 = F.adaptive_avg_pool2d(self.DW_conv0(x[:, :self.embed_dims_0, ...]), output_size=x_size)
        # x_1 = F.adaptive_avg_pool2d(self.DW_conv1(x[:, self.embed_dims_0: self.embed_dims_0 + self.embed_dims_1, ...]),
        #                             output_size=x_size)
        # x_2 = F.adaptive_avg_pool2d(self.DW_conv2(x[:, self.embed_dims - self.embed_dims_2:, ...]), output_size=x_size)
        # x = torch.cat([x_0, x_1, x_2], dim=1)
        # x = self.PW_conv(x)

        x = F.adaptive_avg_pool2d(self.DW_conv(x), output_size=x_size)
        return x


class MultiOrderMaxPool(nn.Module):
    def __init__(self,
                 embed_dims,
                 channel_split=[2,4,2]):
        super(MultiOrderMaxPool, self).__init__()

        self.split_ratio = [i / sum(channel_split) for i in channel_split]
        self.embed_dims_1 = int(self.split_ratio[1] * embed_dims)
        self.embed_dims_2 = int(self.split_ratio[2] * embed_dims)
        self.embed_dims_0 = embed_dims - self.embed_dims_1 - self.embed_dims_2
        self.embed_dims = embed_dims
        assert embed_dims % sum(channel_split) == 0
        # basic DW conv
        # self.DW_conv0 = nn.Sequential(
        #     nn.Conv2d(in_channels=self.embed_dims_0, out_channels=self.embed_dims_0, kernel_size=1),
        #     nn.Conv2d(in_channels=self.embed_dims_0, out_channels=self.embed_dims_0,
        #               kernel_size=3, groups=self.embed_dims_0),
        # )
        # # DW conv 1
        # self.DW_conv1 = nn.Sequential(
        #     nn.Conv2d(in_channels=self.embed_dims_1, out_channels=self.embed_dims_1, kernel_size=1),
        #     nn.Conv2d(in_channels=self.embed_dims_1, out_channels=self.embed_dims_1,
        #               kernel_size=5, groups=self.embed_dims_1)
        # )
        # # DW conv 2
        # self.DW_conv2 = nn.Sequential(
        #     nn.Conv2d(in_channels=self.embed_dims_2, out_channels=self.embed_dims_2, kernel_size=1),
        #     nn.Conv2d(in_channels=self.embed_dims_2, out_channels=self.embed_dims_2,
        #               kernel_size=7,  groups=self.embed_dims_2)
        # )
        # # a channel convolution
        # self.PW_conv = nn.Sequential(
        #     nn.Conv2d(in_channels=embed_dims, out_channels=embed_dims, kernel_size=1),
        #     # nn.GELU(),
        # )
        self.DW_conv = nn.Sequential(
            nn.Conv2d(in_channels=self.embed_dims, out_channels=self.embed_dims, kernel_size=1),
            nn.Conv2d(in_channels=self.embed_dims, out_channels=self.embed_dims,
                      kernel_size=5, padding=2, groups=self.embed_dims),
        )

    def forward(self, x, x_size):
        # x_0 = F.adaptive_max_pool2d(self.DW_conv0(x[:, :self.embed_dims_0, ...]), output_size=x_size)
        # x_1 = F.adaptive_max_pool2d(self.DW_conv1(x[:, self.embed_dims_0: self.embed_dims_0 + self.embed_dims_1, ...]),
        #                             output_size=x_size)
        # x_2 = F.adaptive_max_pool2d(self.DW_conv2(x[:, self.embed_dims - self.embed_dims_2:, ...]), output_size=x_size)
        # x_0 = F.adaptive_avg_pool2d(self.DW_conv0(x[:, :self.embed_dims_0, ...]), output_size=x_size)
        # x_1 = F.adaptive_avg_pool2d(self.DW_conv1(x[:, self.embed_dims_0: self.embed_dims_0 + self.embed_dims_1, ...]),
        #                             output_size=x_size)
        # x_2 = F.adaptive_avg_pool2d(self.DW_conv2(x[:, self.embed_dims - self.embed_dims_2:, ...]), output_size=x_size)
        # x = torch.cat([x_0, x_1, x_2], dim=1)
        # x = self.PW_conv(x)

        x = F.adaptive_avg_pool2d(self.DW_conv(x), output_size=x_size)
        return x


class Attention_conv(nn.Module):
    def __init__(self, dim=48, num_heads=3, bias=True):
        super(Attention_conv, self).__init__()
        self.num_heads = num_heads  # 注意力头的个数
        # vi
        self.temperature_vi = nn.Parameter(torch.ones(num_heads, 1, 1))  # 可学习系数
        self.q_vi = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            # nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=bias),
        )
        self.kv_vi = nn.Conv2d(dim, dim * 2, 1)
        self.project_out_vi = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.MultiOrderPool_vi = MultiOrderAvgPool(dim)
        ##ir
        self.temperature_ir = nn.Parameter(torch.ones(num_heads, 1, 1))  # 可学习系数
        self.q_ir = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            # nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=bias),
        )
        self.kv_ir = nn.Conv2d(dim, dim * 2, 1)
        self.project_out_ir = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.MultiOrderPool_ir = MultiOrderMaxPool(dim)

    def forward(self, ir, vi, x_size):
        ###vi------qkv
        b, c, h, w = vi.shape  # 输入的结构 batch 数，通道数和高宽
        q_vi = self.q_vi(vi)  # .reshape(b, h*w, self.num_heads, c // self.num_heads).permute(0, 2, 1, 3)
        kv_vi = self.kv_vi(self.MultiOrderPool_vi(vi,
                                                  x_size))  # .reshape(b, -1, 2, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        k_vi, v_vi = kv_vi.chunk(2, dim=1)
        # k,v=kv[0],kv[1]
        # print(q.shape, k.shape, v.shape, )
        q_vi = torch.nn.functional.normalize(q_vi, dim=-1)  # C 维度标准化，这里的 C 与通道维度略有不同
        k_vi = torch.nn.functional.normalize(k_vi, dim=-1)
        q_vi = rearrange(q_vi, 'b (head c) h w -> b head (h w) c ', head=self.num_heads)
        k_vi = rearrange(k_vi, 'b (head c) h w -> b head (h w) c', head=self.num_heads)
        v_vi = rearrange(v_vi, 'b (head c) h w -> b head (h w) c', head=self.num_heads)
        # print(q.shape,k.shape,v.shape,)
        ###ir------qkv
        q_ir = self.q_ir(ir)  # .reshape(b, h*w, self.num_heads, c // self.num_heads).permute(0, 2, 1, 3)
        kv_ir = self.kv_ir(self.MultiOrderPool_ir(ir,
                                                  x_size))  # .reshape(b, -1, 2, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        k_ir, v_ir = kv_ir.chunk(2, dim=1)
        # k,v=kv[0],kv[1]
        # print(q.shape, k.shape, v.shape, )
        q_ir = torch.nn.functional.normalize(q_ir, dim=-1)  # C 维度标准化，这里的 C 与通道维度略有不同
        k_ir = torch.nn.functional.normalize(k_ir, dim=-1)
        q_ir = rearrange(q_ir, 'b (head c) h w -> b head (h w) c ', head=self.num_heads)
        k_ir = rearrange(k_ir, 'b (head c) h w -> b head (h w) c', head=self.num_heads)
        v_ir = rearrange(v_ir, 'b (head c) h w -> b head (h w) c', head=self.num_heads)
        ###vi------self attention
        attn_vi = (q_vi @ k_ir.transpose(-2, -1)) * self.temperature_vi
        attn_vi = attn_vi.softmax(dim=-1)
        out_vi = (attn_vi @ v_ir).reshape(b, self.num_heads, h, w, -1).permute(0, 1, 4, 2, 3)  # 注意力图(严格来说不算图)
        out_vi = out_vi.reshape(b, -1, h, w)
        out_vi = self.project_out_vi(out_vi)
        ###ir------self attention
        attn_ir = (q_ir @ k_vi.transpose(-2, -1)) * self.temperature_ir
        attn_ir = attn_ir.softmax(dim=-1)
        out_ir = (attn_ir @ v_vi).reshape(b, self.num_heads, h, w, -1).permute(0, 1, 4, 2, 3)  # 注意力图(严格来说不算图)
        out_ir = out_ir.reshape(b, -1, h, w)
        out_ir = self.project_out_ir(out_ir)

        return out_ir, out_vi


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=1, ffn_expansion_factor=None, bias=False, LayerNorm_type='WithBias'):
        super(TransformerBlock, self).__init__()

        # self.norm_ir = LayerNorm(dim, LayerNorm_type)
        # self.norm_vi = LayerNorm(dim, LayerNorm_type)
        # self.attn = Attention_conv(dim, num_heads, bias)

        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FMultiOrderDWConv(dim)


        self.cov = nn.Conv2d(2 * dim, dim, 1)

    def pc(self,ir, vi):
        max = torch.max(ir, vi)
        # max = (ir+vi)/2
        act = F.sigmoid(max)
        d_ir = ir * act - ir
        d_vi = vi * act - vi
        ir = ir + d_vi
        vi = vi + d_ir
        return ir, vi

    def forward(self, ir, vi, x_size):

        ir_a,vi_a=self.pc(ir, vi)
        x = self.cov(torch.cat((ir_a, vi_a), 1))
        x = x + self.ffn(self.norm2(x))

        return x

class pure(nn.Module):
    def __init__(self, dim,init=0.01):
        super(pure, self).__init__()

        self.pre_process = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1)
        self.pre_process1 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1)
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.act = nn.Sigmoid()

    def forward(self, x):
        x=self.pre_process(x)
        xdp = self.pre_process1(x)
        x_max = xdp * self.act(self.maxpool(xdp))
        out = x_max+ x
        return out



