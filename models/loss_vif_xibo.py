from matplotlib import image
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torchvision.models.vgg import vgg16
import numpy as np
from utils.utils_color import RGB_HSV, RGB_YCbCr
from models.loss_ssim import ssim
import torchvision.transforms.functional as TF


class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, image_A, image_B, image_fused):
        # diff_B = image_fused - image_B
        # diff_A = image_fused - image_A
        # diff = torch.max(diff_A, diff_B)
        diff = torch.max(image_A, image_B)
        diff = image_fused - diff
        # loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps * self.eps)))
        return loss


class EdgeLoss(nn.Module):
    def __init__(self):
        super(EdgeLoss, self).__init__()
        k = torch.Tensor([[.05, .25, .4, .25, .05]])
        self.kernel = torch.matmul(k.t(), k).unsqueeze(0).repeat(1, 1, 1, 1)
        if torch.cuda.is_available():
            self.kernel = self.kernel.cuda()
        self.loss = CharbonnierLoss()
        self.sobel = Sobelxy()

    def conv_gauss(self, img):
        n_channels, _, kw, kh = self.kernel.shape
        img = F.pad(img, (kw // 2, kh // 2, kw // 2, kh // 2), mode='replicate')
        return F.conv2d(img, self.kernel, groups=n_channels)

    def laplacian_kernel(self, current):
        filtered = self.conv_gauss(current)  # filter
        down = filtered[:, :, ::2, ::2]  # downsample
        new_filter = torch.zeros_like(filtered)
        new_filter[:, :, ::2, ::2] = down * 4  # upsample
        filtered = self.conv_gauss(new_filter)  # filter
        diff = current - filtered
        return diff

    def forward(self, image_A, image_B, image_fused):
        # a = self.laplacian_kernel(image_A)
        # b = self.laplacian_kernel(image_B)
        # image_fused = self.laplacian_kernel(image_fused)
        # loss = self.loss(a, b, image_fused)

        a = self.sobel(image_A)
        b = self.sobel(image_B)
        image_fused = self.sobel(image_fused)
        loss = self.loss(a, b, image_fused)
        return loss


class L_color(nn.Module):

    def __init__(self):
        super(L_color, self).__init__()

    def forward(self, x):
        b, c, h, w = x.shape

        mean_rgb = torch.mean(x, [2, 3], keepdim=True)
        mr, mg, mb = torch.split(mean_rgb, 1, dim=1)
        Drg = torch.pow(mr - mg, 2)
        Drb = torch.pow(mr - mb, 2)
        Dgb = torch.pow(mb - mg, 2)
        k = torch.pow(torch.pow(Drg, 2) + torch.pow(Drb, 2) + torch.pow(Dgb, 2), 0.5)
        return k


class L_Grad(nn.Module):
    def __init__(self):
        super(L_Grad, self).__init__()
        self.sobelconv = Sobelxy()

    def forward(self, image_A, image_B, image_fused):
        gradient_A = self.sobelconv(image_A)
        gradient_B = self.sobelconv(image_B)
        gradient_fused = self.sobelconv(image_fused)
        gradient_joint = torch.max(gradient_A, gradient_B)
        # Loss_gradient = F.l1_loss(gradient_fused, gradient_joint)
        Loss_gradient = torch.norm(gradient_fused - gradient_joint)
        return Loss_gradient


class L_SSIM(nn.Module):
    def __init__(self):
        super(L_SSIM, self).__init__()
        self.sobelconv = Sobelxy()

    def forward(self, image_A, image_B, image_fused):
        gradient_A = self.sobelconv(image_A)
        gradient_B = self.sobelconv(image_B)
        weight_A = torch.mean(gradient_A) / (torch.mean(gradient_A) + torch.mean(gradient_B))
        weight_B = torch.mean(gradient_B) / (torch.mean(gradient_A) + torch.mean(gradient_B))
        Loss_SSIM = weight_A * ssim(image_A, image_fused) + weight_B * ssim(image_B, image_fused)
        return Loss_SSIM


class Sobelxy(nn.Module):
    def __init__(self):
        super(Sobelxy, self).__init__()
        kernelx = [[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]]
        kernely = [[1, 2, 1],
                   [0, 0, 0],
                   [-1, -2, -1]]
        kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
        kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
        self.weightx = nn.Parameter(data=kernelx, requires_grad=False).cuda()
        self.weighty = nn.Parameter(data=kernely, requires_grad=False).cuda()

    def forward(self, x):
        sobelx = F.conv2d(x, self.weightx, padding=1)
        sobely = F.conv2d(x, self.weighty, padding=1)
        return torch.abs(sobelx) + torch.abs(sobely)


class L_Intensity(nn.Module):
    def __init__(self):
        super(L_Intensity, self).__init__()

    def forward(self, image_A, image_B, image_fused):
        intensity_joint = torch.max(image_A, image_B)
        Loss_intensity = F.l1_loss(image_fused, intensity_joint)
        return Loss_intensity


class TV_Loss(nn.Module):

    def __init__(self):
        super(TV_Loss, self).__init__()

    def forward(self, IA, IB, IF):
        rA = IA - IF
        rB = IB - IF
        h = rA.shape[2]
        w = rA.shape[3]
        tvAH = torch.pow((rA[:, :, 1:, :] - rA[:, :, :h - 1, :]), 2).mean()
        tvAW = torch.pow((rA[:, :, :, 1:] - rA[:, :, :, :w - 1]), 2).mean()
        tvBH = torch.pow((rB[:, :, 1:, :] - rB[:, :, :h - 1, :]), 2).mean()
        tvBW = torch.pow((rB[:, :, :, 1:] - rB[:, :, :, :w - 1]), 2).mean()
        TV_LossA = tvAH + tvAW
        TV_LossB = tvBH + tvBW
        # TV_Loss = 10*TV_LossA + 20*TV_LossB
        TV_Loss = 0.4 * TV_LossA + 0.6 * TV_LossB
        return TV_Loss


# class fusion_loss_vif(nn.Module):
#     def __init__(self):
#         super(fusion_loss_vif, self).__init__()
#         self.L_Grad = L_Grad()
#         self.L_Inten = L_Intensity()
#         self.L_SSIM = L_SSIM()
#
#         # print(1)
#
#     def forward(self, image_A, image_B, image_fused):
#         loss_TV = torch.tensor(0.0)
#         loss_l1 = 0.01 * self.L_Inten(image_A, image_B, image_fused)
#         loss_gradient = 1 * self.L_Grad(image_A, image_B, image_fused)
#         loss_SSIM = 1 * (1 - self.L_SSIM(image_A, image_B, image_fused))
#         fusion_loss = loss_l1 + loss_gradient + loss_SSIM
#         return fusion_loss, loss_TV, loss_gradient, loss_l1, loss_SSIM
#


class fusion_loss_vif(nn.Module):
    def __init__(self):
        super(fusion_loss_vif, self).__init__()
        self.L_Grad = L_Grad()
        self.L_Inten = L_Intensity()
        self.L_SSIM = L_SSIM()
        self.L_TV = TV_Loss()
        self.L_c = CharbonnierLoss()
        self.edge = EdgeLoss()

    def forward(self, image_A, image_B, image_fused):
        # loss_l1 = torch.tensor(0.0)
        # loss_l1 =  1*self.L_Inten(image_A, image_B, image_fused)
        # loss_gradient = 10*self.L_Grad(image_A, image_B, image_fused)
        # loss_SSIM =5* (1 - self.L_SSIM(image_A, image_B, image_fused))
        # loss_TV = torch.tensor(0.0)
        # loss_TV = self.L_TV(image_A, image_B, image_fused)
        # fusion_loss = loss_l1 + loss_gradient + loss_SSIM + loss_TV

        loss_TV = torch.tensor(0.0)
        loss_SSIM = torch.tensor(0.0)
        loss_lc = self.L_c(image_A, image_B, image_fused)
        loss_edge = self.edge(image_A, image_B, image_fused)
        # fusion_loss = 200*loss_lc + 400*loss_edge
        fusion_loss = 100 * loss_lc + 200 * loss_edge
        return fusion_loss, loss_TV, loss_edge, loss_lc, loss_SSIM
        # return fusion_loss, loss_TV, loss_gradient, loss_l1, loss_SSIM
    # def forward(self, image_A, image_B, image_fused):
    #     # loss_l1 = torch.tensor(0.0)
    #     loss_l1 = 0.1*self.L_Inten(image_A, image_B, image_fused)
    #     loss_gradient =2*self.L_Grad(image_A, image_B, image_fused)
    #     loss_SSIM =2* (1 - self.L_SSIM(image_A, image_B, image_fused))
    #     loss_TV = torch.tensor(0.0)
    #     # loss_TV = self.L_TV(image_A, image_B, image_fused)
    #     fusion_loss = loss_l1 + loss_gradient + loss_SSIM + loss_TV
    #     return fusion_loss, loss_TV, loss_gradient, loss_l1, loss_SSIM

