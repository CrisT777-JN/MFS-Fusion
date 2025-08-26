#
# from matplotlib import image
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import math
# from torchvision.models.vgg import vgg16
# import numpy as np
# from utils.utils_color import RGB_HSV, RGB_YCbCr
# from models.loss_ssim import ssim
# import torchvision.transforms.functional as TF
#
# class L_color(nn.Module):
#
#     def __init__(self):
#         super(L_color, self).__init__()
#
#     def forward(self, x ):
#
#         b,c,h,w = x.shape
#
#         mean_rgb = torch.mean(x,[2,3],keepdim=True)
#         mr,mg, mb = torch.split(mean_rgb, 1, dim=1)
#         Drg = torch.pow(mr-mg,2)
#         Drb = torch.pow(mr-mb,2)
#         Dgb = torch.pow(mb-mg,2)
#         k = torch.pow(torch.pow(Drg,2) + torch.pow(Drb,2) + torch.pow(Dgb,2),0.5)
#         return k
#
# class L_Grad(nn.Module):
#     def __init__(self):
#         super(L_Grad, self).__init__()
#         self.sobelconv=Sobelxy()
#
#     def forward(self, image_A, image_B, image_fused):
#         gradient_A = self.sobelconv(image_A)
#         gradient_B = self.sobelconv(image_B)
#         gradient_fused = self.sobelconv(image_fused)
#         gradient_joint = torch.max(gradient_A, gradient_B)
#         Loss_gradient = F.l1_loss(gradient_fused, gradient_joint)
#         # gradient_joint = torch.max(image_A, image_B)
#         # Loss_gradient = F.l1_loss(image_fused, gradient_joint)
#         return Loss_gradient
#
# class L_SSIM(nn.Module):
#     def __init__(self):
#         super(L_SSIM, self).__init__()
#         self.sobelconv=Sobelxy()
#
#     def forward(self, image_A, image_B, image_fused):
#         gradient_A = self.sobelconv(image_A)
#         gradient_B = self.sobelconv(image_B)
#         weight_A = torch.mean(gradient_A) / (torch.mean(gradient_A) + torch.mean(gradient_B))
#         weight_B = torch.mean(gradient_B) / (torch.mean(gradient_A) + torch.mean(gradient_B))
#         Loss_SSIM = weight_A * (ssim(image_A, image_fused)) + weight_B * (ssim(image_B, image_fused))
#         return Loss_SSIM
# class Sobelxy(nn.Module):
#     def __init__(self):
#         super(Sobelxy, self).__init__()
#         kernelx = [[-1, 0, 1],
#                   [-2,0 , 2],
#                   [-1, 0, 1]]
#         kernely = [[1, 2, 1],
#                   [0,0 , 0],
#                   [-1, -2, -1]]
#         kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
#         kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
#         self.weightx = nn.Parameter(data=kernelx, requires_grad=False).cuda()
#         self.weighty = nn.Parameter(data=kernely, requires_grad=False).cuda()
#     def forward(self,x):
#         sobelx=F.conv2d(x, self.weightx, padding=1)
#         sobely=F.conv2d(x, self.weighty, padding=1)
#         return torch.abs(sobelx)+torch.abs(sobely)
#
# class L_Intensity(nn.Module):
#     def __init__(self):
#         super(L_Intensity, self).__init__()
#
#     def forward(self, image_A, image_B, image_fused):
#         intensity_joint = torch.max(image_A, image_B)
#         Loss_intensity = F.l1_loss(image_fused, intensity_joint)
#         return Loss_intensity
#
#
# class fusion_loss_vif(nn.Module):
#     def __init__(self):
#         super(fusion_loss_vif, self).__init__()
#         self.L_Grad = L_Grad()
#         self.L_Inten = L_Intensity()
#         self.L_SSIM = L_SSIM()
#
#         # print(1)
#     def forward(self, image_A, image_B, image_fused):
#         loss_l1 = 20 * self.L_Inten(image_A, image_B, image_fused)
#         loss_gradient = 20 * self.L_Grad(image_A, image_B, image_fused)
#         loss_SSIM = 10 * (1 - self.L_SSIM(image_A, image_B, image_fused))
#         fusion_loss = loss_l1 + loss_gradient + loss_SSIM
#         return fusion_loss, loss_gradient, loss_l1, loss_SSIM
#
#         # loss_l1 = torch.tensor(0.0)
#         # loss_gradient = 40 * self.L_Grad(image_A, image_B, image_fused)
#         # loss_SSIM = 20 *  (1-self.L_SSIM(image_A, image_B, image_fused))
#         # fusion_loss =  loss_gradient + loss_SSIM
#         return fusion_loss, loss_gradient, loss_l1, loss_SSIM
#

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

'''
#vif
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
        Loss_gradient = torch.norm(gradient_fused - gradient_joint)
        # Loss_gradient = F.l1_loss(gradient_fused, gradient_joint)
        # gradient_joint = torch.max(image_A, image_B)
        # Loss_gradient = F.l1_loss(image_fused, gradient_joint)
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
        Loss_SSIM = 0.5 * ssim(image_A, image_fused) + 0.5 * ssim(image_B, image_fused)
        # 史诗证明，只有单独某一个方向的SSIM——loss得到的结果都是黑色目标
        # Loss_SSIM = ssim(image_B, image_fused) xiao guo bu xing,chu lai de shi hei ren er
        # ##Loss_SSIM = 0.5*ssim(image_B, image_fused)+0.5*(ssim(image_A,image_fused))
        # Loss_SSIM = weight_A * (ssim(image_A, image_fused)) + weight_B * (ssim(image_B, image_fused))
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
        # Loss_intensity = F.l1_loss(image_fused, image_B)
        # Loss_intensity = F.l1_loss(image_fused, image_B)
        # Loss_intensity = F.mse_loss(image_fused,intensity_joint)
        return Loss_intensity


class fusion_loss_vif(nn.Module):
    def __init__(self):
        super(fusion_loss_vif, self).__init__()
        self.L_Grad = L_Grad()
        self.L_Inten = L_Intensity()
        self.L_SSIM = L_SSIM()

    def forward(self, image_A, image_B, image_fused):
        loss_l1 = 20 * self.L_Inten(image_A, image_B, image_fused)
        loss_gradient = 0.01 * self.L_Grad(image_A, image_B, image_fused)
        loss_SSIM = 10 * (1 - self.L_SSIM(image_A, image_B, image_fused))
        fusion_loss = loss_l1 + loss_gradient + loss_SSIM
        return fusion_loss, loss_gradient, loss_l1, loss_SSIM

        # loss_l1 = torch.tensor(0.0)
        # loss_gradient = 20 * self.L_Grad(image_A, image_B, image_fused)
        # loss_SSIM = 10 * (1- self.L_SSIM(image_A, image_B, image_fused))
        # fusion_loss =  loss_gradient + loss_SSIM
        return fusion_loss, loss_gradient, loss_l1, loss_SSIM
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torchvision.models.vgg import vgg16
import numpy as np
import torchvision.transforms.functional as TF
from models.loss_ssim import ssim

import torch
import torch.nn as nn


class Mlp(nn.Module):
    def __init__(self, in_features=128, hidden_features=64, out_features=2, act_layer=nn.ReLU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Brightness_perception(nn.Module):
    def __init__(self):
        super(Brightness_perception, self).__init__()
        self.conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(7, 7), stride=(7, 7))
        self.GAP = nn.AdaptiveAvgPool2d(8)
        self.mlp = Mlp(drop=0.1)

    def forward(self, x, y):
        x = self.GAP(self.conv(x))
        y = self.GAP(self.conv(y))
        B, C, H, W = x.shape
        x = torch.concat([x, y], 1)
        x = x.view(B, -1)
        x = self.mlp(x)
        return x


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
        self.Brigthness = Brightness_perception()

    def forward(self, image_A, image_B, image_fused):
        # out=self.Brigthness(image_A,image_B)
        # pre_A=out[:,0]
        # pre_B=out[:,1]
        # weight_A=pre_A/(pre_A+pre_B)
        # weight_B=1-weight_A#torch.Size([9, 2]) torch.Size([9]) torch.Size([9]) torch.Size([9, 1, 1, 1])
        # #print(F.l1_loss(weight_A[:,None,None,None]*image_fused,weight_A[:,None,None,None]*image_A)
        # gradient_joint = torch.max(image_A, image_B)
        # Loss_gradient = F.l1_loss(image_fused, gradient_joint)
        # Loss_intensity =40*Loss_gradient
        intensity_joint = torch.max(image_A, image_B)
        Loss_intensity = F.l1_loss(image_fused, intensity_joint)
        # image_A=image_A.unsqueeze(0)
        # image_B=image_B.unsqueeze(0)
        # intensity_joint = torch.mean(torch.cat([image_A, image_B]), dim=0)
        # Loss_intensity = F.l1_loss(image_fused, intensity_joint)
        return Loss_intensity


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

class TV_Loss(torch.nn.Module):

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
        TV_Loss = 10*TV_LossA + 20*TV_LossB
        # print("TV_LossA,TV_LossB", TV_LossA, TV_LossB)
        # TV_LossA, TV_LossB tensor(0.0368, device='cuda:0', grad_fn= < AddBackward0 >) tensor(0.0373, device='cuda:0',grad_fn= < AddBackward0 >)
        # loss_tv tensor(0.0741, device='cuda:0', grad_fn= < AddBackward0 >)
        # TV_LossA, TV_LossB
        # tensor(0.0365, device='cuda:0', grad_fn= < AddBackward0 >) tensor(0.0366, device='cuda:0',
        #                                                                   grad_fn= < AddBackward0 >)
        # loss_tv
        # tensor(0.0731, device='cuda:0', grad_fn= < AddBackward0 >)
        return TV_Loss


class fusion_loss_vif(nn.Module):
    def __init__(self):
        super(fusion_loss_vif, self).__init__()
        self.L_Grad = L_Grad()
        self.L_Inten = L_Intensity()
        self.L_SSIM = L_SSIM()
        self.L_TVLoss = TV_Loss()

    def forward(self, image_A, image_B, image_fused):
        loss_tv = 10 * self.L_TVLoss(image_A, image_B, image_fused)
        # print('loss_tv', loss_tv)
        loss_l1 = 10 * self.L_Inten(image_A, image_B, image_fused)
        loss_gradient = 0.1 * self.L_Grad(image_A, image_B, image_fused)
        loss_SSIM = 5 * (1 - self.L_SSIM(image_A, image_B, image_fused))
        fusion_loss = loss_l1 + loss_gradient + loss_SSIM + loss_tv
        return fusion_loss, loss_gradient, loss_l1, loss_SSIM
