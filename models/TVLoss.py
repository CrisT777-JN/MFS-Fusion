import torch


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
        TV_Loss = TV_LossA + TV_LossB

        return TV_Loss
