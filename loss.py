"""
The losses colleted from Deep Shutter Unrolling Network
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import models

from simdeblur.model.build import LOSS_REGISTRY


class Grid_gradient_central_diff():
    def __init__(self, nc, padding=True, diagonal=False):
        self.conv_x = nn.Conv2d(nc, nc, kernel_size=2, stride=1, bias=False)
        self.conv_y = nn.Conv2d(nc, nc, kernel_size=2, stride=1, bias=False)
        self.conv_xy = None
        if diagonal:
            self.conv_xy = nn.Conv2d(
                nc, nc, kernel_size=2, stride=1, bias=False)

        self.padding = None
        if padding:
            self.padding = nn.ReplicationPad2d([0, 1, 0, 1])

        fx = torch.zeros(nc, nc, 2, 2).float().cuda()
        fy = torch.zeros(nc, nc, 2, 2).float().cuda()
        if diagonal:
            fxy = torch.zeros(nc, nc, 2, 2).float().cuda()

        fx_ = torch.tensor([[1, -1], [0, 0]]).cuda()
        fy_ = torch.tensor([[1, 0], [-1, 0]]).cuda()
        if diagonal:
            fxy_ = torch.tensor([[1, 0], [0, -1]]).cuda()

        for i in range(nc):
            fx[i, i, :, :] = fx_
            fy[i, i, :, :] = fy_
            if diagonal:
                fxy[i, i, :, :] = fxy_

        self.conv_x.weight = nn.Parameter(fx)
        self.conv_y.weight = nn.Parameter(fy)
        if diagonal:
            self.conv_xy.weight = nn.Parameter(fxy)

    def __call__(self, grid_2d):
        _image = grid_2d
        if self.padding is not None:
            _image = self.padding(_image)
        dx = self.conv_x(_image)
        dy = self.conv_y(_image)

        if self.conv_xy is not None:
            dxy = self.conv_xy(_image)
            return dx, dy, dxy
        return dx, dy


@LOSS_REGISTRY.register()
class DsunPerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = nn.L1Loss()
        self.model = self.contentFunc()

    def contentFunc(self):
        conv_3_3_layer = 14
        cnn = models.vgg19(pretrained=True).features
        model = nn.Sequential()
        model = model.cuda()
        for i, layer in enumerate(list(cnn)):
            model.add_module(str(i), layer)
            if i == conv_3_3_layer:
                break
        return model

    def forward(self, fakeIm, realIm):
        f_fake = self.model(fakeIm)
        f_real = self.model(realIm)
        f_real_no_grad = f_real.detach()
        loss = self.criterion(f_fake, f_real_no_grad)
        return loss


@LOSS_REGISTRY.register()
class DSUNL1Loss(nn.Module):
    def __init__(self):
        super(DSUNL1Loss, self).__init__()

    def forward(self, output, target, weight=None, mean=False):
        error = torch.abs(output - target)
        if weight is not None:
            error = error * weight.float()
            if mean is not False:
                return error.sum() / weight.float().sum()
        if mean is not False:
            return error.mean()
        return error.sum()


@LOSS_REGISTRY.register()
class VariationLoss(nn.Module):
    def __init__(self, nc, grad_fn=Grid_gradient_central_diff, mean=True):
        super(VariationLoss, self).__init__()
        self.grad_fn = grad_fn(nc)
        self.mean = mean

    def forward(self, image, weight=None):
        dx, dy = self.grad_fn(image)
        variation = dx**2 + dy**2

        if weight is not None:
            variation = variation * weight.float()
            if self.mean is not False:
                return variation.sum() / weight.sum()
        if self.mean is not False:
            return variation.mean()
        return variation.sum()
