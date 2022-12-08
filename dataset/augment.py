import os
import math
import pickle
import random
import numpy as np
import glob
import torch
import cv2


class RandomCrop:
    def __init__(self, size=(256, 256)):
        self.size = size

    def __call__(self, img1, img2=None):
        if img2 is not None:
            assert img1.shape[-2:] == img2.shape[-2:
                                                 ], "Two images should be the same shape (..., h, w)."
        h, w = img1.shape[-2:]
        # crop_t = np.random.randint(0, h - self.size[0])
        # crop_l = np.random.randint(0, w - self.size[1])
        crop_t = random.randint(0, h - self.size[0])
        crop_l = random.randint(0, w - self.size[1])

        img1_croped = img1[..., crop_t: crop_t +
                           self.size[0], crop_l: crop_l + self.size[1]]
        if img2 is not None:
            img2_croped = img2[..., crop_t: crop_t +
                               self.size[0], crop_l: crop_l + self.size[1]]

            return img1_croped, img2_croped

        return img1_croped


class SRRandomCrop:
    def __init__(self, size=(256, 256)):
        self.size = size

    def __call__(self, img1, img2=None):
        if img2 is not None:
            scale_factor_h = img2.shape[-2] / img1.shape[-2]
            scale_factor_w = img2.shape[-1] / img1.shape[-1]
        h, w = img1.shape[-2:]
        crop_t = random.randint(0, h - self.size[0])
        crop_l = random.randint(0, w - self.size[1])

        img1_croped = img1[...,
                           crop_t: crop_t + self.size[0],
                           crop_l: crop_l + self.size[1]]
        if img2 is not None:
            crop_t = int(crop_t * scale_factor_h)
            crop_l = int(crop_l * scale_factor_w)
            crop_b = crop_t + int(self.size[0] * scale_factor_h)
            crop_r = crop_l + int(self.size[1] * scale_factor_w)
            img2_croped = img2[...,
                               crop_t: crop_b,
                               crop_l: crop_r]

            return img1_croped, img2_croped

        return img1_croped


class RandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img1, img2=None):
        # flip = np.random.rand() < self.p
        flip = random.random() < self.p
        if flip:
            img1 = img1[..., ::-1]
        if img2 is not None:
            if flip:
                img2 = img2[..., ::-1]
            return img1, img2

        return img1


class RandomVerticalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img1, img2=None):
        # flip = np.random.rand() < self.p
        flip = random.random() < self.p
        if flip:
            img1 = img1[..., ::-1, :]
        if img2 is not None:
            if flip:
                img2 = img2[..., ::-1, :]
            return img1, img2

        return img1


class RandomRotation90:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img1, img2=None):
        # rot90 = np.random.rand() < self.p
        rot90 = random.random() < self.p
        if rot90:
            img1 = img1.transpose(0, 1, 3, 2)
        if img2 is not None:
            if rot90:
                img2 = img2.transpose(0, 1, 3, 2)
            return img1, img2

        return img1


class RandomReverse:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img1, img2):
        reverse = random.random() < self.p
        if reverse:
            img1 = img1[::-1]
        if img2 is not None:
            if reverse:
                img2 = img2[::-1]
            return img1, img2

        return img1


class Normalize:
    def __init__(self, pixel_max=255):
        self.pixel_max = pixel_max

    def __call__(self, img1, img2=None):
        img1 = img1.astype(np.float)
        if img2 is not None:
            img2 = img2.astype(np.float)
            return img1 / 255.0, img2 / 255.0
        else:
            return img1 / 255.0


def augment(img1, img2, cfg):
    """
    img1 and img2 are with Numpy format.
    """
    transforms = []
    if hasattr(cfg, "RandomCrop"):
        transforms.append(RandomCrop(**cfg["RandomCrop"]))
    if hasattr(cfg, "SRRandomCrop"):
        transforms.append(SRRandomCrop(**cfg["SRRandomCrop"]))
    if hasattr(cfg, "RandomHorizontalFlip"):
        transforms.append(RandomHorizontalFlip(**cfg["RandomHorizontalFlip"]))
    if hasattr(cfg, "RandomVerticalFlip"):
        transforms.append(RandomVerticalFlip(**cfg["RandomVerticalFlip"]))
    if hasattr(cfg, "RandomReverse"):
        transforms.append(RandomReverse(**cfg["RandomReverse"]))
    if hasattr(cfg, "RandomRotation90"):
        transforms.append(RandomRotation90(**cfg["RandomRotation90"]))
    if hasattr(cfg, "Normalize"):
        transforms.append(Normalize(**cfg["Normalize"]))

    for trans in transforms:
        img1, img2 = trans(img1, img2)

    return img1, img2


def generate_2D_grid(H, W):
    x = torch.arange(0, W, 1).float()
    y = torch.arange(0, H, 1).float()

    xx = x.repeat(H, 1)
    yy = y.view(H, 1).repeat(1, W)

    grid = torch.stack([xx, yy], dim=0)

    return grid  # (2,H,W)


def distortion_map(h, w, ref_row, reverse=False):
    grid_row = generate_2D_grid(h, w)[1].float()
    mask = grid_row / (h - 1)
    if reverse:
        mask *= -1.
        ref_row_floor = math.floor(h - 1 - ref_row)
        mask = mask - mask[int(ref_row_floor)] + (h - 1 - ref_row - ref_row_floor) * (1. / (h - 1))
    else:
        ref_row_floor = math.floor(ref_row)
        mask = mask - mask[int(ref_row_floor)] - (ref_row - ref_row_floor) * (1. / (h - 1))

    return mask.unsqueeze(0).unsqueeze(0)  # (1，1，H, W)