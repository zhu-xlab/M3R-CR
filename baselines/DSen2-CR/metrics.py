import math
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size / 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size))
    return window

def SSIM(img1, img2):
    (_, channel, _, _) = img1.size()
    window_size = 11
    window = create_window(window_size, channel).cuda()
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean().item()

def PSNR(img1, img2, mask=None):
    if mask is not None:
        mse = (img1 - img2) ** 2
        B, C, H, W = mse.size()
        mse = torch.sum(mse * mask.float()) / (torch.sum(mask.float()) * C)
    else:
        mse = torch.mean((img1 - img2) ** 2)

    if mse == 0:
        return 100
    PIXEL_MAX = 1
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def MAE(ref, tgt, is_mean = True):
    ref = ref.cpu().detach().numpy()
    tgt = tgt.cpu().detach().numpy()
    pixel_errors = np.absolute(ref - tgt)
    if is_mean:
        return pixel_errors.mean()
    else:
        return pixel_errors

def get_MAE_with_landcover(pixel_errors, landcover, lc_level):

    landcover = landcover.cpu().detach().numpy()

    if lc_level == '1':
        '''
        "0: Forest land", "1: Rangeland", "2: Agriculture land", "3: Urban land", "4: Barren land", "5: Water"
        '''
        labels = [0, 1, 2, 3, 4, 5, 6]
    elif lc_level == '2':
        '''
        "0: Trees", "1: Shrubland", "2: Grassland", "3: Cropland", "4: Built-up", "5: Barren / sparse vegetation", 
        "6: Snow and ice", "7: Open water", "8: Herbaceous wetland", "9: Mangroves", "10: Moss and lichen"
        '''
        labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    maes = []
    for label in labels:
        if label in landcover:
            label_mask = (landcover == label)
            maes.append(np.sum(pixel_errors*label_mask)/(np.sum(label_mask)*pixel_errors.shape[1]))
        else:
            maes.append(None)
    return maes

def SAM(ref, tgt, is_mean = True):

    ref = ref.cpu().detach().numpy()
    tgt = tgt.cpu().detach().numpy()

    ref = ref.transpose(0, 2, 3, 1)
    tgt = tgt.transpose(0, 2, 3, 1)

    kernel = np.einsum('...k,...k', ref, tgt)

    square_norm_ref = np.einsum('...k,...k', ref, ref).clip(min=np.finfo(np.float16).eps)
    square_norm_tgt = np.einsum('...k,...k', tgt, tgt).clip(min=np.finfo(np.float16).eps)
    normalized_kernel = kernel / np.sqrt(square_norm_ref * square_norm_tgt)

    angles = np.arccos(normalized_kernel.clip(min=-1, max=1)) / np.pi * 180

    if is_mean:
        return angles.mean()
    else:
        return angles

def get_SAM_with_landcover(angles, landcover, lc_level):

    landcover = landcover.cpu().detach().numpy()

    if lc_level == '1':
        '''
        "0: Forest land", "1: Rangeland", "2: Agriculture land", "3: Urban land", "4: Barren land", "5: Water"
        '''
        labels = [0, 1, 2, 3, 4, 5, 6]
    elif lc_level == '2':
        '''
        "0: Trees", "1: Shrubland", "2: Grassland", "3: Cropland", "4: Built-up", "5: Barren / sparse vegetation", 
        "6: Snow and ice", "7: Open water", "8: Herbaceous wetland", "9: Mangroves", "10: Moss and lichen"
        '''
        labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    sams = []
    for label in labels:
        if label in landcover:
            label_mask = (landcover == label)
            sams.append(np.sum(angles*label_mask)/np.sum(label_mask))
        else:
            sams.append(None)
    return sams