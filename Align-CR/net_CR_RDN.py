import torch
import torch.nn as nn
from numpy.random import normal
from numpy.linalg import svd
from math import sqrt
import os
import argparse

import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import numpy as np
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

from submodules import df_conv, df_resnet_block, kernel2d_conv

import functools
import model.arch_util as arch_util

from basicsr.archs.edvr_arch import PCDAlignment

class Align_Feature(nn.Module):
    def __init__(self, input_channel):
        super(Align_Feature, self).__init__()
        groups=8
        
        self.lrelu = nn.ReLU(inplace=True)

        self.src_L2_conv1 = nn.Conv2d(input_channel, input_channel*2, kernel_size=3, stride=2, padding=1, bias=True)
        self.src_L3_conv1 = nn.Conv2d(input_channel*2, input_channel*4, kernel_size=3, stride=2, padding=1, bias=True)
        self.ref_L2_conv1 = nn.Conv2d(input_channel, input_channel*2, kernel_size=3, stride=2, padding=1, bias=True)
        self.ref_L3_conv1 = nn.Conv2d(input_channel*2, input_channel*4, kernel_size=3, stride=2, padding=1, bias=True)
        ############### FPN enhance features ###############
        self.src_toplayer = nn.Conv2d(input_channel*4, input_channel, kernel_size=1, stride=1, padding=0)
        self.src_smooth1 = nn.Conv2d(input_channel, input_channel, kernel_size=3, stride=1, padding=1)
        self.src_smooth2 = nn.Conv2d(input_channel, input_channel, kernel_size=3, stride=1, padding=1)
        self.src_latlayer1 = nn.Conv2d(input_channel*2, input_channel, kernel_size=1, stride=1, padding=0)
        self.src_latlayer2 = nn.Conv2d(input_channel*1, input_channel, kernel_size=1, stride=1, padding=0)
        self.ref_toplayer = nn.Conv2d(input_channel*4, input_channel, kernel_size=1, stride=1, padding=0)
        self.ref_smooth1 = nn.Conv2d(input_channel, input_channel, kernel_size=3, stride=1, padding=1)
        self.ref_smooth2 = nn.Conv2d(input_channel, input_channel, kernel_size=3, stride=1, padding=1)
        self.ref_latlayer1 = nn.Conv2d(input_channel*2, input_channel, kernel_size=1, stride=1, padding=0)
        self.ref_latlayer2 = nn.Conv2d(input_channel*1, input_channel, kernel_size=1, stride=1, padding=0)
        ###############        Align        ###############
        self.pcd_align = PCDAlignment(num_feat=input_channel, deformable_groups=groups)

    def _upsample_add(self, x, y):
        return F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False) + y

    def forward(self, src_feature, ref_feature):
        # 
        L1_src_feature = src_feature
        L2_src_feature = self.lrelu(self.src_L2_conv1(L1_src_feature))
        L3_src_feature = self.lrelu(self.src_L3_conv1(L2_src_feature))

        L1_ref_feature = ref_feature
        L2_ref_feature = self.lrelu(self.ref_L2_conv1(L1_ref_feature))
        L3_ref_feature = self.lrelu(self.ref_L3_conv1(L2_ref_feature))
        # 
        L3_src_feature = self.lrelu(self.src_toplayer(L3_src_feature))
        L2_src_feature = self.src_smooth1(self._upsample_add(L3_src_feature, self.src_latlayer1(L2_src_feature)))
        L1_src_feature = self.src_smooth2(self._upsample_add(L2_src_feature, self.src_latlayer2(L1_src_feature)))

        L3_ref_feature = self.lrelu(self.ref_toplayer(L3_ref_feature))
        L2_ref_feature = self.ref_smooth1(self._upsample_add(L3_ref_feature, self.ref_latlayer1(L2_ref_feature)))
        L1_ref_feature = self.ref_smooth2(self._upsample_add(L2_ref_feature, self.ref_latlayer2(L1_ref_feature)))
        #
        src_feature_list = [L1_src_feature.clone(), L2_src_feature.clone(), L3_src_feature.clone()]
        ref_feature_list = [L1_ref_feature.clone(), L2_ref_feature.clone(), L3_ref_feature.clone()]
        #
        aligned_src_feature = self.pcd_align(src_feature_list, ref_feature_list)

        return aligned_src_feature


def pixel_reshuffle(input, upscale_factor):
    r"""Rearranges elements in a tensor of shape ``[*, C, H, W]`` to a
    tensor of shape ``[C*r^2, H/r, W/r]``.

    See :class:`~torch.nn.PixelShuffle` for details.

    Args:
        input (Variable): Input
        upscale_factor (int): factor to increase spatial resolution by

    Examples:
        >>> input = autograd.Variable(torch.Tensor(1, 3, 12, 12))
        >>> output = pixel_reshuffle(input,2)
        >>> print(output.size())
        torch.Size([1, 12, 6, 6])
    """
    batch_size, channels, in_height, in_width = input.size()

    # // division is to keep data type unchanged. In this way, the out_height is still int type
    out_height = in_height // upscale_factor
    out_width = in_width // upscale_factor
    input_view = input.contiguous().view(batch_size, channels, out_height, upscale_factor, out_width, upscale_factor)
    channels = channels * upscale_factor * upscale_factor

    shuffle_out = input_view.permute(0, 1, 3, 5, 2, 4).contiguous()
    return shuffle_out.view(batch_size, channels, out_height, out_width)

class DFG(nn.Module):
    def __init__(self, channels, ks_2d):
        super(DFG, self).__init__()
        ks = 3
        half_channels = channels // 2
        self.fac_warp = nn.Sequential(
            df_conv(channels, half_channels, kernel_size=ks),
            df_resnet_block(half_channels, kernel_size=ks),
            df_resnet_block(half_channels, kernel_size=ks),
            df_conv(half_channels, half_channels * ks_2d ** 2, kernel_size=1))

    def forward(self, opt_f, sar_f):
        concat = torch.cat([opt_f, sar_f], 1)
        out = self.fac_warp(concat)
        return out

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
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

def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.qkv_SAR = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.attn_drop_SAR = nn.Dropout(attn_drop)

        self.attn_fuse_1x1conv = nn.Conv2d(8, 8, kernel_size=1)

        self.proj = nn.Linear(dim, dim)
        self.proj_SAR = nn.Linear(dim, dim)

        self.proj_drop = nn.Dropout(proj_drop)
        self.proj_drop_SAR = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)

        self.softmax = nn.Softmax(dim=-1)
        self.softmax_SAR = nn.Softmax(dim=-1)

    def forward(self, inputs, mask=None):

        [x, x_SAR] = inputs

        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        qkv_SAR = self.qkv_SAR(x_SAR).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()

        q, k, v = qkv[0], qkv[1], qkv[2] 
        q_SAR, k_SAR, v_SAR = qkv_SAR[0], qkv_SAR[1], qkv_SAR[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        q_SAR = q_SAR * self.scale
        attn = (q @ k.transpose(-2, -1))
        attn_SAR = (q_SAR @ k_SAR.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)
        attn_SAR = attn_SAR + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

            attn_SAR = attn_SAR.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn_SAR = attn_SAR.view(-1, self.num_heads, N, N)

            attn_diff_conv = self.attn_fuse_1x1conv(attn_SAR - attn)
            attn_fuse_gate = torch.sigmoid(attn_diff_conv) # compute the gate 

            attn = attn + (attn_SAR - attn)* attn_fuse_gate

            attn = self.softmax(attn)
            attn_SAR = self.softmax_SAR(attn_SAR)
        else:
            attn_diff_conv = self.attn_fuse_1x1conv(attn_SAR - attn)
            attn_fuse_gate = torch.sigmoid(attn_diff_conv) # compute the gate 

            attn = attn + (attn_SAR - attn)* attn_fuse_gate

            attn = self.softmax(attn)
            attn_SAR = self.softmax_SAR(attn_SAR)

        attn = self.attn_drop(attn)
        attn_SAR = self.attn_drop_SAR(attn_SAR)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x_SAR = (attn_SAR @ v_SAR).transpose(1, 2).reshape(B_, N, C)

        x = self.proj(x)
        x_SAR = self.proj_SAR(x_SAR)

        x = self.proj_drop(x)
        x_SAR = self.proj_drop_SAR(x_SAR)

        return [x, x_SAR]

class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.norm1_SAR = norm_layer(dim)

        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.drop_path_SAR = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.norm2_SAR = norm_layer(dim)

        mlp_hidden_dim = int(dim * mlp_ratio)

        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.mlp_SAR = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, inputs):

        [x, x_SAR] = inputs

        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        shortcut_SAR = x_SAR

        x = self.norm1(x)
        x = x.view(B, H, W, C)

        x_SAR = self.norm1_SAR(x_SAR)
        x_SAR = x_SAR.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            shifted_x_SAR = torch.roll(x_SAR, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
            shifted_x_SAR = x_SAR

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        x_SAR_windows = window_partition(shifted_x_SAR, self.window_size)  # nW*B, window_size, window_size, C
        x_SAR_windows = x_SAR_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        [attn_windows, SAR_attn_windows] = self.attn([x_windows, x_SAR_windows], mask=self.attn_mask)

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        SAR_attn_windows = SAR_attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C
        shifted_x_SAR = window_reverse(SAR_attn_windows, self.window_size, H, W)

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
            x_SAR = torch.roll(shifted_x_SAR, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
            x_SAR = shifted_x_SAR

        x = x.view(B, H * W, C)
        x_SAR = x_SAR.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        x_SAR = shortcut_SAR + self.drop_path_SAR(x_SAR)
        x_SAR = x_SAR + self.drop_path_SAR(self.mlp_SAR(self.norm2_SAR(x_SAR)))

        return [x, x_SAR]

class RDB_Conv(nn.Module):
    def __init__(self, inChannels, growRate, kSize):
        super(RDB_Conv, self).__init__()
        Cin = inChannels
        G = growRate

        self.conv = nn.Sequential(*[
            nn.Conv2d(Cin, G, kSize, padding=(kSize - 1) // 2, stride=1),
            nn.ReLU()
        ])
        self.conv_SAR = nn.Sequential(*[
            nn.Conv2d(Cin, G, kSize, padding=(kSize - 1) // 2, stride=1),
            nn.ReLU()
        ])

    def forward(self, inputs):

        [x, x_SAR] = inputs

        x_conv = self.conv(x)
        x_SAR_conv = self.conv_SAR(x_SAR)  # B, growRate0 + c*growRate, H, W --> B, growRate, H, W

        return [torch.cat((x, x_conv), 1), torch.cat((x_SAR, x_SAR_conv), 1)]


class RDB(nn.Module):
    def __init__(self, growRate0, growRate, nConvLayers, kSize, input_size, num_heads, window_size, shift_size):
        super(RDB, self).__init__()
        G0 = growRate0
        G = growRate
        C = nConvLayers

        convs = []
        for c in range(C):
            convs.append(RDB_Conv(inChannels=G0 + c * G, growRate=G, kSize=kSize))
        self.convs = nn.Sequential(*convs)

        # Local Feature Fusion
        self.LFF = nn.Conv2d(G0 + C * G, G0, 3, 1, 1)
        self.LFF_SAR = nn.Conv2d(G0 + C * G, G0, 3, 1, 1)

        #
        self.SwinTransformerBlock = SwinTransformerBlock(G0, [int(input_size/2), int(input_size/2)], num_heads, window_size, shift_size)

    def forward(self, inputs):
        [x, x_SAR] = inputs
        [x_convs, x_SAR_convs] = self.convs(inputs)
        x_convs_LFF = self.LFF(x_convs)
        x_SAR_convs_LFF = self.LFF_SAR(x_SAR_convs)
        
        B, dim, H, W = x_convs_LFF.shape
        x_convs_LFF_unfold = x_convs_LFF.flatten(2).transpose(1, 2)  
        x_SAR_convs_LFF_unfold = x_SAR_convs_LFF.flatten(2).transpose(1, 2)# B, dim, H, W --> B, H*W, dim
        [x_convs_LFF_unfold, x_SAR_convs_LFF_unfold] = self.SwinTransformerBlock([x_convs_LFF_unfold, x_SAR_convs_LFF_unfold])
        _x_convs_LFF = x_convs_LFF_unfold.transpose(1, 2).view(B, dim, H, W)
        _x_SAR_convs_LFF = x_SAR_convs_LFF_unfold.transpose(1, 2).view(B, dim, H, W)

        return [_x_convs_LFF + x, _x_SAR_convs_LFF + x_SAR]


class RDN_residual_CR(nn.Module):
    def __init__(self, input_size):
        super(RDN_residual_CR, self).__init__()
        self.G0 = 96

        # number of RDB blocks, conv layers, out channels
        self.D = 6
        self.C = 5
        self.G = 48

        self.num_heads = 8
        self.window_size = 8

        # Shallow feature extraction net
        self.lrelu = nn.ReLU(inplace=True)
        WARB = functools.partial(arch_util.WideActResBlock, nf=self.G0)
        self.SFE_conv = nn.Conv2d(4*4, self.G0, 3, 1, 1, bias=True)
        self.SFE_module = arch_util.make_layer(WARB, 5)
        self.SFE_conv_SAR = nn.Conv2d(2*4, self.G0, 3, 1, 1, bias=True)
        self.SFE_module_SAR = arch_util.make_layer(WARB, 5)

        # Redidual dense blocks and dense feature fusion
        self.RDBs = nn.ModuleList()
        for i in range(self.D):
            self.RDBs.append(
                RDB(growRate0=self.G0, growRate=self.G, nConvLayers=self.C, kSize=3, 
                    input_size=input_size,num_heads=self.num_heads, window_size=self.window_size,
                    shift_size=0 if (i % 2 == 0) else self.window_size // 2,)
            )

        # fusion
        self.ks_2d = 5

        self.SFAlign = nn.ModuleList() #Align
        for i in range(self.D):
            self.SFAlign.append(Align_Feature(self.G0))

        self.DF = nn.ModuleList() #dynamic filter
        for i in range(self.D):
            self.DF.append(DFG(self.G0*2, self.ks_2d))
        
        self.sar_fuse_1x1conv = nn.ModuleList() #gate for sar, aggregation
        for i in range(self.D):
            self.sar_fuse_1x1conv.append(nn.Conv2d(self.G0, self.G0, kernel_size=1))

        self.opt_distribute_1x1conv = nn.ModuleList() #gate for opt, distribution
        for i in range(self.D):
            self.opt_distribute_1x1conv.append(nn.Conv2d(self.G0, self.G0, kernel_size=1))

        # Global Feature Fusion
        self.GFF = nn.Sequential(*[
            nn.Conv2d(self.D * self.G0, self.G0, 1, padding=0, stride=1),
            nn.Conv2d(self.G0, self.G0, 3, padding=1, stride=1)
        ])

        # Up-sampling net
        self.UPNet = nn.Sequential(*[
            nn.Conv2d(self.G0, 256, 3, padding=1, stride=1),
            nn.PixelShuffle(2),
            nn.Conv2d(64, 4, 3, padding=1, stride=1)
        ])

    def forward(self, cloudy_data, SAR):
        B_shuffle = pixel_reshuffle(cloudy_data, 2)
        f__1 = self.SFE_conv(B_shuffle)
        x = self.SFE_module(self.lrelu(f__1))

        B_shuffle_SAR = pixel_reshuffle(SAR, 2)
        f__1__SAR = self.SFE_conv_SAR(B_shuffle_SAR)
        x_SAR = self.SFE_module_SAR(self.lrelu(f__1__SAR))

        RDBs_out = []
        for i in range(self.D):

            x_SAR = self.SFAlign[i](x_SAR, x)

            [x, x_SAR] = self.RDBs[i]([x,x_SAR])

            x, x_SAR = self.fuse(x, x_SAR, i)

            RDBs_out.append(x)
        
        x = self.GFF(torch.cat(RDBs_out, 1))
        x += f__1

        pred_CloudFree_data = self.UPNet(x) + cloudy_data
        return pred_CloudFree_data

    def fuse(self, OPT, SAR, i):
        
        OPT_m = OPT
        SAR_m = SAR

        kernel_sar = self.DF[i](OPT_m, SAR_m)
        SAR_m = kernel2d_conv(SAR_m, kernel_sar, self.ks_2d)

        sar_s = self.sar_fuse_1x1conv[i](SAR_m - OPT_m)
        sar_fuse_gate = torch.sigmoid(sar_s) # compute the gate 

        new_OPT = OPT + (SAR_m - OPT_m) * sar_fuse_gate # update the optical

        new_OPT_m = new_OPT
        
        opt_s = self.opt_distribute_1x1conv[i](new_OPT_m - SAR_m) 
        opt_distribute_gate = torch.sigmoid(opt_s) # compute the gate 

        new_SAR = SAR + (new_OPT_m - SAR_m) * opt_distribute_gate # update the SAR

        return new_OPT, new_SAR


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument('--crop_size', type=int, default=160)
    opts = parser.parse_args()

    model = RDN_residual_CR(opts.crop_size).cuda()

    planet_cloudy = torch.rand(1, 4, 160, 160).cuda()
    planet_cloudfree = torch.rand(1, 4, 160, 160).cuda()
    s1_sar = torch.rand(1, 2, 160, 160).cuda()

    pred_planet_cloudfree = model(planet_cloudy, s1_sar)

    print(pred_planet_cloudfree.shape)

