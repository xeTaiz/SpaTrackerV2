import numpy as np
import torch.nn as nn
import torch
# from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
from collections import OrderedDict
# from mmseg.ops import resize
from torch.nn.functional import interpolate as resize
# from builder import HEADS
from models.SpaTrackV2.models.depth_refiner.decode_head import BaseDecodeHead, BaseDecodeHead_clips, BaseDecodeHead_clips_flow
# from mmseg.models.utils import *
import attr
from IPython import embed
from models.SpaTrackV2.models.depth_refiner.stablilization_attention import BasicLayer3d3
import cv2
from models.SpaTrackV2.models.depth_refiner.network import *
import warnings
# from mmcv.utils import Registry, build_from_cfg
from torch import nn
from einops import rearrange
import torch.nn.functional as F
from models.SpaTrackV2.models.blocks import (
    AttnBlock, CrossAttnBlock, Mlp
)

class MLP(nn.Module):
    """
    Linear Embedding
    """
    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


def scatter_multiscale_fast(
    track2d: torch.Tensor,
    trackfeature: torch.Tensor,
    H: int,
    W: int,
    kernel_sizes = [1]
) -> torch.Tensor:
    """
    Scatter sparse track features onto a dense image grid with weighted multi-scale pooling to handle zero-value gaps.
    
    This function scatters sparse track features into a dense image grid and applies multi-scale average pooling 
    while excluding zero-value holes. The weight mask ensures that only valid feature regions contribute to the pooling,
    avoiding dilution by empty pixels.

    Args:
        track2d (torch.Tensor): Float tensor of shape (B, T, N, 2) containing (x, y) pixel coordinates
                                for each track point across batches, frames, and points.
        trackfeature (torch.Tensor): Float tensor of shape (B, T, N, C) with C-dimensional features 
                                    for each track point.
        H (int): Height of the target output image.
        W (int): Width of the target output image.
        kernel_sizes (List[int]): List of odd integers for average pooling kernel sizes. Default: [3, 5, 7].

    Returns:
        torch.Tensor: Multi-scale fused feature map of shape (B, T, C, H, W) with hole-resistant pooling.
    """
    B, T, N, C = trackfeature.shape
    device = trackfeature.device

    # 1. Flatten coordinates and filter valid points within image bounds
    coords_flat = track2d.round().long().reshape(-1, 2)  # (B*T*N, 2)
    x = coords_flat[:, 0]  # x coordinates
    y = coords_flat[:, 1]  # y coordinates
    feat_flat = trackfeature.reshape(-1, C)  # Flatten features

    valid_mask = (x >= 0) & (x < W) & (y >= 0) & (y < H)
    x = x[valid_mask]
    y = y[valid_mask]
    feat_flat = feat_flat[valid_mask]
    valid_count = x.shape[0]

    if valid_count == 0:
        return torch.zeros(B, T, C, H, W, device=device)  # Handle no-valid-point case

    # 2. Calculate linear indices and batch-frame indices for scattering
    lin_idx = y * W + x  # Linear index within a single frame (H*W range)
    
    # Generate batch-frame indices (e.g., 0~B*T-1 for each frame in batch)
    bt_idx_raw = (
        torch.arange(B * T, device=device)
        .view(B, T, 1)
        .expand(B, T, N)
        .reshape(-1)
    )
    bt_idx = bt_idx_raw[valid_mask]  # Indices for valid points across batch and frames

    # 3. Create accumulation buffers for features and weights
    total_space = B * T * H * W
    img_accum_flat = torch.zeros(total_space, C, device=device)    # Feature accumulator
    weight_accum_flat = torch.zeros(total_space, 1, device=device) # Weight accumulator (counts)

    # 4. Scatter features and weights into accumulation buffers
    idx_in_accum = bt_idx * (H * W) + lin_idx  # Global index: batch_frame * H*W + pixel_index
    
    # Add features to corresponding indices (index_add_ is efficient for sparse updates)
    img_accum_flat.index_add_(0, idx_in_accum, feat_flat)
    weight_accum_flat.index_add_(0, idx_in_accum, torch.ones((valid_count, 1), device=device))

    # 5. Normalize features by valid weights, keep zeros for invalid regions
    valid_mask_flat = weight_accum_flat > 0  # Binary mask for valid pixels
    img_accum_flat = img_accum_flat / (weight_accum_flat + 1e-6)  # Avoid division by zero
    img_accum_flat = img_accum_flat * valid_mask_flat.float()  # Mask out invalid regions

    # 6. Reshape to (B, T, C, H, W) for further processing
    img = (
        img_accum_flat.view(B, T, H, W, C)
        .permute(0, 1, 4, 2, 3)
        .contiguous()
    )  # Shape: (B, T, C, H, W)

    # 7. Multi-scale pooling with weight masking to exclude zero holes
    blurred_outputs = []
    for k in kernel_sizes:
        pad = k // 2
        img_bt = img.view(B*T, C, H, W)  # Flatten batch and time for pooling
        
        # Create weight mask for valid regions (1 where features exist, 0 otherwise)
        weight_mask = (
            weight_accum_flat.view(B, T, 1, H, W) > 0
        ).float().view(B*T, 1, H, W)  # Shape: (B*T, 1, H, W)

        # Calculate number of valid neighbors in each pooling window
        weight_sum = F.conv2d(
            weight_mask,
            torch.ones((1, 1, k, k), device=device),
            stride=1,
            padding=pad
        )  # Shape: (B*T, 1, H, W)

        # Sum features only in valid regions
        feat_sum = F.conv2d(
            img_bt * weight_mask,  # Mask out invalid regions before summing
            torch.ones((1, 1, k, k), device=device).expand(C, 1, k, k),
            stride=1,
            padding=pad,
            groups=C
        )  # Shape: (B*T, C, H, W)

        # Compute average only over valid neighbors
        feat_avg = feat_sum / (weight_sum + 1e-6)
        blurred_outputs.append(feat_avg)

    # 8. Fuse multi-scale results by averaging across kernel sizes
    fused = torch.stack(blurred_outputs).mean(dim=0)  # Average over kernel sizes
    return fused.view(B, T, C, H, W)  # Restore original shape

#@HEADS.register_module()
class Stabilization_Network_Cross_Attention(BaseDecodeHead_clips_flow):

    def __init__(self, feature_strides, **kwargs):
        super(Stabilization_Network_Cross_Attention, self).__init__(input_transform='multiple_select', **kwargs)
        self.training = False
        assert len(feature_strides) == len(self.in_channels)
        assert min(feature_strides) == feature_strides[0]
        self.feature_strides = feature_strides

        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels

        decoder_params = kwargs['decoder_params']
        embedding_dim = decoder_params['embed_dim']

        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)

        self.linear_fuse = nn.Sequential(nn.Conv2d(embedding_dim*4, embedding_dim, kernel_size=(1, 1), stride=(1, 1), bias=False),\
                                         nn.ReLU(inplace=True))

        self.proj_track = nn.Conv2d(100, 128, kernel_size=(1, 1), stride=(1, 1), bias=True)

        depths = decoder_params['depths']
        
        self.reg_tokens = nn.Parameter(torch.zeros(1, 2, embedding_dim))
        self.global_patch = nn.Conv2d(embedding_dim, embedding_dim, kernel_size=(8, 8), stride=(8, 8), bias=True)

        self.att_temporal = nn.ModuleList(
            [
                AttnBlock(embedding_dim, 8,
                          mlp_ratio=4, flash=True, ckpt_fwd=True)
                for _ in range(8)
            ]
        )
        self.att_spatial = nn.ModuleList(
            [
                AttnBlock(embedding_dim, 8,
                          mlp_ratio=4, flash=True, ckpt_fwd=True)
                for _ in range(8)
            ]
        )
        self.scale_shift_head = nn.Sequential(nn.Linear(embedding_dim, embedding_dim), nn.GELU(), nn.Linear(embedding_dim, 4))


        # Initialize reg tokens
        nn.init.trunc_normal_(self.reg_tokens, std=0.02)

        self.decoder_focal=BasicLayer3d3(dim=embedding_dim,
               input_resolution=(96,
                                 96),
               depth=depths,
               num_heads=8,
               window_size=7,
               mlp_ratio=4.,
               qkv_bias=True,
               qk_scale=None,
               drop=0.,
               attn_drop=0.,
               drop_path=0.,
               norm_layer=nn.LayerNorm,
               pool_method='fc',
               downsample=None,
               focal_level=2,
               focal_window=5,
               expand_size=3,
               expand_layer="all",
               use_conv_embed=False,
               use_shift=False,
               use_pre_norm=False,
               use_checkpoint=False,
               use_layerscale=False,
               layerscale_value=1e-4,
               focal_l_clips=[7,4,2],
               focal_kernel_clips=[7,5,3])

        self.ffm2 = FFM(inchannels= 256, midchannels= 256, outchannels = 128)
        self.ffm1 = FFM(inchannels= 128, midchannels= 128, outchannels = 64)
        self.ffm0 = FFM(inchannels= 64, midchannels= 64, outchannels = 32,upfactor=1)
        self.AO = AO(32, outchannels=3, upfactor=1)
        self._c2 = None
        self._c_further = None

    def buffer_forward(self, inputs, num_clips=None, imgs=None):#,infermode=1):
        
        # input: B T 7 H W  (7 means 3 rgb + 3 pointmap + 1 uncertainty)  normalized
        if self.training:
            assert self.num_clips==num_clips

        x = self._transform_inputs(inputs)  # len=4, 1/4,1/8,1/16,1/32
        c1, c2, c3, c4 = x

        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c4.shape
        batch_size = n // num_clips

        _c4 = self.linear_c4(c4).permute(0,2,1).reshape(n, -1, c4.shape[2], c4.shape[3])
        _c4 = resize(_c4, size=c1.size()[2:],mode='bilinear',align_corners=False)

        _c3 = self.linear_c3(c3).permute(0,2,1).reshape(n, -1, c3.shape[2], c3.shape[3])
        _c3 = resize(_c3, size=c1.size()[2:],mode='bilinear',align_corners=False)

        _c2 = self.linear_c2(c2).permute(0,2,1).reshape(n, -1, c2.shape[2], c2.shape[3])
        _c2 = resize(_c2, size=c1.size()[2:],mode='bilinear',align_corners=False)

        _c1 = self.linear_c1(c1).permute(0,2,1).reshape(n, -1, c1.shape[2], c1.shape[3])
        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))

        _, _, h, w=_c.shape
        _c_further=_c.reshape(batch_size, num_clips, -1, h, w)  #h2w2

        # Expand reg_tokens to match batch size
        reg_tokens = self.reg_tokens.expand(batch_size*num_clips, -1, -1)  # [B, 2, C]
        
        _c2=self.decoder_focal(_c_further, batch_size=batch_size, num_clips=num_clips, reg_tokens=reg_tokens)
        
        assert _c_further.shape==_c2.shape
        self._c2 = _c2
        self._c_further = _c_further
        
        # compute the scale and shift of the global patch
        global_patch = self.global_patch(_c2.view(batch_size*num_clips, -1, h, w)).view(batch_size*num_clips, _c2.shape[2], -1).permute(0,2,1)
        global_patch = torch.cat([global_patch, reg_tokens], dim=1)
        for i in range(8):
            global_patch = self.att_temporal[i](global_patch)
            global_patch = rearrange(global_patch, '(b t) n c -> (b n) t c', b=batch_size, t=num_clips, c=_c2.shape[2])
            global_patch = self.att_spatial[i](global_patch)
            global_patch = rearrange(global_patch, '(b n) t c -> (b t) n c', b=batch_size, t=num_clips, c=_c2.shape[2])

        reg_tokens = global_patch[:, -2:, :]
        s_ = self.scale_shift_head(reg_tokens)
        scale = 1 + s_[:, 0, :1].view(batch_size, num_clips, 1, 1, 1)
        shift = s_[:, 1, 1:].view(batch_size, num_clips, 3, 1, 1)
        shift[:,:,:2,...] = 0 
        return scale, shift

    def forward(self, inputs, edge_feat, edge_feat1, tracks, tracks_uvd, num_clips=None, imgs=None, vis_track=None):#,infermode=1):

        if self._c2 is None:
            scale, shift = self.buffer_forward(inputs,num_clips,imgs)

        B, T, N, _ = tracks.shape

        _c2 = self._c2
        _c_further = self._c_further
        
        # skip and head
        _c_further = rearrange(_c_further, 'b t c h w -> (b t) c h w', b=B, t=T)
        _c2 = rearrange(_c2, 'b t c h w -> (b t) c h w', b=B, t=T)
        
        outframe = self.ffm2(_c_further, _c2)

        tracks_uv = tracks_uvd[...,:2].clone()
        track_feature = scatter_multiscale_fast(tracks_uv/2, tracks, outframe.shape[-2], outframe.shape[-1], kernel_sizes=[1, 3, 5])
        # visualize track_feature as video
        # import cv2
        # import imageio
        # import os
        # BT, C, H, W = outframe.shape
        # track_feature_vis = track_feature.view(B, T, 3, H, W).float().detach().cpu().numpy()
        # track_feature_vis = track_feature_vis.transpose(0,1,3,4,2) 
        # track_feature_vis = (track_feature_vis - track_feature_vis.min()) / (track_feature_vis.max() - track_feature_vis.min() + 1e-6)
        # track_feature_vis = (track_feature_vis * 255).astype(np.uint8)
        # imgs =(imgs.detach() + 1) * 127.5
        # vis_track.visualize(video=imgs, tracks=tracks_uv, filename="test")
        # for b in range(B):
        #     frames = []
        #     for t in range(T):
        #         frame = track_feature_vis[b,t]
        #         frame = cv2.applyColorMap(frame[...,0], cv2.COLORMAP_JET)
        #         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #         frames.append(frame)
        #     # Save as gif
        #     imageio.mimsave(f'track_feature_b{b}.gif', frames, duration=0.1)
        # import pdb; pdb.set_trace()
        track_feature = rearrange(track_feature, 'b t c h w -> (b t) c h w')
        track_feature = self.proj_track(track_feature)
        outframe = self.ffm1(edge_feat1 + track_feature,outframe)
        outframe = self.ffm0(edge_feat,outframe)
        outframe = self.AO(outframe)
        
        return outframe
    
    def reset_success(self):
        self._c2 = None
        self._c_further = None
