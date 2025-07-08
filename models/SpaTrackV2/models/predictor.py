# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F

from tqdm import tqdm
from models.SpaTrackV2.models.SpaTrack import SpaTrack2
from typing import Literal
import numpy as np
from pathlib import Path
from typing import Union, Optional
import cv2
import os
import decord

class Predictor(torch.nn.Module):
    def __init__(self, args=None):
        super().__init__()
        self.args = args
        self.spatrack = SpaTrack2(loggers=[None, None, None], **args)
        self.S_wind = args.Track_cfg.s_wind
        self.overlap = args.Track_cfg.overlap

    def to(self, device: Union[str, torch.device]):
        self.spatrack.to(device)
        if self.spatrack.base_model is not None:
            self.spatrack.base_model.to(device)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, Path],
        *,
        force_download: bool = False,
        cache_dir: Optional[str] = None,
        device: Optional[Union[str, torch.device]] = None,
        model_cfg: Optional[dict] = None,
        **kwargs,
    ) -> "SpaTrack2":
        """
        Load a pretrained model from a local file or a remote repository.

        Args:
            pretrained_model_name_or_path (str or Path):
                - Path to a local model file (e.g., `./model.pth`).
                - HuggingFace Hub model ID (e.g., `username/model-name`).
            force_download (bool, optional):
                Whether to force re-download even if cached. Default: False.
            cache_dir (str, optional):
                Custom cache directory. Default: None (use default cache).
            device (str or torch.device, optional):
                Target device (e.g., "cuda", "cpu"). Default: None (keep original).
            **kwargs:
                Additional config overrides.

        Returns:
            SpaTrack2: Loaded pretrained model.
        """
        # (1) check the path is local or remote
        if isinstance(pretrained_model_name_or_path, Path):
            model_path = str(pretrained_model_name_or_path)
        else:
            model_path = pretrained_model_name_or_path
        # (2) if the path is remote, download it
        if not os.path.exists(model_path):
            raise NotImplementedError("Remote download not implemented yet. Use a local path.")
        # (3) load the model weights

        state_dict = torch.load(model_path, map_location="cpu")
        # (4) initialize the model (can load config.json if exists)
        config_path = os.path.join(os.path.dirname(model_path), "config.json")
        config = {}
        if os.path.exists(config_path):
            import json
            with open(config_path, "r") as f:
                config.update(json.load(f))
        config.update(kwargs)  # allow override the config
        if model_cfg is not None:
            config = model_cfg
        model = cls(config)
        if "model" in state_dict:
            model.spatrack.load_state_dict(state_dict["model"], strict=False)
        else:
            model.spatrack.load_state_dict(state_dict, strict=False)
        # (5) device management
        if device is not None:
            model.to(device)

        return model

    def forward(self, video: str|torch.Tensor|np.ndarray,
                 depth: str|torch.Tensor|np.ndarray=None,
                 unc_metric: str|torch.Tensor|np.ndarray=None,
                 intrs: str|torch.Tensor|np.ndarray=None,
                 extrs: str|torch.Tensor|np.ndarray=None,
                 queries=None, queries_3d=None, iters_track=4,
                 full_point=False, fps=30, track2d_gt=None, 
                 fixed_cam=False, query_no_BA=False, stage=0,
                 support_frame=0, replace_ratio=0.6):
        """
        video: this could be a path to a video, a tensor of shape (T, C, H, W) or a numpy array of shape (T, C, H, W)
        queries: (B, N, 2)
        """

        if isinstance(video, str):
            video = decord.VideoReader(video)
            video = video[::fps].asnumpy()  # Convert to numpy array
            video = np.array(video)  # Ensure numpy array
            video = torch.from_numpy(video).permute(0, 3, 1, 2).float()
        elif isinstance(video, np.ndarray):
            video = torch.from_numpy(video).float()

        if isinstance(depth, np.ndarray):
            depth = torch.from_numpy(depth).float()
        if isinstance(intrs, np.ndarray):
            intrs = torch.from_numpy(intrs).float()
        if isinstance(extrs, np.ndarray):
            extrs = torch.from_numpy(extrs).float()
        if isinstance(unc_metric, np.ndarray):
            unc_metric = torch.from_numpy(unc_metric).float()

        T_, C, H, W = video.shape
        step_slide = self.S_wind - self.overlap
        if T_ > self.S_wind:
            
            num_windows = (T_ - self.S_wind + step_slide) // step_slide 
            T = num_windows * step_slide + self.S_wind
            pad_len = T - T_

            video = torch.cat([video, video[-1:].repeat(T-video.shape[0], 1, 1, 1)], dim=0)
            if depth is not None:
                depth = torch.cat([depth, depth[-1:].repeat(T-depth.shape[0], 1, 1)], dim=0)
            if intrs is not None:
                intrs = torch.cat([intrs, intrs[-1:].repeat(T-intrs.shape[0], 1, 1)], dim=0)
            if extrs is not None:
                extrs = torch.cat([extrs, extrs[-1:].repeat(T-extrs.shape[0], 1, 1)], dim=0)
            if unc_metric is not None:
                unc_metric = torch.cat([unc_metric, unc_metric[-1:].repeat(T-unc_metric.shape[0], 1, 1)], dim=0)
        with torch.no_grad():
            ret = self.spatrack.forward_stream(video, queries, T_org=T_,
                                                depth=depth, intrs=intrs, unc_metric_in=unc_metric, extrs=extrs, queries_3d=queries_3d,
                                                window_len=self.S_wind, overlap_len=self.overlap, track2d_gt=track2d_gt, full_point=full_point, iters_track=iters_track,
                                                fixed_cam=fixed_cam, query_no_BA=query_no_BA, stage=stage, support_frame=support_frame, replace_ratio=replace_ratio) + (video[:T_],)
            
        
        return ret

    
        
        