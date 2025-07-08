import torch
import torch.nn as nn
import torch.nn.functional as F
from models.monoD.depth_anything_v2.dinov2_layers.patch_embed import PatchEmbed
from models.SpaTrackV2.models.depth_refiner.backbone import mit_b3
from models.SpaTrackV2.models.depth_refiner.stablizer import Stabilization_Network_Cross_Attention
from einops import rearrange
class TrackStablizer(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.backbone = mit_b3()

        old_conv = self.backbone.patch_embed1.proj
        new_conv = nn.Conv2d(old_conv.in_channels + 4, old_conv.out_channels, kernel_size=old_conv.kernel_size, stride=old_conv.stride, padding=old_conv.padding)

        new_conv.weight[:, :3, :, :].data.copy_(old_conv.weight.clone())
        self.backbone.patch_embed1.proj = new_conv

        self.Track_Stabilizer = Stabilization_Network_Cross_Attention(in_channels=[64, 128, 320, 512],
                    in_index=[0, 1, 2, 3],
                    feature_strides=[4, 8, 16, 32],
                    channels=128,
                    dropout_ratio=0.1,
                    num_classes=1,
                    align_corners=False,
                    decoder_params=dict(embed_dim=256, depths=4),
                    num_clips=16,
                    norm_cfg = dict(type='SyncBN', requires_grad=True))
                    
        self.edge_conv = nn.Sequential(nn.Conv2d(in_channels=4, out_channels=64, kernel_size=3, padding=1, stride=1, bias=True),\
                                  nn.ReLU(inplace=True))
        self.edge_conv1 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, stride=2, bias=True),\
                                  nn.ReLU(inplace=True))
        self.success = False
        self.x = None

    def buffer_forward(self, inputs, num_clips=16):
        """
            buffer forward for getting the pointmap and image features
        """
        B, T, C, H, W = inputs.shape
        self.x = self.backbone(inputs)
        scale, shift = self.Track_Stabilizer.buffer_forward(self.x, num_clips=num_clips)
        self.success = True
        return scale, shift

    def forward(self, inputs, tracks, tracks_uvd, num_clips=16, imgs=None, vis_track=None):
        
        """
        Args:
            inputs: [B, T, C, H, W], RGB + PointMap + Mask
            tracks: [B, T, N, 4], 3D tracks in camera coordinate + visibility
            num_clips: int, number of clips to use
        """
        B, T, C, H, W = inputs.shape
        edge_feat = self.edge_conv(inputs.view(B*T,4,H,W))
        edge_feat1 = self.edge_conv1(edge_feat)

        if not self.success:
            scale, shift = self.Track_Stabilizer.buffer_forward(self.x,num_clips=num_clips)
            self.success = True
            update = self.Track_Stabilizer(self.x,edge_feat,edge_feat1,tracks,tracks_uvd,num_clips=num_clips, imgs=imgs, vis_track=vis_track)
        else:
            update = self.Track_Stabilizer(self.x,edge_feat,edge_feat1,tracks,tracks_uvd,num_clips=num_clips, imgs=imgs, vis_track=vis_track)

        return update
    
    def reset_success(self):
        self.success = False
        self.x = None
        self.Track_Stabilizer.reset_success()


if __name__ == "__main__":
    # Create test input tensors
    batch_size = 1
    seq_len = 16
    channels = 7  # 3 for RGB + 3 for PointMap + 1 for Mask
    height = 384
    width = 512
    
    # Create random input tensor with shape [B, T, C, H, W]
    inputs = torch.randn(batch_size, seq_len, channels, height, width)
    
    # Create random tracks
    tracks = torch.randn(batch_size, seq_len, 1024, 4)

    # Create random test images
    test_imgs = torch.randn(batch_size, seq_len, 3, height, width)
    
    # Initialize model and move to GPU
    model = TrackStablizer().cuda()
    
    # Move inputs to GPU and run forward pass
    inputs = inputs.cuda()
    tracks = tracks.cuda()
    outputs = model.buffer_forward(inputs, num_clips=seq_len)
    import time
    start_time = time.time()
    outputs = model(inputs, tracks, num_clips=seq_len)
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")
    import pdb; pdb.set_trace()
    # # Print shapes for verification
    # print(f"Input shape: {inputs.shape}")
    # print(f"Output shape: {outputs.shape}")
    
    # # Basic tests
    # assert outputs.shape[0] == batch_size, "Batch size mismatch"
    # assert len(outputs.shape) == 4, "Output should be 4D: [B,C,H,W]"
    # assert torch.all(outputs >= 0), "Output should be non-negative after ReLU"
    
    # print("All tests passed!")

