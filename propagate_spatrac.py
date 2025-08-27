import torch
import torch.nn.functional as F
from torchvision.io import read_image
from torchvision.utils import save_image
import numpy as np
from pathlib import Path
from argparse import ArgumentParser
from PIL import Image
import matplotlib.pyplot as plt
from itertools import count

def preprocess_image(img_tensor, mode="crop", target_size=518, keep_ratio=False):
    """
    Preprocess image tensor(s) to target size with crop or pad mode.
    Args:
        img_tensor (torch.Tensor): Image tensor of shape (C, H, W) or (T, C, H, W), values in [0, 1]
        mode (str): 'crop' or 'pad'
        target_size (int): Target size for width/height
    Returns:
        torch.Tensor: Preprocessed image tensor(s), same batch dim as input
    """
    if mode not in ["crop", "pad"]:
        raise ValueError("Mode must be either 'crop' or 'pad'")
    if img_tensor.dim() == 3:
        tensors = [img_tensor]
        squeeze = True
    elif img_tensor.dim() == 4:
        tensors = list(img_tensor)
        squeeze = False
    else:
        raise ValueError("Input tensor must be (C, H, W) or (T, C, H, W)")
    processed = []
    for img in tensors:
        C, H, W = img.shape
        if mode == "pad":
            if W >= H:
                new_W = target_size
                new_H = round(H * (new_W / W) / 14) * 14
            else:
                new_H = target_size
                new_W = round(W * (new_H / H) / 14) * 14
            out = torch.nn.functional.interpolate(img.unsqueeze(0), size=(new_H, new_W), mode="bicubic", align_corners=False).squeeze(0)
            h_padding = target_size - new_H
            w_padding = target_size - new_W
            pad_top = h_padding // 2
            pad_bottom = h_padding - pad_top
            pad_left = w_padding // 2
            pad_right = w_padding - pad_left
            if h_padding > 0 or w_padding > 0:
                out = torch.nn.functional.pad(
                    out, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=1.0
                )
        else:  # crop
            new_W = target_size
            new_H = round(H * (new_W / W) / 14) * 14
            out = torch.nn.functional.interpolate(img.unsqueeze(0), size=(new_H, new_W), mode="bicubic", align_corners=False).squeeze(0)
            if keep_ratio==False:
                if new_H > target_size:
                    start_y = (new_H - target_size) // 2
                    out = out[:, start_y : start_y + target_size, :]
        processed.append(out)
    result = torch.stack(processed)
    if squeeze:
        return result[0]
    return result

def main(args):
    root = Path(args.dataroot)
    if torch.cuda.is_available() and not args.cpu:
        device = torch.device('cuda')
        dtype = torch.bfloat16
    else:
        device = torch.device('cpu')
        dtype = torch.float32
    image_fns = list(sorted(root.glob('*.jpg')))
    images = [read_image(fn) for fn in image_fns]
    label_fns = list(sorted(Path(str(root).replace('JPEGImages', 'Annotations')).glob('*.png')))
    labels = [read_image(fn) for fn in label_fns]
    rec_4d = np.load(root/'STv2'/'result.npz')
    print('Loaded 4D Reconstruction from SpatialTrackerV2:')
    for k,v in rec_4d.items():
        print(f'{k}: {v.shape}')
    label = labels[0].to(device).float()
    print('label orig', label.shape)
    label = preprocess_image(label).round().to(torch.uint8).squeeze()
    print('label preproc', label.shape, label.min(), label.max())

    tracks = torch.as_tensor(rec_4d['coords']).to(device).to(dtype) # T, N, 3
    points = torch.as_tensor(rec_4d['point_map']).permute(0,2,3,1).to(dtype).to(device) # T, H, W, 3
    grid = torch.as_tensor(rec_4d['grid_pts']).round().long().squeeze().to(device)
    print('tracks', tracks.shape, tracks.min(), tracks.max())
    scene_bb = [points[...,i].max().item() - points[...,i].min().item() for i in range(points.size(-1))]
    print(f'Scene Size: {scene_bb}')
    print('points', points.shape, points.min(), points.max())
    print('grid', grid.shape, grid.min(), grid.max())
    gy, gx = grid[:,1], grid[:,0]
    print('gy', gy.shape, gy.min(), gy.max())
    print('gx', gx.shape, gx.min(), gx.max())

    points_nn = torch.zeros(points.shape[:3], device=device, dtype=torch.long)
    for b in range(points.size(0)):
        points_nn[b] = ((points[b, None, :, :, :] - tracks[b, :, None, None, :]) ** 2).sum(dim=-1).argmin(dim=0)
    print('points_nn', points_nn.shape, points_nn.min(), points_nn.max())
    print('points_nn distirbution:', torch.histc(points_nn, bins=10))
    track_labels = label[gy, gx]
    print('track_labels', track_labels.shape, track_labels.min(), track_labels.max())

    point_labels = track_labels[points_nn].round().to(torch.uint8)
    print('point_labels', point_labels.shape, point_labels.min(), point_labels.max())

    out_dir = root/'spatrac_pred'
    import shutil
    shutil.rmtree(out_dir)
    out_dir.mkdir(exist_ok=True)
    fig, ax = plt.subplots(1,3, dpi=300, tight_layout=True)
    for x in ax: x.set_axis_off()
    ax[0].set_title('Image')
    ax[1].set_title('Prediction')
    ax[2].set_title('Target')
    im1 = ax[0].imshow(preprocess_image(images[0]).permute(1,2,0))
    im2 = ax[1].imshow(point_labels[0].cpu())
    im3 = ax[2].imshow(preprocess_image(labels[0]).round().to(torch.uint8).squeeze())
    for i, im, pred, lab in zip(count(), images, point_labels, labels):
        im1.set_data(preprocess_image(im).permute(1,2,0))
        im2.set_data(pred.cpu())
        im3.set_data(preprocess_image(lab).round().to(torch.uint8).squeeze())
        fig.savefig(out_dir/f'{i:05d}.png')
        plt.close(fig)

if __name__ == '__main__':
    parser = ArgumentParser('Propagat4r')
    parser.add_argument('dataroot', type=Path, help='Path to the video (frame jpgs) and SpatialTracker results')
    parser.add_argument('--cpu', action='store_true', help='Only use CPU')
    args = parser.parse_args()
    main(args)
