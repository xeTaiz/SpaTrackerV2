import torch

def project_to_camera(pts, extri, intri):
    """
    Project 3D points to camera image coordinates.
    
    Args:
        pts (torch.Tensor): 3D points of shape (*, 3) in world coordinates
        extri (torch.Tensor): Extrinsic matrix of shape (4, 4) - world to camera transform
        intri (torch.Tensor): Intrinsic matrix of shape (3, 3)
    
    Returns:
        torch.Tensor: Projected points of shape (*, 3) where [..., 0:2] are image coordinates (u, v) 
                     and [..., 2] is the camera-space depth (Z)
    """
    original_shape = pts.shape[:-1]
    pts_flat = pts.reshape(-1, 3)  # (N, 3)
    
    # Convert to homogeneous coordinates
    pts_homo = torch.cat([pts_flat, torch.ones(pts_flat.shape[0], 1, device=pts.device, dtype=pts.dtype)], dim=1)  # (N, 4)
    
    # Transform to camera coordinates
    pts_cam = (extri @ pts_homo.T).T  # (N, 4) -> (N, 3) after removing homogeneous coordinate
    pts_cam = pts_cam[:, :3]  # (N, 3) - [X_cam, Y_cam, Z_cam]
    
    # Perspective projection
    X, Y, Z = pts_cam[:, 0], pts_cam[:, 1], pts_cam[:, 2]
    fx, fy = intri[0, 0], intri[1, 1]
    cx, cy = intri[0, 2], intri[1, 2]
    
    u = fx * X / Z + cx
    v = fy * Y / Z + cy
    
    # Stack with depth
    projected = torch.stack([u, v, Z], dim=1)  # (N, 3)
    
    # Reshape back to original shape
    return projected.reshape(*original_shape, 3)


# Example usage and test
if __name__ == "__main__":
    # Test the function
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create some test points
    pts = torch.tensor([[0.1, 0.2, 1.0], [-0.1, -0.2, 1.5]], device=device, dtype=torch.float32)
    
    # Identity extrinsics (world = camera coordinates)
    extri = torch.eye(4, device=device, dtype=torch.float32)
    
    # Sample intrinsics
    intri = torch.tensor([[500.0, 0.0, 320.0],
                         [0.0, 500.0, 240.0], 
                         [0.0, 0.0, 1.0]], device=device, dtype=torch.float32)
    
    projected = project_to_camera(pts, extri, intri)
    print(f"Input points: {pts}")
    print(f"Projected (u, v, depth): {projected}")
    
    # Test with batch of points
    pts_batch = torch.randn(5, 10, 3, device=device)  # 5 batches of 10 points each
    projected_batch = project_to_camera(pts_batch, extri, intri)
    print(f"Batch input shape: {pts_batch.shape}")
    print(f"Batch output shape: {projected_batch.shape}")