import torch

# def project_backward(orig_patch, dir_patch, depth_patch, pose_batch):

#     pose_batch = pose_batch.repeat(1, 64, 1, 1).reshape(256 * 64, 4, 3)

#     ones = torch.tensor([[0], [0], [0], [1]], dtype=pose_batch.dtype, device=pose_batch.device).repeat(256 * 64, 1, 1)
#     pose_batch = torch.cat((pose_batch, ones), dim=2) 
#     determinants = torch.det(pose_batch[:, :3, :3])

#     dir_patch = dir_patch / torch.norm(dir_patch, dim=1, keepdim=True)

#     xyz_patch = orig_patch + dir_patch * depth_patch[..., None]

#     # Reshape xyz_patch to match the batch size
#     xyz_patch = xyz_patch.reshape(256 * 64, 3)

#     # Apply the inverse transformation
#     pose_inv = torch.inverse(pose_batch)
#     determinants_inv = torch.det(pose_inv[:, :3, :3])
#     xyz_patch_homogeneous = torch.cat((xyz_patch, torch.ones(xyz_patch.shape[0], 1, device=xyz_patch.device)), dim=1)
#     xyz_patch_transformed = torch.bmm(pose_inv, xyz_patch_homogeneous.unsqueeze(2)).squeeze(2)

#     # Extract the original coordinates
#     test = determinants[:,None]
#     xyz_patch_original = xyz_patch_transformed[:, :3] * determinants[:,None]
    
#     return xyz_patch_original

# def reproject(pose_batch_new, xyz_patch_world, Ks):
#     Ks = torch.tensor(Ks, device=pose_batch_new.device, dtype=xyz_patch_world.dtype)
#     image_height = 2 * Ks[0, -1]
#     image_width = 2 * Ks[1, -1]
#     Ks[0, :] /= image_height
#     Ks[1, :] /= image_width
#     pose_batch_new = pose_batch_new.repeat(1, 64, 1, 1).reshape(256 * 64, 4, 3)

#     ones = torch.tensor([[0], [0], [0], [1]], dtype=pose_batch_new.dtype, device=pose_batch_new.device).repeat(256 * 64, 1, 1)
#     pose_batch_new = torch.cat((pose_batch_new, ones), dim=2)

#     xyz_patch_world_homogeneous = torch.cat((xyz_patch_world, torch.ones(xyz_patch_world.shape[0], 1, device=xyz_patch_world.device)), dim=1)
#     xyz_patch_world_transformed = torch.bmm(pose_batch_new, xyz_patch_world_homogeneous.unsqueeze(2)).squeeze(2)
#     xyz_patch_new = xyz_patch_world_transformed[:, :3]
#     xyz_patch_new = xyz_patch_new @ Ks.transpose(0, 1)
#     orig_patch_new = xyz_patch_new / (-xyz_patch_new[:, -1:])
    
#     return orig_patch_new

# # def reproject_error_render(orig_patch_new):

# #     return orig_patch_new

# def reprojection_coord(orig_patch, dir_patch, viewdir_patch, rgb_patch, depth_patch, pose_batch, pose_batch_new, Ks):
#     xyz_patch_world = project_backward(orig_patch, dir_patch, depth_patch, pose_batch)
#     orig_patch_new = reproject(pose_batch_new, xyz_patch_world, Ks)
#     # rgb_patch_new = reproject_error_render(orig_patch_new)
#     # reprojection_error = torch.norm(rgb_patch - rgb_patch_new, dim=1)

#     orig_patch_new[:, 0] = ((orig_patch_new[:, 0] + 1) / 2 * 1008).round()
#     orig_patch_new[:, 1] = ((orig_patch_new[:, 1] + 1) / 2 * 768).round()

#     return orig_patch_new

# def reprojection_rgb(rgb, orig_patch_new, idx_img):
#     idx_img = torch.tensor(idx_img).repeat(1, 64).reshape(256 * 64)

#     rgb_values = rgb[idx_img, orig_patch_new[:, 0].int(), orig_patch_new[:, 1].int(), :]

#     return rgb_values

def reprojection_error(orig_patch, dir_patch, viewdir_patch, rgb_patch, depth_patch, pose_batch, pose_batch_new, Ks, rgb, idx_img):
    batch_size = [orig_patch.shape[0], orig_patch.shape[1]]
    pose_batch = pose_batch.repeat(1, 64, 1, 1).reshape(256 * 64, 4, 3)
    ones = torch.tensor([[0], [0], [0], [1]], dtype=pose_batch.dtype, device=pose_batch.device).repeat(256 * 64, 1, 1)
    pose_batch = torch.cat((pose_batch, ones), dim=2)
    determinants = torch.det(pose_batch[:, :3, :3])
    dir_patch = dir_patch / torch.norm(dir_patch, dim=1, keepdim=True)
    xyz_patch = orig_patch + dir_patch * depth_patch[..., None]
    xyz_patch = xyz_patch.reshape(256 * 64, 3)
    pose_inv = torch.inverse(pose_batch)
    determinants_inv = torch.det(pose_inv[:, :3, :3])
    xyz_patch_homogeneous = torch.cat((xyz_patch, torch.ones(xyz_patch.shape[0], 1, device=xyz_patch.device)), dim=1)
    xyz_patch_transformed = torch.bmm(pose_inv, xyz_patch_homogeneous.unsqueeze(2)).squeeze(2)
    xyz_patch_world = xyz_patch_transformed[:, :3] * determinants[:, None]

    Ks = torch.tensor(Ks, device=pose_batch_new.device, dtype=xyz_patch_world.dtype)
    image_height = 2 * Ks[0, -1]
    image_width = 2 * Ks[1, -1]
    Ks[0, :] /= image_height
    Ks[1, :] /= image_width
    pose_batch_new = pose_batch_new.repeat(1, 64, 1, 1).reshape(256 * 64, 4, 3)
    ones = torch.tensor([[0], [0], [0], [1]], dtype=pose_batch_new.dtype, device=pose_batch_new.device).repeat(256 * 64, 1, 1)
    pose_batch_new = torch.cat((pose_batch_new, ones), dim=2)
    xyz_patch_world_homogeneous = torch.cat((xyz_patch_world, torch.ones(xyz_patch_world.shape[0], 1, device=xyz_patch_world.device)), dim=1)
    xyz_patch_world_transformed = torch.bmm(pose_batch_new, xyz_patch_world_homogeneous.unsqueeze(2)).squeeze(2)
    xyz_patch_new = xyz_patch_world_transformed[:, :3]
    xyz_patch_new = xyz_patch_new @ Ks.transpose(0, 1)
    orig_patch_new = xyz_patch_new / (-xyz_patch_new[:, -1:])
        
    orig_patch_new[:, 0] = ((orig_patch_new[:, 0] + 1) / 2 * 1008).round()
    orig_patch_new[:, 1] = ((orig_patch_new[:, 1] + 1) / 2 * 768).round()

    idx_img = torch.tensor(idx_img).repeat(1, 64).reshape(256 * 64)
    new_rgb_patch = rgb[idx_img, orig_patch_new[:, 0].long(), orig_patch_new[:, 1].long(), :]
    # new_rgb_patch = new_rgb_patch.unsqueeze(0).repeat(4, 1, 1).reshape(4 * 256 * 64, 3)

    reprojection_error = torch.mean(torch.norm(rgb_patch - new_rgb_patch, dim=1))

    return reprojection_error
