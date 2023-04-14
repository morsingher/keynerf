import numpy as np

def flip_axes(pose):

    flip_yz = np.eye(4)
    flip_yz[1, 1] = -1
    flip_yz[2, 2] = -1
    return pose @ flip_yz

def project_pts_to_cam(pts, c2w, K):

    w2c = np.linalg.inv(c2w)
    R, t = w2c[:3,:3], w2c[:3,3]
    cam_pts = K @ (R @ pts.T + t[:, np.newaxis])

    depth = cam_pts[2,:]
    pix_pts = cam_pts[:2,:] / depth
    u, v = pix_pts[0,:], pix_pts[1,:]

    H, W = (2. * K[1,2]).astype(np.int32), (2. * K[0,2]).astype(np.int32)
    u_valid = (u > 0) & (u < H)
    v_valid = (v > 0) & (v < W)
    z_valid = (depth > 0)
    mask = u_valid & v_valid & z_valid

    return u, v, mask

def check_symmetric(mat, rtol = 1e-05, atol = 1e-08):
    return np.allclose(mat, mat.T, rtol = rtol, atol = atol)

def compute_view_matrix(poses):

    view_dirs = np.zeros((len(poses), 3))
    for i, pose in enumerate(poses):
        d = np.array([0.,0.,1.]) @ pose[:3, :3].T
        view_dirs[i, :] = d / np.linalg.norm(d)

    view_matrix = np.zeros((len(poses), len(poses)))
    for i in range(len(poses)):
        for j in range(i + 1, len(poses)):
            p = np.arccos(np.dot(view_dirs[i], view_dirs[j]))
            view_matrix[i, j] = p
            view_matrix[j, i] = p

    assert check_symmetric(view_matrix)
    return view_matrix