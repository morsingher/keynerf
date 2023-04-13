import argparse
import numpy as np
import open3d as o3d
import os

from colmap.get_colmap_data import get_colmap_keypoints, get_colmap_poses
from visualize_poses import CameraPoseVisualizer

def flip_axes(pose):

    flip_yz = np.eye(4)
    flip_yz[1, 1] = -1
    flip_yz[2, 2] = -1
    return pose @ flip_yz

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type = str, required = True)
    parser.add_argument('--grid_res', type = int, default = 8)
    args = parser.parse_args()

    keypoints = get_colmap_keypoints(args.input_dir)
    print(keypoints.shape)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(keypoints)
    o3d.io.write_point_cloud(os.path.join(args.input_dir, 'kps.ply'), pcd)

    pcd_filt, _ = pcd.remove_statistical_outlier(nb_neighbors = 100, std_ratio = 2.0)
    o3d.io.write_point_cloud(os.path.join(args.input_dir, 'kps_filt.ply'), pcd_filt)

    xyz = np.array(pcd_filt.points)
    grid_min, grid_max = xyz.min(axis = 0), xyz.max(axis = 0)
    print(grid_min, grid_max)

    poses = get_colmap_poses(args.input_dir)
    print(poses.shape)

    t_arr = np.array([pose[:3,-1] for pose in poses])
    mins, maxs = t_arr.min(axis = 0), t_arr.max(axis = 0)
    
    visualizer = CameraPoseVisualizer(
        [mins[0] - 1, maxs[0] + 1], 
        [mins[1] - 1, maxs[1] + 1], 
        [mins[2] - 1, maxs[2] + 1]
    )
    visualizer.add_3d_bbox(grid_min, grid_max, args.grid_res)
    for pose in poses[::4]:
        visualizer.add_pose(flip_axes(pose), 'c', 1)

    visualizer.show()