import argparse
import json
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import cv2
import imageio

class CameraPoseVisualizer:
    
    def __init__(self, xlim, ylim, zlim):

        self.fig = plt.figure(figsize = (18, 7))
        self.ax = self.fig.gca(projection = '3d')
        self.ax.set_aspect('auto')
        self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)
        self.ax.set_zlim(zlim)
        self.ax.set_xlabel('x')
        self.ax.set_ylabel('y')
        self.ax.set_zlabel('z')

    def add_pose(self, extrinsic, color = 'r', f_scale = 5, asp_ratio = 0.3):

        f_scale = -1. * f_scale
        vertex_std = np.array([[0, 0, 0, 1],
                               [f_scale * asp_ratio, -f_scale * asp_ratio, f_scale, 1],
                               [f_scale * asp_ratio, f_scale * asp_ratio, f_scale, 1],
                               [-f_scale * asp_ratio, f_scale * asp_ratio, f_scale, 1],
                               [-f_scale * asp_ratio, -f_scale * asp_ratio, f_scale, 1]])
        vertex_t = vertex_std @ extrinsic.T
        meshes = [[vertex_t[0, :-1], vertex_t[1][:-1], vertex_t[2, :-1]],
                  [vertex_t[0, :-1], vertex_t[2, :-1], vertex_t[3, :-1]],
                  [vertex_t[0, :-1], vertex_t[3, :-1], vertex_t[4, :-1]],
                  [vertex_t[0, :-1], vertex_t[4, :-1], vertex_t[1, :-1]],
                  [vertex_t[1, :-1], vertex_t[2, :-1], vertex_t[3, :-1], vertex_t[4, :-1]]]
        self.ax.add_collection3d(Poly3DCollection(
            meshes, facecolors = color, linewidths = 0.3, edgecolors = color, alpha = 0.35)
        )

    def add_3d_bbox(self, bmin, bmax, N):

        if isinstance(bmin, np.ndarray):
            t_x = np.linspace(bmin[0], bmax[0], N)
            t_y = np.linspace(bmin[1], bmax[1], N)
            t_z = np.linspace(bmin[2], bmax[2], N)
            xs, ys, zs = np.meshgrid(t_x, t_y, t_z)
        else:
            t = np.linspace(bmin, bmax, N)
            xs, ys, zs = np.meshgrid(t, t, t)
        self.ax.scatter(xs, ys, zs, c = 'k', marker = '.')

    def show(self):
        plt.show()

    def save(self, path):
        plt.savefig(path)

def flip_axes(pose):

    flip_yz = np.eye(4)
    flip_yz[1, 1] = -1
    flip_yz[2, 2] = -1
    return pose @ flip_yz

def visualize_poses(poses, grid_min, grid_max, grid_res, path = None):

    t_arr = np.array([pose[:3,-1] for pose in poses])
    mins, maxs = t_arr.min(axis = 0), t_arr.max(axis = 0)
    
    visualizer = CameraPoseVisualizer(
        [mins[0] - 1, maxs[0] + 1], 
        [mins[1] - 1, maxs[1] + 1], 
        [mins[2] - 1, maxs[2] + 1]
    )
    visualizer.add_3d_cube(grid_min, grid_max, grid_res)
    for pose in poses:
        visualizer.add_pose(flip_axes(pose), 'c', 1)

    if path:
        visualizer.save(path)
    else:
        visualizer.show()

def generate_gif(poses, num_opt, path_in, path_out, grid_min, grid_max, grid_res):
    
    with open(path_in, 'r') as f:
        idxs = [int(line.strip()) for line in f]

    init_path = os.path.join(path_out, 'init_poses.png')
    visualize_poses(poses[idxs[:num_opt]], grid_min, grid_max, grid_res, init_path)
    frames_tmp = [cv2.imread(init_path)]

    t_arr = np.array([pose[:3,-1] for pose in poses])
    mins, maxs = t_arr.min(axis = 0), t_arr.max(axis = 0)

    for idx in range(num_opt, len(idxs)):
        visualizer = CameraPoseVisualizer(
            [mins[0] - 1, maxs[0] + 1], 
            [mins[1] - 1, maxs[1] + 1], 
            [mins[2] - 1, maxs[2] + 1]
        )
        visualizer.add_3d_cube(grid_min, grid_max, grid_res)
        cur_poses = poses[idxs[:idx + 1]]
        for pose in cur_poses[:-1]:
            visualizer.add_pose(flip_axes(pose), 'c', 1)
        visualizer.add_pose(flip_axes(cur_poses[-1]), 'r', 1)
        visualizer.save(os.path.join(path_out, f'frame_{idx}.png'))
        frames_tmp.append(cv2.imread(os.path.join(path_out, f'frame_{idx}.png')))

    frames = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames_tmp]
    imageio.mimsave(os.path.join(path_out, 'view_selection.gif'), frames, duration = 0.1)

