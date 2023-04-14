import numpy as np
import cv2
import os

from utils.misc import flip_axes

GRID_MIN = [-1.6, -1.6, -1.6]
GRID_MAX = [1.6, 1.6, 1.6]

def get_blender_poses(meta):

    poses = []
    for frame in meta['frames']:
        poses.append(flip_axes(np.array(frame['transform_matrix'])))
    return np.array(poses)

def get_blender_calibration(meta, path):
    
    frame = meta['frames'][0]
    img_path = os.path.join(path, frame['file_path'] + '.png')
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    H, W = img.shape[:2]
    focal = .5 * W / np.tan(.5 * float(meta['camera_angle_x']))
    K = np.array([
        [focal, 0, 0.5 * W],
        [0, focal, 0.5 * H],
        [0, 0, 1]
    ])
    return K

def get_blender_grid(res):

    t_x = np.linspace(GRID_MIN[0], GRID_MAX[0], res)
    t_y = np.linspace(GRID_MIN[1], GRID_MAX[1], res)
    t_z = np.linspace(GRID_MIN[2], GRID_MAX[2], res)
    pts = np.stack(np.meshgrid(t_x, t_y, t_z), -1).astype(np.float32)
    return pts.reshape([-1, 3]), GRID_MIN, GRID_MAX