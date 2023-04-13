import os
import numpy as np
import imageio
import random
import cv2

from colmap.read_sparse_recon import read_model

def get_colmap_pose(img):

    w2c = np.eye(4)
    w2c[:3, :3] = img.qvec2rotmat()
    w2c[:3, 3] = img.tvec
    c2w = np.linalg.inv(w2c)

    return c2w.astype(np.float32)

def get_colmap_keypoints(input_dir):

    img_dir = os.path.join(input_dir, 'images')
    mask_dir = os.path.join(input_dir, 'masks')
    sparse_dir = os.path.join(input_dir, 'sparse/0')
    assert os.path.exists(img_dir) and os.path.exists(sparse_dir) and os.path.exists(mask_dir)

    _, images, pts = read_model(sparse_dir, '.bin')
    keypoints = []

    for id_im in range(1, len(images) + 1):
        
        mask_path = os.path.join(mask_dir, os.path.splitext(images[id_im].name)[0] + '.png')
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        for id_kp in range(len(images[id_im].xys)):

            id_3D = images[id_im].point3D_ids[id_kp]
            point2D = images[id_im].xys[id_kp] # Pixel coordinates
            u, v = int(point2D[1]), int(point2D[0])
            if mask[u, v] > 200 and id_3D >= 0:
                point3D = pts[id_3D].xyz # Corresponding 3D coordinates
                keypoints.append(point3D)

    return np.unique(np.array(keypoints), axis = 0)

def get_colmap_poses(input_dir):

    img_dir = os.path.join(input_dir, 'images')
    sparse_dir = os.path.join(input_dir, 'sparse/0')
    assert os.path.exists(img_dir) and os.path.exists(sparse_dir)

    cameras, images_col, points3D = read_model(sparse_dir, '.bin')
    num_images = len(images_col)

    # Images and poses

    poses = np.zeros((num_images, 4, 4))
    filenames = [None] * num_images
    for img in images_col.values():
        path = os.path.join(img_dir, img.name)
        poses[img.id - 1] = get_colmap_pose(img)
        filenames[img.id - 1] = path

    # Fix sorting problem of COLMAP

    sort_idx = [i[0] for i in sorted(enumerate(filenames), key = lambda x : x[1])]
    poses = poses[sort_idx]
    idxs_train = [i for i in range(num_images)][::2]

    return poses[idxs_train]

