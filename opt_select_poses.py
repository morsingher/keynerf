import argparse
import json
import time
import os
import numpy as np
import cv2
from ortools.linear_solver import pywraplp

def flip_axes(pose):

    flip_yz = np.eye(4)
    flip_yz[1, 1] = -1
    flip_yz[2, 2] = -1
    return pose @ flip_yz

def read_poses(meta):

    poses = []
    for frame in meta['frames']:
        poses.append(flip_axes(np.array(frame['transform_matrix'])))
    return np.array(poses)

def read_calibration(meta, path):
    
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

def get_3d_grid(bbox_min, bbox_max, grid_res):

    t_x = np.linspace(bbox_min[0], bbox_max[0], grid_res)
    t_y = np.linspace(bbox_min[1], bbox_max[1], grid_res)
    t_z = np.linspace(bbox_min[2], bbox_max[2], grid_res)
    pts = np.stack(np.meshgrid(t_x, t_y, t_z), -1).astype(np.float32)
    return pts.reshape([-1, 3])

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

if __name__ == '__main__':

    # Read and parse input parameters

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type = str, required = True)
    parser.add_argument("--grid_min", type = float, default = -1.6)
    parser.add_argument("--grid_max", type = float, default = 1.6)
    parser.add_argument("--grid_res", type = int, default = 16)
    args = parser.parse_args()

    begin = time.time()

    # Load input data

    file_path = os.path.join(args.input_dir, 'transforms_train.json')
    with open(file_path, 'r') as f:
        meta = json.load(f)
    poses = read_poses(meta)
    K = read_calibration(meta, args.input_dir)
    num_poses = poses.shape[0]
    print(f'Read input poses with shape {poses.shape} and calibration with shape {K.shape}')

    print('===============================')
    print('STEP 1 - INTEGER OPTIMIZATION')
    print('===============================')

    # Generate a 3D uniform grid to approximate the scene 

    pts = get_3d_grid([args.grid_min] * 3, [args.grid_max] * 3, args.grid_res) # [num_pts, 3]
    num_pts = pts.shape[0]
    print(f'Generated uniform grid with {num_pts} points')

    # Build the matrix A: which point is seen by which camera?

    A = np.zeros((num_poses, num_pts)) # [num_poses, num_pts]
    for i, pose in enumerate(poses):
        u, v, mask = project_pts_to_cam(pts, pose, K)
        A[i, :] = mask
    print(f'Computed the visibility matrix between poses and grid with shape {A.shape}')

    # Setup the optimization solver and variables

    solver = pywraplp.Solver.CreateSolver('SCIP')
    assert solver, 'Failed to create ILP solver!'
    x = {}
    for i in range(num_poses):
        x[i] = solver.BoolVar(f'x[{i}]')
    print(f'Created {solver.NumVariables()} variables for the ILP solver')

    # Define the optimization objective

    objective = solver.Objective()
    for i in range(num_poses):
        objective.SetCoefficient(x[i], 1)
    objective.SetMinimization()
    print('Defined the optimization objective correctly')

    # Define the constraints: each point must be seen by at least one camera

    for i in range(num_pts):
        constraint = solver.RowConstraint(1, solver.infinity(), '')
        for j in range(num_poses):
            constraint.SetCoefficient(x[j], A[j,i])
    print(f'Defined {solver.NumConstraints()} constraints to satisfy')

    # Solve the ILP problem

    begin_ilp = time.time()
    status = solver.Solve()
    if status == pywraplp.Solver.OPTIMAL:
        selected_cams = []
        for i in range(num_poses):
            if x[i].solution_value() > 0:
                selected_cams.append(i)
        print(f'Selected {int(solver.Objective().Value())} cameras as initialization: {selected_cams}')
        print(f'Solution computed in {time.time() - begin_ilp} s')
    else:
        print('Failed to find a solution!')

    print('===============================')
    print('STEP 2 - GREEDY SCHEDULING')
    print('===============================')

    # Find the remaining cameras and compute the pairwise view matrix

    all_cams = list(range(len(poses)))
    remaining_cams = [cam for cam in all_cams if not cam in selected_cams]
    assert len(selected_cams) + len(remaining_cams) == len(poses)
    B = compute_view_matrix(poses)

    # Keep adding the most informative camera until the set is empty

    while remaining_cams:
        sub_matrix = B[remaining_cams][:, selected_cams]
        min_score = np.min(sub_matrix, axis = 1)
        idx = np.argmax(min_score)
        new_cam = remaining_cams[idx]
        selected_cams.append(new_cam)
        remaining_cams.remove(new_cam)
        print('Next camera is {} with score: {}'.format(new_cam, min_score[idx]))

    # Save the result

    with open(os.path.join(args.input_dir, 'view_selection.txt'), 'w') as f:
        f.writelines(['{}\n'.format(idx) for idx in selected_cams])
    print('===============================')
    print(f'Done! The whole algorithm completed in {time.time() - begin} s')