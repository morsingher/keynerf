import argparse
import json
import time
import os
import numpy as np
import cv2
from ortools.linear_solver import pywraplp

from utils.camera_visualizer import generate_gif
from utils.misc import project_pts_to_cam, check_symmetric, compute_view_matrix
from utils.blender import get_blender_poses, get_blender_calibration, get_blender_grid
from utils.colmap import get_colmap_poses, get_colmap_calibration, get_colmap_grid

if __name__ == '__main__':

    # Read and parse input parameters

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type = str, required = True)
    parser.add_argument("--dataset", type = str, default = "blender")
    parser.add_argument("--grid_res", type = int, default = 16)
    parser.add_argument("--scale", type = float, default = 1.0)
    parser.add_argument("--gif", action = "store_true")
    args = parser.parse_args()

    assert args.dataset in ["blender", "colmap"]

    # Load input data

    if args.dataset == "blender":
        file_path = os.path.join(args.input_dir, 'transforms_train.json')
        with open(file_path, 'r') as f:
            meta = json.load(f)
        poses = get_blender_poses(meta)
        K = get_blender_calibration(meta, args.input_dir)
        idxs_train = [i for i in range(poses.shape[0])]
    else:
        poses, idxs_train = get_colmap_poses(args.input_dir)
        K = get_colmap_calibration(args.input_dir)

    num_poses = poses.shape[0]
    print(f'Read input poses with shape {poses.shape} and calibration with shape {K.shape}')  

    print('===============================')
    print('STEP 1 - INTEGER OPTIMIZATION')
    print('===============================')

    # Generate a 3D uniform grid to approximate the scene 

    if args.dataset == "blender":
        pts, grid_min, grid_max = get_blender_grid(args.grid_res)
    else:
        pts, grid_min, grid_max = get_colmap_grid(args.input_dir, args.grid_res)

    pts *= args.scale
    num_pts = pts.shape[0]
    print(f'Generated uniform grid with {num_pts} points and bounds ({grid_min}, {grid_max})')

    begin = time.time()

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
        num_opt = len(selected_cams)
    else:
        print('Failed to find a solution!')
        exit()

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

    final_path = os.path.join(args.input_dir, 'view_selection.txt')
    with open(final_path, 'w') as f:
        f.writelines(['{}\n'.format(idxs_train[idx]) for idx in selected_cams])
    print('===============================')
    print(f'Done! The whole algorithm completed in {time.time() - begin} s')

    if args.gif:
        print('Generating GIF for visualization...')
        gif_path = os.path.join(args.input_dir, 'gif')
        os.makedirs(gif_path, exist_ok = True)
        generate_gif(
            poses, num_opt, 
            selected_cams, gif_path,
            grid_min, grid_max, args.grid_res // 2
        )
        print('Done!')