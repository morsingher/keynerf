#!/bin/bash

python3 select_poses.py --dataset colmap --input_dir ~/curriculum_nerf/data/co3d/ball_123_14363_28981
python3 select_poses.py --dataset colmap --input_dir ~/curriculum_nerf/data/co3d/cake_374_42274_84517
python3 select_poses.py --dataset colmap --input_dir ~/curriculum_nerf/data/co3d/hydrant_167_18184_34441
python3 select_poses.py --dataset colmap --input_dir ~/curriculum_nerf/data/co3d/pizza_586_87341_172687 --scale 0.95
python3 select_poses.py --dataset colmap --input_dir ~/curriculum_nerf/data/co3d/plant_247_26441_50907
python3 select_poses.py --dataset colmap --input_dir ~/curriculum_nerf/data/co3d/remote_195_20989_41543
python3 select_poses.py --dataset colmap --input_dir ~/curriculum_nerf/data/co3d/teddybear_34_1479_4753 --scale 0.85
python3 select_poses.py --dataset colmap --input_dir ~/curriculum_nerf/data/co3d/toaster_416_57389_110765