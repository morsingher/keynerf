# KeyNeRF

Official implementation of [Informative Rays Selection for Few-Shot Neural Radiance Fields](https://arxiv.org/abs/2312.17561), to appear at VISAPP 2024. 

## Usage

This repository contains utilities to select a subset of training rays for NeRF, as explained in the paper. We assume to already work in a NeRF-based environment, such as [nerf-pytorch](https://github.com/yenchenlin/nerf-pytorch). Therefore, follow instructions there to download the data and setup the required libraries. 
Then, pixelwise entropies and the associated probability distributions per-image can be computed as:
```
python compute_entropy.py --input_dir <dataset>
```
In the same way, camera poses can be scheduled according to our proposed greedy algorithm with the following script:
```
python select_poses.py --input_dir <dataset>
```
The code above will generate auxiliary data that can be used for sampling rays and pixels in any NeRF codebases, in order to replace the usual `np.random.choice()` and speed-up training with a limited budget of computation.

## Citation

Please cite this paper with the following BibTeX:
```
@inproceedings{orsingher2024informative,
    author = {Marco Orsingher and Anthony Dell'Eva and Paolo Zani and Paolo Medici and Massimo Bertozzi},
    title = {Informative Rays Selection for Few-Shot Neural Radiance Fields},
    booktitle = {International Conference on Computer Vision Theory and Applications (VISAPP 2024)},
    year = {2024}
}
```