import os
import glob
import cv2
import numpy as np
import argparse

from skimage.filters.rank import entropy
from skimage.morphology import disk
neighborhood = disk(5)

toviz = lambda x : (255 * (x - x.min()) / (x.max() - x.min())).astype(np.uint8)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type = str, required = True)
    args = parser.parse_args()

    in_path = os.path.join(folder, 'images')
    assert os.path.exists(in_path)
    out_path = os.path.join(folder, 'entropy')
    os.makedirs(out_path, exist_ok = True)
    img_paths = sorted(glob.glob(os.path.join(in_path, '*.jpg')))

    for i, img_path in enumerate(img_paths):
        print('Computing entropy for image:', i)
        img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        entr = entropy(img_gray, neighborhood)
        entr_norm = entr / entr.sum()
        np.savez(os.path.join(out_path, 'entropy_{}.npz'.format(i)), entr_norm)
        cv2.imwrite(os.path.join(out_path, 'entropy_{}.jpg'.format(i)), toviz(entr))