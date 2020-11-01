import sys
sys.path.append('..')
import os
import json
import cv2
import numpy as np
from tqdm import tqdm
import util.fisheye as fis
import util.bezier as bez
from config.cfg import parse


def save_npz(prefix, image, lines, heatmap_size):
    image_size = (image.shape[1], image.shape[0])
    sx, sy = heatmap_size[0] / image_size[0], heatmap_size[1] / image_size[1]

    lines[:, :, 0] = np.clip(lines[:, :, 0] * sx, 0, heatmap_size[0] - 1e-4)
    lines[:, :, 1] = np.clip(lines[:, :, 1] * sy, 0, heatmap_size[1] - 1e-4)
    juncs = np.concatenate((lines[:, 0], lines[:, -1]))
    juncs = np.unique(juncs, axis=0)

    np.savez_compressed(
        f'{prefix}.npz',
        junc=juncs,
        line=lines
    )
    cv2.imwrite(f'{prefix}.png', image)


def json2npz(src_path, dst_path, cfg, order, plot=False):
    split = 'test'
    os.makedirs(dst_path, exist_ok=True)

    json_file = os.path.join(src_path, f'{split}.json')
    with open(json_file, 'r') as f:
        dataset = json.load(f)

    for data in tqdm(dataset, desc=split):
        filename = data['filename']
        lines = np.asarray(data['lines'])
        K, D, Kc = np.asarray(data['K']), np.asarray(data['D']), np.asarray(data['Kc'])
        coeff = {'K': K, 'D': D, 'Kc': Kc}
        image = cv2.imread(os.path.join(src_path, 'image', filename))

        pts_list = fis.interp_line(lines, coeff, resolution=0.01)
        lines = bez.fit_line(pts_list, order=order)[0]

        prefix = os.path.join(dst_path, filename.split('.')[0])
        save_npz(prefix, image, lines.copy(), cfg.heatmap_size)

        if plot:
            bez.insert_line(image, lines, color=[0, 0, 255])
            bez.insert_point(image, lines, color=[255, 0, 0], thickness=4)
            cv2.namedWindow('image', 0)
            cv2.imshow('image', image)
            cv2.waitKey()


if __name__ == "__main__":
    os.chdir('..')
    # Parameter
    cfg = parse()
    print(cfg)

    # Path
    src_path = cfg.raw_dataset_path
    dst_path = cfg.groundtruth_path
    os.makedirs(dst_path, exist_ok=True)
    order = 6

    json2npz(src_path, dst_path, cfg, order)
