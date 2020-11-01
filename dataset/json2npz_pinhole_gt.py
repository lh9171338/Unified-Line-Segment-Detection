import sys
sys.path.append('..')
import os
import json
import cv2
import numpy as np
from tqdm import tqdm
import util.pinhole as pin
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



def json2npz(src_path, dst_path, cfg, plot=False):
    split = 'test'
    os.makedirs(dst_path, exist_ok=True)

    json_file = os.path.join(src_path, f'{split}.json')
    with open(json_file, 'r') as f:
        dataset = json.load(f)

    for data in tqdm(dataset, desc=split):
        filename = data['filename']
        lines = np.asarray(data['lines'])
        image = cv2.imread(os.path.join(src_path, 'image', filename))

        prefix = os.path.join(dst_path, filename.split('.')[0])
        save_npz(prefix, image, lines.copy(), cfg.heatmap_size)

        if plot:
            pin.insert_line(image, lines, color=[0, 0, 255])
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

    json2npz(src_path, dst_path, cfg)
