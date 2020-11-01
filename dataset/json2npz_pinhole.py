import sys
sys.path.append('..')
import os
import json
import cv2
import numpy as np
import shutil
import skimage.draw
from itertools import combinations
from scipy.ndimage import zoom
import multiprocessing
import time
import util.pinhole as pin
import util.bezier as bez
import util.augment as aug
from config.cfg import parse


def __parallel_handle(f, q_in, q_out):
    while True:
        i, x = q_in.get()
        if i is None:
            break
        print(f'Progress: {i}')
        q_out.put((i, f(x)))


def parmap(f, X, nprocs=multiprocessing.cpu_count(), progress_bar=lambda x: x):
    if nprocs == 0:
        nprocs = multiprocessing.cpu_count()
    q_in = multiprocessing.Queue(1)
    q_out = multiprocessing.Queue()

    proc = [
        multiprocessing.Process(target=__parallel_handle, args=(f, q_in, q_out))
        for _ in range(nprocs)
    ]
    for p in proc:
        p.daemon = True
        p.start()

    try:
        sent = [q_in.put((i, x)) for i, x in enumerate(X)]
        [q_in.put((None, None)) for _ in range(nprocs)]
        res = [q_out.get() for _ in progress_bar(range(len(sent)))]
        [p.join() for p in proc]
    except KeyboardInterrupt:
        q_in.close()
        q_out.close()
        raise
    return [x for i, x in sorted(res)]


def save_npz(prefix, image, lines, centers, cfg):
    n_pts = cfg.order + 1
    image_size = (image.shape[1], image.shape[0])
    heatmap_size = cfg.heatmap_size
    sx, sy = heatmap_size[0] / image_size[0], heatmap_size[1] / image_size[1]

    lines_mask = lines[:, 0, 1] > lines[:, -1, 1]
    lines[lines_mask] = lines[lines_mask, ::-1]
    lines[:, :, 0] = np.clip(lines[:, :, 0] * sx, 0, heatmap_size[0] - 1e-4)
    lines[:, :, 1] = np.clip(lines[:, :, 1] * sy, 0, heatmap_size[1] - 1e-4)
    centers[:, 0] = np.clip(centers[:, 0] * sx, 0, heatmap_size[0] - 1e-4)
    centers[:, 1] = np.clip(centers[:, 1] * sy, 0, heatmap_size[1] - 1e-4)

    jmap = np.zeros((1,) + heatmap_size[::-1], dtype=np.float32)
    joff = np.zeros((2,) + heatmap_size[::-1], dtype=np.float32)
    cmap = np.zeros((1,) + heatmap_size[::-1], dtype=np.float32)
    coff = np.zeros((2,) + heatmap_size[::-1], dtype=np.float32)
    lvec = np.zeros(((n_pts // 2) * 2, 2,) + heatmap_size[::-1], dtype=np.float32)
    lmap = np.zeros((1,) + heatmap_size[::-1], dtype=np.float32)

    juncs = dict()
    lpos = lines.copy()
    lneg, scores = [], []
    linds = []

    if n_pts % 2 == 1:
        lines = np.delete(lines, n_pts // 2, axis=1)

    def to_int(x):
        return tuple(map(int, x))

    def add_junc(juncs, junc):
        ind = len(juncs)
        if junc in juncs:
            ind = juncs[junc]
        else:
            juncs[junc] = ind
        return ind

    for c, pts in zip(centers, lines):
        v0, v1 = pts[0], pts[-1]
        lind = [add_junc(juncs, tuple(v0)), add_junc(juncs, tuple(v1))]
        linds.append(lind)

        cint = to_int(c)
        vint0 = to_int(v0)
        vint1 = to_int(v1)
        jmap[0, vint0[1], vint0[0]] = 1
        jmap[0, vint1[1], vint1[0]] = 1
        joff[:, vint0[1], vint0[0]] = v0 - vint0 - 0.5
        joff[:, vint1[1], vint1[0]] = v1 - vint1 - 0.5
        cmap[0, cint[1], cint[0]] = 1
        coff[:, cint[1], cint[0]] = c - cint - 0.5
        lvec[:, :, cint[1], cint[0]] = pts - c

        rr, cc, value = skimage.draw.line_aa(int(v0[1]), int(v0[0]), int(v1[1]), int(v1[0]))
        lmap[0, rr, cc] = np.maximum(lmap[0, rr, cc], value)

    lvec = lvec.reshape((-1,) + heatmap_size[::-1])
    juncs = np.asarray([np.array(junc) for junc in list(juncs.keys())])
    linds = np.asarray(linds)

    llmap = zoom(lmap[0], [0.5, 0.5])
    lineset = set([frozenset(l) for l in linds])
    for i0, i1 in combinations(range(len(juncs)), 2):
        if frozenset([i0, i1]) not in lineset:
            v0, v1 = juncs[i0], juncs[i1]
            lneg.append([v0, v1])
            v0, v1 = v0 / 2.0, v1 / 2.0
            rr, cc, value = skimage.draw.line_aa(int(v0[1]), int(v0[0]), int(v1[1]), int(v1[0]))
            scores.append(np.average(np.minimum(value, llmap[rr, cc])))

    lpos = np.asarray(lpos)
    lneg = np.asarray(lneg)
    scores = np.asarray(scores)
    indices = np.argsort(-scores)[:2000]
    lneg = lneg[indices]

    np.savez_compressed(
        f'{prefix}.npz',
        junc=juncs,
        lpos=lpos,
        lneg=lneg,
        jmap=jmap,
        joff=joff,
        cmap=cmap,
        coff=coff,
        lvec=lvec,
        lmap=lmap
    )
    cv2.imwrite(f'{prefix}.png', image)


def json2npz(src_path, dst_path, split, cfg, plot=False):

    json_file = os.path.join(src_path, f'{split}.json')
    try:
        with open(json_file, 'r') as f:
            dataset = json.load(f)
    except Exception:
        return

    if os.path.exists(os.path.join(dst_path, split)):
        shutil.rmtree(os.path.join(dst_path, split))
    os.makedirs(os.path.join(dst_path, split), exist_ok=True)

    tfs = [aug.Noop(), aug.HorizontalFlip(), aug.VerticalFlip(),
            aug.Compose([aug.HorizontalFlip(), aug.VerticalFlip()])]

    def call_back(data):
        filename = data['filename']
        lines0 = np.asarray(data['lines'])
        image0 = cv2.imread(os.path.join(src_path, 'image', filename))

        if split == 'train':
            for i in range(len(tfs)):
                image, lines = tfs[i](image0, lines0)
                centers = lines.mean(axis=1)

                prefix = os.path.join(dst_path, split, filename.split('.')[0] + f'_{i}')
                save_npz(prefix, image, lines.copy(), centers, cfg)

                if plot:
                    bez.insert_line(image, lines, color=[0, 0, 255])
                    bez.insert_point(image, lines, color=[255, 0, 0], thickness=2)
                    cv2.namedWindow('image', 0)
                    cv2.imshow('image', image)
                    cv2.waitKey()

        else:
            image, lines = image0.copy(), lines0.copy()
            centers = lines.mean(axis=1)

            prefix = os.path.join(dst_path, split, filename.split('.')[0])
            save_npz(f'{prefix}', image, lines.copy(), centers, cfg)

            if plot:
                bez.insert_line(image, lines, color=[0, 0, 255])
                bez.insert_point(image, lines, color=[255, 0, 0], thickness=2)
                cv2.namedWindow('image', 0)
                cv2.imshow('image', image)
                cv2.waitKey()

    parmap(call_back, dataset, nprocs=multiprocessing.cpu_count())


if __name__ == "__main__":
    os.chdir('..')
    # Parameter
    cfg = parse()
    print(cfg)

    # Path
    src_path = cfg.raw_dataset_path
    dst_path = cfg.train_dataset_path
    os.makedirs(dst_path, exist_ok=True)

    start = time.time()
    for split in ['train', 'test']:
        json2npz(src_path, dst_path, split, cfg)

    end = time.time()
    print('Time: %f s' % (end - start))
