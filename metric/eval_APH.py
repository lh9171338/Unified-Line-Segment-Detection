import sys
sys.path.append('..')
import os
import glob
import shutil
import subprocess
import numpy as np
import time
import scipy.io as sio
import matplotlib.pyplot as plt
from scipy import interpolate
from config.cfg import parse
from metric.eval_metric import plot_pr_curve


output_size = 128
thresh = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.97, 0.99, 0.995, 0.999, 0.9995, 0.9999]


def eval_APH(cfg):
    run_all = False

    image_path = cfg.groundtruth_path
    gt_path = cfg.groundtruth_path
    pred_path = cfg.output_path
    save_path = cfg.figure_path
    output_file = os.path.join(save_path, 'temp.mat')

    if run_all:
        temp_path = os.path.join(cfg.figure_path, 'temp')
        os.makedirs(temp_path, exist_ok=True)
        print(f'intermediate matlab results will be saved at: {temp_path}')

        gt_file_list = glob.glob(os.path.join(gt_path, '*.npz'))
        for gt_file in gt_file_list:
            name = gt_file.split('/')[-1].split('.')[0]
            mat_name = name + '.mat'
            npz = np.load(gt_file)
            lines = npz['line'][:, [0, -1]].reshape(-1, 4)
            os.makedirs(os.path.join(temp_path, 'gt'), exist_ok=True)
            sio.savemat(os.path.join(temp_path, 'gt', mat_name), {'lines': lines})

        pred_file_list = glob.glob(os.path.join(pred_path, '*.npz'))
        for t in thresh:
            for pred_file in pred_file_list:
                name = pred_file.split('/')[-1].split('.')[0]
                mat_name = name + '.mat'
                npz = np.load(pred_file)
                lines = npz['line_pred'][:, [0, -1]].reshape(-1, 4)
                scores = npz['line_score']
                idx = np.where(scores > t)[0]
                os.makedirs(os.path.join(temp_path, str(t)), exist_ok=True)
                sio.savemat(os.path.join(temp_path, str(t), mat_name), {'lines': lines[idx]})

        cmd = 'matlab -nodisplay -nodesktop '
        cmd += '-r "dbstop if error; '
        cmd += "eval_release('{:s}', '{:s}', '{:s}', {:d}); quit;\"".format(
            image_path, temp_path, output_file, output_size
        )
        print('Running:\n{}'.format(cmd))
        os.environ['MATLABPATH'] = 'metric/matlab/'
        subprocess.call(cmd, shell=True)
        shutil.rmtree(temp_path)

    mat = sio.loadmat(output_file)
    tps = mat['sumtp'][:, 0]
    fps = mat['sumfp'][:, 0]
    N = mat['sumgt'][:, 0]

    rcs = (tps / N)
    prs = (tps / (tps + fps))
    mask = np.logical_not(np.isnan(prs))
    rcs = rcs[mask]
    prs = prs[mask]
    indices = np.argsort(rcs)
    rcs = np.sort(rcs[indices])
    prs = np.sort(prs[indices])[::-1]

    recall = np.concatenate(([0.0], rcs, [1.0]))
    precision = np.concatenate(([0.0], prs, [0.0]))
    for i in range(precision.size - 1, 0, -1):
        precision[i - 1] = max(precision[i - 1], precision[i])
    i = np.where(recall[1:] != recall[:-1])[0]
    APH = np.sum((recall[i + 1] - recall[i]) * precision[i + 1]) * 100
    FH = (2 * np.array(prs) * np.array(rcs) / (np.array(prs) + np.array(rcs))).max() * 100

    f = interpolate.interp1d(rcs, prs, kind='linear', bounds_error=False, fill_value='extrapolate')
    x = np.arange(0, 1, 0.01) * rcs[-1]
    y = f(x)

    figure = plot_pr_curve(x, y, title='AP${^{H}}$', legend=['ULSD'])
    figure.savefig(os.path.join(save_path, 'APH.pdf'), format='pdf', bbox_inches='tight')
    sio.savemat(os.path.join(save_path, 'APH-ULSD.mat'), {'rcs': x, 'prs': y, 'AP': APH})
    plt.show()

    return APH, FH


if __name__ == "__main__":
    # Parameter
    os.chdir('..')
    cfg = parse()
    print(cfg)
    os.makedirs(cfg.figure_path, exist_ok=True)

    start = time.time()
    APH, FH = eval_APH(cfg)
    print(f'APH: {APH:.1f} | FH: {FH:.1f}')
    end = time.time()
    print('Time: %f s' % (end - start))
