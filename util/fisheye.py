import numpy as np
import random
import cv2


def distort_coeff(image_size, crop=False):
    """
    Get distortion coefficients
    :param image_size: (width, height)
    :return: coeff: {'K': K, 'D': D, 'Kc', Kc}
    """
    width, height = image_size[0], image_size[1]
    focal = float(np.random.randint(100, 200))
    cx, cy = width / 2.0, height / 2.0
    K = np.array([[focal, 0.0, cx], [0.0, focal, cy], [0.0, 0.0, 1.0]])
    D = 1e-3 * np.random.randn(1, 4)
    Kc = np.array([0.0, 0.0, 1.0, 1.0])
    coeff = {'K': K, 'D': D, 'Kc': Kc}
    inner = random.random() > 0.6

    if crop:
        if inner:
            p0 = np.array([[0.0, 0.0]])
            p0 = distort_point(p0, coeff)[0]
            new_height = height - 2.0 * p0[1]
            new_width = width - 2.0 * p0[0]
            sx, sy = new_width / width, new_height / height
            Kc = np.array([p0[0], p0[1], sx, sy])
            coeff['Kc'] = Kc
        else:
            p0 = np.array([[0.0, cy], [cx, 0.0]])
            p0 = distort_point(p0, coeff)
            new_height = height - 2.0 * p0[1, 1]
            new_width = width - 2.0 * p0[0, 0]
            sx, sy = new_width / width, new_height / height
            Kc = np.array([p0[0, 0], p0[1, 1], sx, sy])
            coeff['Kc'] = Kc

    return coeff


def distort_point(undistorted, coeff):
    """
    Distort points
    :param undistorted: the shape is [N, 2]
    :param coeff: {'K': K, 'D': D, 'Kc', Kc}
    :return: distorted: the shape is [N, 2]
    """
    K, D, Kc = coeff['K'], coeff['D'], coeff['Kc']
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    x0, y0, sx, sy = Kc[0], Kc[1], Kc[2], Kc[3]

    undistorted = undistorted.reshape(-1, 1, 2).astype(np.float32)
    undistorted[:, :, 0] = (undistorted[:, :, 0] - cx) / fx
    undistorted[:, :, 1] = (undistorted[:, :, 1] - cy) / fy
    distorted = cv2.fisheye.distortPoints(undistorted, K, D)
    distorted[:, :, 0] = (distorted[:, :, 0] - x0) / sx
    distorted[:, :, 1] = (distorted[:, :, 1] - y0) / sy
    distorted = np.reshape(distorted, (-1, 2))

    return distorted


def undistort_point(distorted, coeff):
    """
    Undistort points
    :param distorted: the shape is [N, 2]
    :param coeff: {'K': K, 'D': D, 'Kc', Kc}
    :return: undistorted: the shape is [N, 2]
    """
    K, D, Kc = coeff['K'], coeff['D'], coeff['Kc']
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    x0, y0, sx, sy = Kc[0], Kc[1], Kc[2], Kc[3]

    distorted = distorted.reshape(-1, 1, 2).astype(np.float32)
    distorted[:, :, 0] = distorted[:, :, 0] * sx + x0
    distorted[:, :, 1] = distorted[:, :, 1] * sy + y0
    undistorted = cv2.fisheye.undistortPoints(distorted, K, D)
    undistorted[:, :, 0] = undistorted[:, :, 0] * fx + cx
    undistorted[:, :, 1] = undistorted[:, :, 1] * fy + cy
    undistorted = np.reshape(undistorted, (-1, 2))

    return undistorted


def distort_image(image, coeff):
    """
    Distort image
    :param image:
    :param coeff: {'K': K, 'D': D, 'Kc', Kc}
    :param crop:
    :return: image
    """
    height, width = image.shape[0], image.shape[1]

    distorted = np.mgrid[0:width, 0:height].T.reshape(-1, 2)
    undistorted = undistort_point(distorted, coeff)
    undistorted = np.reshape(undistorted, (height, width, 2))
    map1 = undistorted[:, :, 0]
    map2 = undistorted[:, :, 1]
    image = cv2.remap(image, map1, map2, cv2.INTER_CUBIC)

    return image


def undistort_image(image, coeff):
    """
    Distort image
    :param image:
    :param coeff: {'K': K, 'D': D, 'Kc', Kc}
    :param crop:
    :return: image
    """
    height, width = image.shape[0], image.shape[1]

    undistorted = np.mgrid[0:width, 0:height].T.reshape(-1, 2)
    distorted = distort_point(undistorted, coeff)
    distorted = np.reshape(distorted, (height, width, 2))
    map1 = distorted[:, :, 0]
    map2 = distorted[:, :, 1]
    image = cv2.remap(image, map1, map2, cv2.INTER_CUBIC)

    return image


def distort_line(lines, coeff, image_size=None):
    """
    Distort lines
    :param line: the shape is [N, 2, 2]
    :param: coeff: {'K': K, 'D': D, 'Kc', Kc}
    :param: image_size: (width, height)
    :return: lines
    """
    width, height = image_size[0], image_size[1]

    pts = lines.reshape(-1, 2)
    pts = distort_point(pts, coeff)
    lines = pts.reshape(-1, 2, 2)

    pts_list = interp_line(lines, coeff)
    lines_list = []
    for pts in pts_list:
        mask = np.logical_and(np.logical_and(pts[:, 0] >= 0, pts[:, 0] < width),
                              np.logical_and(pts[:, 1] >= 0, pts[:, 1] < height)).astype(np.int)
        mask1 = np.concatenate((mask[:1], mask[1:] - mask[:-1])) == 1
        mask2 = np.concatenate((mask[:-1] - mask[1:], mask[-1:])) == 1
        lines = np.concatenate((pts[mask1][:, None], pts[mask2][:, None]), axis=1)
        lines_list.append(lines)

    lines = np.concatenate(lines_list)

    return lines


def remove_line(lines, thresh):
    """
    Remove short lines
    :param lines: the shape is [N, 2, 2]
    :param thresh:
    :return: lines
    """
    distances = np.max(np.abs(lines[:, 0] - lines[:, 1]), axis=-1)
    mask = distances >= thresh
    lines = lines[mask]
    return lines


def interp_line(lines, coeff, num=None, resolution=0.1):
    """
    Line interpolation
    :param lines: the shape is [N, 2, 2]
    :param coeff: {'K': K, 'D': D, 'Kc', Kc}
    :param num:
    :param resolution:
    :return: pts_list
    """
    pts = lines.reshape(-1, 2)
    pts = undistort_point(pts, coeff)
    lines = pts.reshape(-1, 2, 2)

    pts_list = []
    for line in lines:
        K = int(round(max(abs(line[-1] - line[0])) / resolution)) + 1 if num is None else num
        lambda_ = np.linspace(0, 1, K)[:, None]
        pts = line[1] * lambda_ + line[0] * (1 - lambda_)
        pts = distort_point(pts, coeff)
        pts_list.append(pts)

    return pts_list


def insert_line(image, lines, coeff, color, thickness=0):
    """
    Insert lines into image
    :param image:
    :param lines: the shape is [N, 2, 2]
    :param coeff: {'K': K, 'D': D, 'Kc', Kc}
    :param color:
    :param thickness:
    :return: image
    """
    pts_list = interp_line(lines, coeff, num=None, resolution=0.1)
    for pts in pts_list:
        pts = np.round(pts).astype(np.int32)
        cv2.polylines(image, [pts], isClosed=False, color=color, thickness=thickness)
    return image
