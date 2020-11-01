import numpy as np
import cv2


def uv2xyz(p2ds, image_size):
    """
    Transfer 2D coordinates to 3D coordinates
    :param p2ds: the shape is [N, 2]
    :param image_size: (width, height)
    :return: p3ds: the shape is [N, 3]
    """
    width, height = image_size[0], image_size[1]

    u, v = p2ds[:, 0], p2ds[:, 1]
    lon = np.pi - u / width * 2 * np.pi
    lat = np.pi - v / height * np.pi
    y = np.cos(lat)
    x = np.sin(lat) * np.cos(lon)
    z = np.sin(lat) * np.sin(lon)

    p3ds = np.concatenate((x[:, None], y[:, None], z[:, None]), axis=-1)
    return p3ds


def xyz2uv(p3ds, image_size):
    """
    Transfer 3D coordinates to 2D coordinates
    :param p3ds: the shape is [N, 3]
    :param image_size: (width, height)
    :return: p2ds: the shape is [N, 2]
    """
    width, height = image_size[0], image_size[1]

    x, y, z = p3ds[:, 0], p3ds[:, 1], p3ds[:, 2]
    lat = np.pi - np.arccos(y)
    lon = np.pi - np.arctan2(z, x)
    u = width * lon / (2 * np.pi)
    v = height * lat / np.pi
    u = np.mod(u, width)
    v = np.mod(v, height)

    p2ds = np.concatenate((u[:, None], v[:, None]), axis=-1)
    return p2ds


def interp_arc(arcs, num=None, resolution=0.1):
    """
    Arc interpolation
    :param arcs: the shape is [N, 2, 3]
    :param num:
    :param resolution:
    :return: p3ds_list
    """
    resolution *= np.pi / 180.0

    p3ds_list = []
    for arc in arcs:
        arc_pt1, arc_pt2 = arc[0], arc[1]
        normal = np.cross(arc_pt1, arc_pt2)
        normal /= np.linalg.norm(normal)
        angle = np.arccos(normal[2])
        axes = np.array([-normal[1], normal[0], 0])
        axes /= max(np.linalg.norm(axes), 1e-9)
        rotation_vector = angle * axes
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        pt1 = np.matmul(rotation_matrix.T, arc_pt1[:, None]).flatten()
        pt2 = np.matmul(rotation_matrix.T, arc_pt2[:, None]).flatten()
        min_angle = np.arctan2(pt1[1], pt1[0])
        max_angle = np.arctan2(pt2[1], pt2[0])
        if max_angle < min_angle:
            max_angle += 2 * np.pi

        K = int(round((max_angle - min_angle) / resolution) + 1) if num is None else num
        angles = np.linspace(min_angle, max_angle, K)
        arc_pts = np.concatenate((np.cos(angles)[:, None], np.sin(angles)[:, None], np.zeros((K, 1))), axis=-1)
        arc_pts = np.matmul(rotation_matrix, arc_pts.T).T
        p3ds_list.append(arc_pts)

    return p3ds_list


def interp_line(lines, image_size, num=None, resolution=0.1):
    """
    Line interpolation
    :param lines: the shape is [N, 2, 2]
    :param image_size: (width, height)
    :param num:
    :param resolution:
    :return: p2ds_list
    """
    p2ds = lines.reshape((-1, 2))
    p3ds = uv2xyz(p2ds, image_size)
    arcs = p3ds.reshape((-1, 2, 3))
    p3ds_list = interp_arc(arcs, num, resolution)
    p2ds_list = []
    for p3ds in p3ds_list:
        p2ds = xyz2uv(p3ds, image_size)
        p2ds_list.append(p2ds)

    return p2ds_list


def truncate_line(lines, image_size):
    """
    Truncate lines
    :param lines: the shape is [N, 2, 2]
    :param image_size: (width, height)
    :return: lines
    """
    p2ds_list = interp_line(lines, image_size, num=None, resolution=0.01)
    lines = []
    for p2ds in p2ds_list:
        dx = abs(p2ds[:-1, 0] - p2ds[1:, 0])
        mask = dx > image_size[0] / 2.0
        s = sum(mask)
        assert s <= 1
        if s == 0:
            lines.append([p2ds[0], p2ds[-1]])
        else:
            ind = np.where(mask)[0][0]
            lines.append([p2ds[0], p2ds[ind]])
            lines.append([p2ds[ind + 1], p2ds[-1]])
    lines = np.asarray(lines)
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


def insert_arc(image, arcs, color, thickness=0):
    """
    Insert arcs into image
    :param image:
    :param arcs: the shape is [N, 2, 3]
    :param color:
    :param thickness:
    :return: image
    """
    image_size = (image.shape[1], image.shape[0])

    p3ds_list = interp_arc(arcs, num=None, resolution=0.1)
    for p3ds in p3ds_list:
        p2ds = xyz2uv(p3ds, image_size)
        p2ds = np.round(p2ds).astype(np.int32)
        cv2.polylines(image, [p2ds], isClosed=False, color=color, thickness=thickness)
    return image


def insert_line(image, lines, color, thickness=0):
    """
    Insert lines into image
    :param image:
    :param lines: the shape is [N, 2, 2]
    :param color:
    :param thickness:
    :return: image
    """
    image_size = (image.shape[1], image.shape[0])

    p2ds_list = interp_line(lines, image_size, num=None, resolution=0.1)
    for p2ds in p2ds_list:
        p2ds = np.round(p2ds).astype(np.int32)
        cv2.polylines(image, [p2ds], isClosed=False, color=color, thickness=thickness)
    return image
