import numpy as np
import cv2


def insert_line(image, lines, color, thickness=1):
    """
    Insert lines into image
    :param image:
    :param lines: the shape is [N, 2, 2]
    :param color:
    :param thickness:
    :return: image
    """
    for pts in lines:
        pts = np.round(pts).astype(np.int32)
        cv2.line(image, tuple(pts[0]), tuple(pts[1]), color=color, thickness=thickness)
    return image
