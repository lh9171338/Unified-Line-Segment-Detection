import numpy as np
import cv2
import yaml


class Camera:
    def __init__(self):
        pass

    def load_coeff(self, filename):
        pass

    def distort_point(self, undistorted):
        pass

    def undistort_point(self, distorted):
        pass

    def distort_image(self, image):
        pass

    def undistort_image(self, image):
        pass

    def interp_line(self, lines, num=None, resolution=1.0):
        pass

    def remove_line(self, lines, thresh):
        distances = np.max(np.abs(lines[:, 0] - lines[:, 1]), axis=-1)
        mask = distances >= thresh
        lines = lines[mask]

        return lines

    def insert_line(self, image, pts_list, color, thickness=0):
        for pts in pts_list:
            pts = np.round(pts).astype(np.int32)
            cv2.polylines(image, [pts], isClosed=False, color=color, thickness=thickness)

        return image


class Pinhole(Camera):
    def __init__(self):
        super().__init__()

    def interp_line(self, lines, num=None, resolution=1.0):
        pts_list = []
        for line in lines:
            K = int(round(max(abs(line[1] - line[0])) / resolution)) + 1 if num is None else num
            lambda_ = np.linspace(0, 1, K)[:, None]
            pts = line[1] * lambda_ + line[0] * (1 - lambda_)
            pts_list.append(pts)

        if num is not None:
            return np.asarray(pts_list)
        else:
            return pts_list

    def insert_line(self, image, lines, color, thickness=0):
        pts_list = self.interp_line(lines)
        super().insert_line(image, pts_list, color, thickness)

        return image


class Fisheye(Camera):
    def __init__(self, coeff=None):
        super().__init__()

        self.coeff = coeff

    def load_coeff(self, filename):
        with open(filename, 'r') as f:
            fs = cv2.FileStorage(filename, cv2.FileStorage_READ)
            K = fs.getNode('K').mat()
            D = fs.getNode('D').mat()
        self.coeff = {'K': K, 'D': D}

    def distort_point(self, undistorted):
        K, D = self.coeff['K'], self.coeff['D']
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]

        undistorted = undistorted.reshape(-1, 1, 2).astype(np.float32)
        undistorted[:, :, 0] = (undistorted[:, :, 0] - cx) / fx
        undistorted[:, :, 1] = (undistorted[:, :, 1] - cy) / fy
        distorted = cv2.fisheye.distortPoints(undistorted, K, D)
        distorted = np.reshape(distorted, (-1, 2))

        return distorted

    def undistort_point(self, distorted):
        K, D = self.coeff['K'], self.coeff['D']
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]

        distorted = distorted.reshape(-1, 1, 2).astype(np.float32)
        undistorted = cv2.fisheye.undistortPoints(distorted, K, D)
        undistorted[:, :, 0] = undistorted[:, :, 0] * fx + cx
        undistorted[:, :, 1] = undistorted[:, :, 1] * fy + cy
        undistorted = np.reshape(undistorted, (-1, 2))

        return undistorted

    def distort_image(self, image):
        height, width = image.shape[0], image.shape[1]

        distorted = np.mgrid[0:width, 0:height].T.reshape(-1, 2)
        undistorted = self.undistort_point(distorted)
        undistorted = np.reshape(undistorted, (height, width, 2))
        map1 = undistorted[:, :, 0]
        map2 = undistorted[:, :, 1]
        image = cv2.remap(image, map1, map2, cv2.INTER_CUBIC)

        return image

    def undistort_image(self, image):
        height, width = image.shape[0], image.shape[1]

        undistorted = np.mgrid[0:width, 0:height].T.reshape(-1, 2)
        distorted = self.distort_point(undistorted)
        distorted = np.reshape(distorted, (height, width, 2))
        map1 = distorted[:, :, 0]
        map2 = distorted[:, :, 1]
        image = cv2.remap(image, map1, map2, cv2.INTER_CUBIC)

        return image

    def interp_line(self, lines, num=None, resolution=0.1):
        distorted = lines.reshape(-1, 2)
        undistorted = self.undistort_point(distorted)
        lines = undistorted.reshape(-1, 2, 2)

        pts_list = []
        for line in lines:
            K = int(round(max(abs(line[-1] - line[0])) / resolution)) + 1 if num is None else num
            lambda_ = np.linspace(0, 1, K)[:, None]
            pts = line[1] * lambda_ + line[0] * (1 - lambda_)
            pts = self.distort_point(pts)
            pts_list.append(pts)

        if num is not None:
            return np.asarray(pts_list)
        else:
            return pts_list

    def insert_line(self, image, lines, color, thickness=0):
        pts_list = self.interp_line(lines)
        super().insert_line(image, pts_list, color, thickness)

        return image

    def truncate_line(self, lines, image_size):
        width, height = image_size[0], image_size[1]

        pts_list = self.interp_line(lines, resolution=0.1)
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


class Spherical(Camera):
    def __init__(self, image_size=None):
        super().__init__()

        self.image_size = image_size

    def distort_point(self, undistorted):
        width, height = self.image_size[0], self.image_size[1]

        x, y, z = undistorted[:, 0], undistorted[:, 1], undistorted[:, 2]
        lat = np.pi - np.arccos(y)
        lon = np.pi - np.arctan2(z, x)
        u = width * lon / (2 * np.pi)
        v = height * lat / np.pi
        u = np.mod(u, width)
        v = np.mod(v, height)
        distorted = np.concatenate((u[:, None], v[:, None]), axis=-1)

        return distorted

    def undistort_point(self, distorted):
        width, height = self.image_size[0], self.image_size[1]

        u, v = distorted[:, 0], distorted[:, 1]
        lon = np.pi - u / width * 2 * np.pi
        lat = np.pi - v / height * np.pi
        y = np.cos(lat)
        x = np.sin(lat) * np.cos(lon)
        z = np.sin(lat) * np.sin(lon)
        undistorted = np.concatenate((x[:, None], y[:, None], z[:, None]), axis=-1)

        return undistorted

    def interp_arc(self, arcs, num=None, resolution=0.01):
        resolution *= np.pi / 180.0

        pts_list = []
        for arc in arcs:
            pt1, pt2 = arc[0], arc[1]
            normal = np.cross(pt1, pt2)
            normal /= np.linalg.norm(normal)
            angle = np.arccos(normal[2])
            axes = np.array([-normal[1], normal[0], 0])
            axes /= max(np.linalg.norm(axes), 1e-9)
            rotation_vector = angle * axes
            rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
            pt1 = np.matmul(rotation_matrix.T, pt1[:, None]).flatten()
            pt2 = np.matmul(rotation_matrix.T, pt2[:, None]).flatten()
            min_angle = np.arctan2(pt1[1], pt1[0])
            max_angle = np.arctan2(pt2[1], pt2[0])
            if max_angle < min_angle:
                max_angle += 2 * np.pi

            K = int(round((max_angle - min_angle) / resolution) + 1) if num is None else num
            angles = np.linspace(min_angle, max_angle, K)
            pts = np.concatenate((np.cos(angles)[:, None], np.sin(angles)[:, None], np.zeros((K, 1))), axis=-1)
            pts = np.matmul(rotation_matrix, pts.T).T
            pts_list.append(pts)

        if num is not None:
            return np.asarray(pts_list)
        else:
            return pts_list

    def interp_line(self, lines, num=None, resolution=0.01):
        distorted = lines.reshape((-1, 2))
        undistorted = self.undistort_point(distorted)
        arcs = undistorted.reshape((-1, 2, 3))
        undistorted_list = self.interp_arc(arcs, num, resolution)
        distorted_list = []
        for undistorted in undistorted_list:
            distorted = self.distort_point(undistorted)
            distorted_list.append(distorted)

        if num is not None:
            return np.asarray(distorted_list)
        else:
            return distorted_list

    def insert_line(self, image, lines, color, thickness=0):
        pts_list = self.interp_line(lines)
        super().insert_line(image, pts_list, color, thickness)

        return image

    def truncate_line(self, lines):
        width = self.image_size[0]
        pts_list = self.interp_line(lines, resolution=0.01)
        lines = []
        for pts in pts_list:
            dx = abs(pts[:-1, 0] - pts[1:, 0])
            mask = dx > width / 2.0
            s = sum(mask)
            assert s <= 1
            if s == 0:
                lines.append([pts[0], pts[-1]])
            else:
                ind = np.where(mask)[0][0]
                lines.append([pts[0], pts[ind]])
                lines.append([pts[ind + 1], pts[-1]])
        lines = np.asarray(lines)

        return lines
