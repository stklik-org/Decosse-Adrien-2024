import cv2
import numpy as np
import cProfile

def optimized_unfish(image_path):
    image = cv2.imread(image_path)
    h, w = image.shape[:2]
    rh = h/2
    y_idx, x_idx = np.indices((h, w))
    dy = y_idx - h / 2
    dx = x_idx - w / 2
    dist_sq = dx**2 + dy**2
    valid = dist_sq >= 0.001
    O = np.dstack((dx, np.full_like(dx, rh), dy))
    O_norm = np.linalg.norm(O, axis=2)
    O_unit = O / O_norm[..., np.newaxis]
    r = np.abs(2 / np.pi * rh * np.arccos(O_unit[..., 1])) * valid
    sin_term = np.sin(np.pi / 2 * r / rh)
    sin_term[sin_term == 0] = 1
    x_map = (O_unit[..., 0] / sin_term * r + w/2).astype(np.int32)
    y_map = (h/2 - O_unit[..., 2] / sin_term * r).astype(np.int32)
    x_map = np.clip(x_map, 0, w - 1)
    y_map = np.clip(y_map, 0, h - 1)
    new_image = image[y_map, x_map]
    new_image = np.flipud(new_image)
    return new_image

def main_optimized():
    new_image = optimized_unfish("/home/adrien/Bureau/photo.jpg")
    # cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    # cv2.imshow("image", new_image)
    # cv2.waitKey(0)

cProfile.run('main_optimized()', sort='tottime')