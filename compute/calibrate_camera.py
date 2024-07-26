import cv2
import numpy as np
from tqdm import tqdm

from compute_image_to_space import get_direction_vector
from manual_calibration import manual_calibration

def get_intersect(
        u1: np.array,
        P1: np.array,
        u2: np.array,
        P2: np.array
    ) -> tuple[np.array, np.array]:
    """
    Given two lines described with a point and a unit vector, computes the points of each lines that minimise the distance between them. The two lines must not be parallel

    :param u1: unit vector of the first point. Can be a 4D vector
    :param P1: point of the first line. Must be a 3D vector
    :param u2: unit vector of the second point. Can be a 4D vector
    :param P2: point of the second line. Must be a 3D vector

    :return: the two points in two 3D vectors
    """
    n = np.cross(u1[0:3], u2[0:3])
    n1 = np.cross(n, u2[0:3])
    n2 = np.cross(n, u1[0:3])
    P12 = P2 - P1

    q1 = u1[0:3].dot(n1)
    q2 = u2[0:3].dot(n2)

    assert q1 != 0 and q2 != 0, 'u1 and u2 must not be parallel'

    mu1 = P12.dot(n1)/q1
    mu2 = -P12.dot(n2)/q2

    O1 = P1 + mu1*u1[0:3]
    O2 = P2 + mu2*u2[0:3]

    return O1, O2


def calibrate_camera_new(img, index=0, log=False):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    templates = [
        '/home/adrien/Bureau/Ned/compute/data/templates/circle.jpg',
        '/home/adrien/Bureau/Ned/compute/data/templates/circle2.jpg',
        '/home/adrien/Bureau/Ned/compute/data/templates/circle3.jpg',
        '/home/adrien/Bureau/Ned/compute/data/templates/circle4.jpg',
    ]

    def is_overlapping(box1, box2):
        """
        Check if two boxes are overlapping

        :param box1: the first box in the form (x1_tl, y1_tl, x1_br, y1_br)
        :param box2: the second box in the form (x2_tl, y2_tl, x2_br, y2_br)
        
        :return: True if the boxes are strictly colliding
        """
        x1_tl, y1_tl, x1_br, y1_br = box1
        x2_tl, y2_tl, x2_br, y2_br = box2
        return not (x1_br < x2_tl or x2_br < x1_tl or y1_br < y2_tl or y2_br < y1_tl)

    all_results = []
    for template_path in tqdm(templates):
        template = cv2.imread(template_path, cv2.IMREAD_COLOR)
        template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        h, w = template.shape[:2]

        for scale in np.linspace(0.5, 1.5, 40)[::-1]:
            resized_img = cv2.resize(img_gray, (int(img.shape[1] * scale), int(img.shape[0] * scale)))
            if resized_img.shape[0] < h or resized_img.shape[1] < w:
                break
            res = cv2.matchTemplate(resized_img, template_gray, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)
            all_results.append((max_val, max_loc, scale, template_path))

    all_results = sorted(all_results, key=lambda x: x[0], reverse=True)

    best_results = []
    for max_val, max_loc, scale, template_path in all_results:
        top_left = (int(max_loc[0] / scale), int(max_loc[1] / scale))
        bottom_right = (int((max_loc[0] + w) / scale), int((max_loc[1] + h) / scale))
        if not any(is_overlapping((*tl, *br), (*top_left, *bottom_right)) for _, (tl, br) in best_results):
            best_results.append(((max_val, max_loc, scale, template_path), (top_left, bottom_right)))
        if len(best_results) == 4:
            break

    L_target = []

    for (max_val, max_loc, scale, template_path), (top_left, bottom_right) in best_results:
        template = cv2.imread(template_path, cv2.IMREAD_COLOR)
        template = cv2.resize(template, (bottom_right[0] - top_left[0], bottom_right[1] - top_left[1]))
        L_target.append(((top_left[0]+bottom_right[0])/2, (top_left[1]+bottom_right[1])/2))

    L_target = sorted(L_target, key=lambda x: x[0], reverse=False)
    L_target[:2] = sorted(L_target[:2], key=lambda x: x[1], reverse=False)
    L_target[2:] = sorted(L_target[2:], key=lambda x: x[1], reverse=False)

    L_target = manual_calibration(img, L_target)
    L_target = [(x-1296, 997-y) for x, y in L_target]

    L_target = sorted(L_target, key=lambda x: x[1], reverse=True)
    L_target[:2] = sorted(L_target[:2], key=lambda x: x[0], reverse=False)
    L_target[2:] = sorted(L_target[2:], key=lambda x: x[0], reverse=False)

    A = np.array([L_target[2][0], L_target[2][1]], dtype=np.float128)
    B = np.array([L_target[0][0], L_target[0][1]], dtype=np.float128)
    A_ = np.array([L_target[3][0], L_target[3][1]], dtype=np.float128)
    B_ = np.array([L_target[1][0], L_target[1][1]], dtype=np.float128)

    uA = get_direction_vector(A[0], A[1], 1944/2)[:3]
    uB = get_direction_vector(B[0], B[1], 1944/2)[:3]
    uA_ = get_direction_vector(A_[0], A_[1], 1944/2)[:3]
    uB_ = get_direction_vector(B_[0], B_[1], 1944/2)[:3]

    uA = uA/uA[1]
    uB = uB/uB[1]
    uA_ = uA_/uA_[1]
    uB_ = uB_/uB_[1]

    A = np.array([uA[0], uA[2]], dtype=np.float128)
    B = np.array([uB[0], uB[2]], dtype=np.float128)
    A_ = np.array([uA_[0], uA_[2]], dtype=np.float128)
    B_ = np.array([uB_[0], uB_[2]], dtype=np.float128)

    det = (B[1]-A[1])*(B_[0]-A_[0])-(B[0]-A[0])*(B_[1]-A_[1])
    M = np.array([
        [-(B_[1]-A_[1])/det, (B_[0]-A_[0])/det],
        [-(B[1]-A[1])/det, (B[0]-A[0])/det]
    ], dtype=np.float128)

    L = M.dot(np.array([A_[0]-A[0], A_[1]-A[1]], dtype=np.float128))

    O = A+L[0]*(B-A)
    O = np.array([O[0], 1, O[1]], dtype=np.float128)
    O = O/np.linalg.norm(O)

    r = np.abs(2*1944/np.pi*np.arccos(O[1]))
    if r == 0:
        x, y = 0
    else:
        x, y = O[0]/np.sin(np.pi/2*r/1944)*r, O[2]/np.sin(np.pi/2*r/1944)*r

    r = np.sqrt(x**2 + y**2)
    alpha = np.pi/2*r/1944
    theta = np.arctan2(y, x)

    T = np.array([
        [np.cos(alpha)*np.cos(theta), -np.sin(alpha), np.cos(alpha)*np.sin(theta), 0],
        [np.sin(alpha)*np.cos(theta), np.cos(alpha), np.sin(alpha)*np.sin(theta), 0],
        [-np.sin(theta), 0, np.cos(theta), 0],
        [0, 0, 0, 1]
    ], dtype=np.float128)

    uA = T.dot(np.array([uA[0], uA[1], uA[2], 0]))
    uA_ = T.dot(np.array([uA_[0], uA_[1], uA_[2], 0]))

    uA = uA/uA[1]
    uA_ = uA_/uA_[1]

    uAA_ = uA_-uA
    beta = np.arctan2(uAA_[2], uAA_[0])

    T = np.array([
        [np.cos(beta), 0, np.sin(beta), 0],
        [0, 1, 0, 0],
        [-np.sin(beta), 0, np.cos(beta), 0],
        [0, 0, 0, 1]
    ], dtype=np.float128).dot(T)

    gamma = [np.pi, -np.pi/2, 0, np.pi/2][index]

    T = np.array([
        [np.cos(gamma), np.sin(gamma), 0, 0],
        [-np.sin(gamma), np.cos(gamma), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ], dtype=np.float128).dot(T)

    # print(T)

    A = np.array([L_target[2][0], L_target[2][1]], dtype=np.float128)
    B = np.array([L_target[0][0], L_target[0][1]], dtype=np.float128)
    A_ = np.array([L_target[3][0], L_target[3][1]], dtype=np.float128)
    B_ = np.array([L_target[1][0], L_target[1][1]], dtype=np.float128)

    # print(f"A: {A}")
    # print(f"B: {B}")
    # print(f"A_: {A_}")
    # print(f"B_: {B_}")

    uA = get_direction_vector(A[0], A[1], 1944/2)[:3]
    uB = get_direction_vector(B[0], B[1], 1944/2)[:3]
    uA_ = get_direction_vector(A_[0], A_[1], 1944/2)[:3]
    uB_ = get_direction_vector(B_[0], B_[1], 1944/2)[:3]

    uA = -T.dot(np.array([uA[0], uA[1], uA[2], 0]))
    uB = -T.dot(np.array([uB[0], uB[1], uB[2], 0]))
    uA_ = -T.dot(np.array([uA_[0], uA_[1], uA_[2], 0]))
    uB_ = -T.dot(np.array([uB_[0], uB_[1], uB_[2], 0]))

    uA = uA/np.linalg.norm(uA)
    uB = uB/np.linalg.norm(uB)
    uA_ = uA_/np.linalg.norm(uA_)
    uB_ = uB_/np.linalg.norm(uB_)

    uO, uA, uB, uC = [
        (uB_, uA_, uB, uA),
        (uB, uB_, uA, uA_),
        (uA, uB, uA_, uB_),
        (uA_, uA, uB_, uB)
    ][index]

    O = np.array([0.164, -0.091, 0], dtype=np.float128)
    A = np.array([0.167, 0.084, 0], dtype=np.float128)
    B = np.array([0.338, -0.093, 0], dtype=np.float128)
    C = np.array([0.338, 0.078, 0], dtype=np.float128)

    Lo = []
    Lo += list(get_intersect(uO, O, uA, A))
    # print(uA, A, uB, B)
    Lo += list(get_intersect(uO, O, uB, B))
    Lo += list(get_intersect(uO, O, uC, C))

    # print(Lo)

    X = np.array([0, 0, 0], dtype=np.float128)
    for x in Lo:
        # print(x)
        X = X + np.array(x, dtype=np.float128)
    X *= 1/6

    if log:
        sigma = 0
        for x in Lo:
            sigma += np.linalg.norm(np.array(x)-X)**2
        sigma = np.sqrt(sigma/6)
        print(f"Precision on the position: {sigma*100:.2f}cm")
    
    T[0][3] = X[0]
    T[1][3] = X[1]
    T[2][3] = X[2]

    if log:
        print(T)

    return T


if __name__ == '__main__':
    img = cv2.imread('/home/adrien/Bureau/Ned/compute/data/photos/photo_minnie.jpg', cv2.IMREAD_COLOR)
    calibrate_camera_new(img, index=0, log=True)