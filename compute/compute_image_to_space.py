import numpy as np

def compute_image_to_space(
        x1: float,
        y1: float,
        rh1: float,
        B1: np.ndarray,
        x2: float,
        y2: float,
        rh2: float,
        B2: np.ndarray,
        log=False,
    ) -> np.ndarray:
    '''
    :param x: x coordinate from the center of the point in pixel
    :param y: y coordinate from the center of the point in pixel
    :param rh: distance from the center of the horizon in pixel
    :param B: baseline of the camera in meters

    :return: the found position in a 4D vector
    '''
    assert rh1 > 0 and rh2 > 0, 'rh1 and rh2 must be greater than 0'
    assert B1.shape == (4, 4) and B2.shape == (4, 4), 'B1 and B2 must be 4x4 matrices'
    assert x1**2 + y1**2 <= rh1**2, 'The point must be inside the horizon'
    assert x2**2 + y2**2 <= rh2**2, 'The point must be inside the horizon'
    # assert B1[3] == [0, 0, 0, 1] and B2[3] == [0, 0, 0, 1], 'The last row of B1 and B2 must be [0, 0, 0, 1]'
    assert np.linalg.det(B1.copy().astype(np.float64)) != 0 and np.linalg.det(B2.copy().astype(np.float64)) != 0, 'The determinant of B1 and B2 must be different from 0'

    u1 = get_direction_vector(x1, y1, rh1)
    u2 = get_direction_vector(x2, y2, rh2)

    u1 = change_basis(B1, u1)
    u2 = change_basis(B2, u2)

    # print(u1, u2)

    P1 = B1[0:3, 3]
    P2 = B2[0:3, 3]

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

    O = (O1 + O2)/2

    delta = np.linalg.norm(O1-O2)/2
    if log:
        print(f"Precision on the position: {delta*100:.2f}cm")
        # print(f"Position1: {O1}")
        # print(f"Position2: {O2}")

    return np.array([O[0], O[1], O[2], 1])



def get_direction_vector(x: float, y: float, rh: float) -> np.ndarray:
    '''
    :param x: x coordinate from the center of the point in pixel
    :param y: y coordinate from the center of the point in pixel
    :param rh: distance from the center of the horizon in pixel
    '''

    r = np.sqrt(x**2 + y**2)
    if r == 0:
        return np.array([0, 1, 0, 0])
    
    u = np.array([
        x/r*np.sin(np.pi/2*r/rh),
        np.cos(np.pi/2*r/rh),
        y/r*np.sin(np.pi/2*r/rh),
        1
    ])

    return u

def change_basis(B: np.ndarray, u: np.ndarray) -> np.ndarray:
    '''
    :param B: 4x4 matrix
    :param u: 4x1 vector
    '''
    u_ = np.dot(B[0:3, 0:3], u[0:3])
    return np.array([u_[0], u_[1], u_[2], 1])


if __name__=="__main__":
    print(
        compute_image_to_space(
            0.5,
            0,
            1,
            np.array([
                [1, 0, 0, 0],
                [0, 1, 0, -1],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ]),
            0,
            0,
            1,
            np.array([
                [0, -1, 0, 1],
                [1, 0, 0, 0],
                [0, 0, 1, 0.001],
                [0, 0, 0, 1]
            ]),
            log=True
        )
    )