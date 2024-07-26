import numpy as np
import os

D1 = 0.174
D2 = 0.220
D3 = 0.245
D4 = 0.042

E1 = 0.032
E2 = 0.009

def joints_to_coordinate(j1, j2, j3, j4, j5, j6):
    '''
    This program computes the theorical position and orientation of the arm given the angle made by each joints
    return: the base of the arm in a (4, 4) matrix
    '''
    T0_1 = np.array([
        [np.cos(j1), -np.sin(j1), 0, 0],
        [np.sin(j1), np.cos(j1), 0, 0],
        [0, 0, 1, D1],
        [0, 0, 0, 1]
    ])

    T1_2 = np.array([
        [np.cos(j2), 0, -np.sin(j2), -D2*np.sin(j2)],
        [0, 1, 0, 0],
        [np.sin(j2), 0, np.cos(j2), D2*np.cos(j2)],
        [0, 0, 0, 1]
    ])

    T2_3 = np.array([
        [np.cos(j3), 0, -np.sin(j3), D3*np.cos(j3)-E1*np.sin(j3)],
        [0, 1, 0, 0],
        [np.sin(j3), 0, np.cos(j3), D3*np.sin(j3)+E1*np.cos(j3)],
        [0, 0, 0, 1]
    ])

    T3_4 = np.array([
        [1, 0, 0, 0],
        [0, np.cos(j4), -np.sin(j4), 0],
        [0, np.sin(j4), np.cos(j4), 0],
        [0, 0, 0, 1]
    ])

    T4_5 = np.array([
        [np.cos(j5), 0, -np.sin(j5), D4*np.cos(j5)-E2*np.sin(j5)],
        [0, 1, 0, 0],
        [np.sin(j5), 0, np.cos(j5), D4*np.sin(j5)+E2*np.cos(j5)],
        [0, 0, 0, 1]
    ])

    T5_6 = np.array([
        [1, 0, 0, 0],
        [0, np.cos(j6), -np.sin(j6), 0],
        [0, np.sin(j6), np.cos(j6), 0],
        [0, 0, 0, 1]
    ])

    T0_6 = np.dot(np.dot(np.dot(np.dot(np.dot(T0_1, T1_2), T2_3), T3_4), T4_5), T5_6)

    # os.system("clear")
    # print(f"X: {T0_6[0][3]}")
    # print(f"Y: {T0_6[1][3]}")
    # print(f"Z: {T0_6[2][3]}")
    # print(f"Roll: {np.arctan2(T0_6[2][1], T0_6[2][2])}")
    # print(f"Pitch: {np.arctan2(-T0_6[2][0], np.sqrt(T0_6[2][1]**2 + T0_6[2][2]**2))}")
    # print(f"Yaw: {np.arctan2(T0_6[1][0], T0_6[0][0])}")
    return [
        T0_6[0][3],
        T0_6[1][3],
        T0_6[2][3],
        np.arctan2(T0_6[2][1], T0_6[2][2]),
        np.arctan2(-T0_6[2][0], np.sqrt(T0_6[2][1]**2 + T0_6[2][2]**2)),
        np.arctan2(T0_6[1][0], T0_6[0][0])
    ]

if __name__=="__main__":
    t = np.pi/2
    print(joints_to_coordinate(0, 0, 0, 0, -np.pi/2, 0))