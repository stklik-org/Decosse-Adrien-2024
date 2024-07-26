import numpy as np

from compute.compute_image_to_space import get_direction_vector

index = 0

A = np.array([1392, 1321])
B = np.array([1361, 1198])
A_ = np.array([1708, 1282])
B_ = np.array([1587, 1186])

uA = get_direction_vector(A[0]-2592/2, 1944/2-A[1], 1944/2)[:3]
uB = get_direction_vector(B[0]-2592/2, 1944/2-B[1], 1944/2)[:3]
uA_ = get_direction_vector(A_[0]-2592/2, 1944/2-A_[1], 1944/2)[:3]
uB_ = get_direction_vector(B_[0]-2592/2, 1944/2-B_[1], 1944/2)[:3]

uA = uA/uA[1]
uB = uB/uB[1]
uA_ = uA_/uA_[1]
uB_ = uB_/uB_[1]

A = np.array([uA[0], uA[2]])
B = np.array([uB[0], uB[2]])
A_ = np.array([uA_[0], uA_[2]])
B_ = np.array([uB_[0], uB_[2]])

M = np.array([
    [B[0]-A[0], -(B_[0]-A_[0])],
    [B[1]-A[1], -(B_[1]-A_[1])]
])

L = np.linalg.inv(M).dot(np.array([A_[0]-A[0], A_[1]-A[1]]))

# print(A+L[0]*(B-A))

O = A+L[0]*(B-A)
O = np.array([O[0], 1, O[1]])
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
])

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
]).dot(T)

gamma = [np.pi, -np.pi/2, 0, np.pi/2][index]

T = np.array([
    [np.cos(gamma), np.sin(gamma), 0, 0],
    [-np.sin(gamma), np.cos(gamma), 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
]).dot(T)

A = np.array([1392, 1321])
B = np.array([1361, 1198])
A_ = np.array([1708, 1282])
B_ = np.array([1587, 1186])
print(T)

print(f"A: {A[0]-2592/2, 1944/2-A[1]}")
print(f"B: {B[0]-2592/2, 1944/2-B[1]}")
print(f"A_: {A_[0]-2592/2, 1944/2-A_[1]}")
print(f"B_: {B_[0]-2592/2, 1944/2-B_[1]}")

uA = get_direction_vector(A[0]-2592/2, 1944/2-A[1], 1944/2)[:3]
uB = get_direction_vector(B[0]-2592/2, 1944/2-B[1], 1944/2)[:3]
uA_ = get_direction_vector(A_[0]-2592/2, 1944/2-A_[1], 1944/2)[:3]
uB_ = get_direction_vector(B_[0]-2592/2, 1944/2-B_[1], 1944/2)[:3]

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

O = np.array([0.164, -0.091, 0])
A = np.array([0.167, 0.084, 0])
B = np.array([0.338, -0.093, 0])
C = np.array([0.338, 0.078, 0])


n = np.cross(uO[0:3], uA[0:3])
n1 = np.cross(n, uA[0:3])
n2 = np.cross(n, uO[0:3])
P12 = A - O

q1 = uO[0:3].dot(n1)
q2 = uA[0:3].dot(n2)

assert q1 != 0 and q2 != 0, 'u1 and u2 must not be parallel'

mu1 = P12.dot(n1)/q1
mu2 = -P12.dot(n2)/q2

O1 = O + mu1*uO[0:3]
O2 = A + mu2*uA[0:3]

print(O1)
print(O2)

n = np.cross(uO[0:3], uB[0:3])
n1 = np.cross(n, uB[0:3])
n2 = np.cross(n, uO[0:3])
P12 = B - O

q1 = uO[0:3].dot(n1)
q2 = uB[0:3].dot(n2)

assert q1 != 0 and q2 != 0, 'u1 and u2 must not be parallel'

mu1 = P12.dot(n1)/q1
mu2 = -P12.dot(n2)/q2

O1 = O + mu1*uO[0:3]
O2 = B + mu2*uB[0:3]

print(O1)
print(O2)

n = np.cross(uO[0:3], uC[0:3])
n1 = np.cross(n, uC[0:3])
n2 = np.cross(n, uO[0:3])
P12 = C - O

q1 = uO[0:3].dot(n1)
q2 = uC[0:3].dot(n2)

assert q1 != 0 and q2 != 0, 'u1 and u2 must not be parallel'

mu1 = P12.dot(n1)/q1
mu2 = -P12.dot(n2)/q2

O1 = O + mu1*uO[0:3]
O2 = C + mu2*uC[0:3]

print(O1)
print(O2)

# print(X1)
# print(X2)
# print(X3)