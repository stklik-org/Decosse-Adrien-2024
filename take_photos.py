import numpy as np
import pyniryo
import os
import matplotlib.pyplot as plt
import cv2


ned = pyniryo.NiryoRobot("169.254.200.200")
ned.calibrate_auto()
alpha = 0.25
ned.move_joints([0, -alpha, -3.141592/4, 0, 3.141592/4+alpha, 0])

def click_event(event, x, y, flags, param):
    global points, img
    if event == cv2.EVENT_LBUTTONDBLCLK and len(points) < 2:
        points.append((x, y))
        cv2.circle(img, (x, y), 5, (255, 0, 0), -1)
        cv2.imshow("image", img)

f = open('compute/logs/inside.log', 'w')

L_points = []
for _ in range(10):
    ned.move_joints([3.141592/4, -alpha, -3.141592/4, 0, 3.141592/4+alpha, 0])
    ned.move_joints([0, -alpha, -3.141592/4, 0, 3.141592/4+alpha, 0])

    img = ned.get_img_compressed()
    img = pyniryo.uncompress_image(img)
    points = []
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.imshow("image", img)
    cv2.setMouseCallback("image", click_event)

    while True:
        if cv2.waitKey(1) & 0xFF == 13:
            break

    cv2.destroyAllWindows()

    L_points.append(points)
    f.write(f"{L_points[-1][0][0], L_points[-1][0][1], L_points[-1][1][0], L_points[-1][1][1]}\n")

# print(L_points)
f.close()

# plt.imsave('compute/data/niryo_photo/photo.png', img)