import cv2
import numpy as np

selected_point = None
L_target = []
img = None

def select_point(event, x, y, flags, param):
    global selected_point, L_target
    if event == cv2.EVENT_LBUTTONUP and selected_point is None:
        for i, point in enumerate(L_target):
            if abs(x - point[0]) < 10 and abs(y - point[1]) < 10:
                selected_point = i
                break
    elif event == cv2.EVENT_LBUTTONUP:
        selected_point = None
    elif event == cv2.EVENT_MOUSEMOVE:
        if selected_point is not None:
            L_target[selected_point] = (x, y)
    redraw_lines()

def redraw_lines():
    global L_target, img, selected_point
    img_copy = img.copy()
    if selected_point is not None:
        x, y = L_target[selected_point]
        zoom_size = 100  # size of the zoomed region
        try:
            zoom_img = img[max(0, y-zoom_size//2):min(img.shape[0], y+zoom_size//2), max(0, x-zoom_size//2):min(img.shape[1], x+zoom_size//2)]
            zoom_img = cv2.resize(zoom_img, (zoom_size*2, zoom_size*2))
            img_copy[max(0, y-zoom_size):min(img.shape[0], y+zoom_size), max(0, x-zoom_size):min(img.shape[1], x+zoom_size)] = zoom_img
        except:
            pass
    cv2.line(img_copy, (int(L_target[0][0]), int(L_target[0][1])), (int(L_target[1][0]), int(L_target[1][1])), (0, 0, 255), 1)
    cv2.line(img_copy, (int(L_target[1][0]), int(L_target[1][1])), (int(L_target[3][0]), int(L_target[3][1])), (0, 0, 255), 1)
    cv2.line(img_copy, (int(L_target[3][0]), int(L_target[3][1])), (int(L_target[2][0]), int(L_target[2][1])), (0, 0, 255), 1)
    cv2.line(img_copy, (int(L_target[2][0]), int(L_target[2][1])), (int(L_target[0][0]), int(L_target[0][1])), (0, 0, 255), 1)
    cv2.imshow('Matched Areas', img_copy)

def manual_calibration(img_, L_target_):
    global selected_point, L_target, img
    selected_point = None
    L_target = L_target_
    img = img_
    cv2.namedWindow('Matched Areas', cv2.WINDOW_NORMAL)
    cv2.setMouseCallback('Matched Areas', select_point)

    while True:
        redraw_lines()
        if cv2.waitKey(0) & 0xFF == 13:
            break

    cv2.destroyAllWindows()
    return L_target