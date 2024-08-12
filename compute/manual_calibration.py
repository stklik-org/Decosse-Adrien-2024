import cv2
import numpy as np

selected_point = None
L_target = []
img = None

## when calibrating the camera, the position of the markers on the image can be manually selected

def select_point(event, x, y, flags, param): # processing the modification of the position of one marker
    global selected_point, L_target
    if event == cv2.EVENT_LBUTTONUP and selected_point is None: # select the closest point when clicking
        for i, point in enumerate(L_target):
            if abs(x - point[0]) < 10 and abs(y - point[1]) < 10: # but only if close enough
                selected_point = i
                break
    elif event == cv2.EVENT_LBUTTONUP: # click again to stop modifying the point
        selected_point = None
    elif event == cv2.EVENT_MOUSEMOVE: # move the point when moving the mouse
        if selected_point is not None:
            L_target[selected_point] = (x, y)
    redraw_lines()

def redraw_lines(): # redraw the line after modification
    global L_target, img, selected_point
    img_copy = img.copy()
    if selected_point is not None: # zoom when a point is selected in order to get a better precision
        x, y = L_target[selected_point]
        zoom_size = 100  # size of the zoomed region
        try:
            zoom_img = img[max(0, y-zoom_size//2):min(img.shape[0], y+zoom_size//2), max(0, x-zoom_size//2):min(img.shape[1], x+zoom_size//2)]
            zoom_img = cv2.resize(zoom_img, (zoom_size*2, zoom_size*2))
            img_copy[max(0, y-zoom_size):min(img.shape[0], y+zoom_size), max(0, x-zoom_size):min(img.shape[1], x+zoom_size)] = zoom_img
        except: # in the case the zoomed square exceeds the image
            pass
    cv2.line(img_copy, (int(L_target[0][0]), int(L_target[0][1])), (int(L_target[1][0]), int(L_target[1][1])), (0, 0, 255), 1)
    cv2.line(img_copy, (int(L_target[1][0]), int(L_target[1][1])), (int(L_target[3][0]), int(L_target[3][1])), (0, 0, 255), 1)
    cv2.line(img_copy, (int(L_target[3][0]), int(L_target[3][1])), (int(L_target[2][0]), int(L_target[2][1])), (0, 0, 255), 1)
    cv2.line(img_copy, (int(L_target[2][0]), int(L_target[2][1])), (int(L_target[0][0]), int(L_target[0][1])), (0, 0, 255), 1)
    cv2.imshow('Matched Areas', img_copy) # and drawing it, of course

def manual_calibration(img_, L_target_): # open the image and proceed to the manual calibration
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