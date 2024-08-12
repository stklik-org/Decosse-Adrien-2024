import cv2
import numpy as np
import subprocess
import cProfile

def get_center_green_cube(
        image,
        rh: int,
        display: bool=False
    ) -> tuple[float, float]:
    """
    :param image: the image in wich the green square must be found
    :param rh: distance from the center of the horizon in pixel
    :param display: if True, show the image with the found square
    """

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    width, height, _ = image.shape

    lower_green = np.array([40, 110, 60]) # the range of the detected green in the image in the hsv format
    upper_green = np.array([70, 255, 255]) # change those values in order to calibrate the detection

    mask = cv2.inRange(hsv, lower_green, upper_green)

    green_image = cv2.bitwise_and(image, image, mask=mask)
    green_image = cv2.cvtColor(green_image, cv2.COLOR_BGR2HSV) # the selected composante (green here) of the image

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # find contour of the detection

    image_center = np.array([height/2, width/2])
    filtered_contours = []
    for contour in contours: # filtering the noises
        inside = True
        center_contour = np.mean(contour, axis=0)[0]
        if np.linalg.norm(center_contour - image_center) > rh/2.5: # remove detection too far from the center
            inside = False

        if inside and cv2.contourArea(contour) > 30*rh/1944: # remove very small detection
            filtered_contours.append(contour)
            # print(center_contour, np.linalg.norm(center_contour - image_center), rh/3)
    # print(image_center)

    for contour in filtered_contours: # change contour for rectangles
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.intp(box)
        cv2.drawContours(image, [box], 0, (255, 0, 0), 2)

    if filtered_contours == []:
        raise Exception("No green square found") # sad situation, must change lower_green and upper_green
    merged_contours = np.concatenate(filtered_contours) # detecting oonly one object
    hull = cv2.convexHull(merged_contours)

    rect = cv2.minAreaRect(hull)
    hull = cv2.boxPoints(rect)
    hull = np.intp(hull)
    cv2.drawContours(image, [hull], 0, (0, 0, 255), 2)
    (center_x, center_y), (width, height), _ = rect
    cv2.circle(image, (int(center_x), int(center_y)), 2, (255, 0, 0), -1) # draw all the result in the image

    if display: # for a verification of the result
        cv2.namedWindow('Green Image', cv2.WINDOW_NORMAL)
        cv2.imshow('Green Image', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return center_x-width/2, center_y-height/2 # return the center of the found square

def main(log=False): # detect the green square on the image
    results = []
    for i in range(100):
        print(f"Photo number {i+1}")
        redo = True
        nb = 1
        while redo:
            try:
                subprocess.check_call(f"~/Bureau/Ned/photo.sh mickey", shell=True)
                image = cv2.imread('/home/adrien/Bureau/photo.jpg') # must be change by the wanted path
                results.append(get_center_green_cube(image, len(image), display=log))
                redo = False
            except Exception as e:
                print(e)
                print(f"Error - {nb}-th redo of the photo number {i+1}")
                nb += 1

    mean = (0, 0)
    for x, y in results:
        mean = (mean[0]+x, mean[1]+y)
    mean = (mean[0]/len(results), mean[1]/len(results))

    sigma_x = 0
    sigma_y = 0
    for x, y in results:
        sigma_x += (mean[0]-x)**2
        sigma_y += (mean[1]-y)**2
    sigma_x = np.sqrt(sigma_x/len(results))
    sigma_y = np.sqrt(sigma_y/len(results))

    print(f"mean={mean}")
    print(f"sigma_x={sigma_x}")
    print(f"sigma_y={sigma_y}")

if __name__=="__main__":
    cProfile.run('main(log=True)',sort='tottime')