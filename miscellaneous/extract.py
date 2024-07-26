import cv2

img = cv2.imread('/home/adrien/Bureau/Ned/compute/data/photos/photo_mickey.jpg', cv2.IMREAD_COLOR)

x1, y1 = 1437, 1310  # Coordonnées du coin supérieur gauche
x2, y2 = 1496, 1335  # Coordonnées du coin inférieur droit

roi = img[y1:y2, x1:x2]

# cv2.imshow('Region of Interest', roi)
# cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite('/home/adrien/Bureau/Ned/compute/data/templates/circle4.jpg', roi)