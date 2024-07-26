import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Input
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

points_cibles = np.array([
    [1793, 827, 1833, 827, 1793, 873, 1833, 873],
    [1413, 790, 1456, 790, 1413, 849, 1456, 849],
    [1456, 763, 1522, 763, 1456, 829, 1522, 829],
    [1856, 815, 1897, 815, 1856, 866, 1897, 866],
    [1420, 811, 1447, 811, 1420, 858, 1447, 858],
    [1768, 839, 1794, 839, 1768, 878, 1794, 878],
    [1196, 1195, 1248, 1195, 1196, 1246, 1448, 1246],
    [1667, 1166, 1716, 1166, 1667, 1209, 1716, 1209],
    [1641, 856, 1695, 856, 1641, 910, 1695, 910],
    [1153, 836, 1207, 836, 1153, 895, 1207, 895],
    [852, 771, 896, 771, 852, 821, 896, 821],
    [1310, 734, 1358, 734, 1310, 796, 1358, 796],
    [1526, 980, 1565, 980, 1526, 1018, 1565, 1018],
    [1179, 976, 1220, 976, 1179, 1014, 1220, 1014],
    [1430, 864, 1487, 864, 1430, 906, 1487, 906],
    [1800, 890, 1838, 890, 1800, 926, 1838, 926],
])
points_cibles = points_cibles / np.array([2592, 1944] * 4)
points_cibles = np.array([[p[0], p[1], p[2], p[5]] for p in points_cibles])

images = [cv2.imread(f"./compute/data/train_photo/photo{i}.jpg", cv2.IMREAD_GRAYSCALE) for i in range(1, len(points_cibles)+1)]
images = [cv2.resize(image, (int(2592/4), int(1944/4))) for image in images]
images = np.array(images) / 255.0

points_cibles *= 10
images *= 10

X_train, X_test, y_train, y_test = train_test_split(images, points_cibles, test_size=0.2)

def max_absolute_error(y_true, y_pred):
    return tf.reduce_max(tf.abs(y_true - y_pred))

model = Sequential([
    Input(shape=(int(2592/4), int(1944/4), 1)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(4)  # 4 points * 2 coordonnées (x, y) pour chaque point
])

model.compile(optimizer=Adam(learning_rate=0.0001), loss="mean_squared_error")

model.fit(X_train, y_train, epochs=10000, validation_data=(X_test, y_test))

print(model.predict(images[0].reshape(1, int(1944/4), int(2592/4), 1)) * (np.array([2592, 1944] * 4)))
print(points_cibles[0] * (np.array([2592, 1944] * 4)))