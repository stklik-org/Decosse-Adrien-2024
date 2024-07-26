# miscellaneous

This folder contains different programs independant from the main programs, but which are tools which can be used or attempts.

### `extract.py`

This script take a photo and extract a part by defining a square.

### `joints_to_coordinate.py`

Computes the theorical position and orientation of the arm given the angles made by each joints. Some approximation have been made in the shape or in the length, due to an inexisting datas (they might exists somewhere, but have usually been replaced by the Ned 2's one).

### `solver.py`

Have been made in first in order to test and verify the mathematical models before divided it into more formal programs.

### `train.py`

This is an attempt to use machine learning in order to detect the position of the markers or of the arm. The results wheren't concluding enough, and demands lot of datas and computation (due to the size of the photo).

### `unfish.py`

A program in order to transform a photo taken by the raspberry pi into a photo without the deformation due to the geometry of a fisheye.