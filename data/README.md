# Local data

Use real footage for the final submission. A phone camera aimed at boxes moving across a table is acceptable.

Recommended capture:

1. Lock the camera and use the same resolution and focus for the video and checkerboard images.
2. Capture 10–15 checkerboard views with the board visible from different angles. Tilt matters: if every view is taken head-on, the calibration is under-determined and the focal length is meaningless. Tilt the board roughly 20–45° and rotate it around its own axis across views. `calibrate.py` prints the view spread and warns when it is below 10°.
3. Capture at least one checkerboard image with the board lying flat on the belt plane. The plane homography is fitted to the first view only, so that view defines the world scale. Measuring a box that is not on that plane carries a few percent error.
4. Record 15–30 seconds of at least three box types, including separated and touching boxes.
5. Record the real width and height of one reference box in centimeters.
6. Put crops into `recognition/<class_name>/` for Module E.

`tools/make_test_video.py` creates a synthetic smoke-test clip only. Do not use it as final project footage.
