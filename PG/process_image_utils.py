import imutils
import cv2
import numpy as np


def get_object_coordinates(image):
    # load the image, convert it to grayscale, blur it slightly,
    # and threshold it
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]

    # find contours in the thresholded image
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)

    cnts = cnts[0] if imutils.is_cv2() else cnts[1]

    # This vector stores the coordinates of the two paddles and the ball
    object_coord = [0, 0, 0, 0, 0, 0]

    # loop over the contours
    for c in cnts:

        # Compute the width of the object
        flatten_c = np.ndarray.flatten(np.array(c))
        y_coords = flatten_c[1::2]
        y_range = np.ptp(y_coords)

        # Compute the center coordinates (0 ~ 255) of each object
        M = cv2.moments(c)
        eps = 1e-6
        cX = int(M["m10"] / (M["m00"] + eps))
        cY = int(M["m01"] / (M["m00"] + eps))

        # Manual thresholding to identify the object

        # Object is left paddle
        if cX < 19 and cX > 15 and y_range > 4:
            object_coord[0] = cX
            object_coord[1] = cY

        # Object is right paddle
        elif cX > 139 and cX < 143 and y_range > 4:
            object_coord[4] = cX
            object_coord[5] = cY

        # Object is ball
        else:
            object_coord[2] = cX
            object_coord[3] = cY

    # Return the object coordinates
    return np.array(object_coord)
