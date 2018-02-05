from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time

import numpy as np
from  skimage.color import rgb2gray
from skimage.transform import resize
import imutils
import cv2

def get_log_dir(log_base, suffix):
    """
    For each run create a direction
    :param log_base:
    :param suffix:
    :return:
    """
    run = time.strftime('%Y-%m-%d_%H-%M-%S')
    log_dir = '%s/%s_%s' % (log_base, run, suffix)
    os.mkdir(log_dir)
    return log_dir


def preprocess_cartpole(frame):
    """
    add dimension...
    :param frame:
    :return:
    """
    return frame[np.newaxis,:,np.newaxis]


def preprocess_atari(frame):
    """
    Preprocess the image as suggested on kharpathys blog
    :param state:
    :return:
    """
    frame = frame[35:195,:]
    frame = rgb2gray(frame)*255.
    frame = resize(frame,(84,84))
    return frame[:,:,np.newaxis].astype(np.uint8)


def preprocess_atari_no_crop(frame):
    """
    Preprocess the image as suggested on kharpathys blog
    :param state:
    :return:
    """
    frame = rgb2gray(frame)*255.
    frame = resize(frame,(84,84))
    return frame[:,:,np.newaxis].astype(np.uint8)


def preprocess_atari_crop(I):
  """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
  I = I[35:195]  # crop
  I[I[:, :, 0] == 144, :] = 0  # erase background (background type 1)
  I[I[:, :, 0] == 109, :] = 0  # erase background (background type 2)
  I[157:, :, :] = 0
  #I[I != 0] = 255  # everything else (paddles, ball) just set to 1
  return I


def scale_coord(coord):
    """ Scale coordinates in range 0 ~ 255 """
    return coord / 79.5 - 1.0
    return coord

def preprocess_atari_object(image):
     # load the image, convert it to grayscale, blur it slightly,
    # and threshold it

    I = preprocess_atari_crop(image)

    #object_coord = np.array([1., 1., 1., 1., 1., 1.])
    object_coord = np.zeros((9, ))

    for idx, target in enumerate([92, 236, 213]):
        coord_tuple = np.where(I[:, :, 0] == target)

        if len(coord_tuple[0]) > 0 and len(coord_tuple[1]) > 0:
            object_coord[idx*3] = scale_coord(np.mean(coord_tuple[1]))
            object_coord[idx*3+1] = scale_coord(np.mean(coord_tuple[0]))
            object_coord[idx*3+2] = 1




            '''
            if idx == 1:
                object_coord[idx*2] = object_coord[0] - scale_coord(np.mean(coord_tuple[1]))
                object_coord[idx*2 + 1] = object_coord[1] - scale_coord(np.mean(coord_tuple[0]))

            else:
                object_coord[idx*2] = scale_coord(np.mean(coord_tuple[1]))
                object_coord[idx*2 + 1] = scale_coord(np.mean(coord_tuple[0]))
            '''

    '''
    image = preprocess_atari_crop(image)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]

    # find contours in the thresholded image
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)

    cnts = cnts[0] if imutils.is_cv2() else cnts[1]

    # This vector stores the coordinates of the two paddles and the ball
    object_coord = np.array([0., 0., 0., 0., 0., 0.])

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
            object_coord[0] = scale_coord(cX)
            object_coord[1] = scale_coord(cY)

        # Object is right paddle
        elif cX > 139 and cX < 143 and y_range > 4:
            object_coord[4] = scale_coord(cX)
            object_coord[5] = scale_coord(cY)

        # Object is ball
        else:
            object_coord[2] = scale_coord(cX)
            object_coord[3] = scale_coord(cY)

        '''


    # Return the object coordinates
    return object_coord.reshape((1, 9, 1))
