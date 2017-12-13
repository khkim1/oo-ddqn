import os

CLASSES = ['marine', 'man', 'shark', 'foe', 'missle', 'sharkman']
OUTPUT_DIR = 'output'
WEIGHT_DIR = 'yolo/weights/save.ckpt-251721'
#
# model parameter
#
ALPHA = 0.1
IMAGE_SIZE = 84

CELL_SIZE = 10

BOXES_PER_CELL = 1

LOSS_AVERAGE = 0.999

OBJECT_SCALE = 1.0
NOOBJECT_SCALE = 1.0
CLASS_SCALE = 2.0
COORD_SCALE = 5.0


GPU = ''

LEARNING_RATE = 1e-4
LEARNING_RATE_MIN = 1e-7
DECAY_RATE = 0.96
STAIRCASE = True



BATCH_SIZE = 16
MAX_BATCH = {'train': 27969, 'dev': 136}

MAX_EPOCH = 12

SUMMARY_ITER = 100

SAVE_ITER = 27969


#
# test parameter
#

THRESHOLD = 0.05

IOU_THRESHOLD = 0.5
