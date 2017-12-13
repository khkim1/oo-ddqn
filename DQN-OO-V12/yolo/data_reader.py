import os
import numpy as np
import cv2
import pickle as cPickle
import copy
import yolo.config as cfg


class batch_reader():
    def __init__(self, name):
        self.name = name
        self.batch_size = cfg.BATCH_SIZE
        self.image_size = cfg.IMAGE_SIZE
        self.cell_size = cfg.CELL_SIZE
        self.classes = cfg.CLASSES
        self.num_classes = len(cfg.CLASSES)
        self.max_batch = cfg.MAX_BATCH[name]
        self.cursor = 0
        self.epoch = 1

    def get_epoch(self):
        return self.epoch
    def get_batch(self):
        images = np.zeros((self.batch_size, self.image_size, self.image_size, 3))
        labels = np.zeros((self.batch_size, self.cell_size, self.cell_size, self.num_classes + 5))
        count = 0
        batch = cPickle.load(open('data/%s_set/batch_%d'%(self.name, self.cursor), 'rb'))
        while count < self.batch_size:
            images[count, :, :, :] = batch[count]['img']/255.0
            labels[count, :, :, :] = self.gt_labels(batch[count]['obj'])
            count += 1
        self.cursor += 1
        if self.cursor >= self.max_batch:
            self.epoch += 1
            self.cursor = 0
        return images, labels

    def gt_labels(self, obj):

        label = np.zeros((self.cell_size, self.cell_size, self.num_classes + 5))

        num_obj = obj.shape[1]
        for i in range(num_obj):
            cls_ind, x, y, w, h = obj[:, i]
            boxes = [x + w/2.0, y + h/2.0, w, h]
            x_ind = int(boxes[0] * self.cell_size / self.image_size)
            y_ind = int(boxes[1] * self.cell_size / self.image_size)
            if label[y_ind, x_ind, 0] == 1:
                continue
            label[y_ind, x_ind, 0] = 1
            label[y_ind, x_ind, 1:5] = boxes
            label[y_ind, x_ind, 5 + int(cls_ind)] = 1
        return label
