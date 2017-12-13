import tensorflow as tf
import numpy as np
import os
import cv2
import yolo.config as cfg
from yolo.yolo import YOLONet
from timer import Timer
import yolo.data_reader as reader
import matplotlib.pyplot as plt


class Detector(object):

    def __init__(self, net, weight_file):
        self.net = net
        self.weights_file = weight_file

        self.classes = cfg.CLASSES
        self.num_class = len(self.classes)
        self.image_size = cfg.IMAGE_SIZE
        self.cell_size = cfg.CELL_SIZE
        self.boxes_per_cell = cfg.BOXES_PER_CELL
        self.threshold = cfg.THRESHOLD
        self.iou_threshold = cfg.IOU_THRESHOLD
        self.boundary1 = self.cell_size * self.cell_size * self.num_class
        self.boundary2 = self.boundary1 + self.cell_size * self.cell_size * self.boxes_per_cell

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        print('Restoring weights from: ' + self.weights_file)
        self.saver = tf.train.Saver()
        self.saver.restore(self.sess, self.weights_file)

    def draw_result(self, img, result, obj):
        font = cv2.FONT_HERSHEY_SIMPLEX
        for i in range(len(result)):
            x = int(result[i][1])
            y = int(result[i][2])
            w = int(result[i][3]/ 2)
            h = int(result[i][4]/ 2)
            cv2.rectangle(img, (x-w, y-h), (x + w, y + h), (10, 10, 10), 1)
            cv2.putText(img,str(result[i][0]), (x + w + 2, y - h), font, 0.3, (20, 250, 10),1,cv2.LINE_AA)
        for s1 in range(self.cell_size):
            for s2 in range(self.cell_size):
                result = obj[s1, s2, :]
                if result[0] == 0:
                    continue
                x = int(result[1])
                y = int(result[2])
                w = int(result[3]/ 2)
                h = int(result[4]/ 2)
                #cv2.rectangle(img, (x-w, y-h), (x + w, y + h) , (20, 250, 10), 1)

    def cut_scale(self, img):
        img = img[30:150, 7:, :] # Cutting image
        img = cv2.resize(img, (84, 84), interpolation = cv2.INTER_AREA)/255.
        return np.expand_dims(img, 0)

    def detect(self, img):
        inputs = self.cut_scale(img)
        net_output = self.sess.run(self.net.logits,
        feed_dict={self.net.images: inputs})
        net_output = net_output.reshape((1, 1100, 1))
        return net_output

    def detect_from_cvmat(self, inputs):
        net_output = self.sess.run(self.net.logits,
                                   feed_dict={self.net.images: inputs})
        results = []
        for i in range(net_output.shape[0]):
            results.append(self.interpret_output(net_output[i]))

        return results

    def interpret_output(self, output):
        probs = np.zeros((self.cell_size, self.cell_size,
                          self.boxes_per_cell, self.num_class))

        scales = np.reshape(output[:, :, 0], (self.cell_size, self.cell_size, self.boxes_per_cell))
        boxes = np.reshape(output[:, :, 1:5], (self.cell_size, self.cell_size, self.boxes_per_cell, 4))
        class_probs = np.reshape(output[:, :, 5:], (self.cell_size, self.cell_size, self.num_class))

        #class_probs = np.reshape(output[0:self.boundary1], (self.cell_size, self.cell_size, self.num_class))
        #scales = np.reshape(output[self.boundary1:self.boundary2], (self.cell_size, self.cell_size, self.boxes_per_cell))
        #boxes = np.reshape(output[self.boundary2:], (self.cell_size, self.cell_size, self.boxes_per_cell, 4))
        offset = np.transpose(np.reshape(np.array([np.arange(self.cell_size)] * self.cell_size * self.boxes_per_cell),
                                         [self.boxes_per_cell, self.cell_size, self.cell_size]), (1, 2, 0))

        boxes[:, :, :, 0] += offset
        boxes[:, :, :, 1] += np.transpose(offset, (1, 0, 2))
        boxes[:, :, :, :2] = 1.0 * boxes[:, :, :, 0:2] / self.cell_size
        boxes[:, :, :, 2:] = np.square(boxes[:, :, :, 2:])

        boxes *= self.image_size

        for i in range(self.boxes_per_cell):
            for j in range(self.num_class):
                probs[:, :, i, j] = np.multiply(
                    class_probs[:, :, j], scales[:, :, i])

        filter_mat_probs = np.array(probs >= self.threshold, dtype='bool')
        filter_mat_boxes = np.nonzero(filter_mat_probs)
        boxes_filtered = boxes[filter_mat_boxes[0],
                               filter_mat_boxes[1], filter_mat_boxes[2]]
        probs_filtered = probs[filter_mat_probs]
        classes_num_filtered = np.argmax(filter_mat_probs, axis=3)[filter_mat_boxes[
            0], filter_mat_boxes[1], filter_mat_boxes[2]]

        argsort = np.array(np.argsort(probs_filtered))[::-1]
        boxes_filtered = boxes_filtered[argsort]
        probs_filtered = probs_filtered[argsort]
        classes_num_filtered = classes_num_filtered[argsort]

        for i in range(len(boxes_filtered)):
            if probs_filtered[i] == 0:
                continue
            for j in range(i + 1, len(boxes_filtered)):
                if self.iou(boxes_filtered[i], boxes_filtered[j]) > self.iou_threshold:
                    probs_filtered[j] = 0.0

        filter_iou = np.array(probs_filtered > 0.0, dtype='bool')
        boxes_filtered = boxes_filtered[filter_iou]
        probs_filtered = probs_filtered[filter_iou]
        classes_num_filtered = classes_num_filtered[filter_iou]

        result = []
        for i in range(len(boxes_filtered)):
            result.append([classes_num_filtered[i], boxes_filtered[i][0], boxes_filtered[
                          i][1], boxes_filtered[i][2], boxes_filtered[i][3], probs_filtered[i]])

        return result

    def iou(self, box1, box2):
        tb = min(box1[0] + 0.5 * box1[2], box2[0] + 0.5 * box2[2]) - \
            max(box1[0] - 0.5 * box1[2], box2[0] - 0.5 * box2[2])
        lr = min(box1[1] + 0.5 * box1[3], box2[1] + 0.5 * box2[3]) - \
            max(box1[1] - 0.5 * box1[3], box2[1] - 0.5 * box2[3])
        if tb < 0 or lr < 0:
            intersection = 0
        else:
            intersection = tb * lr
        return intersection / (box1[2] * box1[3] + box2[2] * box2[3] - intersection)

    def image_detector(self, batch_img, batch_label, wait=0):
        detect_timer = Timer()
        batch_num = batch_img.shape[0]
        for i in range(batch_num):
            image = batch_img[i, :, :, :]
            obj = batch_label[i, :, :, :]
            detect_timer.tic()
            result = self.detect(image)
            detect_timer.toc()
            self.draw_result(image, result, obj)
            plt.imshow(image)
            plt.show()
        print('Average detecting time: {:.3f}s'.format(detect_timer.average_time))



def main():

    #os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    yolo = YOLONet(False)
    detector = Detector(yolo, cfg.WEIGHT_DIR)
    data_dev = reader.batch_reader('dev')
    for i in range(1):
        images, labels = data_dev.get_batch()
        # detect from image file

        detector.image_detector(images, labels)


if __name__ == '__main__':
    main()
