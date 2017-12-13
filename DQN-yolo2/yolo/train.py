import tensorflow as tf
import datetime
import os
import argparse
import config as cfg
from yolo import YOLONet
import data_reader as reader
from timer import Timer
import numpy as np

class Solver(object):

    def __init__(self, net, data_train, data_dev):
        self.net = net
        self.data = data_train
        self.data_dev = data_dev
        self.dev_trial = 10 # Trial to calculate loss of valdiation set

        self.max_epoch = cfg.MAX_EPOCH
        self.max_iter   = self.max_epoch * cfg.MAX_BATCH['train']
        self.initial_learning_rate = cfg.LEARNING_RATE
        self.decay_rate = cfg.DECAY_RATE
        self.staircase = cfg.STAIRCASE
        self.decay_steps = np.log(self.decay_rate)/np.log(cfg.LEARNING_RATE_MIN/cfg.LEARNING_RATE) * cfg.MAX_EPOCH * cfg.MAX_BATCH['train']
        self.summary_iter = cfg.SUMMARY_ITER
        self.save_iter = cfg.SAVE_ITER
        self.weights_file = cfg.WEIGHT_DIR
        self.loss_average = cfg.LOSS_AVERAGE
        self.output_dir = os.path.join(cfg.OUTPUT_DIR)
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        self.save_cfg() #Saving configuration file

        self.variable_to_restore = tf.global_variables()
        self.restorer = tf.train.Saver(self.variable_to_restore, max_to_keep=None)
        self.saver = tf.train.Saver(self.variable_to_restore, max_to_keep=None)
        self.ckpt_file = os.path.join(self.output_dir, 'save.ckpt')
        self.summary_op = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter(self.output_dir, flush_secs=60)

        self.global_step = tf.get_variable(
            'global_step', [], initializer=tf.constant_initializer(0), trainable=False)
        self.learning_rate = tf.train.exponential_decay(
            self.initial_learning_rate, self.global_step, self.decay_steps,
            self.decay_rate, self.staircase, name='learning_rate')
        self.optimizer = tf.train.AdamOptimizer(
            learning_rate=self.learning_rate).minimize(
            self.net.total_loss, global_step=self.global_step)
        self.ema = tf.train.ExponentialMovingAverage(decay=0.9999)
        self.averages_op = self.ema.apply(tf.trainable_variables())
        with tf.control_dependencies([self.optimizer]):
            self.train_op = tf.group(self.averages_op)

        self.sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
        self.sess.run(tf.global_variables_initializer())

        if self.weights_file is not None:
            print('Restoring weights from: ' + self.weights_file)
            self.restorer.restore(self.sess, self.weights_file)

        self.writer.add_graph(self.sess.graph)

    def train(self):

        train_timer = Timer()
        step = 0

        train_loss = 0
        valid_loss = 0
        beta = self.loss_average

        while self.data.get_epoch() < self.max_epoch:
            step += 1
            images, labels = self.data.get_batch() # Loading data
            feed_dict = {self.net.images: images, self.net.labels: labels} # Train all at one time!!!

            if step % self.summary_iter == 0:
                if step % (self.summary_iter * 10) == 0:
                    train_timer.tic()
                    summary_str, temp_loss, _ = self.sess.run(
                        [self.summary_op, self.net.total_loss, self.train_op],
                        feed_dict=feed_dict)
                    train_timer.toc()
                    train_loss = beta * temp_loss + (1 - beta) * train_loss

                    # Computing validation loss
                    for dev in range(self.dev_trial):
                        images, labels = self.data_dev.get_batch() # Loading data
                        feed_dict = {self.net.images: images, self.net.labels: labels}
                        loss_temp = self.sess.run([self.net.total_loss], feed_dict=feed_dict)[0]
                        valid_loss = beta * loss_temp + (1 - beta) * valid_loss

                    log_str = ('{} Epoch: {}, Step: {}, Learning rate: {},'
                        ' Loss: {:5.3f} Loss Val: {:5.3f} Speed: {:.3f}s/iter,'
                        ' Remain: {}').format(
                        datetime.datetime.now().strftime('%m/%d %H:%M:%S'),
                        self.data.get_epoch(),
                        int(step),
                        round(self.learning_rate.eval(session=self.sess), 6),
                        train_loss,
                        valid_loss,
                        train_timer.average_time,
                        train_timer.remain(step, self.max_iter))
                    print(log_str)
                else:
                    train_timer.tic()
                    summary_str, _ = self.sess.run(
                        [self.summary_op, self.train_op],
                        feed_dict=feed_dict)
                    train_timer.toc()
                self.writer.add_summary(summary_str, step)
            else:
                train_timer.tic()
                _, temp_loss= self.sess.run([self.train_op, self.net.total_loss], feed_dict=feed_dict)
                train_loss = beta * temp_loss + (1 - beta) * train_loss
                train_timer.toc()
            if step % self.save_iter == 0:
                print('{} Saving checkpoint file to: {}'.format(
                    datetime.datetime.now().strftime('%m/%d %H:%M:%S'),
                    self.output_dir))
                self.saver.save(self.sess, self.ckpt_file,
                                global_step=self.global_step)

    def save_cfg(self):

        with open(os.path.join(self.output_dir, 'config.txt'), 'w') as f:
            cfg_dict = cfg.__dict__
            for key in sorted(cfg_dict.keys()):
                if key[0].isupper():
                    cfg_str = '{}: {}\n'.format(key, cfg_dict[key])
                    f.write(cfg_str)



def main():


    #os.environ['CUDA_VISIBLE_DEVICES'] = cfg.GPU

    yolo = YOLONet()
    data_train = reader.batch_reader('train')
    data_dev = reader.batch_reader('dev')
    solver = Solver(yolo, data_train, data_dev)

    print('Start training ...')
    solver.train()
    print('Done training.')

if __name__ == '__main__':

    # python train.py --weights YOLO_small.ckpt --gpu 0
    main()