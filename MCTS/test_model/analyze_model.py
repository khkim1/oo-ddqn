import os, sys
import ipdb
from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf


def main():
  prefix = sys.argv[1]

# Load model
  with tf.Session() as sess:
    saver = tf.train.import_meta_graph(prefix + '.meta')
    saver.restore(sess, tf.train.latest_checkpoint(os.path.dirname(prefix)))
    graph = sess.graph

    print('\n\n---- Model Variables ----\n')
    for x in graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
      print(x)

    print('\n\n---- Model Ops ----\n')
    for x in graph.get_operations():
      print(x.name)

if __name__ == '__main__':
  main()
