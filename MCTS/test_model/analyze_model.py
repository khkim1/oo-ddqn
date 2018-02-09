import os, sys
import ipdb
from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf

# Generate some data with noise



def main():
  print('\n\n---- Model Variables ----\n')
  prefix = sys.argv[1]

# Load model
  with tf.Session() as sess:
    saver = tf.train.import_meta_graph(prefix + '.meta')
    saver.restore(sess, tf.train.latest_checkpoint(os.path.dirname(prefix)))
    graph = sess.graph

    for x in graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
      print(x)


if __name__ == '__main__':
  main()
