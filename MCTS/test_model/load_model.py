import ipdb
from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf

# Generate some data with noise
N = 10
w, b = 3.0, -1.0
X = np.linspace(0, 10, N)
Y = w * X + b

# Saved Variables
#   wb (weights), ph_x, ph_y, pred, loss

# Load model
with tf.Session() as sess:
  saver = tf.train.import_meta_graph('test_model.meta')
  saver.restore(sess, tf.train.latest_checkpoint('./'))
  g = sess.graph

  ph_x = g.get_tensor_by_name('ph_x:0')
  ph_y = g.get_tensor_by_name('ph_y:0')
  pred = g.get_tensor_by_name('pred:0')
  loss = g.get_tensor_by_name('loss:0')

  pred_val = sess.run(pred, feed_dict={ph_x: X})
  print('pred:', pred_val)
  print('true:', Y)
  print('loss:', ((Y-pred_val)**2).mean())


