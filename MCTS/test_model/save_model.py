import ipdb
from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf

# Generate some data with noise
N = 101
w, b = 3.0, -1.0
X = np.linspace(0, 10, N)
Y = w * X + b + np.random.randn(N)

print('X & Y Mean: %.3f & %.3f' % (X.mean(), Y.mean()))
print('MSE (mean): %.3f' % np.mean((Y.mean()-Y)**2))
print('MSE (regr): %.3f' % np.mean((w*X+b - Y)**2))

# Define model
wb = tf.get_variable('wb', shape=[2, 1], dtype=tf.float32)
ph_x = tf.placeholder(tf.float32, shape=[None], name='ph_x')
ph_y = tf.placeholder(tf.float32, shape=[None], name='ph_y')
x_with_bias = tf.stack([ph_x, tf.ones_like(ph_x)], axis=1)
pred = tf.reshape(tf.matmul(x_with_bias, wb), [-1], name='pred')
loss = tf.reduce_mean((ph_y-pred)**2, name='loss')

# Train model
adam = tf.train.AdamOptimizer(learning_rate=0.5)
train_op = adam.minimize(loss)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
sess.run(tf.assign(wb, [[0.0],[0.0]]))
for it in range(100):
  _, lval, wbval = sess.run(
      [train_op, loss, wb],
      feed_dict={ph_x: X, ph_y: Y})
  wval, bval = wbval
  if it % 30 == 0 :
    print('It %03d loss %.3f w %.3f b %.3f' % (it, lval, wval, bval))
print('It %03d loss %.3f w %.3f b %.3f' % (it, lval, wval, bval))

# Save model
tf.train.Saver(tf.trainable_variables()).save(sess, 'test_model')
print('Model saved!')
