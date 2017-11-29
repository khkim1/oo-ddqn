""" Trains an agent with (stochastic) Policy Gradients on Pong. Uses OpenAI Gym. """
import numpy as np
import cPickle as pickle
import gym


from matplotlib import pyplot as plt
import scipy.misc
from process_image_utils import get_object_coordinates
import cv2
from PIL import Image

# hyperparameters
H = 200  # number of hidden layer neurons
batch_size = 10  # every how many episodes to do a param update?
learning_rate = 1e-4
gamma = 0.99  # discount factor for reward
decay_rate = 0.99  # decay factor for RMSProp leaky sum of grad^2
resume = False  # resume from previous checkpoint?
render = False

# model initialization
D = 80 * 80  # input dimensionality: 80x80 grid
if resume:
  model = pickle.load(open('save.p', 'rb'))
else:
  model = {}
  model['W1'] = np.random.randn(H, D) / np.sqrt(D)  # "Xavier" initialization
  model['W2'] = np.random.randn(H) / np.sqrt(H)

grad_buffer = {k: np.zeros_like(v) for k, v in model.iteritems()}  # update buffers that add up gradients over a batch
rmsprop_cache = {k: np.zeros_like(v) for k, v in model.iteritems()}  # rmsprop memory


def sigmoid(x):
  return 1.0 / (1.0 + np.exp(-x))  # sigmoid "squashing" function to interval [0,1]


def prepro(I):
  """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
  I = I[35:195]  # crop
  # I = I[::2, ::2, 0]  # downsample by factor of 2
  #I = I[:, :, 0]

  I[I[:, :, 0] == 144, :] = 0  # erase background (background type 1)
  I[I[:, :, 0] == 109, :] = 0  # erase background (background type 2)
  I[I != 0] = 255  # everything else (paddles, ball) just set to 1
  return I


def discount_rewards(r):
  """ take 1D float array of rewards and compute discounted reward """
  discounted_r = np.zeros_like(r)
  running_add = 0
  for t in reversed(xrange(0, r.size)):
    if r[t] != 0:
      running_add = 0  # reset the sum, since this was a game boundary (pong specific!)
    running_add = running_add * gamma + r[t]
    discounted_r[t] = running_add
  return discounted_r


def policy_forward(x):
  h = np.dot(model['W1'], x)
  h[h < 0] = 0  # ReLU nonlinearity
  logp = np.dot(model['W2'], h)
  p = sigmoid(logp)
  return p, h  # return probability of taking action 2, and hidden state


def policy_backward(eph, epdlogp):
  """ backward pass. (eph is array of intermediate hidden states) """
  dW2 = np.dot(eph.T, epdlogp).ravel()
  dh = np.outer(epdlogp, model['W2'])
  dh[eph <= 0] = 0  # backpro prelu
  dW1 = np.dot(dh.T, epx)
  return {'W1': dW1, 'W2': dW2}


env = gym.make("Pong-v0")

observation = env.reset()

for i in range(60):
  observation, reward, done, info = env.step(1)

  I = prepro(observation)

  object_coord = get_object_coordinates(I)

  print(object_coord)

  if object_coord is not None:
    for idx in range(3):
      cv2.circle(I, (object_coord[2 * idx], object_coord[2 * idx + 1]), 1, (255, 0, 0), -1)
    cv2.imshow('Image', I)
    cv2.waitKey(0)


#scipy.misc.imsave('figure1.png', prepro(observation))
