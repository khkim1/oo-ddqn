""" Trains an agent with (stochastic) Policy Gradients on Pong. Uses OpenAI Gym. """
import numpy as np
import cPickle as pickle
import gym
from matplotlib import pyplot as plt


from matplotlib import pyplot as plt
import scipy.misc
from process_image_utils import get_object_coordinates
import cv2


# hyperparameters
H = 200  # number of hidden layer neurons
batch_size = 1  # every how many episodes to do a param update?
learning_rate = 1e-3
gamma = 0.99  # discount factor for reward
decay_rate = 0.99  # decay factor for RMSProp leaky sum of grad^2
resume = False  # resume from previous checkpoint?
render = True

# model initialization
D = 80 * 80  # input dimensionality: 80x80 grid

model = pickle.load(open('trained_models/save.p', 'rb'))

def sigmoid(x):
  return 1.0 / (1.0 + np.exp(-x))  # sigmoid "squashing" function to interval [0,1]


def prepro(I):
  """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
  I = I[35:195]  # crop
  I = I[::2, ::2, 0]  # downsample by factor of 2
  I[I == 144] = 0  # erase background (background type 1)
  I[I == 109] = 0  # erase background (background type 2)
  I[I != 0] = 1  # everything else (paddles, ball) just set to 1
  return I.astype(np.float).ravel()


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
prev_x = None  # used in computing the difference frame
xs, hs, dlogps, drs = [], [], [], []
running_reward = None
reward_sum = 0
episode_number = 0

while True:
  if render:
    env.render()

  # preprocess the observation, set input to network to be difference image
  cur_x = prepro(observation)
  x = cur_x - prev_x if prev_x is not None else np.zeros(D)
  prev_x = cur_x

  #cv2.imshow('Image', observation)
  #cv2.waitKey(1)

  # forward the policy network and sample an action from the returned probability
  aprob, h = policy_forward(x)
  action = 2 if np.random.uniform() < aprob else 3  # roll the dice!

  # step the environment and get new measurements
  observation, reward, done, info = env.step(action)

  if done:
    observation = env.reset()
