import gym
import cv2
import numpy as np


def preprocess_atari_crop(I):
  """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
  I = I[35:195]  # crop
  I[I[:, :, 0] == 144, :] = 0  # erase background (background type 1)
  I[I[:, :, 0] == 109, :] = 0  # erase background (background type 2)
  I[157:, :, :] = 0
  #I[I != 0] = 255  # everything else (paddles, ball) just set to 1
  return I


env = gym.make('PongDeterministic-v4')

observation = env.reset()

while True:
    action = np.random.randint(6)
    state, _, done, _ = env.step(action)

    I = preprocess_atari_crop(state)


    for target in [92, 236, 213]:
        coord_tuple = np.where(I[:, :, 0] == target)

        if len(coord_tuple[0]) > 0 and len(coord_tuple[1]) > 0:
            y = np.mean(coord_tuple[0])
            x = np.mean(coord_tuple[1])
            cv2.circle(I, (int(x), int(y)), 2, (255, 0, 0), -1)

    cv2.imshow('Image', I)
    cv2.waitKey(0)

    if done:
        state = env.reset()
