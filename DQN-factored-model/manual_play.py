import gym
from matplotlib import pyplot as plt

env = gym.make('PongDeterministic-v4')
env.reset()

while True:
    action = int(input("input action: "))
    state, _, _, _ = env.step(action)
    #plt.imshow(state)
    env.render()
