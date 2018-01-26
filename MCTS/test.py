import sys
import ipdb
import numpy as np
from tqdm import trange
from matplotlib import pyplot as plt
import seaborn as sns; sns.palplot(sns.color_palette("husl", 8))

from model import GymModel
from mcts import MCTS

def run_episode(model, mcts, max_horizon=200, action_map=None, render=False):
  mcts.reset()
  total_reward = 0
  done = False
  steps = 0

  while steps < max_horizon and not done:
    steps += 1
    
    # Plan with MCTS
    q_values = mcts.run()
    a, value = max(q_values, key=lambda x: x[1]) 

    # Take a step in the model
    r, done = model.step(a)
    total_reward += r

    if render:
      a = action_map[a] if action_map else str(a)
      print('[Step %05d] action: %s, reward: %f, Q: [%s]' %
          (steps, a, r, ', '.join([str(x[1]) for x in q_values])))
      model.render()

    mcts.reset()

  return total_reward


def main():
  assert len(sys.argv) > 1, ("Usage: %s <env_name>" % sys.argv[0])
  model = GymModel(sys.argv[1])

  result = []
  x = [10, 20, 50, 100, 200]
  for nr in x:
    mcts = MCTS(model, num_rollouts=nr)
    scores = []
    for _ in range(20):
      model.reset()
      scores.append(run_episode(model, mcts, render=False))
    print(scores)
    result.append(scores)

  result = np.vstack(result).T
  y = result.mean(axis=0)
  std = result.std(axis=0)
  plt.clf()
  plt.plot(x, y)
  plt.fill_between(x, y-std, y+std, alpha=0.5)
  plt.show()


if __name__ == '__main__':
  main()
