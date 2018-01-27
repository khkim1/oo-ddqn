import argparse, os, sys, time
import numpy as np

from matplotlib import pyplot as plt
import seaborn as sns; sns.palplot(sns.color_palette("husl", 8))

from model import GymModel
from mcts import MCTS

def run_episode(
    env_name, max_horizon=None, max_depth=None, num_rollouts=None,
    action_map=None, render=False):
  assert max_depth and num_rollouts
  model = GymModel(env_name)
  model.reset()
  mcts = MCTS(model, num_rollouts=num_rollouts, max_depth=max_depth)
  mcts.reset()

  rewards, actions = [], []
  done = False
  steps = 0

  while not (done or max_horizon and steps < max_horizon):
    steps += 1
    
    # Plan with MCTS
    q_values = mcts.plan()
    action, value = max(q_values, key=lambda x: x[1]) 

    # Take a step in the actual model
    reward, done = model.step(action)
    actions.append(action)
    rewards.append(reward)

    if render:
      action_name = action_map[action] if action_map else str(action)
      print('[Step %05d] action: %s, reward: %f, Q: [%s]' %
          (steps, action_name, r, ', '.join([str(x[1]) for x in q_values])))
      model.render()

    mcts.reset(action=action)

  return rewards, actions, steps


def main():
  # Parse arguments
  parser = argparse.ArgumentParser(description='Run MCTS experiments.')
  parser.add_argument('env_name', help='Name of the Gym environment')
  parser.add_argument('-d', '--max_depth', type=int, default=300,
                      help='Max rollout depth')
  parser.add_argument('-r', '--num_rollouts', type=int, default=500,
                      help='Number rollouts to use to plan a single step')
  parser.add_argument('-l', '--max_horizon', type=int, default=None,
                      help='Max number of steps before episode is stopped')
  parser.add_argument('-e', '--num_episodes', type=int, default=50,
                      help='Number of episodes to run the experiment for')
  args = parser.parse_args()

  # Set up experiment directory and log file
  dir_name = '%s_%s' % (args.env_name, time.strftime('%Y%m%d_%H%M%S'))
  if not os.path.exists(dir_name):
    os.makedirs(dir_name)
  logfile = open('%s/log.txt' % dir_name, 'w')
  def writelog(msg):
    logfile.write(msg + '\n')
    logfile.flush()
  writelog('Hyperparameters')
  writelog('\n'.join([('  %15s: %s' % ('--'+arg, str(val)))
    for arg, val in vars(args).iteritems()]))

  scores = []
  for ep in range(1, args.num_episodes+1):
    ep_start_time = time.time()
    rewards, actions, steps = run_episode(
        args.env_name, 
        max_horizon=args.max_horizon,
        num_rollouts=args.num_rollouts,
        max_depth=args.max_depth)
    writelog('Ep %03d / %03d finished in %d steps (%.2f sec)'
        % (ep, args.num_episodes, steps, time.time()-ep_start_time))
    np.save('%s/ep_%03d_rewards.npy' % (dir_name, ep), np.array(rewards))
    np.save('%s/ep_%03d_actions.npy' % (dir_name, ep), np.array(actions))

  # result = np.vstack(result).T
  # y = result.mean(axis=0)
  # std = result.std(axis=0)
  # plt.figure(figsize=(8,6))
  # plt.plot(x, y)
  # plt.fill_between(x, y-std, y+std, alpha=0.5)
  # plt.savefig('plot_%s.pdf' % sys.argv[1])


if __name__ == '__main__':
  main()
