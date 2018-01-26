"""Different models to be used with MCTS."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc, copy, itertools
import gym
from gym.envs.toy_text import frozen_lake, discrete
from gym.envs.registration import register



class Model:
  __metaclass__ = abc.ABCMeta

  @abc.abstractmethod
  def step(self, action):
    """Returns sampled (reward, is_terminal) pair."""

  @abc.abstractmethod
  def actions(self):
    """Returns the list of all actions of the environment."""

  @abc.abstractmethod
  def snapshot(self):
    """Returns a copy of current state to be restored later."""

  @abc.abstractmethod
  def restore(self, snapshot):
    """Restores the environment from the given snapshot."""


class GymModel(Model):
  """Model wrapper for OpenAI Gym environments."""
  def __init__(self, env):
    if isinstance(env, str):
      self.env = gym.make(env)
      self.env.reset()
    elif isinstance(env, gym.Env):
      self.env = env
    else:
      raise NotImplementedError

  def step(self, action):
    _, reward, done, _ = self.env.step(action)
    return reward, done

  @property
  def actions(self):
    space = self.env.action_space
    if isinstance(space, gym.spaces.Discrete):
      return range(space.n)
    elif isinstance(space, gym.spaces.Tuple):
      # Return the Cartesian product of all actions
      return list(itertools.product(*[list(range(x.n)) for x in space.spaces]))
    else:
      raise NotImplementedError

  def snapshot(self):
    self.env.render(close=True)
    return copy.deepcopy(self.env)

  def restore(self, snapshot):
    self.env = snapshot

  def reset(self):
    self.env.reset()

  def render(self, *args, **kwargs):
    self.env.render(*args, **kwargs)


# Register some environments for testing and debugging.
register(
    id='Deterministic-4x4-FrozenLake-v0',
    entry_point='gym.envs.toy_text.frozen_lake:FrozenLakeEnv',
    kwargs={'map_name': '4x4', 'is_slippery': False})
register(
    id='Deterministic-8x8-FrozenLake-v0',
    entry_point='gym.envs.toy_text.frozen_lake:FrozenLakeEnv',
    kwargs={'map_name': '8x8', 'is_slippery': False})
