"""Implementation of Monte Carlo Search Tree (MCTS) for deterministic MDP"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from math import sqrt, log
import random, time


# For debugging statements
_DEBUG = False
def debug(msg):
  global _DEBUG
  if _DEBUG:
    timestr = '[%s] ' % time.strftime('%Y-%m-%d %H:%M:%S')
    indent = len(timestr) * ' '
    lines = msg.split('\n')
    if lines:
      print(timestr + lines[0])
      for line in lines[1:]:
        print(indent + line)


class Node(object):
  """
  Represents a single node of the search tree, corresponding to a state.
  Children of a node are succesor states obtained for all actions.
  Each node keeps its own value estimate Q(s,a).
  """
  def __init__(self, action=None, parent=None):
    """
    Args:
      action: action used to generate this node (None for root)
      parent: parent of this node (if exists)
    """
    self.action = action
    self.parent = parent
    self.children = []
    self.total_value = 0.0
    self.visit_count = 0

  @property
  def value(self):
    if self.visit_count == 0:
      return 0.0
    else:
      return self.total_value / self.visit_count

  def is_leaf(self):
    return len(self.children) == 0

  def is_root(self):
    return self.parent is None

  def update(self, reward):
    """
    Updates the statistics for this node from a new trajectory.
    """
    self.visit_count += 1
    self.total_value += reward

  def expand(self, actions):
    """
    Expands this node by taking all possible actions and creating children.
    """
    self.children = [Node(parent=self, action=action) for action in actions]

  # For debugging, etc.
  @property
  def num_nodes(self):
    queue = [self]
    cnt = 1
    while queue:
      node = queue.pop()
      cnt += len(node.children)
      queue.extend(node.children)
    return cnt


class MCTS(object):
  def __init__(self, model, max_depth=300, num_rollouts=500,
      tree_policy="ucb1", ucb1_coeff=1.0):
    """
    Args
      model: instance of Model class
      max_depth: if given, truncates rollouts to this many steps
      tree_policy: tree policy to use ("greedy", "random", "ucb1")
      ucb1_coeff: exploration parameter for UCB1 tree policy
    """
    self.model = model
    self.max_depth = max_depth
    self.num_rollouts = num_rollouts
    self.root = Node()

    if tree_policy == "random":
      self.score_fn = lambda x: random.random()
    elif tree_policy == "greedy":
      self.score_fn = lambda x: x.value
    elif tree_policy == "ucb1":
      assert ucb1_coeff
      self.score_fn = self._get_ucb1_func(ucb1_coeff)
    else:
      raise ValueError("Invalid tree policy specified.")
    self.tree_policy = lambda node: max(node.children, key=self.score_fn)

  def _get_ucb1_func(self, coeff):
    """
    Returns the UCB1 value function for the given coefficient.
    """
    def func(node):
      if node.visit_count == 0:
        return float('inf')
      else:
        return (node.value
            + coeff * sqrt(log(node.parent.visit_count) / node.visit_count))
    return func

  def reset(self, action=None):
    """
    Resets the search tree.
    If action is given, subtree starting with that action is preserved.
    """
    if action:
      for child in self.root.children:
        if child.action == action:
          self.root = child
          break
    else:
      self.root = Node()

  def step(self):
    """
    Runs a single complete step of MCTS algorithm:
      Select leaf -> Expand -> Simulate -> Backprop
    """
    snapshot = self.model.snapshot()
    total_reward = 0
    node = self.root

    # Step 1: select
    debug('  --> first action scores: [%s]'
        % ', '.join([('%.2f' % self.score_fn(x)) for x in node.children]))
    while not node.is_leaf():
      node = self.tree_policy(node)
      reward, _ = self.model.step(node.action)
      total_reward += reward

    # Step 2:  expand
    node.expand(self.model.actions)
    debug('  --> number of nodes: %d' % self.root.num_nodes)

    # Step 3: rollout
    step_cnt = 0
    while step_cnt < self.max_depth:
      # Take random action
      reward, done = self.model.step(random.choice(self.model.actions))
      total_reward += reward
      step_cnt += 1
      if done:
        break
    debug('  --> rollout steps: %d, total reward: %.2f '
        % (step_cnt, total_reward))

    # Step 4: backpropagate
    while node:
      node.update(total_reward)
      node = node.parent

    self.model.restore(snapshot)

  def plan(self):
    for it in range(self.num_rollouts):
      debug('Iteration %d' % (it+1))
      self.step()

    return [(x.action, x.value) for x in self.root.children]

