from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import matplotlib.pyplot as plt
plt.ion()

from plot_util import init_figure, update_figure

import tensorflow as tf
import numpy as np
from agent import QAgent
from configs import pong_config, breakout_config

if __name__ == '__main__':

    config = pong_config
    config['state_memory']=1 # prevent allocating of a huge chunk of memory

    # Build the graph on CPU to keep gpu for training....
    with tf.device('/cpu:0'):
        agent = QAgent(config=config, log_dir=None)

    rewards = []

    for idx in range(4, 9):
        load_episode = idx * config['episodes_save_interval']
        epsilon = 0.05 # The epsilon for the strategy

        # Restore the values....
        tf.train.Saver().restore(agent.session,'log/2017-10-28_20-06-09_PongDeterministic-v4_True/episode_%d.ckpt'%(load_episode))

        # Save validation reward to textfile
        cur_reward = agent.session.run(agent.validation_reward)

        rewards.append(cur_reward)

    f = open('rewards.txt', 'a')
    for item in rewards:
        f.write('%d\n' % item)
    f.close()
