from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from agent import QAgent
from configs import object_seaquest_config
from util import get_log_dir


if __name__ == '__main__':
    config = object_seaquest_config
    log_dir = get_log_dir('log', config['game']+'_'+str(config['double_q'])) # Name of logging directory
    agent = QAgent(config=config, log_dir=log_dir)
    saver = tf.train.Saver(max_to_keep=None)

    saver.restore(agent.session, '%s/episode_%d.ckpt'%("log/log/2017-12-09_23-40-34_SeaquestDeterministic-v4_True",800))

    print('Validate....\n==============')
    scores = agent.validate_episode(epsilon=0, visualise=True)
        
