from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import matplotlib.pyplot as plt
plt.ion()

from plot_util import init_figure, update_figure

import tensorflow as tf
import numpy as np
from agent import QAgent
from configs import object_pong_config


config = object_pong_config
load_episode = 500
epsilon = 0.05 # The epsilon for the strategy

# Build the graph on CPU to keep gpu for training....
agent = QAgent(config=config, log_dir=None)

# Restore the values....
tf.train.Saver().restore(agent.session,'saves/trained_Q/episode_%d.ckpt'%load_episode)

# Reset the model functions
self.model_pro = config['model'](
    batch_size=config['batch_size'],
    state_shape=config['model_state_shape']+[config['state_time']],
    output_state_shape=3,
    name='Model_pro',
    summaries=True,
    **config['q_params']
    )

self.model_ball = config['model'](
    batch_size=config['batch_size'],
    state_shape=config['model_state_shape']+[config['state_time']],
    output_state_shape=3,
    name='Model_ball',
    summaries=True,
    **config['q_params']
    )

self.model_ant = config['model'](
    batch_size=config['batch_size'],
    state_shape=config['model_state_shape']+[config['state_time']],
    output_state_shape=3,
    name='Model_ant',
    summaries=True,
    **config['q_params']
    )


for episode in range(config['episodes']):
    # Store the rewards...
    cur_trng_reward, loss = agent.train_episode()
    agent._update_training_reward(cur_trng_reward)
    reward_list.append(cur_trng_reward)

    print('episode: %d, step: %d, eps: %.4f, model loss (ant, ball, pro): %.4f, %.4f, %.4f' % (episode, agent.steps, agent.epsilon, loss[0], loss[1], loss[2]))

    if episode > 10:
        del reward_list[0]

    avg_trng_reward = np.mean(reward_list)

    if episode % config['episodes_validate']==0 and episode != 0:
        agent.validate_model(epsilon=0.05)

    #if agent.steps % config['steps_validate'] == 0:
        print('Validate....\n==============')
        scores = [agent.validate_episode(epsilon=0.05) for i in range(config['episodes_validate_runs'])]
        agent._update_validation_reward(np.mean(scores))
        print(scores)

        '''
        # Record scores
        f = open('learning_curves/trial1/rewards.txt', 'a')
        f.write('%d, %d, %f, %f, %f, %f, %f\n' %(agent.steps, episode, avg_trng_reward, np.mean(scores), loss[0], loss[1], loss[2]))
        f.close()
        '''

        '''
        if episode % 200 == 0 and episode != 0:
            agent.validate_episode(epsilon=0.05, visualise=True)
        '''

    '''
    # Store every validation interval
    if episode % config['episodes_save'] == 0 and episode != 0:
        saver.save(agent.session,'%s/episode_%d.ckpt'%(log_dir,episode))
    '''
