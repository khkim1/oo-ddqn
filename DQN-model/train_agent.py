from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from agent import QAgent
from configs import pong_config, object_pong_config, breakout_config
from util import get_log_dir


if __name__ == '__main__':
    config = object_pong_config
    log_dir = get_log_dir('log', config['game']+'_'+str(config['double_q'])) # Name of logging directory
    agent = QAgent(config=config, log_dir=log_dir)
    saver = tf.train.Saver(max_to_keep=None)
    reward_list = []

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
            f = open('learning_curves/trial1/rewards.txt', 'a')
            f.write('%d, %d, %f, %f, %f, %f, %f\n' %(agent.steps, episode, avg_trng_reward, np.mean(scores), loss[0], loss[1], loss[2]))
            f.close()

            '''
            if episode % 200 == 0 and episode != 0:
                agent.validate_episode(epsilon=0.05, visualise=True)
            '''

        # Store every validation interval
        if episode % config['episodes_save'] == 0 and episode != 0:
            saver.save(agent.session,'%s/episode_%d.ckpt'%(log_dir,episode))

