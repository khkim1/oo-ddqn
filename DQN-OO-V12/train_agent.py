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
    #save.restore(agent.session,'%s/episode_%d.ckpt'%(log_dir,episode))
    reward_list = []
    grad_list = []
    avg_trng_reward = 0
    avg_grad = 0
    for episode in range(config['episodes']):
        print('\nepisode: %d, step: %d, eps: %.4f, avg_reward: %.4f, avg_grad: %.4f\n\n---------------------' % (episode, agent.steps, agent.epsilon, avg_trng_reward, avg_grad))
        # Store the rewards...
        cur_trng_reward, grad_cur = agent.train_episode()
        agent._update_training_reward(cur_trng_reward)
        reward_list.append(cur_trng_reward)
        if grad_cur > 0:
            grad_list.append(grad_cur)
            avg_grad = np.mean(grad_list)
        if episode > 10:
            del reward_list[0]
        if len(grad_list) > 10:
            del grad_list[0]

        avg_trng_reward = np.mean(reward_list)
        

        tol = 1e-5
        if episode % config['episodes_validate']==0 and episode != 0:
        #if agent.steps % config['steps_validate'] == 0:
            eps = 0.1 + np.min([40.0/(episode+tol), 0.9])
            print('Validate....\n==============')
            scores = [agent.validate_episode(epsilon=eps) for i in range(config['episodes_validate_runs'])]
            agent._update_validation_reward(np.mean(scores))
            print('epsilon: %f' %eps)
            print(scores)
            f = open('learning_curves/trial1/rewards3.txt', 'a')
            f.write('%d, %d, %f, %f\n' %(agent.steps, episode, avg_trng_reward, np.mean(scores)))
            f.close()

        
        # Store every validation interval
        if episode% config['episodes_save_interval']==0:
            saver.save(agent.session,'%s/episode_%d.ckpt'%(log_dir,episode))
        
