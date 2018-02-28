from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import gym
from replay import Experience
from util import preprocess_atari_crop
import cv2
from operator import add


class QAgent(object):
    def __init__(self, config, log_dir):
        self.config = config # dictionary with game-specific information
        self.log_dir = log_dir # saving directory

        self.env = gym.make(config['game'])
        self.ale_lives = None

        # Instantiate replay buffer
        self.replay_memory = Experience(
            memory_size=config['state_memory'],
            state_shape=config['state_shape'],
            dtype=config['state_dtype']
        )

        # Instantiate the Q-networks (Running & Target)
        self.net = config['q'](
            batch_size=config['batch_size'],
            state_shape=config['state_shape']+[config['state_time']],
            num_actions=config['actions'],
            summaries=True,
            **config['q_params']
        )

        # Instantiate class-wise models
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

        # Disable the target network if needed
        if not self.config['double_q']:
            self.net.t_batch = self.net.q_batch

        with tf.variable_scope('RL_summary'):
            self.episode = tf.Variable(0.,name='episode')
            self.training_reward = tf.Variable(0.,name='training_reward')
            self.validation_reward = tf.Variable(0.,name='validation_reward')
            tf.summary.scalar(name='training_reward',tensor=self.training_reward)
            tf.summary.scalar(name='validation_reward',tensor=self.validation_reward)

        # Create tensorflow variables
        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())
        self.update_target_network()

        self.summaries = tf.summary.merge_all()
        if log_dir is not None:
            self.train_writer = tf.summary.FileWriter(self.log_dir, self.session.graph)

        self.epsilon = 1.0
        self.steps = 0

    def update_target_network(self):
        """
        Update the parameters of the target network
        Assigns the parameter of the Q-Network to the Target network
        :return:
        """
        if self.config['double_q']:
            self.session.run(self.net.assign_op)

    def _update_training_reward(self,reward):
        """
        set the value of the training reward.
        This ensures it is stored and visualised on the tensorboard
        """
        self.session.run(self.training_reward.assign(reward))

    def _update_validation_reward(self,reward):
        """
        set the value of the validation reward.
        This ensures it is stored and visualised on the tensorboard
        """
        self.session.run(self.validation_reward.assign(reward))

    def get_training_state(self):
        """
        Get the last state
        :return:
        """
        return self.replay_memory.get_last_state(self.config['state_time'])

    def sample_action(self,state,epsilon):
        """
        Sample an action for the state according to the epsilon greedy strategy

        :param state:
        :param epsilon:
        :return:
        """
        if np.random.rand() <= epsilon:
            return np.random.randint(0,self.config['actions'])
        else:
            return self.session.run(self.net.q, feed_dict={self.net.x: state[np.newaxis].astype(np.float32)})[0].argmax()

    def update_state(self, old_state, new_frame):
        """

        :param old_state:
        :param new_frame:
        :return:
        """
        # Remove oldest frame and add new frame (84 x 84 x 1)
        '''
        if np.any(old_state[:, -6:, :] == 0.0) or np.any(new_frame == 0.0):
            return np.concatenate([
            old_state[:, 12:, :],
            new_frame - old_state[:, -6:, :],
            new_frame], axis=1)
        '''
        return np.concatenate([
            old_state[:, 9:, :],
            new_frame], axis=1)


    def reset_to_zero_state(self):
        """
        Reset the state history to zeros and reset the environment to fill the first state
        :return:
        """

        return np.concatenate([
                    np.zeros((1, self.config['state_shape'][1] - 9, 1)),
                    self.config['frame'](self.env.reset())
                ],axis=1)


    def reset_to_zero_state_model(self):
        """
        Reset the state history to zeros and reset the environment to fill the first state
        :return:
        """

        state = self.env.reset()
        for idx in range(15):
            state, _, _, _ = self.env.step(1)

        return np.concatenate([
                    np.zeros((1, self.config['state_shape'][1] - 9, 1)),
                    self.config['frame'](state)
                ],axis=1)


    def update_epsilon_and_steps(self):
        if self.steps > self.config['step_startrl']:
            self.epsilon = max(self.config['eps_minval'],self.epsilon*self.config['step_eps_mul']-self.config['step_eps_min'])
        self.steps += 1


    def train_episode(self):
        # Load the last state and add a reset
        state = self.update_state(
            self.get_training_state(),
            self.config['frame'](self.env.reset())
        )

        # Store the starting state in memory
        self.replay_memory.add(
            state = state[:,:,-1],
            action = np.random.randint(self.config['actions']),
            reward = 0.,
            done = False
        )

        # Use flags to signal the end of the episode and for pressing the start button
        done = False
        press_fire = True
        total_reward = 0
        agg_loss = np.zeros((3, ))
        loop_count = 0

        while not done:
            if press_fire: # start the episode
                press_fire = False
                new_frame,reward,done, info = self.act(state,-1,True)
                if info.has_key('ale.lives'):
                    self.ale_lives = info['ale.lives']
            else:
                self.update_epsilon_and_steps()
                new_frame,reward,done, info = self.act(state,self.epsilon,True)
            state = self.update_state(state, new_frame)
            #print(state.reshape(72, ))

            #print(state.reshape((24, )))
            total_reward += reward

            # Perform online Q-learning updates
            if self.steps > self.config['step_startrl']:
                loop_count += 1
                summaries,_,loss = self.train_batch(loop_count)
                for dummy_idx in range(25):
                    self.train_model_batch(loop_count)
                agg_loss += np.array(loss).reshape((3, ))

                if self.steps % self.config['tensorboard_interval'] == 0:
                    self.train_writer.add_summary(summaries, global_step=self.steps)
                if self.steps % self.config['double_q_freq'] == 0.:
                    print("double q swap")
                    self.update_target_network()

        if loop_count == 0:
            return total_reward, agg_loss
        else:
            return total_reward, agg_loss/loop_count


    def validate_episode(self,epsilon,visualise=False):
        state = self.reset_to_zero_state()
        #model_state = state
        done = False
        press_fire = True
        total_reward = 0
        while not done:
            if press_fire:
                press_fire = False
                new_frame, reward, done,_ = self.act(state=state, epsilon=-1, store=False)
            else:
                new_frame, reward, done,_ = self.act(state=state, epsilon=epsilon, store=False, debug=False)
                #model_frame, new_frame, reward, done,_ = self.act(model_state=model_state, state=state, epsilon=epsilon, store=False, debug=True)
            state = self.update_state(old_state=state, new_frame=new_frame)
            #model_state = self.update_state(old_state=model_state, new_frame=model_frame)

            if reward == 0.1:
                reward = 0

            total_reward += reward
            if visualise:
                self.env.render()

        return total_reward


    def act(self, state, epsilon, store=False, debug=False, rs=False):
        """
        Perform an action in the environment.

        If it is an atari game and there are lives, it will end the episode in the replay memory if a life is lost.
        This is important in games lik

        :param epsilon: the epsilon for the epsilon-greedy strategy. If epsilon is -1, the no-op will be used in the atari games.
        :param state: the state for which to compute the action
        :param store: if true, the state is added to the replay memory
        :return: the observed state (processed), the reward and whether the state is final
        """
        if epsilon == -1:
            action = 1
        else:
            action = self.sample_action(state=state,epsilon=epsilon)

        raw_frame, reward, done, info = self.env.step(action)


        # Clip rewards to -1,0,1
        reward = np.sign(reward)

        # Preprocess the output state
        new_frame = self.config['frame'](raw_frame)


        if debug:

            action_input = np.zeros((1, 1, self.config['actions'], 1))
            action_input[0, 0, action, 0] = 1

            m_state = state[np.newaxis].astype(np.float32)
            m_state = m_state[:, :, -108:, :]

            model_input = np.concatenate((m_state, action_input.astype(np.float32)), axis=2)

            # Class-wise state prediction
            model_ant_prediction, model_ball_prediction, model_pro_prediction = self.session.run([self.model_ant.next_state, self.model_ball.next_state, self.model_pro.next_state], feed_dict={self.model_ant.x: model_input, self.model_ball.x: model_input, self.model_pro.x: model_input})


            model_ant_prediction = self.update_presence(model_ant_prediction)
            model_pro_prediction = self.update_presence(model_pro_prediction)
            model_ball_prediction = self.update_presence(model_ball_prediction)

            # Concatenate into a joint model state
            model_prediction = np.concatenate((model_ant_prediction, model_ball_prediction, model_pro_prediction), axis=1)

            cur_x = new_frame.reshape((self.config['actions']+3, ))
            model_predicted_state = model_prediction.reshape((9, ))

            I = preprocess_atari_crop(raw_frame)

            for idx in range(3):

                # Display opponent paddle in Blue (opencv is BGR format)
                if idx == 0 and model_predicted_state[3*idx+2] == 1:
                    cv2.rectangle(I, (np.round((model_predicted_state[3*idx]+1)*79.5).astype(int) - 2,
                                      np.round((model_predicted_state[3*idx+1]+1)*79.5).astype(int) - 8),
                                     (np.round((model_predicted_state[3*idx]+1)*79.5).astype(int) + 2,
                                      np.round((model_predicted_state[3*idx+1]+1)*79.5).astype(int) + 8), (255, 0, 0), 3)

                # Display ball in Green
                elif idx == 1 and model_predicted_state[3*idx+2] == 1:
                    cv2.rectangle(I, (np.round((model_predicted_state[3*idx]+1)*79.5).astype(int) - 1,
                                      np.round((model_predicted_state[3*idx+1]+1)*79.5).astype(int) - 2),
                                     (np.round((model_predicted_state[3*idx]+1)*79.5).astype(int) + 1,
                                      np.round((model_predicted_state[3*idx+1]+1)*79.5).astype(int) + 2), (0, 255, 0), 3)

                # Display your paddle in Red
                elif idx == 2 and model_predicted_state[3*idx+2] == 1:
                    cv2.rectangle(I, (np.round((model_predicted_state[3*idx]+1)*79.5).astype(int) - 2,
                                      np.round((model_predicted_state[3*idx+1]+1)*79.5).astype(int) - 8),
                                     (np.round((model_predicted_state[3*idx]+1)*79.5).astype(int) + 2,
                                      np.round((model_predicted_state[3*idx+1]+1)*79.5).astype(int) + 8), (0, 0, 255), 3)

                cv2.circle(I, (np.round((cur_x[3*idx]+1)*79.5).astype(int),
                               np.round((cur_x[3*idx+1]+1)*79.5).astype(int)), 1, (255, 255, 255), -1)

            print(cur_x)
            print(model_predicted_state)
            print('--------------------')
            I = cv2.resize(I, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
            cv2.imshow('Image', I)
            cv2.waitKey(0)

        # If needed, store the last frame
        if store:
            # End episodes if lives are involved in the atari game.
            # This is important in e.g. breakout
            # If this is not included, dropping the ball is not penalised.
            # By marking the end of the reward propagation, the maximum reward is limited
            # This makes learning faster.
            store_done = done
            if self.ale_lives is not None and info.has_key('ale.lives') and info['ale.lives']<self.ale_lives:
                store_done = True
                self.ale_lives = info['ale.lives']
            self.replay_memory.add(state[:,:,-1],action,reward,store_done)

        return new_frame, reward, done, info


    def validate_model(self, epsilon):
        state = self.reset_to_zero_state_model()
        #model_state = state
        done = False
        press_fire = True
        total_reward = 0
        count = 0

        while not done:

            if count > 400:
                done = True
            elif press_fire:
                press_fire = False
                new_frame = self.act_with_model(state=state, epsilon=-1, debug=True)
            else:
                new_frame = self.act_with_model(state=state, epsilon=epsilon, debug=True)

            if new_frame[:, 2, :] == 0:
                state = self.reset_to_zero_state_model()
            else:
                state = self.update_state(old_state=state, new_frame=new_frame)

            count += 1

        return

    def act_with_model(self, state, epsilon, debug):
        if epsilon == -1:
            action = 1
        else:
            action = self.sample_action(state=state,epsilon=epsilon)

        action_input = np.zeros((1, 1, self.config['actions'], 1))
        action_input[0, 0, action, 0] = 1

        state = state[np.newaxis].astype(np.float32)
        state = state[:, :, -108:, :]

        model_input = np.concatenate((state, action_input.astype(np.float32)), axis=2)

        # Class-wise state prediction
        model_ant_prediction, model_ball_prediction, model_pro_prediction = self.session.run([self.model_ant.next_state, self.modedl_ball.next_state, self.model_pro.next_state], feed_dict={self.model_ant.x: model_input, self.model_ball.x: model_input, self.model_pro.x: model_input})


        model_ant_prediction = self.update_presence(model_ant_prediction)
        model_pro_prediction = self.update_presence(model_pro_prediction)
        model_ball_prediction = self.update_presence(model_ball_prediction)

        # Concatenate into a joint model state
        model_prediction = np.concatenate((model_ant_prediction, model_ball_prediction, model_pro_prediction),
                                          axis=1)

        if debug:
            model_predicted_state = model_prediction.reshape((9, ))

            I = np.zeros((160, 160, 3))

            for idx in range(3):

                # Display opponent paddle in Blue (opencv is BGR format)
                if idx == 0 and model_predicted_state[3*idx+2] == 1:
                    cv2.rectangle(I, (np.round((model_predicted_state[3*idx]+1)*79.5).astype(int) - 2,
                                      np.round((model_predicted_state[3*idx+1]+1)*79.5).astype(int) - 8),
                                     (np.round((model_predicted_state[3*idx]+1)*79.5).astype(int) + 2,
                                      np.round((model_predicted_state[3*idx+1]+1)*79.5).astype(int) + 8), (255, 0, 0), 3)

                # Display ball in Green
                elif idx == 1 and model_predicted_state[3*idx+2] == 1:
                    cv2.rectangle(I, (np.round((model_predicted_state[3*idx]+1)*79.5).astype(int) - 1,
                                      np.round((model_predicted_state[3*idx+1]+1)*79.5).astype(int) - 2),
                                     (np.round((model_predicted_state[3*idx]+1)*79.5).astype(int) + 1,
                                      np.round((model_predicted_state[3*idx+1]+1)*79.5).astype(int) + 2), (0, 255, 0), 3)

                # Display your paddle in Red
                elif idx == 2 and model_predicted_state[3*idx+2] == 1:
                    cv2.rectangle(I, (np.round((model_predicted_state[3*idx]+1)*79.5).astype(int) - 2,
                                      np.round((model_predicted_state[3*idx+1]+1)*79.5).astype(int) - 8),
                                     (np.round((model_predicted_state[3*idx]+1)*79.5).astype(int) + 2,
                                      np.round((model_predicted_state[3*idx+1]+1)*79.5).astype(int) + 8), (0, 0, 255), 3)
                '''
                cv2.circle(I, (int((model_predicted_state[3*idx]+1)*79.5),
                               int((model_predicted_state[3*idx+1]+1)*79.5)), 1, (255, 255, 255), -1)
                '''

            I = cv2.resize(I, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)

            print(model_predicted_state)
            print('--------------------')
            cv2.imshow('Image', I)
            cv2.waitKey(0)

        model_prediction = model_prediction.reshape((1, 9, 1))

        #model_prediction = np.round((model_prediction + 1)*79.5) / 79.5 - 1.0

        return model_prediction


    def update_presence(self, prediction):
        if prediction[0, 2] > 0:
            prediction[0, 2] = 1
        else:
            prediction[0, 2] = 0
        return prediction


    def train_batch(self, loop_count):
        """
        Sample a batch of training samples from the replay memory.
        Compute the target Q values
        Perform one SGD update step

        :return: summaries, step
        summaries: the tensorflow summaries that can be put into a log.
        step, the global step from tensorflow. This represents the number of updates
        """

        # Sample experience
        xp_states, xp_actions, xp_rewards, xp_done, xp_next = self.replay_memory.sample_experience(
            self.config['batch_size'],
            self.config['state_time']
        )

        # Create a mask on which output to update
        q_mask = np.zeros((self.config['batch_size'], self.config['actions']))
        action_input = np.zeros((self.config['batch_size'], 1, self.config['actions'], 1))
        for idx, a in enumerate(xp_actions):
            q_mask[idx, a] = 1
            action_input[idx, 0, a, 0] = 1

        # Use the target network to value the next states
        next_actions, next_values = self.session.run(
            [self.net.q_batch,self.net.t_batch],
            feed_dict={self.net.x_batch: xp_next.astype(np.float32)}
        )

        # Combine the reward and the value of the next state into the q-targets
        q_next = np.array([
            next_values[idx,next_actions[idx].argmax()]
            for idx in range(self.config['batch_size'])
        ])
        q_next *= (1.-xp_done)
        q_targets = (xp_rewards + self.config['gamma']*q_next)

        '''
        # Perform the update
        feed = {
            self.net.x_batch: xp_states.astype(np.float32),
            self.net.q_targets: q_targets[:,np.newaxis]*q_mask,
            self.net.q_mask: q_mask
        }
        _, summaries, step = self.session.run([self.net._train_op, self.summaries, self.net.global_step], feed_dict=feed)
        '''

        # Perform update on class-wise model

        # State batch = same for all models
        xp_states = xp_states.astype(np.float32)
        m_batch = np.concatenate((xp_states[:, :, -108:, :], action_input.astype(np.float32)), axis=2)

        # Joint next state batch
        m_targets = xp_next.astype(np.float32)

        # Antagonist model target
        m_targets_ant = m_targets[:, :, -9:-6, :]
        m_targets_ant = m_targets_ant.reshape((self.config['batch_size'], 3))

        # Ball model target
        m_targets_ball = m_targets[:, :, -6:-3, :]
        m_targets_ball = m_targets_ball.reshape((self.config['batch_size'], 3))

        # Protagonist model target
        m_targets_pro = m_targets[:, :, -3:, :]
        m_targets_pro = m_targets_pro.reshape((self.config['batch_size'], 3))

        feed = {
            self.net.x_batch: xp_states.astype(np.float32),
            self.net.q_targets: q_targets[:,np.newaxis]*q_mask,
            self.net.q_mask: q_mask,
            self.model_ant.x_batch: m_batch,
            self.model_ant.m_targets: m_targets_ant,
            self.model_ball.x_batch: m_batch,
            self.model_ball.m_targets: m_targets_ball,
            self.model_pro.x_batch: m_batch,
            self.model_pro.m_targets: m_targets_pro
        }
        '''
        feed_model = {
            self.model.x_batch: np.concatenate((xp_states.astype(np.float32), action_input.astype(np.float32)), axis=2),
            self.model.m_targets: m_targets
        }
        '''

        # Train all networks
        _, _, _, _, next_states, loss_ant, loss_ball, loss_pro, summaries, step, _, _, _ = self.session.run([self.net._train_op, self.model_ant._train_op, self.model_ball._train_op, self.model_pro._train_op, self.model_ball.next_state_batch, self.model_ant.loss, self.model_ball.loss, self.model_pro.loss, self.summaries, self.net.global_step, self.model_ant.global_step, self.model_ball.global_step, self.model_pro.global_step], feed_dict=feed)

        if loop_count == 500:
            #m_batch = m_batch.reshape((self.config['batch_size'], self.config['model_state_shape'][1]))
            #print(m_batch[0:10, :])
            print(m_targets[0:5, :, -9:, :].reshape((5, 9)))
            print(next_states[0:5, :])

        loss = [loss_ant, loss_ball, loss_pro]

        return summaries, step, loss


    def train_model_batch(self, loop_count):
        """
        Sample a batch of training samples from the replay memory.
        Compute the target Q values
        Perform one SGD update step

        :return: summaries, step
        summaries: the tensorflow summaries that can be put into a log.
        step, the global step from tensorflow. This represents the number of updates
        """

        # Sample experience
        xp_states, xp_actions, xp_rewards, xp_done, xp_next = self.replay_memory.sample_experience(
            self.config['batch_size'],
            self.config['state_time']
        )

        # Create a mask on which output to update
        action_input = np.zeros((self.config['batch_size'], 1, self.config['actions'], 1))
        for idx, a in enumerate(xp_actions):
            action_input[idx, 0, a, 0] = 1

        # Perform update on class-wise model

        # State batch = same for all models
        xp_states = xp_states.astype(np.float32)
        m_batch = np.concatenate((xp_states[:, :, -108:, :], action_input.astype(np.float32)), axis=2)

        # Joint next state batch
        m_targets = xp_next.astype(np.float32)

        # Antagonist model target
        m_targets_ant = m_targets[:, :, -9:-6, :]
        m_targets_ant = m_targets_ant.reshape((self.config['batch_size'], 3))

        # Ball model target
        m_targets_ball = m_targets[:, :, -6:-3, :]
        m_targets_ball = m_targets_ball.reshape((self.config['batch_size'], 3))

        # Protagonist model target
        m_targets_pro = m_targets[:, :, -3:, :]
        m_targets_pro = m_targets_pro.reshape((self.config['batch_size'], 3))

        feed = {
            self.model_ant.x_batch: m_batch,
            self.model_ant.m_targets: m_targets_ant,
            self.model_ball.x_batch: m_batch,
            self.model_ball.m_targets: m_targets_ball,
            self.model_pro.x_batch: m_batch,
            self.model_pro.m_targets: m_targets_pro
        }
        '''
        feed_model = {
            self.model.x_batch: np.concatenate((xp_states.astype(np.float32), action_input.astype(np.float32)), axis=2),
            self.model.m_targets: m_targets
        }
        '''

        # Train all networks
        _, _, _, next_states, loss_ant, loss_ball, loss_pro = self.session.run([self.model_ant._train_op, self.model_ball._train_op, self.model_pro._train_op, self.model_ball.next_state_batch, self.model_ant.loss, self.model_ball.loss, self.model_pro.loss], feed_dict=feed)

        loss = [loss_ant, loss_ball, loss_pro]

        return loss

