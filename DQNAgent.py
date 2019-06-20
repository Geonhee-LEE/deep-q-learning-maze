import tensorflow as tf
import numpy as np
import random
from collections import deque

class DQNAgent:
    # Initialization
    def __init__(self, num_actions, num_features, lr, gamma, epsilon, update_operation,
                 memory_size, batch_size):
        self.num_actions = num_actions
        self.num_features = num_features
        self.learning_rate = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.update_num = update_operation
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.cnt = 0
        self.memory = np.zeros((self.memory_size, num_features * 2 + 2))
        self.base_model()
        self.sess = tf.Session()
        self.save_cost = []
        self.sess.run(tf.global_variables_initializer())
        target_parameter = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target')
        eval_parameter = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='eval')

        # replace network
        with tf.variable_scope('update_operation'):
            self.update_operation = [tf.assign(t, e) for t, e in zip(target_parameter, eval_parameter)] # update target and evaluation network

    def base_model(self):
        # input setting
        self.s = tf.placeholder(tf.float32, [None, self.num_features], name='state')
        self.a = tf.placeholder(tf.int32, [None, ], name='action')
        self.r = tf.placeholder(tf.float32, [None, ], name='reward')
        self.s_ = tf.placeholder(tf.float32, [None, self.num_features], name='next_state')

        init_weight, init_bias = tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1) # initialize weight with appropriate value

        # evaluation network setting
        with tf.variable_scope('eval'):
            eval1 = tf.layers.dense(self.s, 20, tf.nn.relu, kernel_initializer=init_weight, bias_initializer=init_bias,
                                    name='eval1')
            self.q_eval = tf.layers.dense(eval1, self.num_actions, kernel_initializer=init_weight,
                                          bias_initializer=init_bias, name='eval2')

        # target network setting
        with tf.variable_scope('target'):
            target1 = tf.layers.dense(self.s_, 20, tf.nn.relu, kernel_initializer=init_weight,
                                      bias_initializer=init_bias, name='target1')
            self.q_next = tf.layers.dense(target1, self.num_actions, kernel_initializer=init_weight,
                                          bias_initializer=init_bias, name='target2')

        with tf.variable_scope('q_target'):
            q_target = self.r + self.gamma * tf.reduce_max(self.q_next, axis=1, name='Qmax') # set q value
            self.q_target = tf.stop_gradient(q_target) # protect q function
        with tf.variable_scope('q_eval'):
            a_indices = tf.stack([tf.range(tf.shape(self.a)[0], dtype=tf.int32), self.a], axis=1)
            self.q_eval_wrt_a = tf.gather_nd(params=self.q_eval, indices=a_indices) # set q value w.r.t action
        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval_wrt_a, name='TD_error'))
        with tf.variable_scope('train'):
            self.train_optimizer = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss) # optimize network via TD_error

    # memorize transition
    def add_memory(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
        transition = np.hstack((s, [a, r], s_))
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition # update replay memory
        self.memory_counter += 1

    # choose action from state observation
    def choose_action(self, observation):
        observation = observation[np.newaxis, :]

        if np.random.uniform() < self.epsilon:
            actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0, self.num_actions) # exploration
        return action

    # train & replace network
    def train(self):
        if self.cnt % self.update_num == 0:
            self.sess.run(self.update_operation)

        if self.memory_counter > self.memory_size: # if the replay memory is bigger than memory size, choose number one of upto memory size for batching
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        _, cost = self.sess.run([self.train_optimizer, self.loss], # train network
                                feed_dict={
                                    self.s: batch_memory[:, :self.num_features],
                                    self.a: batch_memory[:, self.num_features],
                                    self.r: batch_memory[:, self.num_features + 1],
                                    self.s_: batch_memory[:, -self.num_features:],
                                })

        self.save_cost.append(cost)
        self.cnt += 1
