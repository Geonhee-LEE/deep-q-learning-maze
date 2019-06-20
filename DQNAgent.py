import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
import numpy as np
import random
import os
from collections import deque

seed = 0
np.random.seed(seed)
random.seed(seed)

class DQNAgent:
    # Initialization
    def __init__(self, n_action, obs_dim, lr, discount_factor, epsilon, 
                 memory_size, train_start = 200, batch_size = 64, 
                 seed=0, epsilon_decay = 0.999, epsilon_min = 0.01, hidden_unit_size = 4):

        self.seed = seed 
 
        # Environment Information
        self.n_action = n_action
        self.obs_dim = obs_dim
        self.discount_factor = discount_factor

        # Epsilon Greedy Policy
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # Network Hyperparameters
        self.hidden_unit_size = hidden_unit_size
        self.learning_rate = lr
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.train_start = train_start

        # Experience Replay
        self.memory = deque(maxlen=memory_size)

        # Define Computational Graph in TF
        self.g = tf.Graph()
        with self.g.as_default():
            self.build_placeholders()
            self.build_model()
            self.build_loss()
            self.build_update_operation()
            self.init_session() # Initialize all parameters in graph
            
    # Input 및 output place holder 선언
    def build_placeholders(self): 
        self.obs_ph = tf.placeholder(tf.float32, (None, self.obs_dim), 'obs') # Input state
        self.target_ph = tf.placeholder(tf.float32, (None, self.n_action), 'target') # TD target
        self.learning_rate_ph = tf.placeholder(tf.float32, (), 'lr')        
    
    # Networks 선언
    def build_model(self): 
        hid1_size = self.hidden_unit_size
        hid2_size = self.hidden_unit_size
        
         # Prediction Network / Two layered perceptron / Training Parameters
        with tf.variable_scope('q_prediction'):
            out = tf.layers.dense(self.obs_ph, hid1_size, tf.nn.relu, # Relu Activation
                                  kernel_initializer=tf.random_normal_initializer(stddev=0.01,seed=self.seed), name='hidden1')
            out = tf.layers.dense(out, hid2_size, tf.nn.relu, # Relu Activation
                                  kernel_initializer=tf.random_normal_initializer(stddev=0.01,seed=self.seed), name='hidden2')
            self.q_predict = tf.layers.dense(out, self.n_action, # Linear Layer
                                  kernel_initializer=tf.random_normal_initializer(stddev=0.01,seed=self.seed), name='q_predict')
                        
        # Target Network / Two layered perceptron / Old Parameters                        
        with tf.variable_scope('q_target'): 
            out = tf.layers.dense(self.obs_ph, hid1_size, tf.nn.relu, # Relu Activation
                                  kernel_initializer=tf.random_normal_initializer(stddev=0.01,seed=self.seed), name='hidden1')
            out = tf.layers.dense(out, hid2_size, tf.nn.relu, # Relu Activation
                                  kernel_initializer=tf.random_normal_initializer(stddev=0.01,seed=self.seed), name='hidden2')
            self.q_predict_old = tf.layers.dense(out, self.n_action, # Linear Layer
                                  kernel_initializer=tf.random_normal_initializer(stddev=0.01,seed=self.seed), name='q_predict')
        
        self.weights = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='q_prediction') # Get Prediction network's Parameters
        self.weights_old = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='q_target') # Get Target network's Parameters
              
    # Loss(MSE) 선언, optimizer(RMS) 선언
    def build_loss(self):
        self.loss = 0.5*tf.reduce_mean(tf.square(self.target_ph - self.q_predict)) # Squared Error
        self.optim = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate_ph).minimize(self.loss) # AdamOptimizer (Gradient Descent Algorithm)
    
    # Prediction Network를 Target Network로 update(TF Graph 갱신)하기 위해 parameter 정의
    def build_update_operation(self): 
        update_ops = [] 
        for var, var_old in zip(self.weights, self.weights_old): # Update Target Network's Parameter with Prediction Network
            update_ops.append(var_old.assign(var))
        self.update_ops = update_ops            

    # TF 세션 초기화
    def init_session(self):
        self.sess = tf.Session(config=tf.ConfigProto(log_device_placement=True), graph=self.g) # Initialize session
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(self.update_ops)

        # Tensor board를 사용하기 위한 변수 선언(Summary writer)
        summary_q = tf.summary.scalar('max_Q_predict', tf.reduce_max(self.q_predict))
        summary_q_old = tf.summary.scalar('max_Q_target', tf.reduce_max(self.q_predict_old))
        summary_loss = tf.summary.scalar('loss', self.loss)
        self.merge_q_step = 0
        self.merge_q = tf.summary.merge([summary_q, summary_q_old])
        self.merge_loss_step = 0
        self.merge_loss = tf.summary.merge([summary_loss])
        self.summary_writer = tf.summary.FileWriter('./tf_logs/dqn', graph=self.sess.graph)

    #  Target update하기 위해 함수 선언
    def update_target(self): 
        self.sess.run(self.update_ops)            
    
    # Policy update, epsilon decaying policy사용.
    def update_policy(self):
        if self.epsilon > self.epsilon_min: # Update epsilon
            self.epsilon *= self.epsilon_decay      

    # Target network에서 Q-value 획득
    def get_prediction_old(self, obs): 
        q_value_old, summary = self.sess.run([self.q_predict_old,self.merge_q],feed_dict={self.obs_ph:obs}) 
        
        # Summary Q value
        self.merge_q_step += 1
        self.summary_writer.add_summary(summary,self.merge_q_step)       
        
        return q_value_old

    # Prediction network에서 Q value 획득
    def get_prediction(self, obs): 
        q_value, summary = self.sess.run([self.q_predict,self.merge_q],feed_dict={self.obs_ph:obs}) 
        
        # Summary Q value
        self.merge_q_step += 1
        self.summary_writer.add_summary(summary,self.merge_q_step)       
        
        return q_value

     # Epsilon Greedy policy에서 action 선택.
    def get_action(self, obs):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.n_action)
        else:
            q_value = self.get_prediction([obs])
            return np.argmax(q_value[0])

    # Replay memory에 추가
    def add_experience(self, obs, action, reward, next_obs, done): 
        self.memory.append((obs, action, reward, next_obs, done))

    # Batch를 이용하여 학습진행. 초반에 replay memory를 쌓아놓은 뒤에 batch보다 클 때 학습 진행하도록 구성.
    def train(self):
        loss = np.nan
        n_entries = len(self.memory)   
        if n_entries >= self.train_start: 
            # Random하게 batch 샘플
            mini_batch = random.sample(self.memory, self.batch_size)
            
            observations = np.zeros((self.batch_size, self.obs_dim))
            next_observations = np.zeros((self.batch_size, self.obs_dim))
            actions, rewards, dones = [], [], []

            # Batch size만큼 obs, act, reward, next_obs, done 저장
            for i in range(self.batch_size):
                observations[i] = mini_batch[i][0]
                actions.append(mini_batch[i][1])
                rewards.append(mini_batch[i][2])
                next_observations[i] = mini_batch[i][3]
                dones.append(mini_batch[i][4])

            # Prediction network에서 Q-value 획득
            target = self.get_prediction(observations)
            # Target network에서 Q-value 획득
            next_q_value = self.get_prediction_old(next_observations)

            # Bellman update rule
            for i in range(self.batch_size):
                #  Positive, negative grid에 도달시에는 reward만을 이용하여 target(feed_dict)에 저장
                if dones[i] == 'DONE' or dones[i] == 'FAIL':
                    target[i][actions[i]] = rewards[i]
                #  Target network 중에서 가장 큰 Q값에 reward를 더하여 target(feed_dict)에 저장
                else:
                    target[i][actions[i]] = rewards[i] + self.discount_factor * (np.amax(next_q_value[i]))

            # Loss 계산, optimizer 실행하여 back propagation, tensor board를 이용하기 위해 summary 반환
            loss, _, summary = self.sess.run([self.loss, self.optim, self.merge_loss], 
                                    feed_dict={self.obs_ph:observations,self.target_ph:target,self.learning_rate_ph:self.learning_rate})                        
            
            # tensor board를 이용하기 위해 summary 작성.
            self.merge_loss_step += 1
            self.summary_writer.add_summary(summary, self.merge_loss_step)       
        return loss
