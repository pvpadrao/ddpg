'''
Nomenclature of variables:
Lillicrap, Timothy P, Hunt, Jonathan J, Pritzel, Alexander, Heess, Nicolas, Erez, Tom, Tassa,
Yuval, Silver, David, and Wierstra, Daan. Continuous control with deep reinforcement learning.
'''

import os
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras .layers import Dense

class CriticNetwork(keras.Model):
    # fc_dims = fully connected layers; chkpt_dir = checkpoint directory
    def __init__(self, fc1_dims=512, fc2_dims=512, name='critic', chkpt_dir='tmp/ddpg'):
        # The super() builtin returns a proxy object (temporary object of the superclass) that allows us
        # to access methods of the base class. Allows us to avoid using the base class name explicitly.
        super(CriticNetwork, self).__init__()
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims

        self.model_name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, self.model_name+'_ddpg.h5')

        # implement the network
        self.fc1 = Dense(fc1_dims, activation='relu')
        self.fc2 = Dense(fc2_dims, activation='relu')
        # output layer, nomenclature from the paper
        self.q = Dense(1, activation = None)

        # forward propagation
        def call(self, state, action):
            action_value = self.fc1(tf.concat([state, action]), axis=1)
            action_value = self.fc2(action_value)
            q = self.q(action_value)

            return q

class ActorNetwork(keras.Model):
    # fc_dims = fully connected layers; chkpt_dir = checkpoint directory
    def __init__(self, fc1_dims=512, fc2_dims=512, n_actions =2, name='actor', chkpt_dir='tmp/ddpg'):
        # The super() builtin returns a proxy object (temporary object of the superclass) that allows us
        # to access methods of the base class. Allows us to avoid using the base class name explicitly.
        super(ActorNetwork, self).__init__()
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.model_name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, self.model_name+'_ddpg.h5')

        # implement the network
        self.fc1 = Dense(fc1_dims, activation='relu')
        self.fc2 = Dense(fc2_dims, activation='relu')
        self.mu = Dense(self.n_actions, activation='tanh')

        # forward propagation
        def call(self, state):
            # prob = not actually probability
            prob = self.fc1(state)
            prob = self.fc2(prob)
            # if action bounds not +/-1, we can multiply it here, so we have bounded actions
            mu = self.mu(prob)

            return mu
