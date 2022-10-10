'''
Nomenclature of variables:
Lillicrap, Timothy P, Hunt, Jonathan J, Pritzel, Alexander, Heess, Nicolas, Erez, Tom, Tassa,
Yuval, Silver, David, and Wierstra, Daan. Continuous control with deep reinforcement learning.
'''
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.optimizers import Adam
from buffer import ReplayBuffer
from networks import ActorNetwork, CriticNetwork


class Agent:
    # alpha = learning rate for actor network
    # beta = learning rate for critic network
    # gamma = discount factor for update equation
    # max_size = maximum size of replay buffer
    # tau = update factor for target networks
    def __init__(self, input_dims, alpha=0.001, beta=0.002, env=None,
                 gamma=0.99, n_actions=2, max_size=1000000, tau=0.005,
                 fc1=400, fc2=300, batch_size=64, noise=0.1):
        self.gamma = gamma
        self.tau = tau
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size
        self.n_actions = n_actions
        self.noise = noise
        self.max_action = env.action_space.high[0]
        self.min_action = env.action_space.low[0]

        self.actor = ActorNetwork(n_actions=n_actions, name='actor')
        self.critic = CriticNetwork(name='critic')
        self.target_actor = ActorNetwork(n_actions=n_actions,
                                         name='target_actor')
        self.target_critic = CriticNetwork(name='target_critic')

        self.actor.compile(optimizer=Adam(learning_rate=alpha))
        self.critic.compile(optimizer=Adam(learning_rate=beta))
        self.target_actor.compile(optimizer=Adam(learning_rate=alpha))
        self.target_critic.compile(optimizer=Adam(learning_rate=beta))

        self.update_network_parameters(tau=1)

    def update_network_parameters(self, tau=None):
        # base case: if we don't supply a value for tau, we use tau = 1
        if tau is None:
            tau = self.tau

        # lines 16, 17, 18 from the paper
        weights = []
        targets = self.target_actor.weights
        for i, weight in enumerate(self.actor.weights):
            weights.append(weight * tau + targets[i] * (1 - tau))
        self.target_actor.set_weights(weights)

        weights = []
        targets = self.target_critic.weights
        for i, weight in enumerate(self.critic.weights):
            weights.append(weight * tau + targets[i] * (1 - tau))
        self.target_critic.set_weights(weights)

    # function to store transitions in the replay buffer
    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def save_models(self):
        print('... saving models ...')
        self.actor.save_weights(self.actor.checkpoint_file)
        self.target_actor.save_weights(self.target_actor.checkpoint_file)
        self.critic.save_weights(self.critic.checkpoint_file)
        self.target_critic.save_weights(self.target_critic.checkpoint_file)

    def load_models(self):
        print('... loading models ...')
        self.actor.load_weights(self.actor.checkpoint_file)
        self.target_actor.load_weights(self.target_actor.checkpoint_file)
        self.critic.load_weights(self.critic.checkpoint_file)
        self.target_critic.load_weights(self.target_critic.checkpoint_file)

    # evaluate is a flag used to choose between TESTING the agent or TRAINING the agent
    def choose_action(self, observation, evaluate=False):

        state = tf.convert_to_tensor([observation], dtype=tf.float32)
        actions = self.actor(state)
        # If we are training... = if we are not evaluating
        if not evaluate:
            # add some Gaussian noise to the action set
            actions += tf.random.normal(shape=[self.n_actions],
                                        mean=0.0, stddev=self.noise)
        # note that if the env has an action > 1, we have to multiply by
        # max action at some point
        # we clip it here so that we guarantee we are not passing any illegal action to the environment.
        actions = tf.clip_by_value(actions, self.min_action, self.max_action)

        # in a tensor, the zeroth element is the value of the tensor
        return actions[0]
    # if the memory is not full enough to be batched, don't batch. This means we are not learning anything, but we are
    # waiting for the memory to get full enough.
    def learn(self):
        if self.memory.mem_counter < self.batch_size:
            return

        state, action, reward, new_state, done = self.memory.sample_buffer(self.batch_size)

        states = tf.convert_to_tensor(state, dtype=tf.float32)
        states_ = tf.convert_to_tensor(new_state, dtype=tf.float32)
        rewards = tf.convert_to_tensor(reward, dtype=tf.float32)
        actions = tf.convert_to_tensor(action, dtype=tf.float32)

        # the GradientTape records operations for automatic differentiation.
        # Code below follows line 13, 14, and 15 of the DDPG algorithm

        # For the Critic
        with tf.GradientTape() as tape:
            target_actions = self.target_actor(states_)
            critic_value_ = tf.squeeze(self.target_critic(states_, target_actions), 1)
            critic_value = tf.squeeze(self.critic(states, actions), 1)
            # (1 - done) = (1 - boolean value). Done can be 1 (last episode) or 0 (keep updating the target)
            target = rewards + self.gamma * critic_value_ * (1 - done)
            critic_loss = keras.losses.MSE(target, critic_value)
        # trainable_variables come from the fact that we derive our Critic/Network class from the
        # keras Network class
        critic_network_gradient = tape.gradient(critic_loss,
                                                self.critic.trainable_variables)
        # The zip() function returns a zip object, which is an iterator of tuples where the first item in each
        # passed iterator is paired together, and then the second item in each passed iterator are paired together etc.
        self.critic.optimizer.apply_gradients(zip(
            critic_network_gradient, self.critic.trainable_variables))
        # For the actor Loss
        with tf.GradientTape() as tape:
            new_policy_actions = self.actor(states)
            # the loss is negative, because we are trying to maximize the total score.
            # By default, we do gradient descent but with the minus sign, it becomes gradient ascent.
            actor_loss = -self.critic(states, new_policy_actions)
            # tf.math.reduce_mean = computes the mean of elements across dimensions of a tensor.
            actor_loss = tf.math.reduce_mean(actor_loss)

        actor_network_gradient = tape.gradient(actor_loss,
                                               self.actor.trainable_variables)
        self.actor.optimizer.apply_gradients(zip(
            actor_network_gradient, self.actor.trainable_variables))

        # soft update of target networks
        self.update_network_parameters()


