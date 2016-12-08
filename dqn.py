import tensorflow as tf
import tflearn
from collections import deque

class DQN():
    def __init__(self, sess=None, input_shape=None, num_outputs=4):

        # Cannot use mutable type for default, as this does not create a new shape every invocation, but only once
        # for all instances. Therefore we must set the default from within, and use a Sentinel value to denote
        # when an input_shape is not passed.
        if input_shape is None:
            input_shape = [None, 25]

        self.s = sess if sess is not None else tf.Session()

        """Placeholder"""
        self.input_shape = input_shape
        self.num_outputs = num_outputs

        self.experience_replay_size = 500
        # Create experience replay to a maximum size N.
        self.experience_replay = deque(maxlen=self.experience_replay_size)

        # Used for updating target Q Network's Weights.
        self.target_q_network_update_frequency = 100
        self.target_q_network_update_coefficient = 0.01 # How much to update Target Q weights, based on loss.

        # Initialise action-value Q function
        self.q_network = self.create_model()
        # Initialise target action-value Q function; Weights will be frozen, and only periodically updated.
        self.target_q_network = self.create_model()

        # Initialise all tf variables.
        self.s.run(tf.initialize_all_variables())

    def create_model(self):
        net = tflearn.input_data( shape=self.input_shape, name="input")
        net = tflearn.fully_connected(net, 50, activation='tanh')
        net = tflearn.fully_connected(net, 50, activation='tanh')
        net = tflearn.fully_connected(net, self.num_outputs, activation='softmax')
        net = tflearn.regression(net,
                                 optimizer='adam',
                                 loss='categorical_crossentropy',
                                 learning_rate=0.001,
                                 name='target')
        return net
