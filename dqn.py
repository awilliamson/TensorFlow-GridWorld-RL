import tensorflow as tf
import tflearn

class DQN():
    def __init__(self, input_shape=None, num_outputs=4):

        # Cannot use mutable type for default, as this does not create a new shape every invocation, but only once
        # for all instances. Therefore we must set the default from within, and use a Sentinel value to denote
        # when an input_shape is not passed.
        if input_shape is None:
            input_shape = [None, 5, 5, 1]

        """Placeholder"""
        self.input_shape = input_shape
        self.num_outputs = num_outputs

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
