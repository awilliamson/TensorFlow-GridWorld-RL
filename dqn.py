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
        self.target_q_network_update_frequency = tf.constant( 100 )

        # How much to update Target Q weights, based on loss.
        self.target_q_network_update_coefficient = tf.constant( 0.01 )

        # Initialise action-value Q function
        with tf.name_scope("q_network"):
            self.q_network, self.q_network_layers = self.create_model()

        with tf.name_scope("target_q_network"):
            # Initialise target action-value Q function; Weights will be frozen, and only periodically updated.
            self.target_q_network, self.target_q_network_layers = self.create_model()
        # Initialise all tf variables.
        self.s.run(tf.initialize_all_variables())

        #model_vars = tflearn.variables.get_all_trainable_variable()
        #with self.s.as_default():
        #    for x in model_vars: print(tflearn.variables.get_value( x ))


        with tf.name_scope("target_network_update"):
            self.target_q_network_update_op = []
            #For each layer in the architecture ( which has variables )
            for q, t in zip( self.q_network_layers, self.target_q_network_layers ):
                # For each of those variables
                for vq, vt in zip( q, t ):
                    # Assign an update operation to bring them close together, by subtracting the difference.
                    self.target_q_network_update_op.append( vt.assign_sub( vt - vq ) )

            # Group all the sub-operations, into a big super-operation to be applied.
            self.target_q_network_update_op = tf.group(*self.target_q_network_update_op)

        # Update our Target Q Network
        self.s.run( self.target_q_network_update_op )

        # Testing if our Target Network equals our current ( Was our update successful ).
        for a, b in zip( self.q_network_layers, self.target_q_network_layers):
            for x, y in zip( a, b ):
                assert( tflearn.variables.get_value( x ) == tflearn.variables.get_value( y ) )


    def create_model(self):
            layers_array = []

            input = tflearn.input_data( shape=self.input_shape, name="input")
            fc1 = tflearn.fully_connected(input, 50, activation='tanh', name='fc1')
            fc2 = tflearn.fully_connected(fc1, 50, activation='tanh', name='fc2')
            read_out = tflearn.fully_connected(fc2, self.num_outputs, activation='softmax', name='readout')
            regression = tflearn.regression(read_out,
                                     optimizer='adam',
                                     loss='categorical_crossentropy',
                                     learning_rate=0.001,
                                     name='target')

            layers_array.append(tflearn.variables.get_layer_variables_by_name('fc1'))
            layers_array.append(tflearn.variables.get_layer_variables_by_name('fc2'))
            layers_array.append(tflearn.variables.get_layer_variables_by_name('readout'))

            return regression, layers_array
