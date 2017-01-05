import tensorflow as tf
import tflearn
from tflearn import Evaluator
from collections import deque
import random

class DQN():
    def __init__(self, sess=None, input_shape=None, num_outputs=4, experience_replay_size=50):

        # Cannot use mutable type for default, as this does not create a new shape every invocation, but only once
        # for all instances. Therefore we must set the default from within, and use a Sentinel value to denote
        # when an input_shape is not passed.
        if input_shape is None:
            input_shape = [None, 25]

        self.s = sess if sess is not None else tf.Session()
        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.001)

        """Placeholder"""
        self.input_shape = input_shape
        self.num_outputs = num_outputs

        self.minibatch_size = 32

        self.experience_replay_size = experience_replay_size
        # Create experience replay to a maximum size N.
        self.experience_replay = deque(maxlen=self.experience_replay_size)

        # Used for updating target Q Network's Weights.
        self.target_q_network_update_frequency = 100
        self.target_q_network_update_count = 0

        # How much to update Target Q weights, based on loss.
        self.target_q_network_update_coefficient = tf.constant( 0.01 )

        # Initialise action-value Q function
        with tf.name_scope("q_network"):
            self.q_network, self.q_network_layers, self.q_network_layers_variables = self.create_model()

        with tf.name_scope("q_network_forward"):
            self.observation = tf.placeholder(tf.float32, list(self.input_shape), name="observation")
            self.action_scores = tf.identity( self.observation, name="action_scores")
            print( self.action_scores.get_shape() )
            #tf.histogram_summary("action_scores", self.action_scores)
            self.predicted_actions = tf.argmax(self.action_scores, dimension=1, name="predicted_actions")

        with tf.name_scope("target_q_network"):
            # Initialise target action-value Q function; Weights will be frozen, and only periodically updated.
            self.target_q_network, self.target_q_network_layers, self.target_q_network_layers_variables = self.create_model()
            #print( self.target_q_network_layers )

        # Initialise all tf variables.
        self.s.run(tf.initialize_all_variables())

        #for n in self.s.graph.get_operations():
        #    print( n )

        #model_vars = tflearn.variables.get_all_trainable_variable()
        #with self.s.as_default():
        #    for x in model_vars: print(tflearn.variables.get_value( x ))
        with tf.name_scope("target_q_network_operations"):
            self.t_observation = tf.placeholder(tf.float32, list(self.input_shape),
                                                   name="t_observation")
            #self.t_observation_mask = tf.placeholder(tf.float32, (None,), name="next_observation_mask")

            # Define some operation which will be responsible for evaluating the given model via the placeholder t_observation, which will be populated, later.
            #self.t_action_scores = tf.stop_gradient( Evaluator(self.target_q_network).predict(feed_dict={self.t_ob: self.t_observation}) )

            self.t_action_scores = tf.stop_gradient(sum([tf.matmul(x, W) for x, W in zip([self.t_observation], [self.target_q_network_layers[-1].W])]) + self.target_q_network_layers[-1].b)
            # Not sure that this will work, as the observations must be propagated from previous layers rather than input layer?
            # This is a derived equivalent from https://github.com/nivwusquorum/tensorflow-deepq/blob/master/tf_rl/controller/discrete_deepq.py#L142

            #tf.histogram_summary("target_action_scores", self.t_action_scores)
            self.rewards = tf.placeholder(tf.float32, (None,), name="rewards")
            target_values = tf.reduce_max(self.t_action_scores, reduction_indices=[1, ])
            self.future_rewards = self.rewards + 0.99 * target_values

        with tf.name_scope("q_value_prediction"):
            # FOR PREDICTION ERROR
            self.action_mask = tf.placeholder(tf.float32, (None, self.num_outputs), name="action_mask")
            print(self.action_mask.get_shape())
            self.masked_action_scores = tf.reduce_sum(self.action_scores * self.action_mask, reduction_indices=[1, ])
            temp_diff = self.masked_action_scores - self.future_rewards
            self.prediction_error = tf.reduce_mean(tf.square(temp_diff))
            gradients = self.optimizer.compute_gradients(self.prediction_error)
            for i, (grad, var) in enumerate(gradients):
                if grad is not None:
                    gradients[i] = (tf.clip_by_norm(grad, 5), var)
            # Add histograms for gradients.
            for grad, var in gradients:
                tf.histogram_summary(var.name, var)
                if grad is not None:
                    tf.histogram_summary(var.name + '/gradients', grad)
            self.train_op = self.optimizer.apply_gradients(gradients)

        with tf.name_scope("target_network_update"):
            self.target_q_network_update_op = []
            #For each layer in the architecture ( which has variables )
            for q, t in zip(self.q_network_layers_variables, self.target_q_network_layers_variables):
                # For each of those variables
                for vq, vt in zip( q, t ):
                    # Assign an update operation to bring them close together, by subtracting the difference.
                    self.target_q_network_update_op.append( vt.assign_sub( vt - vq ) )

            # Group all the sub-operations, into a big super-operation to be applied.
            self.target_q_network_update_op = tf.group(*self.target_q_network_update_op)

        # Update our Target Q Network
        self.s.run( self.target_q_network_update_op )

        # Testing if our Target Network equals our current ( Was our update successful ).
        for a, b in zip(self.q_network_layers_variables, self.target_q_network_layers_variables):
            for x, y in zip( a, b ):
                assert( tflearn.variables.get_value( x ) == tflearn.variables.get_value( y ) )

    def action(self, state):
        return random.randint(0, self.num_outputs - 1)

    def train(self):

        #print( "i: ", self.target_q_network_update_count, self.target_q_network_update_count % self.target_q_network_update_frequency == 0 )
        # Sample from MiniBatch
        # If we're due to train
        if self.target_q_network_update_count % self.target_q_network_update_frequency == 0:

            #print("We're TRAINING!")
            if len(self.experience_replay) < self.experience_replay_size:
                # Not enough experience replay to pool, don't train.
                return

            minibatch = random.sample(self.experience_replay, self.minibatch_size)
            # St, At, Rt, St+1
            print( minibatch )
            print( len( minibatch ) )
            print( "Are we actually training?" )

            y_t = 0
            for experience in minibatch:
                if experience[4]:
                    # Terminal
                    y_t = experience[ 1 ] # Rt
                else:
                    y_t = experience[ 1 ]


            # Update Target Q Network weights
            self.s.run( self.target_q_network_update_op )

            # Assertion test to ensure our weights are now identical.
            for a, b in zip(self.q_network_layers_variables, self.target_q_network_layers_variables):
                for x, y in zip(a, b):
                    assert (tflearn.variables.get_value(x) == tflearn.variables.get_value(y))

        self.target_q_network_update_count += 1

    def store(self, experience):
        self.experience_replay.append( experience )

    def create_model(self):

            layers = []
            layers_variables_array = []

            input = tflearn.input_data( shape=self.input_shape, name="input")
            fc1 = tflearn.fully_connected(input, 50, activation='tanh', name='fc1')
            fc2 = tflearn.fully_connected(fc1, 50, activation='tanh', name='fc2')
            read_out = tflearn.fully_connected(fc2, self.num_outputs, activation='softmax', name='readout')
            regression = tflearn.regression(read_out,
                                     optimizer='adam',
                                     loss='categorical_crossentropy',
                                     learning_rate=0.001,
                                     name='target')

            layers.append(fc1)
            layers.append(fc2)
            layers.append(read_out)

            layers_variables_array.append(tflearn.variables.get_layer_variables_by_name('fc1'))
            layers_variables_array.append(tflearn.variables.get_layer_variables_by_name('fc2'))
            layers_variables_array.append(tflearn.variables.get_layer_variables_by_name('readout'))

            return regression, layers, layers_variables_array
