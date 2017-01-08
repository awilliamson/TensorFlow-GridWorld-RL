import tensorflow as tf
#import tflearn
from collections import deque
import random
import numpy as np

class DQN():
    def create_model(self, keep_prob=0.5):

        q_in = tf.placeholder(tf.float32, [None, 25])

        # Original Q-Network
        with tf.name_scope("weights_and_biases"):
            weights = {
                "fc1": tf.Variable(tf.random_normal([self.input_shape, 512]), name="fc1_w"),
                "fc2": tf.Variable(tf.random_normal([512, 512]), name="fc2_w"),
                "fc3": tf.Variable(tf.random_normal([512, 512]), name="fc3_w"),
                "out": tf.Variable(tf.random_normal([512, self.num_outputs]), name="out_w")
            }
            bias = {
                "fc1": tf.Variable(tf.random_normal([512]), name="fc1_b"),
                "fc2": tf.Variable(tf.random_normal([512]), name="fc2_b"),
                "fc3": tf.Variable(tf.random_normal([512]), name="fc3_b"),
                "out": tf.Variable(tf.random_normal([self.num_outputs]), name="out_b")
            }

        # fc1
        with tf.name_scope("fc1"):
            q_fc1 = tf.nn.tanh(tf.add(tf.matmul(q_in, weights["fc1"]), bias["fc1"]), name="fc1")

        with tf.name_scope("drop1"):
            q_fc1_drop = tf.nn.dropout(q_fc1, keep_prob)

        # fc2
        with tf.name_scope("fc2"):
            q_fc2 = tf.nn.tanh(tf.add(tf.matmul(q_fc1_drop, weights["fc2"]), bias["fc2"]), name="fc2")

        with tf.name_scope("drop2"):
            q_fc2_drop = tf.nn.dropout(q_fc2, keep_prob)

        with tf.name_scope("fc3"):
            q_fc3 = tf.nn.tanh(tf.add(tf.matmul(q_fc2_drop, weights["fc3"]), bias["fc3"]), name="fc3")

        with tf.name_scope("drop3"):
            q_fc3_drop = tf.nn.dropout(q_fc3, keep_prob)

        with tf.name_scope("out"):
            q_out = tf.nn.softmax(tf.add(tf.matmul(q_fc3_drop, weights["out"]), bias["out"]), name="out")

        return q_out, q_in, [q_fc1, q_fc1_drop, q_fc2, q_fc2_drop, q_fc3, q_fc3_drop, q_out], [weights, bias]

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

        # Counter for training
        self.training_n = 0
        self.training_frequency = 50

        # Used for updating target Q Network's Weights.
        self.target_q_network_update_frequency = 100

        # How much to update Target Q weights, based on loss.
        self.target_q_network_update_coefficient = tf.constant( 0.01 )

        self.optimiser = tf.train.AdamOptimizer(learning_rate=0.001)

        # Initialise action-value Q function
        with tf.name_scope("q_network"):
            self.q_out, self.q_in, self.q_network, self.q_network_Wsb = self.create_model()

        # Initialise Target Q Network and variables. Under the hood of 'target_q_network' name.
        # Eg target_q_network/weights_and_biases/fc1_w
        with tf.name_scope("target_q_network"):
            # Initialise target action-value Q function; Weights will be frozen, and only periodically updated.
            self.target_q_out, self.target_q_in, self.target_q_network, self.target_q_network_Wsb = self.create_model()

        # Qt+1 = Qt + alpha( rt+1 + gamma(Qt(St+1, At) - Qt) )
        # New value = Old Value + alpha( reward + discount_factor( estimate of future value ) - old value )
        with tf.name_scope('calculations'):

            with tf.name_scope("taking_action"):
                #self.observation = tf.placeholder(tf.float32, [None, self.input_shape], name="observation")
                tf.summary.histogram("action_scores", self.q_out)
                self.predicted_actions = tf.argmax(self.q_out, dimension=1, name="predicted_actions")

            with tf.name_scope("estimating_future_rewards"):
                self.next_observation = tf.placeholder(tf.float32, [None, self.input_shape], name="next_observation")

                #self.next_action_scores = self.target_q_out
                tf.summary.histogram("target_action_scores", self.target_q_out)
                self.rewards = tf.placeholder(tf.float32, (None,), name="rewards")
                target_values = tf.reduce_max(self.target_q_out, reduction_indices=[1, ])

                # Qt <- Qt + alpha( self.rewards + 0.99 * target_values )
                self.future_rewards = self.rewards + 0.99 * target_values

            with tf.name_scope("q_value_prediction"):
                # FOR PREDICTION ERROR
                self.q_prediction_action_scores = tf.reduce_sum(self.q_out, reduction_indices=[1, ])

                temp_diff = self.q_prediction_action_scores - self.future_rewards
                self.prediction_error = tf.reduce_mean(tf.square(temp_diff))
                tf.summary.scalar('prediction_error', self.prediction_error)
                self.train_op = self.optimiser.minimize( self.prediction_error)

        tf.summary.scalar("prediction_error", self.prediction_error)

        # We can use no_op if we're not wanting to generate a summary. Makes logic somewhat easier.
        # Session run( summarize and self.summarize or self.no_op1 ).
        self.summarize = tf.summary.merge_all() # Tensorboard summaries
        self.no_op1 = tf.no_op()

        # Define OP to Initialise all tf variables.
        init = tf.global_variables_initializer() # Latest 0.12rc API variant of tf.initialize_all_variables()
        #self.s.run(tf.initialize_all_variables())

        with tf.name_scope("target_network_update"):
            self.target_q_network_update_op = []

            #Wsb contains [Weights, Bias] therefore, we need to calculate these differences twice.
            for i in range( len(self.q_network_Wsb )):
                for q, t in zip( self.q_network_Wsb[i].values(), self.target_q_network_Wsb[i].values() ):

                    # Obtain variables, eg 'out' for the weights, in zipped format.
                    # Create a subtraction op equal to the distance.
                    # q -= t-q (difference).
                    self.target_q_network_update_op.append(q.assign_sub(t - q))

            # Group all the sub-operations, into a big super-operation to be applied.
            self.target_q_network_update_op = tf.group(*self.target_q_network_update_op)

        # Update our Target Q Network
        self.s.run(init)
        self.s.run(self.target_q_network_update_op) #Apply immediately!

    def add_experience(self, s, a, r, st):
        # Automatically popleft's when "maxlen=self.experience_replay_size"
        self.experience_replay.append((s, a, r, st))

    def training(self):
        if len(self.experience_replay) < self.experience_replay_size:
            return

        if self.training_n % self.training_frequency == 0:
            # We can train

            samples = random.sample(range(len(self.experience_replay)), 1)
            samples = [self.experience_replay[i] for i in samples]

            states_t = np.empty(len(samples))
            states_t2 = np.empty(len(samples))
            rewards = np.empty((len(samples),))

            for i, ( s, a, r, st ) in samples:
                states_t[i] = s
                rewards[i] = r
                if st is not None:
                    states_t2[i] = st
                else:
                    states_t2[i] = 0

            summarize = False
            self.s.run( [self.prediction_error, self.train_op, self.summarize if summarize else self.no_op1],
                        {self.q_in: states_t,
                        self.target_q_in: states_t2,
                        self.rewards: rewards}
                        )

            if self.training_n % self.target_q_network_update_frequency == 0:
                self.s.run(self.target_q_network_update_op)

            self.training_n += 1


