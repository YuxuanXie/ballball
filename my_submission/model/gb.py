# Model taken from https://arxiv.org/pdf/1810.08647.pdf,
# INTRINSIC SOCIAL MOTIVATION VIA CAUSAL
# INFLUENCE IN MULTI-AGENT RL


# model is a single convolutional layer with a kernel of size 3, stride of size 1, and 6 output
# channels. This is connected to two fully connected layers of size 32 each

import tensorflow as tf
from ray.rllib.models.tf.misc import normc_initializer
from ray.rllib.models.tf.recurrent_tf_modelv2 import RecurrentTFModelV2, TFModelV2
from ray.rllib.utils.annotations import override
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.policy.rnn_sequencing import add_time_dimension
import numpy as np



class GoBigger(RecurrentTFModelV2):
    """Custom model for policy gradient algorithms."""

    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name):
        super(ConvToFCNet, self).__init__(obs_space, action_space, num_outputs, model_config, name)
        hiddens_size = 32 
        self.cell_size = self.model_config['lstm_cell_size'] 
        visual_size = np.prod(obs_space.shape)

        # Input layers for lstm
        state_in_h = tf.keras.layers.Input(shape=(self.cell_size, ), name="h")
        state_in_c = tf.keras.layers.Input(shape=(self.cell_size, ), name="c")
        seq_in = tf.keras.layers.Input(shape=(), name="seq_in", dtype=tf.int32)

        # Input layers for obs
        inputs = tf.keras.layers.Input(shape=(None, visual_size), name="visual_inputs")
        input_visual = inputs

        input_visual = tf.reshape(input_visual, [-1, obs_space.shape[0], obs_space.shape[1], obs_space.shape[2]])
        conv1 = tf.keras.layers.Conv2D(6, [3,3], 1, activation=tf.nn.relu)(input_visual)
        conv1 = tf.keras.layers.Flatten()(conv1) # tf.reshape(conv1, [-1, np.prod(conv1.shape.as_list()[1:])])
        dense1 = tf.keras.layers.Dense(hiddens_size, name="fc1", activation=tf.nn.relu, kernel_initializer=normc_initializer(1.0))(conv1)
        dense2 = tf.keras.layers.Dense(hiddens_size, name="fc2", activation=tf.nn.relu, kernel_initializer=normc_initializer(1.0))(dense1)
        dense2 = tf.reshape(dense2, [-1, tf.shape(inputs)[1], dense2.shape.as_list()[-1]])

        import pdb; pdb.set_trace()
        lstm_out, state_h, state_c = tf.keras.layers.LSTM(self.cell_size, return_sequences=True, return_state=True, name="lstm")(
                inputs=dense2, initial_state=[state_in_h, state_in_c])
        
        values = tf.keras.layers.Dense(1, name="values", activation=None, kernel_initializer=normc_initializer(0.01))(lstm_out)
        logits = tf.keras.layers.Dense(num_outputs, name="logits", activation=tf.keras.activations.linear, kernel_initializer=normc_initializer(0.01))(lstm_out)

        # Create the RNN model
        self.rnn_model = tf.keras.Model(
            inputs=[inputs, seq_in, state_in_h, state_in_c],
            outputs=[logits, values, state_h, state_c])
        self.register_variables(self.rnn_model.variables)
        # self.rnn_model.summary()

    @override(RecurrentTFModelV2)
    def forward_rnn(self, inputs, state, seq_lens):
        model_out, self._value_out, h, c = self.rnn_model([inputs, seq_lens] +
                                                          state)
        return model_out, [h, c]

    @override(ModelV2)
    def value_function(self):
        return tf.reshape(self._value_out, [-1])

    @override(ModelV2)
    def get_initial_state(self):
        return [
                np.zeros(self.cell_size, np.float32),
                np.zeros(self.cell_size, np.float32)
        ]

    def metrics(self):
        return {}


