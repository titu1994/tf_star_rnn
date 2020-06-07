import tensorflow as tf
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.training.tracking.data_structures import NoDependency

import initializers


class STARCell(tf.keras.layers.GRUCell):

    def __init__(self, units, t_max,
                 activation='tanh',
                 use_bias=True,
                 kernel_initializer='orthogonal',
                 recurrent_initializer='orthogonal',
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 bias_regularizer=None,
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 bias_constraint=None,
                 dropout=0.,
                 recurrent_dropout=0.,
                 **kwargs):
        super(STARCell, self).__init__(units,
                                       activation=activation,
                                       use_bias=use_bias,
                                       kernel_initializer=kernel_initializer,
                                       recurrent_initializer=recurrent_initializer,
                                       kernel_regularizer=kernel_regularizer,
                                       recurrent_regularizer=recurrent_regularizer,
                                       bias_regularizer=bias_regularizer,
                                       kernel_constraint=kernel_constraint,
                                       recurrent_constraint=recurrent_constraint,
                                       bias_constraint=bias_constraint,
                                       dropout=dropout,
                                       recurrent_dropout=recurrent_dropout,
                                       **kwargs)

        self.t_max = t_max
        self.state_size = NoDependency([self.state_size])

    @tf_utils.shape_type_conversion
    def build(self, input_shape):
        input_dim = input_shape[-1]
        self.kernel = self.add_weight(
            shape=(input_dim, self.units),
            name='kernel_xh_K',
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint)
        self.kernel_OBS = self.add_weight(
            shape=(input_dim, self.units),
            name='kernel_xh_z',
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint
        )
        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.units),
            name='recurrent_kernel',
            initializer=self.recurrent_initializer,
            regularizer=self.recurrent_regularizer,
            constraint=self.recurrent_constraint)

        if self.use_bias:
            if self.t_max is None:
                # Zeros initializer
                self.bias = self.add_weight(shape=[2 * self.units],
                                            name='bias',
                                            initializer=initializers.StarBiasInitializer(num_gates=2),
                                            regularizer=self.bias_regularizer,
                                            constraint=self.bias_constraint)

            else:
                self.bias = self.add_weight(shape=[2 * self.units],
                                            name='bias',
                                            initializer=initializers.StarChronoInitializer(num_gates=2,
                                                                                           t_max=self.t_max),
                                            regularizer=self.bias_regularizer,
                                            constraint=self.bias_constraint)

        else:
            self.bias = None
        self.built = True

    def call(self, inputs, states, training=None):
        h_tm1 = states[0]  # previous memory state

        all_inputs = tf.concat([inputs, h_tm1], 1)
        weights_K = tf.concat([self.kernel, self.recurrent_kernel], 0)

        if self.use_bias:
            bias_K = self.bias[self.units:, ...]
            bias_OBS = self.bias[:self.units, ...]
        else:
            bias_K = None
            bias_OBS = None

        f = tf.matmul(all_inputs, weights_K)
        j = tf.matmul(inputs, self.kernel_OBS)

        if self.use_bias:
            f = tf.nn.bias_add(f, bias_K)
            j = tf.nn.bias_add(j, bias_OBS)

        beta = 1.0
        new_h = tf.sigmoid(f) * h_tm1 + (1. - tf.sigmoid(f - beta)) * self.activation(j)
        new_h = self.activation(new_h)
        return new_h, [new_h]

    def get_config(self):
        config = {
            't_max': self.t_max,
        }
        base_config = super(STARCell, self).get_config()
        base_config.pop('recurrent_activation')
        base_config.pop('bias_initializer')
        return dict(list(base_config.items()) + list(config.items()))
