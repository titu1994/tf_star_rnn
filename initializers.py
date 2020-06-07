import numpy as np
import tensorflow as tf


class StarChronoInitializer(tf.keras.initializers.Initializer):

    def __init__(self, num_gates, t_max, seed=None):
        super(StarChronoInitializer, self).__init__()

        self.num_gates = num_gates
        self.t_max = t_max
        self.seed = seed

    def __call__(self, shape, dtype=None):
        """Returns a tensor object initialized as specified by the initializer.

        Args:
          shape: Shape of the tensor.
          dtype: Optional dtype of the tensor. If not provided will return tensor
           of `tf.float32`.
        """
        num_units = shape[0] // self.num_gates
        uni_vals = tf.math.log(tf.random.uniform([num_units], minval=1.0,
                                                 maxval=self.t_max, dtype=dtype,
                                                 seed=self.seed))

        bias_j = tf.zeros(num_units)
        bias_f = uni_vals

        return tf.concat([bias_j, bias_f], 0)

    def get_config(self):
        """Returns the configuration of the initializer as a JSON-serializable dict.

        Returns:
          A JSON-serializable Python dict.
        """
        config = {
            'num_gates': self.num_gates,
            't_max': self.t_max,
            'seed': self.seed,
        }

        return config


class StarBiasInitializer(tf.keras.initializers.Initializer):

    def __init__(self, num_gates):
        super(StarBiasInitializer, self).__init__()

        self.num_gates = num_gates

    def __call__(self, shape, dtype=None):
        """Returns a tensor object initialized as specified by the initializer.

        Args:
          shape: Shape of the tensor.
          dtype: Optional dtype of the tensor. If not provided will return tensor
           of `tf.float32`.
        """
        p = np.zeros(shape)
        num_units = int(shape[0] // self.num_gates)
        # `forget` gate : ones init
        p[-num_units:] = np.ones(num_units)
        return tf.constant(p, dtype)

    def get_config(self):
        """Returns the configuration of the initializer as a JSON-serializable dict.

        Returns:
          A JSON-serializable Python dict.
        """
        config = {
            'num_gates': self.num_gates,
        }

        return config
