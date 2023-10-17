from rlts.networks.search_transformer import SearchTransformer

import silence_tensorflow.auto  # noqa
import tensorflow as tf


class SearchQNetwork(tf.keras.Model):
    """
    Transformer-based model for Q-value estimation of search-tree tokens.

    The model takes as input a batch of search-tree tokens, and outputs a
    batch of Q-values for each possible computational actions.

    The advantage of using a transformer-based model is that it
    treats the search-tree tokens as a set, and thus is invariant to the
    order of the tokens. The structure of the search tree is encoded in the tokens.
    """

    def __init__(self,
                 n_heads: int = 3,
                 d_model: int = 16,
                 n_layers: int = 2,
                 name='search_q_model',
                 **kwargs):
        super().__init__(name=name, **kwargs)

        self.transformer = SearchTransformer(n_heads, d_model, n_layers)

        kernel_init = tf.keras.initializers.RandomUniform(
            minval=-0.03, maxval=0.03
        )
        bias_init = tf.keras.initializers.Constant(-0.2)

        self.to_q_vals = tf.keras.Sequential([
            tf.keras.layers.Dense(
                1,
                kernel_initializer=kernel_init,
                bias_initializer=bias_init),
            tf.keras.layers.Flatten()
        ], name='to_q_vals')

    def call(self, inputs, training=False):
        tokens = self.transformer(inputs, training=training)
        return self.to_q_vals(tokens, training=training)

    def get_config(self):
        config = super().get_config()
        config.update(self.transformer.get_config())
        return config
