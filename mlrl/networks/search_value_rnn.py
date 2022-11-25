from mlrl.networks.search_rnn_transformer import SearchRNNTransformer
from mlrl.meta.meta_env import MetaEnv

import silence_tensorflow.auto  # noqa
import tensorflow as tf
from tf_agents.networks import network


class ValueSearchRNN(network.Network):

    def __init__(self,
                 input_tensor_spec,
                 n_heads: int = 3,
                 n_transformer_layers: int = 3,
                 d_model: int = 64,
                 n_lstm_layers: int = 1,
                 name='value_search_rnn',
                 **_):
        """
        Args:
            input_tensor_spec: A nest of `tensor_spec.TensorSpec` representing
                the input observations.
            n_heads: Number of heads in the transformer.
            n_transformer_layers: Number of transformer layers.
            d_model: Dimension of tokens in the transformer model.
            n_lstm_layers: Number of lstm layers.
        """

        self.tokens_spec = input_tensor_spec[MetaEnv.SEARCH_TOKENS_KEY]
        self.transformer = SearchRNNTransformer(
            self.tokens_spec,
            n_heads=n_heads,
            n_transformer_layers=n_transformer_layers,
            d_model=d_model,
            n_lstm_layers=n_lstm_layers
        )

        self.to_value = tf.keras.layers.Dense(
            1,
            activation=None,
            kernel_initializer=tf.random_uniform_initializer(
                minval=-0.03, maxval=0.03))

        super(ValueSearchRNN, self).__init__(
            input_tensor_spec=input_tensor_spec,
            state_spec=self.transformer.state_spec,
            name=name)

    def compute_value(self,
                      search_tokens,
                      step_type,
                      network_state=(),
                      training=False):

        tokens, network_state = self.transformer(search_tokens,
                                                 step_type=step_type,
                                                 network_state=network_state,
                                                 training=training)
        x = tf.reduce_sum(tokens, axis=-2)
        return tf.squeeze(self.to_value(x), axis=-1), network_state

    def call(self,
             observation,
             step_type=None,
             network_state=(),
             training=False):

        if tf.nest.is_nested(observation):
            tokens = observation[MetaEnv.SEARCH_TOKENS_KEY]
        else:
            tokens = observation

        return self.compute_value(tokens, step_type, network_state, training=training)
