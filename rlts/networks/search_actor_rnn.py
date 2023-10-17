from rlts.networks.search_rnn_transformer import SearchRNNTransformer
from rlts.meta.meta_env import MetaEnv

import silence_tensorflow.auto  # noqa
import tensorflow as tf
import tensorflow_probability as tfp
from tf_agents.networks import network


class ActionSearchRNN(network.Network):
    """
    Network that takes in a search token sequence
    and outputs a distribution over actions.
    Using a recurrent transformer to encode the sequence.
    """

    def __init__(self,
                 input_tensor_spec,
                 n_heads: int = 3,
                 n_transformer_layers: int = 3,
                 d_model: int = 64,
                 n_lstm_layers: int = 1,
                 name='action_search_rnn',
                 **_):
        """
        Args:
            input_tensor_spec: A nest of `tensor_spec.TensorSpec` representing
                the input observations.
            n_heads: Number of heads in the transformer.
            n_transformer_layers: Number of transformer layers.
            d_model: Dimension of tokens in the transformer model.
            n_lstm_layers: Number of lstm layers.

        Note: unlike normal actor networks, this does not need to
        take the action spec as an argument. This is because the number of
        actions is determined by the size of the search token sequence.
        """

        self.tokens_spec = input_tensor_spec[MetaEnv.SEARCH_TOKENS_KEY]
        self.transformer = SearchRNNTransformer(
            self.tokens_spec,
            n_heads=n_heads,
            n_transformer_layers=n_transformer_layers,
            d_model=d_model,
            n_lstm_layers=n_lstm_layers
        )

        self.to_logit = tf.keras.layers.Dense(
            1,
            activation=None,
            kernel_initializer=tf.random_uniform_initializer(
                minval=-0.03, maxval=0.03))

        super(ActionSearchRNN, self).__init__(
            input_tensor_spec=input_tensor_spec,
            state_spec=self.transformer.state_spec,
            name=name)

    def call(self,
             observation,
             step_type=None,
             network_state=(),
             training=False):

        if tf.nest.is_nested(observation):
            tokens = observation[MetaEnv.SEARCH_TOKENS_KEY]
            mask = tf.cast(observation[MetaEnv.ACTION_MASK_KEY], tf.bool)
        else:
            tokens = observation

        tokens, network_state = self.transformer(tokens,
                                                 step_type=step_type,
                                                 network_state=network_state,
                                                 training=training)

        action_logits = tf.squeeze(self.to_logit(tokens, training=training), -1)
        action_logits = tf.where(mask, action_logits, tf.float32.min)

        action_distribution = tfp.distributions.Categorical(
            logits=action_logits, dtype=tf.int64
        )
        return action_distribution, network_state
