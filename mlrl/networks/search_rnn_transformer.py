from mlrl.meta.search_networks import SearchTransformer

import tensorflow as tf
from official.nlp.modeling.layers import transformer

from tf_agents.keras_layers import dynamic_unroll_layer
from tf_agents.networks import network
from tf_agents.networks import lstm_encoding_network
from tf_agents.trajectories import time_step
from tf_agents.utils import nest_utils
from tf_agents.specs import tensor_spec


class SearchRNNTransformer(network.Network):
    """
    Network that combines a transformer with an LSTM.
    A decode block projects information from the search tokens
    onto a one-dimensional vector.
    This is used as input to the LSTM along with the previous state.
    The output of the LSTM is then added to each token and the
    result is passed through a transformer.
    """

    def __init__(self,
                 input_tensor_spec,
                 n_heads: int = 3,
                 n_transformer_layers: int = 3,
                 d_model: int = 64,
                 n_lstm_layers: int = 1,
                 name='search_rnn_transformer',
                 dtype=tf.float32,
                 **kwargs):
        """
        Args:
            input_tensor_spec: A nest of `tensor_spec.TensorSpec` representing
                the input observations.
            n_heads: Number of heads in the transformer.
            n_transformer_layers: Number of transformer layers.
            d_model: Dimension of tokens in the transformer model.
            n_lstm_layers: Number of lstm layers.
        """
        super(SearchRNNTransformer, self).__init__(**kwargs)

        self.decoder_block = transformer.TransformerDecoderBlock(
            num_attention_heads=n_heads,
            intermediate_size=d_model * n_heads,
            intermediate_activation='relu',
            dropout_rate=0.1,
            attention_dropout_rate=0.1,
            use_bias=False,
            norm_first=True,
            norm_epsilon=1e-6,
            intermediate_dropout=0.1,
            attention_initializer=tf.keras.initializers.RandomUniform(
                minval=1., maxval=1.)
        )

        cell = tf.keras.layers.StackedRNNCells([
            tf.keras.layers.LSTMCell(d_model * n_heads, dtype=dtype,
                                     implementation=lstm_encoding_network.KERAS_LSTM_FUSED)
            for _ in range(n_lstm_layers)
        ])
        self.lstm_network = dynamic_unroll_layer.DynamicUnroll(cell)

        self.project_search_tokens = tf.keras.layers.Dense(
            d_model * n_heads, name='project_search_tokens'
        )

        self.transformer = SearchTransformer(
            n_heads=n_heads, n_layers=n_transformer_layers, d_model=d_model
        )

        counter = [-1]

        def create_spec(size):
            counter[0] += 1
            return tensor_spec.TensorSpec(
                size, dtype=dtype, name='network_state_%d' % counter[0])

        state_spec = tf.nest.map_structure(create_spec,
                                           self.lstm_network.cell.state_size)

        super(SearchRNNTransformer, self).__init__(
            input_tensor_spec=input_tensor_spec, state_spec=state_spec, name=name)

    def call(self, search_tokens, step_type, network_state=(), training=False):
        num_outer_dims = nest_utils.get_outer_rank(search_tokens,
                                                   self.input_tensor_spec)
        if num_outer_dims not in (1, 2):
            raise ValueError(
                'Input observation must have a batch or batch x time outer shape.')

        has_time_dim = num_outer_dims == 2
        if not has_time_dim:
            # Add a time dimension to the inputs.
            search_tokens = tf.nest.map_structure(lambda t: tf.expand_dims(t, 1),
                                                  search_tokens)
            step_type = tf.nest.map_structure(lambda t: tf.expand_dims(t, 1),
                                              step_type)

        # resets network state for timesteps with step_type == FIRST
        network_kwargs = {
            'reset_mask': tf.equal(step_type, time_step.StepType.FIRST, name='mask')
        }

        # Prepare lstm inputs for each timestep.
        # This means using a decoder transformer layer from the search tokens at
        # each timestep to single token that servers as the input for the lstm.
        batch_size = tf.shape(search_tokens)[0]
        n_time_steps = tf.shape(search_tokens)[1]
        n_tokens = tf.shape(search_tokens)[2]

        # TODO: switch this to a tf.map_fn
        lstm_inputs_across_time = []
        for t in range(n_time_steps):
            inputs = tf.zeros((batch_size, 1, 192))
            decoder_inputs = [inputs, search_tokens[:, t], None, None]
            lstm_inputs, _ = self.decoder_block(decoder_inputs)
            lstm_inputs_across_time.append(lstm_inputs)

        lstm_inputs_across_time = tf.concat(lstm_inputs_across_time, axis=1)

        # Apply inputs and network state to the lstm
        memory_encoding, network_state = self.lstm_network(
            inputs=lstm_inputs_across_time,
            initial_state=network_state,
            training=training,
            **network_kwargs)

        # reshape the output of the lstm to be added to each of the search tokens
        # thus encorporating information from previous timesteps into the tokens
        memory_encoding = tf.expand_dims(memory_encoding, 2)
        memory_encoding = tf.repeat(memory_encoding, n_tokens, axis=2)

        tokens = self.project_search_tokens(search_tokens)
        tokens = tokens + memory_encoding

        # swap the time and batch dimensions to be able to apply the transformer
        # to each of the token sets across time
        tokens = tf.einsum('btnd->tbnd', tokens)
        tokens = tf.map_fn(self.transformer, tokens)
        tokens = tf.einsum('tbnd->btnd', tokens)

        if not has_time_dim:
            # Remove time dimension from the state.
            tokens = tf.squeeze(tokens, [1])

        return tokens, network_state
