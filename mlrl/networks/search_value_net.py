from mlrl.networks.search_transformer import SearchTransformer

import tensorflow as tf
from tf_agents.networks.value_network import ValueNetwork
from tf_agents.specs import tensor_spec


def create_value_network(observation_tensor_spec: tensor_spec.TensorSpec,
                         **network_kwargs) -> ValueNetwork:
    return ValueNetwork(
        observation_tensor_spec,
        preprocessing_layers=SearchTransformer(**network_kwargs),
        preprocessing_combiner=tf.keras.layers.Lambda(lambda x: tf.reduce_sum(x[0], axis=-2)),
        batch_squash=True,
        fc_layer_params=None
    )


class SearchValueNetwork(tf.keras.Model):
    def __init__(self,
                 n_heads=3,
                 d_model=16,
                 n_layers=2,
                 name='search_value_network',
                 **kwargs):
        super().__init__(name=name, **kwargs)
        self.transformer = SearchTransformer(n_heads, d_model, n_layers)
        self.to_value = tf.keras.layers.Dense(1)

    def compute_value(self, search_tokens, training=False):
        tokens = self.transformer(search_tokens, training=training)
        x = tf.reduce_sum(tokens, axis=-2)
        return tf.squeeze(self.to_value(x), axis=-1)

    def call(self, inputs, training=False):
        if inputs.shape.rank == 4:
            # Handle multiple time steps independently
            return tf.map_fn(
                lambda tokens: self.compute_value(tokens,
                                                  training=training),
                inputs
            )

        return self.compute_value(inputs, training=training)

    def get_config(self):
        config = super().get_config()
        config.update(self.transformer.get_config())
        return config
