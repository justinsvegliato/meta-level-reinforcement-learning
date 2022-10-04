import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from official.nlp.modeling.layers import Transformer

from tf_agents.networks import network
from tf_agents.specs import tensor_spec
from tf_agents.specs import distribution_spec
from tf_agents.networks.actor_distribution_network import ActorDistributionNetwork
from tf_agents.networks.value_network import ValueNetwork


class SearchTransformer(tf.keras.Model):
    """
    Transformer model for search tree tokens
    """

    def __init__(self, n_heads, head_dim, n_layers, **kwargs):
        super().__init__(**kwargs)
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.n_layers = n_layers

        self.project_tokens = tf.keras.layers.Dense(head_dim * n_heads,
                                                    activation='relu')

        self.transformer_layers = [
            Transformer(n_heads, head_dim, 'relu',
                        dropout_rate=0.1,
                        attention_dropout_rate=0.1)
            for _ in range(n_layers)
        ]

    def call(self, inputs, training=False):

        if tf.nest.is_nested(inputs):
            tokens = inputs['search_tree_tokens']
        else:
            tokens = inputs

        tokens = self.project_tokens(tokens,
                                     training=training)

        attention_mask = tokens[:, :, 0]
        attention_mask = attention_mask[:, tf.newaxis, tf.newaxis, :]
        attention_mask = tf.repeat(attention_mask, self.n_heads, axis=1)
        attention_mask = tf.repeat(attention_mask, tf.shape(attention_mask)[-1], axis=2)

        for layer in self.transformer_layers:
            tokens = layer([tokens, attention_mask], training=training)

        return tokens

    def get_config(self):
        config = super().get_config()
        config.update({
            'n_heads': self.n_heads,
            'head_dim': self.head_dim,
            'n_layers': self.n_layers
        })
        return config


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
                 head_dim: int = 16,
                 n_layers: int = 2,
                 name='search_q_model',
                 **kwargs):
        super().__init__(name=name, **kwargs)

        self.transformer = SearchTransformer(n_heads, head_dim, n_layers)

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


class SearchValueNetwork(tf.keras.Model):
    def __init__(self,
                 n_heads=3,
                 head_dim=16,
                 n_layers=2,
                 name='search_value_network',
                 **kwargs):
        super().__init__(name=name, **kwargs)
        self.transformer = SearchTransformer(n_heads, head_dim, n_layers)
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


def get_action_mask(search_tokens: tf.Tensor) -> tf.Tensor:
    return tf.cast(search_tokens[:, :, 1], tf.bool)


class SearchActorNetwork(tf.keras.Model):

    def __init__(self,
                 n_heads=3,
                 head_dim=16,
                 n_layers=2,
                 temperature=1.0,
                 name='search_actor_model',
                 relaxed_one_hot=False,
                 **kwargs):
        super().__init__(name=name, **kwargs)
        self.temperature = temperature
        self.relaxed_one_hot = relaxed_one_hot

        self.transformer = SearchTransformer(n_heads, head_dim, n_layers)

        self.to_logits = tf.keras.Sequential([
            tf.keras.layers.Dense(1),
            tf.keras.layers.Flatten()
        ], name='to_logits')

    def compute_logits(self, search_tokens, training=False):
        mask = get_action_mask(search_tokens)
        tokens = self.transformer(search_tokens, training=training)
        action_logits = self.to_logits(tokens, training=training)
        action_logits = tf.where(mask, action_logits, tf.float32.min)
        return action_logits

    def call(self, inputs, training=False):
        if inputs.shape.rank == 4:
            logits = tf.map_fn(
                lambda x: self.compute_logits(x, training=training),
                inputs
            )
        else:
            logits = self.compute_logits(inputs, training=training)

        if self.relaxed_one_hot:
            action_dist = tfp.distributions.RelaxedOneHotCategorical(
                self.temperature, logits=logits
            )
        else:
            action_dist = tfp.distributions.Categorical(
                logits=logits, dtype=tf.int64
            )

        return action_dist

    def get_config(self):
        config = super().get_config()
        config.update({
            'temperature': self.temperature,
            'relaxed_one_hot': self.relaxed_one_hot,
            **self.transformer.get_config()
        })
        return config


class SearchActorLogitsNetwork(tf.keras.Model):

    def __init__(self,
                 n_heads=3,
                 head_dim=16,
                 n_layers=2,
                 name='search_actor_logits',
                 **kwargs):
        super().__init__(name=name, **kwargs)

        self.transformer = SearchTransformer(n_heads, head_dim, n_layers)

        self.to_logits = tf.keras.Sequential([
            tf.keras.layers.Dense(1),
            tf.keras.layers.Flatten()
        ], name='to_logits')

    def compute_logits(self, search_tokens, training=False):
        mask = get_action_mask(search_tokens)
        tokens = self.transformer(search_tokens, training=training)
        action_logits = self.to_logits(tokens, training=training)
        action_logits = tf.where(mask, action_logits, tf.float32.min)
        return action_logits

    def call(self, inputs, training=False):
        if inputs.shape.rank == 4:
            logits = tf.map_fn(
                lambda x: self.compute_logits(x, training=training),
                inputs
            )
        else:
            logits = self.compute_logits(inputs, training=training)

        return logits

    def get_config(self):
        config = super().get_config()
        config.update(self.transformer.get_config())
        return config


class CategoricalNetwork(network.DistributionNetwork):

    def __init__(self,
                 sample_spec,
                 name='CategoricalNetwork'):

        unique_num_actions = np.unique(sample_spec.maximum - sample_spec.minimum + 1)
        if len(unique_num_actions) > 1 or np.any(unique_num_actions <= 0):
            raise ValueError('Bounds on discrete actions must be the same for all '
                             'dimensions and have at least 1 action. Projection '
                             'Network requires num_actions to be equal across '
                             'action dimensions. Implement a more general '
                             'categorical projection if you need more flexibility.')

        output_shape = sample_spec.shape.concatenate([int(unique_num_actions)])
        output_spec = self._output_distribution_spec(output_shape, sample_spec,
                                                     name)

        super(CategoricalNetwork, self).__init__(
            # We don't need these, but base class requires them.
            input_tensor_spec=None,
            state_spec=(),
            output_spec=output_spec,
            name=name)

        if not tensor_spec.is_bounded(sample_spec):
            raise ValueError(
                'sample_spec must be bounded. Got: %s.' % type(sample_spec))

        if not tensor_spec.is_discrete(sample_spec):
            raise ValueError('sample_spec must be discrete. Got: %s.' % sample_spec)

        self._sample_spec = sample_spec
        self._output_shape = output_shape

    def _output_distribution_spec(self, output_shape, sample_spec, network_name):
        input_param_spec = {
            'logits':
                tensor_spec.TensorSpec(
                    shape=output_shape,
                    dtype=tf.float32,
                    name=network_name + '_logits')
        }

        return distribution_spec.DistributionSpec(
            tfp.distributions.Categorical,
            input_param_spec,
            sample_spec=sample_spec,
            dtype=sample_spec.dtype)

    def call(self, logits, outer_rank, training=False, mask=None):
        return self.output_spec.build_distribution(logits=logits), ()


def create_action_distribution_network(observation_tensor_spec: tensor_spec.TensorSpec,
                                       action_tensor_spec: tensor_spec.TensorSpec,
                                       **network_kwargs) -> ActorDistributionNetwork:
    custom_objects = {
        'SearchActorLogitsNetwork': SearchActorLogitsNetwork,
        'SearchTransformer': SearchTransformer
    }

    with tf.keras.utils.custom_object_scope(custom_objects):
        return ActorDistributionNetwork(
            observation_tensor_spec, action_tensor_spec,
            preprocessing_layers=SearchActorLogitsNetwork(**network_kwargs),
            fc_layer_params=None,
            discrete_projection_net=lambda spec: CategoricalNetwork(spec)
        )


def create_value_network(observation_tensor_spec: tensor_spec.TensorSpec,
                         **network_kwargs) -> ValueNetwork:
    return ValueNetwork(
        observation_tensor_spec,
        preprocessing_layers=SearchTransformer(**network_kwargs),
        preprocessing_combiner=tf.keras.layers.Lambda(lambda x: tf.reduce_sum(x[0], axis=-2)),
        batch_squash=True,
        fc_layer_params=None
    )