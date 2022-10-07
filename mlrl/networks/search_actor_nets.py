from mlrl.networks.search_transformer import SearchTransformer

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from tf_agents.networks import network
from tf_agents.specs import tensor_spec
from tf_agents.specs import distribution_spec
from tf_agents.networks.actor_distribution_network import ActorDistributionNetwork


def get_action_mask(search_tokens: tf.Tensor) -> tf.Tensor:
    return search_tokens[:, :, 1] > 0.5


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


class SearchActorNetwork(tf.keras.Model):

    def __init__(self,
                 n_heads=3,
                 d_model=16,
                 n_layers=2,
                 temperature=1.0,
                 name='search_actor_model',
                 relaxed_one_hot=False,
                 **kwargs):
        super().__init__(name=name, **kwargs)
        self.temperature = temperature
        self.relaxed_one_hot = relaxed_one_hot

        self.transformer = SearchTransformer(n_heads, d_model, n_layers)

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
                 d_model=16,
                 n_layers=2,
                 name='search_actor_logits',
                 **kwargs):
        super().__init__(name=name, **kwargs)

        self.transformer = SearchTransformer(n_heads, d_model, n_layers)

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

        if mask is not None:
            # If the action spec says each action should be shaped (1,), add another
            # dimension so the final shape is (B, 1, A), where A is the number of
            # actions. This will make Categorical emit events shaped (B, 1) rather
            # than (B,). Using axis -2 to allow for (B, T, 1, A) shaped q_values.
            if mask.shape.rank < logits.shape.rank:
                mask = tf.expand_dims(mask, -2)

            # Overwrite the logits for invalid actions to a very large negative
            # number. We do not use -inf because it produces NaNs in many tfp
            # functions.
            almost_neg_inf = tf.constant(logits.dtype.min, dtype=logits.dtype)
            logits = tf.compat.v2.where(tf.cast(mask, tf.bool), logits, almost_neg_inf)

        return self.output_spec.build_distribution(logits=logits), ()
