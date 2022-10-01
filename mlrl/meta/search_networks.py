import tensorflow as tf
import tensorflow_probability as tfp
from official.nlp.modeling.layers import Transformer


class PrependTerminateToken(tf.keras.layers.Layer):
    """
    Simple layer to prepend a zero vector to a sequence of tokens
    """

    def __init__(self):
        super().__init__()

    def call(self, inputs, training=False):
        batch_size = tf.shape(inputs)[0]
        n_tokens = tf.shape(inputs)[1]
        token_dim = tf.shape(inputs)[2]

        terminate_token = tf.concat([
            tf.zeros((batch_size, 1, token_dim)),
            tf.ones((batch_size, 1, 1))
        ], axis=-1)

        inputs_with_terminate = tf.concat([
            inputs, tf.zeros((batch_size, n_tokens, 1))
        ], axis=-1)

        return tf.concat([
            terminate_token, inputs_with_terminate
        ], axis=1)


class PrependTerminateMask(tf.keras.layers.Layer):
    """
    Simple layer to prepend the mask
    """

    def __init__(self):
        super().__init__()

    def call(self, inputs, training=False):
        batch_size = tf.shape(inputs)[0]
        return tf.concat([
            tf.ones((batch_size, 1)), inputs
        ], axis=1)


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

        self.prepend_terminate_token = PrependTerminateToken()
        self.prepend_terminate_mask = PrependTerminateMask()

        self.transformer_layers = [
            Transformer(n_heads, head_dim, 'relu',
                        dropout_rate=0.1,
                        attention_dropout_rate=0.1)
            for _ in range(n_layers)
        ]

    def call(self, inputs, training=False):
        tokens = self.prepend_terminate_token(inputs, training=training)
        tokens = self.project_tokens(tokens,
                                     training=training)

        attention_mask = self.prepend_terminate_mask(inputs[:, :, 0],
                                                     training=training)
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

    def call(self, search_tokens, training=False):
        tokens = self.transformer(search_tokens, training=training)
        x = tf.reduce_sum(tokens, axis=-2)
        return self.to_value(x)

    def get_config(self):
        config = super().get_config()
        config.update(self.transformer.get_config())
        return config


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

        self.prep_mask = PrependTerminateMask()

    def call(self, search_tokens, training=False):
        mask = tf.cast(self.prep_mask(search_tokens[:, :, 1]), tf.bool)

        tokens = self.transformer(search_tokens, training=training)
        action_logits = self.to_logits(tokens, training=training)
        action_logits = tf.where(mask, action_logits, tf.float32.min)

        if self.relaxed_one_hot:
            action_dist = tfp.distributions.RelaxedOneHotCategorical(
                self.temperature, logits=action_logits
            )
        else:
            action_dist = tfp.distributions.Categorical(
                logits=action_logits, dtype=tf.int64
            )

        return action_dist

    def get_config(self):
        config = super().get_config()
        config.update({
            'transformer': self.transformer.get_config(),
            'temperature': self.temperature,
            'relaxed_one_hot': self.relaxed_one_hot
        })
        return config
