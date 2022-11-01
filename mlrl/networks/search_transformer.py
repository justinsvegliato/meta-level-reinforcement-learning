import silence_tensorflow.auto  # pylint: disable=unused-import
import tensorflow as tf
from official.nlp.modeling.layers import Transformer


class SearchTransformer(tf.keras.Model):
    """
    Transformer model for search tree tokens
    """

    def __init__(self,
                 n_heads: int = 3,
                 d_model: int = 16,
                 n_layers: int = 2,
                 name='search_transformer',
                 **kwargs):
        super().__init__(name=name, **kwargs)
        self.n_heads = n_heads
        self.d_model = d_model
        self.n_layers = n_layers

        self.project_tokens = tf.keras.layers.Dense(d_model * n_heads,
                                                    activation='relu')

        self.transformer_layers = [
            Transformer(n_heads, d_model, 'relu',
                        dropout_rate=0.1,
                        attention_dropout_rate=0.1)
            for _ in range(n_layers)
        ]

    def call(self, inputs, training=False):

        from mlrl.meta.meta_env import MetaEnv
        if tf.nest.is_nested(inputs):
            tokens = inputs[MetaEnv.SEARCH_TOKENS_KEY]
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
            'd_model': self.d_model,
            'n_layers': self.n_layers
        })
        return config
