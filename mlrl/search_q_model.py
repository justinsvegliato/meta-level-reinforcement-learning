import tensorflow as tf
from official.nlp.modeling.layers import Transformer


class PrependTerminateToken(tf.keras.layers.Layer):
    """
    Simple layer to prepend a zero vector to a sequence of tokens
    """

    def __init__(self):
        super().__init__()

    def call(self, inputs, training=False):
        batch_size = tf.shape(inputs)[0]
        token_dim = tf.shape(inputs)[-1]
        return tf.concat([tf.zeros((batch_size, 1, token_dim)), inputs], axis=1)


class SearchQModel(tf.keras.Model):
    """
    Transformer-based model for Q-value estimation of search-tree tokens.

    The model takes as input a batch of search-tree tokens, and outputs a
    batch of Q-values for each possible computational actions.

    The advantage of using a transformer-based model is that it 
    treats the search-tree tokens as a set, and thus is invariant to the order of the tokens.
    The structure of the search tree is encoded in the tokens.
    """

    def __init__(self,
                 n_object_actions: int = 4,
                 n_heads: int = 3,
                 head_dim: int = 16):
        super().__init__()

        self.q_network = tf.keras.Sequential([
            tf.keras.layers.Dense(
                head_dim * n_heads * n_object_actions,
                activation='relu'
            ),  # project to the correct dimension for the transformer
            tf.keras.layers.Reshape((-1, head_dim * n_heads)),
                # (batch_size, n_object_actions * max_tree_size, head_dim * n_heads)
            PrependTerminateToken(),  # adds a token for the terminate action
            Transformer(n_heads, head_dim, 'relu',
                        dropout_rate=0.1,
                        attention_dropout_rate=0.1),
            Transformer(n_heads, head_dim, 'relu',
                        dropout_rate=0.1,
                        attention_dropout_rate=0.1),
            tf.keras.layers.Dense(
                1,
                kernel_initializer=tf.keras.initializers.RandomUniform(minval=-0.03, maxval=0.03),
                bias_initializer=tf.keras.initializers.Constant(-0.2)),
            tf.keras.layers.Flatten()
        ], name='q_network')

    def call(self, inputs, training=False):
        return self.q_network(inputs, training=training)
