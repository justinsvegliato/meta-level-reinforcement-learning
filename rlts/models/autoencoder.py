import tensorflow as tf


class Autoencoder(tf.keras.Model):

    def __init__(self, enc_dim=64, n_channels=3):
        super(Autoencoder, self).__init__()
        
        self.encoding_dim = enc_dim
        
        self.encoder = tf.keras.Sequential([
            tf.keras.layers.Conv2D(64, (3, 3), 
                                   activation='relu', 
                                   input_shape=(64, 64, n_channels)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu', strides=2),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu', strides=2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(enc_dim, activation='relu'),
        ])
        
        self.decoder = tf.keras.Sequential([
            tf.keras.layers.Dense(units=8*8*32, activation='relu'),
            tf.keras.layers.Reshape(target_shape=(8, 8, 32)),
            tf.keras.layers.Conv2DTranspose(
                filters=64, kernel_size=3, strides=2, padding='same',
                activation='relu'),
            tf.keras.layers.Conv2DTranspose(
                filters=32, kernel_size=3, strides=2, padding='same',
                activation='relu'),
            tf.keras.layers.Conv2DTranspose(
                filters=3, kernel_size=3, strides=2, padding='same',
                activation='sigmoid'
            ),
        ])
        
    def encode(self, inputs, training=False):
        input_encodings = self.encoder(inputs, training=training)
        return input_encodings

    def decode(self, encoding, training=False):
        return self.decoder(encoding, training=training)
    
    def call(self, inputs, training=False):
        enc = self.encode(inputs, training=training)
        return self.decode(enc, training=training)
