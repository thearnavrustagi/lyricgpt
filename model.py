import tensorflow as tf
import numpy as np
from layers import Encoder, Decoder

from constants import DEPTH, ATTN_LAYERS, ATTN_HEADS
from constants import DROPOUT_RATE, FEED_FORWARD_DIMENSIONS, VOCAB_SIZE

class Transformer (tf.keras.Model):
    def __init__ (self, *, num_layers, d_model, num_heads, dff, vocab_size, dropout_rate=0.1):

        super().__init__()

        self.encoder = Encoder(num_layers = num_layers,
                               d_model = d_model,
                               num_heads=num_heads,
                               dff=dff,
                               vocab_size=vocab_size,
                               dropout_rate=dropout_rate)
        
        self.decoder = Decoder(num_layers=num_layers,
                               d_model = d_model,
                               num_heads=num_heads,
                               dff=dff,
                               vocab_size=vocab_size,
                               dropout_rate=dropout_rate)

        self.final_layer = tf.keras.layers.Dense(vocab_size)
        
    def train_step (self, data):
        x,y = data
        with tf.GradientTape() as tape:
            y_pred = self(data, training=True)  # Forward pass
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}

    def call (self, inputs):
        context, x = inputs

        context = self.encoder (context)
        x = self.decoder(x,context)

        logits = self.final_layer(x)

        try:
            del logits._keras_mask
        except AttributeError: 
            pass

        return logits


def create_transformer():
    transformer = Transformer(
        num_layers=ATTN_LAYERS,
        d_model=DEPTH,
        num_heads=ATTN_HEADS,
        dff=FEED_FORWARD_DIMENSIONS,
        vocab_size=VOCAB_SIZE,
        dropout_rate=DROPOUT_RATE
            )

    return transformer


if __name__ == "__main__":
    transformer = create_transformer()

    transformer((np.array([np.array([1,2,3,4])]),np.array([np.array([1,2,3])])))

    transformer.summary()
