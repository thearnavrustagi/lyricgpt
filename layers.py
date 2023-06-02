import tensorflow as tf

class PositionalEmbedding (tf.keras.layers.Layer):
    def __init__ (self, vocab_size, d_model):

