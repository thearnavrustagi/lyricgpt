# ml imports
import tensorflow as tf
import numpy as np

# local imports
from constants import LYRIC_LENGTH

class PositionalEmbedding (tf.keras.layers.Layer):
    """
    Why ?
    attention layers often dont see input data in order, they see it as a set, 
    "how are you" is the same as "you how are" for the attention layers, so to ensure that
    the model sees the order, we use positional embedding, which helps the attention
    layers understand the order of the words
    """

    @staticmethod
    def positional_encoding (length, depth):
        depth = depth/2

        positions = np.arange(length)[:, np.newaxis]
        depths = np.arange(depth)[np.newaxis, :]/depth

        angle_rates = 1 / (1000**depths)
        angle_rads  = positions * angle_rates

        pos_encoding = np.concatenate (
                [ np.sin(angle_rads), np.cos(angle_rads)],
                axis = -1)

        return tf.cast(pos_encoding, dtype=tf.float32)

    def __init__ (self, vocab_size, d_model):
        super().__init__()
        self.d_model = d_model
        self.embedding = tf.keras.layers.Embedding(vocab_size, d_model, mask_zero = True)
        self.pos_embedding = tf.keras.layers.Embedding(LYRIC_LENGTH, d_model)
        self.add = tf.keras.layers.Add()

    def compute_mask (self, *args, **kwargs):
        return self.embedding.compute_mask (*args, **kwargs)

    def call (self, x):
        length = tf.shape(x)[1]
        x = self.embedding(x)
        x = self.add([x,self.pos_embedding(tf.convert_to_tensor(tf.range(length)))])

        return x

class BaseAttention (tf.keras.layers.Layer):
    def __init__ (self, **kwargs):
        super().__init__()
        self.mha = tf.keras.layers.MultiHeadAttention(**kwargs)
        self.layernorm = tf.keras.layers.LayerNormalization()
        self.add = tf.keras.layers.Add()


class CrossAttention (BaseAttention):
    """
    This attention layer is the most basic attention layer which can be used
    In this each query, can see all the key/value pairs in the context
    """
    def call (self, x, context):
        attn_output, attn_scores = self.mha (
                query = x,
                key = context,
                value = context,
                return_attention_scores = True
                )

        self.last_attn_scores = attn_scores

        x = self.add ([x, attn_output])
        x = self.layernorm (x)

        return x

class GlobalSelfAttention (BaseAttention):
    """
    In this attention layer, all of the query is used to independently create
    the solution, without looking at the key/value pairs

    i.e. It attends to the query and the query only, "self" attention
    """
    def call (self, x):
        attn_output = self.mha (
                query = x,
                value = x,
                key   = x)
        
        x = self.add([x, attn_output])
        x = self.layernorm(x)

        return x

class CausalSelfAttention (BaseAttention):
    """
    This is similar to global self attention, the main difference being this operates
    on the output sequence, this ensures that the next token generated depends
    on the previous token generated
    """

    def call (self, x):
        attn_output = self.mha (
                query = x,
                value = x,
                key   = x,
                use_causal_mask = True)
        x = self.add([x, attn_output])
        x = self.layernorm(x)

        return x

class FeedForward (tf.keras.layers.Layer):
    """
    simple feedforward CNN
    """
    def __init__ (self, d_model, dff, dropout_rate = 0.1):
        super().__init__()

        self.seq = tf.keras.Sequential ([
            tf.keras.layers.Dense (dff, activation='relu'),
            tf.keras.layers.Dense (d_model),
            tf.keras.layers.Dropout (dropout_rate),
            ])

        self.add = tf.keras.layers.Add()
        self.layer_norm = tf.keras.layers.LayerNormalization()

    def call (self, x):
        x = self.add([x, self.seq(x)])
        x = self.layer_norm(x)

        return x

class EncoderLayer (tf.keras.layers.Layer):
    """
    This is a combination of gsa and ffn, so that one can encode the query
    into a vector
    """
    def __init__ (self, *, d_model, num_heads, dff, dropout_rate=0.1):
        super().__init__()

        self.self_attention = GlobalSelfAttention(
                    num_heads = num_heads,
                    key_dim = d_model,
                    dropout = dropout_rate
                )

        self.ffn = FeedForward(d_model, dff)
    
    def call (self, x):
        x = self.self_attention(x)
        x = self.ffn(x)

        return x

class Encoder (tf.keras.layers.Layer):
    def __init__ (self, *, num_layers, d_model, num_heads, dff, vocab_size, dropout_rate=0.1):
        super().__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.pos_embedding = PositionalEmbedding (vocab_size, d_model = d_model)
        self.enc_layers = [
                EncoderLayer (d_model=d_model,
                              num_heads=num_heads,
                              dff=dff,
                              dropout_rate=dropout_rate)
                for _ in range(num_layers)]

        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call (self, x):
        x = self.pos_embedding(x)

        x = self.dropout(x)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x)

        return x

class DecoderLayer (tf.keras.layers.Layer):
    def __init__ (self, *, d_model, num_heads, dff, dropout_rate=0.1):
        super(DecoderLayer, self).__init__()

        self.causal_self_attention = CausalSelfAttention (
            num_heads = num_heads,
            key_dim = d_model,
            dropout = dropout_rate)

        self.cross_attention = CrossAttention (
            num_heads = num_heads,
            key_dim = d_model,
            dropout = dropout_rate)

        self.ffn = FeedForward(d_model, dff)

    def call (self, x, context):
        x = self.causal_self_attention(x=x)
        x = self.cross_attention(x=x, context=context)

        self.last_attn_scores = self.cross_attention.last_attn_scores

        x = self.ffn(x)
        return x


class Decoder (tf.keras.layers.Layer):
    def __init__ (self, *, num_layers, d_model, num_heads, dff, vocab_size, dropout_rate=0.1):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.pos_embedding = PositionalEmbedding (vocab_size = vocab_size, d_model=d_model)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.dec_layers = [
                DecoderLayer(d_model=d_model,
                             num_heads=num_heads,
                             dff=dff,
                             dropout_rate=dropout_rate)
                for _ in range(num_layers)
                ]

        self.last_attn_scores = None

    def call (self, x, context):
        x = self.pos_embedding(x)
        x = self.dropout(x)

        for i in range(self.num_layers):
            x = self.dec_layers[i](x, context)

        self.last_attn_scores = self.dec_layers[-1].last_attn_scores

        return x

if __name__ == "__main__":
    print("checking positional embedding")


