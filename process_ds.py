# this program is made to process our dataset into tensors
# ml imports
import tensorflow as tf
import numpy as np

# local imports
from translator import Translator, standardize
from dataset import load_simple_ds
from constants import LYRIC_LENGTH


def process_chunk(dataset, translator):
    converted = []
    for _,row in dataset.iterrows():
        print(row)
        context = tf.convert_to_tensor(
            translator.tokenizer(row[0]).numpy(), dtype=tf.float32
        )
        x = tf.convert_to_tensor(translator.tokenizer(row[6]).numpy(), dtype=tf.float32)
        converted.append((context, x))

    max_length = LYRIC_LENGTH
    padded_tensors = []
    print("converting")
    for context, x in converted:
        context = tf.pad(
            context, paddings=[[0, max_length - context.shape[0]]], constant_values=0.0
        )
        if max_length - x.shape[0] > 0:
            x = tf.pad(x, paddings=[[0, max_length - x.shape[0]]], constant_values=0.0)
        else:
            x = x[:max_length]
        padded_tensors.append((context, x))
    ctx, x = zip(*padded_tensors)  # Convert to tuple of tensors
    print("conversion done")

    # Convert tuple of tensors to TensorFlow tensors
    ctx = tf.stack(ctx)
    x = tf.stack(x)
    return tf.data.Dataset.from_tensor_slices((ctx, x))

def get_processed_ds(s):
    dataset = load_simple_ds(s)
    translator = Translator.load()
    print("starting conversion")
    data = process_chunk(dataset, translator)

    return data, translator


if __name__ == "__main__":
    get_processed_ds()
