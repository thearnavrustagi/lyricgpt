# this program is made to process our dataset into tensors
# ml imports
import tensorflow as tf
import numpy as np

# local imports
from translator import Translator, standardize
from dataset import load_simple_ds
from constants import LYRIC_LENGTH

def convert (dataset, translator):
    converted = []
    for row in dataset:
        context = tf.convert_to_tensor(translator.tokenizer(row[0]).numpy(),dtype=tf.float32)
        x = tf.convert_to_tensor(translator.tokenizer(row[1]).numpy(),dtype=tf.float32)
        converted.append((context,x))
        print(type(context), context)
        
    max_length = LYRIC_LENGTH
    padded_tensors = []
    for context, x in converted:
        context = tf.pad(context, paddings=[[0, max_length - context.shape[0]]], constant_values=0.0)
        x = tf.pad(x, paddings=[[0, max_length - x.shape[0]]], constant_values=0.0)
        padded_tensors.append((context, x))
    ctx, x = zip(*padded_tensors)  # Convert to tuple of tensors
    
    # Convert tuple of tensors to TensorFlow tensors
    ctx = tf.stack(ctx)
    print(ctx.shape)
    x = tf.stack(x)
    return tf.data.Dataset.from_tensor_slices((ctx, x))

def get_processed_ds ():
    dataset = load_simple_ds("./dataset/lyrics.small.csv")
    translator = Translator.load()
    print("starting conversion")
    data = convert(dataset, translator)

    return (data)

if __name__ == "__main__":
    get_processed_ds()
