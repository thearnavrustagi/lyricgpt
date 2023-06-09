import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pandas as pd

#from metrics import masked_loss, masked_accuracy, optimizer
from model import create_transformer
from translator import standardize
from process_ds import process_chunk
from translator import Translator, standardize
from constants import LYRIC_DS, CHUNKSIZE, BATCH_SIZE, LYRIC_LENGTH
from constants import START

from metrics import masked_accuracy, masked_loss, TransformerSchedule

def simple_gen(translator, transformer, query , temperature=0.5):
  print(translator.tokenizer.get_vocabulary()[:10])
  initial = translator.word2index([["START"]]) # (batch, sequence)
  arr = initial.numpy()[0]
  print("initial",arr)
  query = np.array([translator.tokenizer(query).numpy(),])
  initial = pad_sequences(initial, maxlen=LYRIC_LENGTH, padding='post', truncating='post')
  initial = tf.cast(initial, tf.float32)
  print(initial,query)

  query = pad_sequences(query, maxlen=LYRIC_LENGTH, padding='post', truncating='post')
  query = tf.cast(query, tf.float32)

  print("Initial shape:", initial.shape)
  print("Query shape:", query.shape)

  print(query.shape, initial.shape)

  tokens = initial # (batch, sequence)
  for n in range(LYRIC_LENGTH):
    preds = transformer((query, tokens)).numpy()  # (batch, sequence, vocab)
    preds = preds[:,-1, :]  #(batch, vocab)
    if temperature==0:
        next = tf.argmax(preds, axis=-1)[:, tf.newaxis]  # (batch, 1)
    else:
        next = tf.random.categorical(preds/temperature, num_samples=1)  # (batch, 1)
    arr = np.append(arr,next)
    print(" ".join([bytes.decode(a) for a in translator.index2word([arr,])[0].numpy().tolist()]))
    initial = np.array([arr,])
    initial = pad_sequences(initial, maxlen=LYRIC_LENGTH, padding='post', truncating='post')

    if next[0] == translator.word2index('[END]'):
      break
  words = translator.index2word(tokens[0, 1:-1])
  result = tf.strings.reduce_join(words, axis=-1, separator=' ')
  return result.numpy().decode()

if __name__ == "__main__":
    translator = Translator.load()
    transformer = tf.keras.models.load_model("./model/transformer", custom_objects={'masked_loss':masked_loss, 'masked_accuracy':masked_accuracy, "TransformerSchedule" : TransformerSchedule})
    transformer.summary()

    while 1:
        print(simple_gen(translator, transformer, input(" song name : ")))
