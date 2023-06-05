import tensorflow as tf
import numpy as np
import pandas as pd

from metrics import masked_loss, masked_accuracy, optimizer
from model import create_transformer
from translator import standardize
from process_ds import process_chunk
from translator import Translator, standardize
from constants import LYRIC_DS, CHUNKSIZE, BATCH_SIZE

def simple_gen(translator, transformer, query , temperature=0.5):
  initial = translator.word2index([['[START]']]) # (batch, sequence)
  query = np.array([translator.tokenizer(query).numpy(),])

  tokens = initial # (batch, sequence)
  for n in range(128):
    preds = transformer((query, tokens)).numpy()  # (batch, sequence, vocab)
    preds = preds[:,-1, :]  #(batch, vocab)
    if temperature==0:
        next = tf.argmax(preds, axis=-1)[:, tf.newaxis]  # (batch, 1)
    else:
        next = tf.random.categorical(preds/temperature, num_samples=1)  # (batch, 1)
    tokens = tf.concat([tokens, next], axis=1) # (batch, sequence) 

    if next[0] == translator.word2index('[END]'):
      break
  words = translator.index2word(tokens[0, 1:-1])
  result = tf.strings.reduce_join(words, axis=-1, separator=' ')
  return result.numpy().decode()

def chunk_generator (fname=LYRIC_DS, translator=None):
    for chunk in pd.read_csv(fname,chunksize=256):
        lyrics = chunk.pop("lyrics").numpy()
        title = chunk.pop("title").numpy()
        yield np.column_stack(np.array(translator.tokenizer([title, lyrics])))
        yield chunk

def start_training (transformer, translator):
    i = 1
    for _ in range(5):
        for chunk in pd.read_csv(LYRIC_DS,chunksize=CHUNKSIZE):
            pchunks = process_chunk(chunk,translator).batch(BATCH_SIZE)
            #for a in pchunks: print(a)
            transformer.fit(pchunks, epochs=5)
            tf.keras.models.save_model(transformer,'./model/transformer')
            print(f"chunk {i} done")
            i+=1


if __name__ == "__main__":
    transformer = create_transformer()
    translator = Translator.load()
    transformer.compile(
        loss=masked_loss, optimizer=optimizer, metrics=[masked_accuracy]
    )

    start_training(transformer,translator)

    while 1:
        print(simple_gen(translator, transformer, input(" : ")))
