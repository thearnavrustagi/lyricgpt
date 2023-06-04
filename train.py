import tensorflow as tf
import numpy as np
from metrics import masked_loss, masked_accuracy, optimizer
from model import create_transformer
from translator import standardize
from process_ds import get_processed_ds

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

if __name__ == "__main__":
    dataset, translator = get_processed_ds("./dataset/lyrics.medium.csv")
    dataset = dataset.batch(64)
    for element in dataset:
        print(element)
    transformer = create_transformer()

    transformer.compile(
        loss=masked_loss, optimizer=optimizer, metrics=[masked_accuracy]
    )

    transformer.fit(dataset, epochs=50)
    #transformer.save('./model/transformer')

    while 1:
        print(simple_gen(translator, transformer, input(" : ")))
