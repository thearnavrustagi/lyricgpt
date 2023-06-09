import tensorflow as tf
import numpy as np
import pandas as pd

#from metrics import masked_loss, masked_accuracy, optimizer
from model import create_transformer
from translator import standardize
from process_ds import process_chunk
from translator import Translator, standardize
from constants import LYRIC_DS, CHUNKSIZE, BATCH_SIZE
from metrics import optimizer, masked_loss, masked_accuracy, TransformerSchedule

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
            print("data")
            for x,y in pchunks:
                print(x,y)
            transformer.fit(pchunks, epochs=5)
            tf.keras.models.save_model(transformer,'./model/transformer')
            print(f"chunk {i} done")
            with open(".chunkno",'w') as file:
                file.write(str(i))
            i+=1


if __name__ == "__main__":
    transformer = create_transformer()
#    transformer = tf.keras.models.load_model ("./model/transformer", custom_objects={'masked_loss':masked_loss, 'masked_accuracy':masked_accuracy, "TransformerSchedule":TransformerSchedule})
    translator = Translator.load()
    transformer.compile(
            loss = masked_loss,
            optimizer = optimizer,
            metrics = [ masked_accuracy ]
            )
    """
        loss=  tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.Adam(beta_1=0.9,beta_2=0.98,epsilon=1e-9), 
        metrics=[ tf.keras.metrics.sparse_categorical_accuracy]
    )"""

    start_training(transformer,translator)

    while 1:
        print(simple_gen(translator, transformer, input(" : ")))
