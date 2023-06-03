# ml imports
import tensorflow as tf
import pandas as pd
import numpy as np

# python imports
import re
import string
import pickle

# local imports
from constants import VOCAB_SIZE, TRANSLATOR_FNAME
from constants import TYPE, END_TYPE, BACKGROUND, END_BACKGROUND
from constants import NEWLINE, START, END
from dataset import load_lyrics

def standardize (arg):
    arg = tf.strings.lower(arg)
    arg = tf.strings.regex_replace(arg,"\[",TYPE)
    arg = tf.strings.regex_replace(arg,"\]",END_TYPE)

    arg = tf.strings.regex_replace(arg,"\(",BACKGROUND)
    arg = tf.strings.regex_replace(arg,"\)",END_BACKGROUND)

    arg = tf.strings.regex_replace(arg,"\n",NEWLINE)

    arg = tf.strings.regex_replace(arg, f'[{re.escape(string.punctuation)}]',' ')

    return START + arg + END


class Translator:
    """
    This class is supposed to handle a tokenizer, word to index convertor and the index to word
    convertor

    it requires the data to adapt the tokenizer in np.array data format
    """
    def __init__(self,data,vocabulary_size=VOCAB_SIZE, 
                 standardize = standardize, ragged = True):
        if type(data) == type(dict()):
            self.tokenizer = tf.keras.layers.TextVectorization.from_config(data['config'])
            self.tokenizer.adapt(tf.data.Dataset.from_tensor_slices(["a"]))
            self.tokenizer.set_weights(data['weights'])
        else:
            self.vocabulary_size = vocabulary_size
            self.standardize = standardize
            self.ragged = ragged
            self.tokenizer = self.__initialize_tokenizer(data)

        print(self.tokenizer.get_vocabulary())
        self.word2index = self.__initialize_w2i()
        self.index2word = self.__initialize_i2w()

    def __make_tokenizer (self):
        tokenizer = tf.keras.layers.TextVectorization(
                max_tokens = self.vocabulary_size,
                standardize= self.standardize,
                ragged=self.ragged
                )
        return tokenizer

    def __initialize_tokenizer (self,data):
        tokenizer = self.__make_tokenizer()
        tokenizer.adapt(data)

        return tokenizer

    def __initialize_w2i (self,invert = False):
        return tf.keras.layers.StringLookup (
                mask_token = "",
                vocabulary = self.tokenizer.get_vocabulary(),
                invert = invert)

    def __initialize_i2w (self):
        return self.__initialize_w2i(True)

    def save (self, fname=TRANSLATOR_FNAME):
        with open(fname, "wb") as handle:
            pickle.dump({'config'  : translator.tokenizer.get_config(),
                         'weights' : translator.tokenizer.get_weights()},
                        handle)

    @staticmethod
    def load (fname=TRANSLATOR_FNAME):
       with open(fname, "rb") as handle:
           data = pickle.load(handle)
           return Translator(data)


if __name__ == "__main__":
    lyrics = load_lyrics("./dataset/lyrics.medium.csv")
    print("LOADING DONE")

    translator = Translator(lyrics)
    print(f"vocab : {translator.tokenizer.get_vocabulary()[:10]}")
    t = translator.tokenizer(["[chorus] Look What you made Me do (Look What you MaDe me do)"])
    print(f"tokenized : {t}")
    print(f"to words  : {translator.index2word(t)}")
    print("SAVING")

    translator.save()

    print("TESTING POST LOAD")
    translator = Translator.load()
    print(f"vocab : {translator.tokenizer.get_vocabulary()[:10]}")
    t = translator.tokenizer(["[chorus] Look What you made Me do (Look What you MaDe me do)"])
    print(f"tokenized : {t}")
    print(f"to words  : {translator.index2word(t)}")

