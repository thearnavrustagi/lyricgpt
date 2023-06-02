# ml imports
import tensorflow as tf
import pandas as pd
import numpy as np

# python imports
import re
import string
import pickle

# local imports
from constants import VOCAB_SIZE
from constants import TYPE, END_TYPE, BACKGROUND, END_BACKGROUND
from constants import NEWLINE, START, END

def standardize (arg):
    arg = tf.strings.lower(arg)
    arg = tf.strings.regex_replace(arg,"\[",TYPE)
    arg = tf.strings.regex_replace(arg,"\]",END_TYPE)

    arg = tf.strings.regex_replace(arg,"\(",BACKGROUND)
    arg = tf.strings.regex_replace(arg,"\)",END_BACKGROUND)

    arg = tf.strings.regex_replace(arg,"\n",NEWLINE)

    arg = tf.strings.regex_replace(arg, f'[{re.escape(string.punctuation)}]','')

    return START + arg + END


class Translator:
    """
    This class is supposed to handle a tokenizer, word to index convertor and the index to word
    convertor

    it requires the data to adapt the tokenizer in np.array data format
    """
    def __init__(self,data,vocabulary_size=VOCAB_SIZE, standardize = standardize, ragged = True):
        self.vocabulary_size = vocabulary_size
        self.standardize = standardize
        self.ragged = ragged
        
        self.tokenizer = self.__initialize_tokenizer(data)
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


if __name__ == "__main__":
    print(standardize("[CHORUS] Look what you made me do\n (Look WHat you made me do) [VERSE 2] Bye"))
