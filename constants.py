# files
LYRIC_DS = "./dataset/lyrics.csv"
PROCESSED_DS = "./dataset/lyrics.processed.csv"

# tokenizer
VOCAB_SIZE = 50000
TRANSLATOR_FNAME = "./model/translator.pickle"

# text constants
START = " START "
NEWLINE = " NEWLINE "
END = " END "
TYPE, END_TYPE = " TYPE ", " ENDTYPE "
BACKGROUND, END_BACKGROUND = " BG ", " ENDBG "

# model
LYRIC_LENGTH = 128
DEPTH = 256
ATTN_LAYERS = 16
ATTN_HEADS = 32
DROPOUT_RATE = 0.2
FEED_FORWARD_DIMENSIONS = 16

# training and data streaming
CHUNKSIZE = 4
BATCH_SIZE = 2

