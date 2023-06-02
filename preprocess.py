# ml imports
import tensorflow as tf
import pandas as pd

# general imports
import re
import string
import math

# local imports
from constants import LYRIC_DS,NEWLINE,START,END, PROCESSED_DS

def restructure_lyrics (dataframe):
    curr_song = ""
    old_row = None
    lyrics = ""
    f_df = pd.DataFrame(columns=["year","lyrics"])
    for index, row in dataframe.iterrows():
        if type(row['lyric']) == type(1.0) and math.isnan(row['lyric']): continue
        if curr_song != row["track_title"]:
            curr_song = row["track_title"]
            append_to_formatted_dataframe(f_df,old_row,lyrics)
            lyrics = ""
        lyrics += standardize(row["lyric"])
        old_row = row

    f_df.to_csv(PROCESSED_DS)

def standardize (s):
    s = tf.strings.lower(s)
    s = tf.strings.regex_replace(s, f'[{re.escape(string.punctuation)}]', '')
    return s + "\n"

def append_to_formatted_dataframe (formatted_df, row, lyrics):
    try:
        lyrics = tf.strings.regex_replace(lyrics,'[\n]'," "+NEWLINE+" ")
        lyrics = START + " " + lyrics + " " + END
        formatted_df.loc[len(formatted_df.index)] = [ row['year'], tensor_to_str(lyrics)]
        print(formatted_df)
    except Exception as e:
        print(e)

def tensor_to_str (eagertensor):
    return bytes.decode(eagertensor.numpy())

if __name__ == "__main__":
    df = pd.read_csv(LYRIC_DS, encoding="us-ascii")
    restructure_lyrics(df)
