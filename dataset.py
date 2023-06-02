# ml imports
import tensorflow as tf
import pandas as pd

# local imports
from constants import LYRIC_DS

# [title, tag, artist, year, views, features, id]
def create_dataset (features, lyrics):
    data = []
    for f, l in zip(features.iterrows(),lyrics.to_numpy()):
        data.append(((f[1].to_numpy()),l))
    print(data[0])


def get_features_and_lyrics(fpath=LYRIC_DS):
    df = pd.read_csv(fpath)
    features = df.copy()
    lyrics = features.pop("lyrics")

    return features, lyrics

def load_ds (fpath=LYRIC_DS):
    fnl = get_features_and_lyrics(fpath)
    datset = create_dataset(*fnl)
    

if __name__ == "__main__":
    load_ds("./dataset/lyrics.small.csv")


