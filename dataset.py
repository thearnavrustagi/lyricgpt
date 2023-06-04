# ml imports
import tensorflow as tf
import pandas as pd
import numpy as np

# local imports
from constants import LYRIC_DS


# [title, tag, artist, year, views, features, id]
def create_dataset(features, lyrics):
    data = []
    for f, l in zip(features.iterrows(), lyrics.to_numpy()):
        data.append(((f[1].to_numpy()), l))
    print(data[0])

    return np.array(data)


def get_features_and_lyrics(fpath=LYRIC_DS):
    df = pd.read_csv(fpath)
    features = df.copy()
    lyrics = features.pop("lyrics")

    return features, lyrics


def load_ds(fpath=LYRIC_DS):
    fnl = get_features_and_lyrics(fpath)
    dataset = create_dataset(*fnl)

    return dataset


def load_simple_ds(fpath=LYRIC_DS):
    df = pd.read_csv(fpath)
    features = df.copy()
    lyrics = features.pop("lyrics")
    features = features.pop("title")

    data = []
    for f, l in zip(features.to_numpy(), lyrics.to_numpy()):
        data.append([f, l])
    print(data[0])

    return np.array(data)


def load_lyrics(fpath=LYRIC_DS):
    df = pd.read_csv(fpath)
    lyrics = df.pop("lyrics").to_numpy()
    lyrics = [l if type(l) == type("") else " " for l in lyrics]

    return lyrics


if __name__ == "__main__":
    load_ds("./dataset/lyrics.small.csv")
