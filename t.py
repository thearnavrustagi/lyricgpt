l = [int(a) for a in """ 39 14958 49911  4973 14299 37191 31780 29205 23070 33431 11091   914
 11948 24702  1553 28440  5644 15450 23052  4613 44636 16614 46570""".split()]

from translator import Translator, standardize

t = Translator.load()
print(" ".join([bytes.decode(a) for a in t.index2word([l,])[0].numpy().tolist()]))
