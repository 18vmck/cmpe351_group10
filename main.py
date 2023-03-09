import csv
import hashlib
import collections
from itertools import chain
import numpy as np

import pandas as pd

csv_data = r'final_mfrc_data.csv'
read_data = pd.read_csv(csv_data, header=0, names=["text","subreddit","bucket","annotator","annotation","confidence"])
read_data.head()
######preprocessing steps to isolate data we need#########
list_words = []
# for row in read_data["annotation"]:
#     words = row.split(',')
#     word_counts = collections.Counter(words)
#     list_words.append(words)
#     flat_list = list(chain.from_iterable(list_words))
# unique_words = set(flat_list)
# print(collections.Counter(flat_list)) ###visualizaion


#read_data.insert(0,"id", np.zeros_like(read_data["text"]))


hashes = []

for row in read_data["text"]:
    hashes.append(hashlib.md5(row.encode()).hexdigest()[0:7])


read_data.insert(0,"id", hashes)
print(read_data.head())










#print(unique_words)





