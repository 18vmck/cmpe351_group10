import csv
import hashlib
import collections
from itertools import chain
import hashlib
import numpy as np
import pandas as pd

csv_data = r'final_mfrc_data.csv'
read_data = pd.read_csv(csv_data, header=0, names=["text","subreddit","bucket","annotator","annotation","confidence"])
read_data.head()

'''
Made faster implementation of the code below 

######preprocessing steps to isolate data we need#########
list_words = []
for row in read_data["annotation"]:
    words = row.split(',')
    word_counts = collections.Counter(words)
    list_words.append(words)
    flat_list = list(chain.from_iterable(list_words))
unique_words = set(flat_list)
print(collections.Counter(flat_list)) ###visualizaion
'''

# Get the unique words
annotator_data = [{'annotator': row[0], 'annotation': row[1]} for row in read_data[['annotator', 'annotation']].values]
# Get the counts of each word
labels = [{'Non-Moral': row['annotation'].count('Non-Moral'), 'Thin Morality': row['annotation'].count('Thin Morality'), 'Care': row['annotation'].count('Care'), 'Equality': row['annotation'].count('Equality'), 'Authority': row['annotation'].count('Authority'), 'Proportionality': row['annotation'].count('Proportionality'), 'Loyalty': row['annotation'].count('Loyalty'), 'Purity': row['annotation'].count('Purity')} for row in annotator_data]
# Convert to a dataframe
labels = pd.DataFrame(labels)
# Sum the counts
labels = labels.sum(axis=0)


hashes = []

for row in read_data["text"]:
    hashes.append(hashlib.md5(row.encode()).hexdigest()[0:7])

read_data.insert(0,"id", hashes)
print(read_data.head(100))






