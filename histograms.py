import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# read data from data/ATIS and open train_full.json and test.json
train = pd.read_json('data/SNIPS/train.json')
test = pd.read_json('data/SNIPS/test.json')
dev = pd.read_json('data/SNIPS/valid.json')

# create the histogram for intent distribution the entire dataset
# stack the dataframes and count the number of occurrences of each intent as percentage of the total
intent_distribution_perc = pd.concat([train, test, dev]).intent.value_counts(normalize=True)
# and as number of occurrences
intent_distribution = pd.concat([train, test, dev]).intent.value_counts()
# plot the histogram
# put the number of occurrences on top of each bar
for i, v in enumerate(intent_distribution):
    plt.text(i, v, str(v), color='black', fontweight='bold')
# plot the histogram with grey bars
intent_distribution.plot(kind='bar', color='grey')
# set the title
plt.title('Intent Distribution')
# set the x label
plt.xlabel('Intent')
# set the y label
plt.ylabel('Number of Occurrences')
# show the plot
plt.show()

