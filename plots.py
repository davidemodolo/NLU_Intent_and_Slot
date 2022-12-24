slot_results = [0.9334745014999117, 0.9322512350035287, 0.9323705530116239, 0.932958932958933, 0.9297506146821215]
intent_results = [0.9529675251959686, 0.9652855543113102, 0.975363941769317, 0.9686450167973124, 0.9507278835386338]

import numpy as np
# compute the mean and standard deviation of the results
slot_mean = np.mean(slot_results)
slot_std = np.std(slot_results)

print("Slot F1: {:.6f} ± {:.6f}".format(slot_mean, slot_std))
print("Intent F1: {:.6f} ± {:.6f}".format(np.mean(intent_results), np.std(intent_results)))

# SLOTS RESULTS
# ModelIAS	92,8386	0,2939
# EncoderDecoder	94,9467	0,1174
# BidirectionalIAS	94,4575	0,1758
# BERT	93,2161	0,1282
# INTENT RESULTS
# ModelIAS	93,617	0,4308
# EncoderDecoder	96,4165	0,2554
# BidirectionalIAS	95,7671	0,2493
# BERT	96,2598	0,9385
# put this in a dataframe with model, slot_accuracy, slot_std, intent_accuracy, intent_std

import pandas as pd

df = pd.DataFrame({
    'model': ['ModelIAS', 'EncoderDecoder', 'BidirectionalIAS', 'BERT'],
    'slot_accuracy': [92.8386, 94.9467, 94.4575, 93.2161],
    'slot_std': [0.2939, 0.1174, 0.1758, 0.1282],
    'intent_accuracy': [93.617, 96.4165, 95.7671, 96.2598],
    'intent_std': [0.4308, 0.2554, 0.2493, 0.9385]
})

# plot data as scatterplot with error bars
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.5)

fig, ax = plt.subplots(figsize=(10, 6))
ax.errorbar(df.slot_accuracy, df.intent_accuracy, xerr=df.slot_std, yerr=df.intent_std, fmt='o', color='black')
ax.set_xlabel('Slot F1')
ax.set_ylabel('Intent Accuracy')
ax.set_xlim(92, 98)
ax.set_ylim(92, 98)
ax.set_title('ATIS')

for i, txt in enumerate(df.model):
    # annotate the points with the model name with a small offset
    ax.annotate(txt, (df.slot_accuracy[i] + 0.1, df.intent_accuracy[i] + 0.1))
# add a vertical line at baseline accuracy + 2%
ax.axvline(x=92.8386 + 2, color='grey', linestyle='--')
# add a label with goal under the line
ax.text(92.8386 + 2+0.1, 92+0.1, 'Goal', color='grey')
# add a horizontal line at baseline accuracy + 2%
ax.axhline(y=93.617 + 2, color='grey', linestyle='--')
# add a label with goal under the line
ax.text(92+0.1, 93.617 + 2+0.1, 'Goal', color='grey')
plt.show()