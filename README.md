**Davide Modolo 229297**

# NLU Project - Joint Intent Classification and Slot Filling Sentence Level

## Theory

### Intent Classification

Intent classification is a text classification task in which the objective is to assign an intent for a given sentence or utterance.

> _Utterance_: Can you help me find out about flights?
>
> _Intent_: InfoRequest

### Slot Filling (Slot F1)
Slot filling is a sequence labelling task where the objective is to map a given sentence or utterance to a sequence of domain-slot labels.

> _Utterance_: I want to travel from nashville to tacoma
>
> _Concepts_: O O O O O B-fromloc.city_name O B-toloc.city_name

## Task

Implement a neural network that predicts intents and slots in a multitask learning setting.

<u>Since the datasets are tiny, you have to train and test your model from scratch at least 5 times. Report average and standard deviation.</u>

**Datasets**: ATIS and SNIPS

**Goal**: Improve baseline results by at least 2/3%:

- ATIS -> Slot F1: 92.0%, Intent Acc.: 94.0%

- SNIPS -> Slot F1: 80.0%, Intent Acc.: 96.0%

**PROJECT TODO**:

1. Implement baseline methods

2. Build different architectures (Seq2Seq, Bi-LSTM + CRF, etc.)

3. Try to use pre-trained models (e.g. BERT, GPT2, T5, etc.)

**MY TODO**:

- [x] download and import datasets

- [x] prepare validation dataset for ATIS (since only SNIPS has it) - ~10% of the train but intents with only one instance remain in training

- [x] implement baseline methods

- [x] implement architectures from scratch (PyTorch)

- [x] implement pre-trained models (PyTorch) BERT & ERNIE

- [x] data visualization

- [x] write paper

## Repository content
```
project
│   README.md
│   NLU_Report_Modolo.pdf: report on this project
│   conll.py: script to evaluate results
│   modolo_davide.ipynb: python notebook containing the baseline model, the bi-directional one and ED
│   pretrainedBERT.ipynb: python notebook containing the BERT model
│   pretrainedERNIE.ipynb: python notebook containing the ERNIE model
│
└───data
    └───ATIS
    │   test.json
    │   train_full.json
    │   train.json
    │   valid.json
    │
    └───SNIPS
        test.json
        train.json
        valid.json
```