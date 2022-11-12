PAD_TOKEN = 0
import json
import torch
from torch.utils.data import Dataset, DataLoader
import os
device = 'CUDA:0' if torch.cuda.is_available() else 'CPU'

def word2id(raw_dataset):
# returns a dictionary of words and their ids
    words = []
    for entry in raw_dataset:
       words.extend(entry['utterance'].split())
    words = list(set(words))
    words_dict = {'pad': PAD_TOKEN}
    words_dict.update({w:i+1 for i, w in enumerate(words)})
    words_dict['unk'] = len(words_dict)
    return words_dict

def slot2id(raw_dataset):
# returns a dictionary of slots and their ids
    slots = ['pad']
    for entry in raw_dataset:
       slots.extend(entry['slots'].split())
    slots = list(set(slots))
    slots_dict = {s:i for i, s in enumerate(slots)}
    return slots_dict

def intent2id(raw_dataset):
# returns a dictionary of intents and their ids
    intents = [entry['intent'] for entry in raw_dataset]
    intents = list(set(intents))
    intents_dict = {inte:i for i, inte in enumerate(intents)}
    return intents_dict

class Lang():
    def __init__(self, train_raw, dev_raw, test_raw):
        self.word2id = word2id(train_raw)
        self.slot2id = slot2id(train_raw + dev_raw + test_raw)
        self.intent2id = intent2id(train_raw + dev_raw + test_raw)
        self.id2word = {v:k for k, v in self.word2id.items()}
        self.id2slot = {v:k for k, v in self.slot2id.items()}
        self.id2intent = {v:k for k, v in self.intent2id.items()}
        # self.intent_list = list(set(list(self.intent2id.keys())))
        # self.slot_list = list(set(list(self.slot2id.keys())))
def load_data(path):
    dataset = []
    with open(path) as f:
        dataset = json.loads(f.read())
    return dataset

class IntentsAndSlots (Dataset):
    # Mandatory methods are __init__, __len__ and __getitem__
    def __init__(self, dataset, lang, unk='unk'):
        self.utterances = []
        self.intents = []
        self.slots = []
        self.unk = unk
        
        for x in dataset:
            self.utterances.append(x['utterance'])
            self.slots.append(x['slots'])
            self.intents.append(x['intent'])

        self.utt_ids = self.mapping_seq(self.utterances, lang.word2id)
        self.slot_ids = self.mapping_seq(self.slots, lang.slot2id)
        self.intent_ids = self.mapping_lab(self.intents, lang.intent2id)

    def __len__(self):
        return len(self.utterances)

    def __getitem__(self, idx):
        utt = torch.Tensor(self.utt_ids[idx])
        slots = torch.Tensor(self.slot_ids[idx])
        intent = self.intent_ids[idx]
        sample = {'utterance': utt, 'slots': slots, 'intent': intent}
        return sample
    
    # Auxiliary methods
    
    def mapping_lab(self, data, mapper):
        return [mapper[x] if x in mapper else mapper[self.unk] for x in data]
    
    def mapping_seq(self, data, mapper): # Map sequences to number
        res = []
        for seq in data:
            tmp_seq = []
            for x in seq.split():
                if x in mapper:
                    tmp_seq.append(mapper[x])
                else:
                    tmp_seq.append(mapper[self.unk])
            res.append(tmp_seq)
        return res

def collate_fn(data):
    def merge(sequences):
        '''
        merge from batch * sent_len to batch * max_len 
        '''
        lengths = [len(seq) for seq in sequences]
        max_len = 1 if max(lengths)==0 else max(lengths)
        # Pad token is zero in our case
        # So we create a matrix full of PAD_TOKEN (i.e. 0) with the shape 
        # batch_size X maximum length of a sequence
        padded_seqs = torch.LongTensor(len(sequences),max_len).fill_(PAD_TOKEN)
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq # We copy each sequence into the matrix
        # print(padded_seqs)
        padded_seqs = padded_seqs.detach()  # We remove these tensors from the computational graph
        return padded_seqs, lengths
    # Sort data by seq lengths
    data.sort(key=lambda x: len(x['utterance']), reverse=True) 
    new_item = {}
    for key in data[0].keys():
        new_item[key] = [d[key] for d in data]
    # We just need one length for packed pad seq, since len(utt) == len(slots)
    src_utt, _ = merge(new_item['utterance'])
    y_slots, y_lengths = merge(new_item["slots"])
    intent = torch.LongTensor(new_item["intent"])
    
    src_utt = src_utt.to(device) # We load the Tensor on our seleceted device
    y_slots = y_slots.to(device)
    intent = intent.to(device)
    y_lengths = torch.LongTensor(y_lengths).to(device)
    
    new_item["utterances"] = src_utt
    new_item["intents"] = intent
    new_item["y_slots"] = y_slots
    new_item["slots_len"] = y_lengths
    return new_item

def prepare_data(dataset):
    train_raw = load_data(os.path.join('data', dataset, 'train.json'))
    test_raw = load_data(os.path.join('data', dataset, 'test.json'))
    dev_raw = load_data(os.path.join('data', dataset, 'valid.json'))
    
    lang = Lang(train_raw, dev_raw, test_raw)

    ##############################
    train_dataset = IntentsAndSlots(train_raw, lang)
    dev_dataset = IntentsAndSlots(dev_raw, lang)
    test_dataset = IntentsAndSlots(test_raw, lang)
    
    ##############################
    train_loader = DataLoader(train_dataset, batch_size=128, collate_fn=collate_fn,  shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=64, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=64, collate_fn=collate_fn)
    
    return train_loader, dev_loader, test_loader, lang