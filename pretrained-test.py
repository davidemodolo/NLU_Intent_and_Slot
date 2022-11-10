# imports and PAD_TOKEN
PAD_TOKEN = 0
import os
import json

import torch
device = 'cuda:0' # cuda:0 means we are using the GPU with id 0, if you have multiple GPU
os.environ['CUDA_LAUNCH_BLOCKING'] = "1" # Used to report errors on CUDA side
import torch.utils.data as data

from torch.utils.data import DataLoader

import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# Lang class
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
# datasets
def load_data(path):
    dataset = []
    with open(path) as f:
        dataset = json.loads(f.read())
    return dataset
class IntentsAndSlots (data.Dataset):
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
#import bert
from transformers import BertTokenizer, BertForSequenceClassification, BertConfig
from conll import evaluate
from sklearn.metrics import classification_report
import torch.nn.functional as F
def init_weights(mat):
    for m in mat.modules():
        if type(m) in [nn.GRU, nn.LSTM, nn.RNN]:
            for name, param in m.named_parameters():
                if 'weight_ih' in name:
                    for idx in range(4):
                        mul = param.shape[0]//4
                        torch.nn.init.xavier_uniform_(param[idx*mul:(idx+1)*mul])
                elif 'weight_hh' in name:
                    for idx in range(4):
                        mul = param.shape[0]//4
                        torch.nn.init.orthogonal_(param[idx*mul:(idx+1)*mul])
                elif 'bias' in name:
                    param.data.fill_(0)
        else:
            if type(m) in [nn.Linear]:
                torch.nn.init.uniform_(m.weight, -0.01, 0.01)
                if m.bias != None:
                    m.bias.data.fill_(0.01)
def train_loop(data, optimizer, criterion_slots, criterion_intents, model):
    model.train()
    loss_array = []
    for sample in data:
        optimizer.zero_grad() # Zeroing the gradient
        bert = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=out_int)
        bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        # bert_config = BertConfig.from_pretrained('bert-base-uncased')
        # first tokenize the input  
        tokenized_input = bert_tokenizer(sample['utterances'], return_tensors='pt')
        # then pass the input to the model
        bert_output = bert(**tokenized_input)
        # get the logits
        logits = bert_output.logits
        # get the predictions
        predictions = torch.argmax(logits, dim=1)
        # get the loss
        loss = criterion_intents(predictions, sample['intents'])
        loss.backward()
        optimizer.step()
        loss_array.append(loss.item())

        # slots, intent = model(sample['utterances'], sample['slots_len'])
        # loss_intent = criterion_intents(intent, sample['intents'])
        # loss_slot = criterion_slots(slots, sample['y_slots'])
        # loss = loss_intent + loss_slot
        # loss_array.append(loss.item())
        # loss.backward() # Compute the gradient, deleting the computational graph
        # # clip the gradient to avoid explosioning gradients
        # # torch.nn.utils.clip_grad_norm_(model.parameters(), clip)  
        # optimizer.step() # Update the weights
    return loss_array

def eval_loop(data, criterion_slots, criterion_intents, model, lang):
    model.eval()
    loss_array = []
    
    ref_intents = []
    hyp_intents = []
    
    ref_slots = []
    hyp_slots = []
    #softmax = nn.Softmax(dim=1) # Use Softmax if you need the actual probability
    with torch.no_grad(): # It used to avoid the creation of computational graph
        for sample in data:
            slots, intents = model(sample['utterances'], sample['slots_len'])
            loss_intent = criterion_intents(intents, sample['intents'])
            loss_slot = criterion_slots(slots, sample['y_slots'])
            loss = loss_intent + loss_slot 
            loss_array.append(loss.item())
            # Intent inference
            # Get the highest probable class
            out_intents = [lang.id2intent[x] 
                           for x in torch.argmax(intents, dim=1).tolist()] 
            gt_intents = [lang.id2intent[x] for x in sample['intents'].tolist()]
            ref_intents.extend(gt_intents)
            hyp_intents.extend(out_intents)
            
            # Slot inference 
            output_slots = torch.argmax(slots, dim=1)
            for id_seq, seq in enumerate(output_slots):
                length = sample['slots_len'].tolist()[id_seq]
                utt_ids = sample['utterance'][id_seq][:length].tolist()
                gt_ids = sample['y_slots'][id_seq].tolist()
                gt_slots = [lang.id2slot[elem] for elem in gt_ids[:length]]
                utterance = [lang.id2word[elem] for elem in utt_ids]
                to_decode = seq[:length].tolist()
                ref_slots.append([(utterance[id_el], elem) for id_el, elem in enumerate(gt_slots)])
                tmp_seq = []
                for id_el, elem in enumerate(to_decode):
                    tmp_seq.append((utterance[id_el], lang.id2slot[elem]))
                hyp_slots.append(tmp_seq)
    try:            
        results = evaluate(ref_slots, hyp_slots)
    except Exception as ex:
        # Sometimes the model predics a class that is not in REF
        print(ex)
        ref_s = set([x[1] for x in ref_slots])
        hyp_s = set([x[1] for x in hyp_slots])
        print(hyp_s.difference(ref_s))
        
    report_intent = classification_report(ref_intents, hyp_intents, 
                                          zero_division=False, output_dict=True)
    return results, report_intent, loss_array


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from tqdm import tqdm
import numpy as np

class AfterBert(nn.Module):
    def __init__(self, hid_size, out_slot, out_int, emb_size, vocab_len, n_layer=1, pad_index=0,  dropout=0.1):
        super(AfterBert, self).__init__()
        self.slot_out = nn.Linear(hid_size, out_slot)
        self.intent_out = nn.Linear(hid_size, out_int)
        self.dropout = 0.1

    def forward(self, utterance, seq_lengths):

        packed_output, self.hidden = self.bert(utterance) 

        # packed_output, input_sizes = pad_packed_sequence(packed_output, seq_lengths.cpu().numpy())

        # packed_output = F.dropout(packed_output, self.dropout)
        # self.hidden = F.dropout(self.hidden, self.dropout)
        last_hidden = self.hidden[-1,:,:]
        slots = self.slot_out(packed_output)
        intent = self.intent_out(last_hidden)
        slots = slots.permute(1,2,0)
        return slots, intent



train_loader, dev_loader, test_loader, lang = prepare_data('ATIS')
hid_size = 200
emb_size = 300

lr = 0.0001 # learning rate
clip = 5 # Clip the gradient

out_slot = len(lang.slot2id)
out_int = len(lang.intent2id)
vocab_len = len(lang.word2id)
# use bert and bert tokenizer to get intents and slots






model.apply(init_weights)

optimizer = optim.Adam(model.parameters(), lr=lr)
criterion_slots = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
criterion_intents = nn.CrossEntropyLoss() # Because we do not have the pad token

n_epochs = 20
patience = 5

losses_train = []
losses_dev = []
sampled_epochs = []
best_f1 = 0

for x in tqdm(range(1,n_epochs)):
    loss = train_loop(train_loader, optimizer, criterion_slots, 
                    criterion_intents, model)
    if x % 5 == 0:
        sampled_epochs.append(x)
        losses_train.append(np.asarray(loss).mean())
        results_dev, intent_res, loss_dev = eval_loop(dev_loader, criterion_slots, 
                                                    criterion_intents, model, lang)
        losses_dev.append(np.asarray(loss_dev).mean())
        f1 = results_dev['total']['f']
        
        if f1 > best_f1:
            best_f1 = f1
        else:
            patience -= 1
        if patience <= 0: # Early stopping with patience
            break # Not nice but it keeps the code clean

results_test, intent_test, _ = eval_loop(test_loader, criterion_slots, 
                                        criterion_intents, model, lang)

print('Slot F1: ', results_test['total']['f'])
print('Intent Accuracy:', intent_test['accuracy'])