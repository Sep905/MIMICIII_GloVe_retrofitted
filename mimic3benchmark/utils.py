from __future__ import absolute_import
from __future__ import print_function
import codecs
import common_utils
import numpy as np
import torch
import os
from torch.utils.data import Dataset, DataLoader
import string


UNK_TOKEN = "<unk>"
PAD_TOKEN = "<pad>"
SOS_TOKEN = "<s>"
EOS_TOKEN = "</s>"


class Vocabulary:
    """
        Creates a vocabulary from a word2vec file. 
    """
    def __init__(self):
        self.idx_to_word = {0: PAD_TOKEN, 1: UNK_TOKEN, 2: SOS_TOKEN, 3: EOS_TOKEN}
        self.word_to_idx = {PAD_TOKEN: 0, UNK_TOKEN: 1, SOS_TOKEN: 2, EOS_TOKEN: 3}
        self.word_freqs = {}
       
    
    def __getitem__(self, key):
        return self.word_to_idx[key] if key in self.word_to_idx else self.word_to_idx[UNK_TOKEN]
    
    def word(self, idx):
        return self.idx_to_word[idx]
    
    def size(self):
        return len(self.word_to_idx)
    
    
    def from_data(input_file, vocab_size, emb_size):
      
        vocab = Vocabulary()
        vocab_size = vocab_size + len(vocab.idx_to_word)
        weight = np.zeros((vocab_size, emb_size))
        with codecs.open(input_file, 'rb')  as f:
         
          for l in f:
            line = l.decode().split()
            token = line[0]
            if token not in vocab.word_to_idx:
              idx = len(vocab.word_to_idx)
              vocab.word_to_idx[token] = idx
              vocab.idx_to_word[idx] = token
            
              vect = np.array(line[1:]).astype(np.float)
              weight[idx] = vect
          # average embedding for unk word
          avg_embedding = np.mean(weight, axis=0)
          weight[1] = avg_embedding
                            
        return vocab, weight

class MIMICTextDataset(Dataset):
    """
       Loads a list of sentences into memory from a text file,
       split by newlines. 
    """
    def __init__(self, reader, discretizer, normalizer, 
            notes_output='sentence', max_w=25, max_s=500, max_d=500,
            target_repl=False, batch_labels=False):
        self.data = []
        self.y  = []
        self.max_w = max_w
        self.max_s = max_s
        self.max_d = max_d
        N = reader.get_number_of_examples()

        ret = common_utils.read_chunk(reader, N)
        data = ret["X"]
        notes_text = ret["text"]
        notes_info = ret["text_info"]
        ts = ret["t"]
        labels = ret["y"]
        names = ret["name"]
        data = [discretizer.transform(X, end=t)[0] for (X, t) in zip(data, ts)]
        if normalizer is not None:
            data = [normalizer.transform(X) for X in data]

        self.notes = []
        tmp_data = []
        tmp_labels = []

        self.doc_categories = [] 

        if notes_output == 'sentence':
            # [N, W] patients, words
            for patient_notes, _x, l  in zip(notes_text, data, labels):
                tmp_notes = []
                for doc in sorted(patient_notes):
                    sentences = patient_notes[doc]
                    for sentence in sentences:
                        tmp_notes.extend(sentence)
                if len(tmp_notes) > 0 and len(tmp_notes) <= self.max_w:
                    self.notes.append(' '.join(tmp_notes))
                    tmp_data.append(_x)
                    tmp_labels.append(l)

        elif notes_output == 'sentence-max':
             # [N, W] patients, words
            for patient_notes, _x, l  in zip(notes_text, data, labels):
                tmp_notes = []
                for doc in sorted(patient_notes):
                    sentences = patient_notes[doc]
                    for sentence in sentences:
                        tmp_notes.extend(sentence)
                if len(tmp_notes) > 0 and len(tmp_notes) <= self.max_w:
                    self.notes.append(' '.join(tmp_notes))
                    tmp_data.append(_x)
                    tmp_labels.append(l)
                elif len(tmp_notes) > 0:
                    self.notes.append(' '.join(tmp_notes[:self.max_w]))
                    tmp_data.append(_x)
                    tmp_labels.append(l)

        elif notes_output == 'doc':
            # [N, S, W] patients, sentences, words
            for patient_notes,notex_category, _x, l  in zip(notes_text,notes_info, data, labels):
                tmp_notes = []

                tmp_categories = [] 

                for doc in sorted(patient_notes):
                    sentences = patient_notes[doc]
                    for sentence in sentences:
                        if len(sentence) > 0 and len(sentence) <= max_w:
                            tmp_notes.append(sentence)
                            tmp_categories.append(notex_category[doc][0]) 
                        elif len(sentence) > 0:
                            tmp_notes.append(sentence[:max_w])
                            tmp_categories.append(notex_category[doc][0]) 
                if len(tmp_notes) > 0 and len(tmp_notes) <= max_s:
                    self.notes.append(tmp_notes)
                    self.doc_categories.append(tmp_categories) 
                    tmp_data.append(_x)
                    tmp_labels.append(l)
                elif len(tmp_notes) > 0:
                    self.notes.append(tmp_notes[:max_s])
                    self.doc_categories.append(tmp_categories) 
                    tmp_data.append(_x)
                    tmp_labels.append(l)

        self.x = np.array(tmp_data, dtype=np.float32)   
        self.T = self.x.shape[1]
        if batch_labels:
            self.y = np.array([[l] for l in tmp_labels], dtype=np.float32)
        else:
            self.y = np.array(tmp_labels, dtype=np.float32)


    def _extend_labels(self, labels):
        # (B,)
        labels = labels.repeat(self.T, axis=1)  # (B, T)
        return labels

    def __len__(self):
        # overide len to get number of instances
        return len(self.x)

    def __getitem__(self, idx):
        # get words and label for a given instance index
        return self.x[idx], self.notes[idx], self.y[idx], self.doc_categories[idx] 


def doc_collate(batch):
    data = np.array([item[0] for item in batch])
    data = torch.tensor(data)
    notes = [item[1] for item in batch]
    target = np.array([item[2] for item in batch])
    target = torch.tensor(target)
    category = [item[3] for item in batch]    
    return [data, notes, target,category]



def create_doc_batch(docs, vocab, device,categories):
    """
    """

    sent_seq_lengths = np.array([len(doc) for doc in docs])
    word_seq_lengths = [[len(sent) for sent in doc] for doc in docs]
    b = len(docs)
    sent_max_len = max(sent_seq_lengths)
    word_max_len = max([max(w_seq) for w_seq in word_seq_lengths])
    pad_id = vocab[PAD_TOKEN]
    unk_id = vocab[UNK_TOKEN]

    pad_id_input = np.zeros((b, sent_max_len, word_max_len), dtype=int)

    semantic_group_input = np.zeros((b, sent_max_len, word_max_len), dtype=int)

    word_seq_length = np.ones((b, sent_max_len), dtype=int)
    categories_input = np.ones((b, sent_max_len), dtype=float)   
    
    
    for i, w_lens in enumerate(word_seq_lengths):
        for j, w_len in enumerate(w_lens):
            word_seq_length[i][j] = w_len
    #pad and find ids for words given the word2vec vocab
    for idx_doc, doc in enumerate(docs):
        for i in range(sent_max_len):
            tmp_sent = []
            if i < sent_seq_lengths[idx_doc]:

                categories_input[idx_doc][i] = categories[idx_doc][i]    
                
                sent = doc[i]
                for j in range(word_max_len):
                    if j < word_seq_lengths[idx_doc][i]: #[idx_doc][i]
                        try:
                            token_id = vocab[sent[j]]
                        except KeyError:
                            token_id = unk_id
                    else:
                        token_id = pad_id
                    pad_id_input[idx_doc][i][j] = token_id
            else:

                categories_input[idx_doc][i] = 14  

                for j in range(word_max_len):
                    pad_id_input[idx_doc][i][j] = pad_id

    # Convert everything to PyTorch tensors.
    batch_input = torch.tensor(pad_id_input)
    sent_seq_length = torch.tensor(sent_seq_lengths)
    word_seq_length = torch.tensor(word_seq_length.flatten())
    categories_input = torch.tensor(categories_input,dtype=torch.long)   
    
    # Move all tensors to the given device.
    batch_input = batch_input.to(device)
    #seq_mask = seq_mask.to(device)
    sent_seq_length = sent_seq_length.to(device)
    word_seq_length = word_seq_length.to(device)

    categories_input = categories_input.to(device)      
    
    return batch_input, sent_seq_length, word_seq_length, categories_input  

