#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import spacy
import torch
import torch.nn as nn
from torch.optim import Adam
from gensim.models import word2vec
from torch.utils.data import DataLoader, Dataset
import os
import sys


# In[2]:


class Vocab:
    def __init__(self, w2v):
        self._idx2token = [token for token, _ in w2v]
        self._token2idx = {token: idx for idx,
                           token in enumerate(self._idx2token)}
        self.PAD, self.UNK = self._token2idx["<PAD>"], self._token2idx["<UNK>"]

    def trim_pad(self, tokens, seq_len):
        return tokens[:min(seq_len, len(tokens))] + [self.PAD] * (seq_len - len(tokens))

    def convert_tokens_to_indices(self, tokens):
        return [
            self._token2idx[token]
            if token in self._token2idx else self.UNK
            for token in tokens]

    def __len__(self):
        return len(self._idx2token)


def to_arr(sen, vocab):
    sen = vocab.trim_pad(sen,30)
    sen = vocab.convert_tokens_to_indices(sen)
    return sen

class hw5_train(Dataset):
    
    def __init__(self, sentences, label):
        self.sentences = sentences
        self.label = label
        
    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        data = self.sentences[idx]
        label = self.label[idx]
        return data, label

class Example_Net(nn.Module):
    def __init__(self, pretrained_embedding, hidden_size, n_layers, bidirectional, dropout, padding_idx):
        super(Example_Net, self).__init__()
        
        pretrained_embedding = torch.FloatTensor(pretrained_embedding)
        self.embedding = nn.Embedding(
            pretrained_embedding.size(0),
            pretrained_embedding.size(1),
            padding_idx=padding_idx)
        # Load pretrained embedding weight
        self.embedding.weight = torch.nn.Parameter(pretrained_embedding)

        self.rnn = nn.GRU(
            input_size=pretrained_embedding.size(1),
            hidden_size=hidden_size,
            num_layers=n_layers,
            dropout=dropout,
            bidirectional=bidirectional,
            batch_first=True)
        self.classifier = nn.Linear(
            hidden_size * (1+bidirectional), 1)

    def forward(self, batch):
        batch = self.embedding(batch)
        output, _ = self.rnn(batch)
        output = output.mean(1)
        logit = self.classifier(output)
        return logit

class hw5_test(Dataset):
    def __init__(self, sentences):
        self.sentences = sentences
        
    def __len__(self):
        return 860

    def __getitem__(self, idx):
        data = self.sentences[idx]
        return data


# In[3]:


if __name__ == '__main__':
    if  len(sys.argv) == 5 and sys.argv[4]=='--train':
        train_csv = pd.read_csv(sys.argv[1])
        test_csv = pd.read_csv( sys.argv[3])
        label = pd.read_csv( sys.argv[2])['label']

        doc = train_csv.append(test_csv)['comment']
        nlp = spacy.load('en_core_web_sm')
        tokenizer = spacy.lang.en.English().Defaults().create_tokenizer(nlp)
        sentences = [tokenizer(sen) for sen in doc]
        tokens = [ [ token.orth_ for token in sen] for sen in sentences]
#         sentences = word2vec.LineSentence("corpus_seg.csv")

        # model = word2vec.Word2Vec(
        #     sentences=sentences,
        #     size=args.size,
        #     window=args.window,
        #     iter=args.iter,
        #     negative=args.negative,
        #     workers=args.workers,
        #     min_count=args.min_count,
        #     compute_loss=True,
        #     callbacks=[callback(args.iter)])

        size = 300
        w2v_model = word2vec.Word2Vec(
            sentences=tokens,
            size=size,
            window=3,
        #     iter=args.iter,
            sg=1,
            iter=20,
            negative=10,
            workers=8,
            seed=2266,
        #     seed=9527,
            min_count=2,
        #     compute_loss=True,
        #     callbacks=[callback(args.iter)],
            )
        w2v_model.save("word2vec_seg_iter20_rand2266.model")

        w2v = []
        for _, key in enumerate(w2v_model.wv.vocab):
            w2v.append((key, w2v_model.wv[key]))
        special_tokens = ["<PAD>", "<UNK>"]
        for token in special_tokens:
            w2v.append((token, [0.0] * size)) # Initialize <PAD>&<UNK> as 0s
        #     w2v.append((token, [0.0] * args.size)) # Initialize <PAD>&<UNK> as 0s
        vocab = Vocab(w2v)
        pretrained_embedding = [ emb for word, emb in w2v]
        # data = [ to_arr(sen,vocab) for sen in word_corpus ]
        tokens = [vocab.trim_pad(sen, 30) for sen in tokens]
        tokens = [ vocab.convert_tokens_to_indices(tok) for tok in tokens ]

        train_dataset = hw5_train(np.array(tokens,dtype=np.int64),label)
        train_loader = DataLoader(train_dataset, batch_size=50, shuffle=False)


        hidden_size=60
        n_layers=1
        bidirectional=False
        dropout=0
        padding_idx=-1

        model = Example_Net(pretrained_embedding, hidden_size=hidden_size, n_layers=n_layers,bidirectional=bidirectional,
                            dropout=dropout,padding_idx=padding_idx)
        model.embedding.weight.requires_grad=False

        use_gpu = torch.cuda.is_available()
        if use_gpu:
            model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
        loss_fn = nn.BCEWithLogitsLoss()

        num_epoch = 14
        for epoch in range(num_epoch):
                model.train()
                train_loss = []
                train_acc = []

                for idx, (sent, label_) in enumerate(train_loader):
                    if use_gpu:
                        sent = torch.LongTensor(sent).cuda()
                        label_ = label_.reshape(len(label_),1).cuda()
                    optimizer.zero_grad()
                    output = model(sent)
                    loss = loss_fn(output, label_.float())
                    loss.backward()
                    optimizer.step()

        #             predict = torch.max(output, 1)[1]
                    predict = (output > 0).long()
                    acc = np.mean((label_ == predict).cpu().numpy())
                    train_acc.append(acc)
                    train_loss.append(loss.item())
        #         if (epoch+1) % 5 == 0:
#                 print("Epoch: {}, train Loss: {:.4f}, train Acc: {:.4f}".format(epoch + 1, np.mean(train_loss), np.mean(train_acc)))
        torch.save(model.state_dict(), "gru.pkl")
    
    if sys.argv[3] == '--test':
        test_csv = pd.read_csv(sys.argv[1])
        doc = test_csv['comment']
        nlp = spacy.load('en_core_web_sm')
        tokenizer = spacy.lang.en.English().Defaults().create_tokenizer(nlp)
        sentences = [tokenizer(sen) for sen in doc]
        tokens = [ [ token.orth_ for token in sen] for sen in sentences]

#         test_csv['comment'].to_csv('test_.csv', index=False)
#         sentences = word2vec.LineSentence('test_.csv')

        w2v_model = word2vec.Word2Vec.load("word2vec_seg_iter20_rand2266.model")
        w2v = []
        for _, key in enumerate(w2v_model.wv.vocab):
            w2v.append((key, w2v_model.wv[key]))
        special_tokens = ["<PAD>", "<UNK>"]
        for token in special_tokens:
            w2v.append((token, [0.0] * 300)) # Initialize <PAD>&<UNK> as 0s
        #     w2v.append((token, [0.0] * args.size)) # Initialize <PAD>&<UNK> as 0s

        vocab = Vocab(w2v)
        # data = [ to_arr(sen,vocab) for sen in word_corpus ]
#         data = [ to_arr(sen,vocab) for sen in sentences ]
        tokens = [vocab.trim_pad(sen, 30) for sen in tokens]
        tokens = [ vocab.convert_tokens_to_indices(tok) for tok in tokens ]
        
        pretrained_embedding = [ emb for word, emb in w2v]

        hidden_size=60
        n_layers=1
        bidirectional=False
        dropout=0
        padding_idx=-1

        model = Example_Net(pretrained_embedding, hidden_size=hidden_size, n_layers=n_layers,bidirectional=bidirectional,
                            dropout=dropout,padding_idx=padding_idx)
        model.load_state_dict(torch.load("gru.pkl"))

        test_dataset = hw5_test(np.array(tokens[::],dtype=np.int64))
        test_loader = DataLoader(test_dataset, batch_size=50, shuffle=False)
        
        use_gpu = torch.cuda.is_available()
        if use_gpu:
            model.cuda()
        model.eval()
        pred_list = []
        for idx, sent in enumerate(test_loader):
            if use_gpu:
                sent = torch.LongTensor(sent).cuda()
            output = model(sent)
            pred = output.detach().cpu().numpy()
            pred = pred>0
            pred_list.append(pred)
        pred = np.concatenate(pred_list).reshape(-1)
        output = pd.DataFrame(list(enumerate(pred.astype(int))), columns=['id','label'])
        output.to_csv(sys.argv[2], index=False)

