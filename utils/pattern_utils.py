import os
import torch


def pad_sequence(batch):
    sorted_batch = sorted(batch, key=lambda x: x[0].shape[0], reverse=True)
    sequences = [x[0] for x in sorted_batch]
    sequences_padded = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True)
    lengths = torch.LongTensor([len(x) for x in sequences])
    labels = torch.LongTensor(map(lambda x: x[1], sorted_batch))
    return sequences_padded, lengths, labels


def read_file(patterns):
    p = open(patterns, "r")
    p = p.readlines()
    pat = parse_patterns(p)
    return pat


def parse_patterns(p):
    patterns = []
    for el in p:
        out = el[:el.find(']') + 1]
        out = out.replace('[', '').replace(']', '').replace("'", '').replace(',', ' ')
        out = out.split()
        patterns.append((out, int(el[el.find(']') + 1:].replace('\n',''))))
    return patterns


def read_directory(path):
    all_patterns = [read_file(path+f) for f in os.listdir(path) if '_intervals.txt' in f]
    return all_patterns


def process_patterns(pat):
    docs=[]
    for doc in pat:
        docu=[]
        for d in doc:
            #docu.append('_'.join(d[0]))
            docu.append(d[0])
        docs.append(docu)
    return docs


def create_dictionary(path):
    patterns = read_directory(path)
    songs = process_patterns(patterns)
    dictionary =[]
    vocab = [dictionary.append(p) for d in songs for p in d if p not in dictionary]
    return dictionary


def find_sublist(s,l):
    result=[]
    sll=len(s)
    for ind in (i for i,e in enumerate(l) if e==s[0]):
        if l[ind:ind+sll]==s:
            result.append((ind,ind+sll-1))
    return result