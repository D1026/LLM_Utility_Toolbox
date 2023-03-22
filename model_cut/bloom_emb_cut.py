"""
cut by your own char set
"""
import csv
import string
from zhon.hanzi import punctuation
import pandas as pd
from collections import Counter
import torch
from transformers import AutoTokenizer, BloomModel
from tqdm.auto import tqdm, trange

model_path = '/*/models/bloom560m'
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = BloomModel.from_pretrained(model_path)


def is_retain_token(k):
    for c in k:
        if (u'\u4e00' <= c <= u'\u9fff') or (c in punctuation):
            continue
        # if c.isalpha() or c.isdigit():
        if (' ' <= c <= '~') or c.isdigit():
            continue
        if c in (list(string.punctuation) + ['\n', '\t', ' ']):
            continue
        return False
    return True
# print('【','---', is_retain_token('【'))
# for i in range(1000, 1100):
#     print(tokenizer.decode([i]),'---', is_retain_token(tokenizer.decode([i])))


def msize(m):
    return sum(p.numel() for p in m.parameters())

# you can use print(model) look up the name of every layer
print("模型embeding大小：", msize(model.word_embeddings) / msize(model))

pieces_list = []
new_idx = 0
for i in range(tokenizer.vocab_size):
    if is_retain_token(tokenizer.decode([i])):
        # pieces_list.append([tokenizer.decode([i]), new_idx, i])
        pieces_list.append([tokenizer.convert_ids_to_tokens(i), new_idx, i])
        new_idx += 1


new_size = len(pieces_list)
print("old_size:", tokenizer.vocab_size)
print("new_size:", new_size)
new_emb = torch.nn.Embedding(new_size, model.word_embeddings.embedding_dim)
# new_head = torch.nn.Linear(in_features=model.lm_head.in_features, out_features=new_size, bias=False)
for _, new_id, old_id in pieces_list:
    new_emb.weight.data[new_id] = model.word_embeddings.weight.data[old_id]
    # new_head.weight.data[new_id] = model.lm_head.weight.data[old_id]
model.word_embeddings.weight = new_emb.weight
# model.lm_head.weight = new_head.weight

model.config.__dict__['vocab_size'] = new_size
model.config.__dict__['_name_or_path'] = 'bigscience/bloom-560m-en_zh'

model.save_pretrained('./bloom-560m-en_zh')


import json
with open("/*/models/bloom560m/tokenizer.json", "r") as f:
    d = json.load(f)
new_vocab = {}
for t, new_id, _ in pieces_list:
    new_vocab[t] = new_id
d['model']['vocab'] = new_vocab

with open("*/tokenizer.json", "w") as f:
    json.dump(d, f, ensure_ascii=False)

# from transformers import BloomTokenizerFast
# tokenizer = BloomTokenizerFast.from_pretrained("bigscience/bloom")
# tokenizer.get_vocab()['<unk>']
# tokenizer.save_vocabulary()
# tokenizer.save_vocabulary()
#
# new_tokenizer = T5Tokenizer('resources/your_sp.model', extra_ids=0)
# new_tokenizer.save_pretrained('resources/your-t5-base')
# model.save_pretrained('resources/your-t5-base')

#tf_model = TFMT5ForConditionalGeneration.from_pretrained('resources/keept5-base', from_pt=True)

#tf_model..save_pretrained('keep_saved_model', saved_model=True)
