"""
if you have a BloomForCausalLM model that have been fine-tuned, you can cut it by this code.
you should notice, the lm_head and embedding layer are two independent layer.
it means, parameters are not shared on input embedding and output embedding, this is different from T5

cut by your own char set,
bloom use BPE tokenizer
"""
import csv
import string
from zhon.hanzi import punctuation
import pandas as pd
from collections import Counter
import torch
from transformers import BloomForCausalLM, AutoTokenizer
from tqdm.auto import tqdm, trange

model_path = '/*/models/checkpoint-93600'
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = BloomForCausalLM.from_pretrained(model_path)


# rewrite this function to filter out the tokens that You won't use
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
transformer = model.transformer
lm_head = model.lm_head
print("模型embeding大小：", msize(transformer.word_embeddings) / msize(model))


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
new_emb = torch.nn.Embedding(new_size, transformer.word_embeddings.embedding_dim)
new_head = torch.nn.Linear(in_features=lm_head.in_features, out_features=new_size, bias=False)

for _, new_id, old_id in pieces_list:
    new_emb.weight.data[new_id] = transformer.word_embeddings.weight.data[old_id]
    # print("new_id", new_id, "old_id", old_id)
    new_head.weight.data[new_id] = lm_head.weight.data[old_id]
# model.word_embeddings.weight = new_emb.weight
del transformer.word_embeddings.weight
del lm_head.weight
transformer.word_embeddings = new_emb
lm_head.weight = new_head.weight

model.config.__dict__['vocab_size'] = new_size
model.config.__dict__['_name_or_path'] = 'bigscience/bloomLM-93600-en_zh'

model.save_pretrained('./bloomLM-93600-en_zh')


import json
with open("/*/models/checkpoint-93600/tokenizer.json", "r") as f:
    d = json.load(f)
new_vocab = {}

# --- can't work, maybe broken Byte-pairs
# for t, new_id, _ in pieces_list:
#     new_vocab[t] = new_id
# d['model']['vocab'] = new_vocab
#
# merge = d['model']['merges']
# new_merge = []
# for m in new_merge:
#     if m in new_vocab:
#         new_merge.append(m)
# d['model']['merges'] = new_merge
#
# merge = d['model']['merges']
# new_merge = []
# for m in new_merge:
#     if m in new_vocab:
#         new_merge.append(m)
# d['model']['merges'] = new_merge


# --- re-index all tokens, remain token - new_index, dropped token - 0, 会错误分词
# cnt = 0
# for t, new_id, _ in pieces_list:
#     if t in d['model']['vocab']:
#         new_vocab[t] = new_id
#         cnt += 1
# print('re-index:', cnt)
#
# for t, old in d['model']['vocab'].items():
#     if t not in new_vocab:
#         new_vocab[t] = 0
#
# print('final vocab:', len(new_vocab))
#
# d['model']['vocab'] = new_vocab

# --- re-index all tokens, remain token - new_index, dropped token -> cnt+=1,
# worked, but tokenizer.vocab still contains all tokens
cnt = 0
for t, new_id, _ in pieces_list:
    if t in d['model']['vocab']:
        new_vocab[t] = new_id
        cnt += 1
print('re-index:', cnt)

for t, old in d['model']['vocab'].items():
    if t not in new_vocab:
        new_vocab[t] = cnt
        cnt += 1

print('final vocab:', len(new_vocab))

d['model']['vocab'] = new_vocab


with open("/*/models/bloomLM-93600-en_zh/tokenizer.json", "w", encoding='utf-8') as f:
    json.dump(d, f, ensure_ascii=False, indent=1)

# from transformers import BloomTokenizerFast
# tokenizer = BloomTokenizerFast.from_pretrained("bigscience/bloom")
# tokenizer.get_vocab()['<unk>']
# tokenizer.save_vocabulary()
# tokenizer.save_vocabulary()

# old model
from transformers import BloomForCausalLM, AutoTokenizer
model_path = '/*/models/checkpoint-93600'
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = BloomForCausalLM.from_pretrained(model_path)

inputs = tokenizer.encode("南京是中华民国的什么？", return_tensors="pt")
print(inputs)
print(tokenizer.convert_ids_to_tokens(inputs[0]))
outputs = model.generate(inputs)
print(outputs)
print(tokenizer.decode(outputs[0]))

# new model
path = '/*/models/bloomLM-93600-en_zh'
tk = AutoTokenizer.from_pretrained(path)
light = BloomForCausalLM.from_pretrained(path)

inputs = tk.encode("南京是中华民国的什么？", return_tensors="pt")
print(inputs)
print(tk.convert_ids_to_tokens(inputs[0]))
outputs = light.generate(inputs)
print(outputs)
print(tk.decode(outputs[0]))
