"""
cut by your own char set,
bloom use BPE tokenizer
todo: there is still a problem, but easy to avoid.
tokenizer can encode out a out of model vocab_size ids, you can limit it before put 'inputs' Tensor into model.
for example:
inputs = tokenizer.encode("한국어南京是中华民国的什么？", return_tensors="np")
inputs[inputs >= model.config.vocab_size] = 0
"""
import string
from zhon.hanzi import punctuation
import torch
from transformers import AutoTokenizer, BloomModel

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
print("model embedding params ratio：", msize(model.word_embeddings) / msize(model))

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
    # print("new_id", new_id, "old_id", old_id)
    # new_head.weight.data[new_id] = model.lm_head.weight.data[old_id]
# model.word_embeddings.weight = new_emb.weight
del model.word_embeddings.weight
model.word_embeddings = new_emb
# model.lm_head.weight = new_head.weight

model.config.__dict__['vocab_size'] = new_size
model.config.__dict__['_name_or_path'] = 'bigscience/bloom-560m-en_zh'

model.save_pretrained('./bloom-560m-en_zh')


import json
with open("/*/models/bloom560m/tokenizer.json", "r") as f:
    d = json.load(f)
new_vocab = {}

# --- only keep useful tokens in vocab and merges. can't work, maybe broken Byte-pairs
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


with open("/*/models/bloom-560m-en_zh/tokenizer.json", "w", encoding='utf-8') as f:
    json.dump(d, f, ensure_ascii=False, indent=1)

# from transformers import BloomTokenizerFast
# tokenizer = BloomTokenizerFast.from_pretrained("bigscience/bloom")
# tokenizer.get_vocab()['<unk>']
# tokenizer.save_vocabulary()
# tokenizer.save_vocabulary()


from transformers import BloomForCausalLM, AutoTokenizer

# --- new model
path = '/*/models/bloom-560m-en_zh'
tk = AutoTokenizer.from_pretrained(path)
light = BloomForCausalLM.from_pretrained(path)

inputs = tk.encode("南京是中华民国的什么？", return_tensors="pt")
print(inputs)
print(tk.convert_ids_to_tokens(inputs[0]))
outputs = light.generate(inputs)
print(outputs)
print(tk.decode(outputs[0]))

# old model
model_path = '/*/models/bloom560m'
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = BloomForCausalLM.from_pretrained(model_path)
inputs = tokenizer.encode("南京是中华民国的什么？", return_tensors="pt")
print(inputs)
print(tokenizer.convert_ids_to_tokens(inputs[0]))
outputs = model.generate(inputs)
print(outputs)
print(tokenizer.decode(outputs[0]))