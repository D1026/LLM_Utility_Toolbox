"""
cut by your own corpus
"""
import torch
from transformers import MT5ForConditionalGeneration, T5Tokenizer
import pandas as pd
import csv
from collections import Counter
from tqdm.auto import tqdm, trange


tokenizer = T5Tokenizer.from_pretrained("resources/mt5-base")
model = MT5ForConditionalGeneration.from_pretrained("resources/mt5-base")


def msize(m):
    return sum(p.numel() for p in m.parameters())


print("模型共享embeding大小：", msize(model.shared) / msize(model))
print(msize(model.lm_head) / msize(model))


# filter out the tokens needed through your corpus
df_yours = pd.read_csv("data/new_corpus.dat", sep='\t', header=None, names=["text"], quoting=csv.QUOTE_NONE)
cnt_keep = Counter()
for text in tqdm(df_yours.text):
    cnt_keep.update(tokenizer.encode(text))
print("keep tokens 数量：{}, 占比: {}".format(len(cnt_keep), len(cnt_keep)/tokenizer.vocab_size))
for top in 10_000, 20_000, 30_000:
    print(top, sum(v for k, v in cnt_keep.most_common(top)) / sum(cnt_keep.values()))

df_en = pd.read_csv("/*/eng-com_web-public_2018_1M/eng-com_web-public_2018_1M-sentences.txt",
                    sep='\t', header=None, quoting=csv.QUOTE_NONE, names=["idx", "text"])
cnt_en = Counter()
for text in tqdm(df_en.text):
    cnt_en.update(tokenizer.encode(text))

new_tokens = set(range(1000))
for i, (k, v) in enumerate(cnt_en.most_common(10_000)):
    if k not in new_tokens:
        new_tokens.add(k)
for i, (k, v) in enumerate(cnt_keep.most_common(30_000)):
    if len(new_tokens) == 34_900:
        print(i, 'Keep tokens are included')
        break
    if k not in new_tokens:
        new_tokens.add(k)

for t in range(tokenizer.vocab_size - 100, tokenizer.vocab_size):  # 100 <extra_id_xx> that mT5 uses
    new_tokens.add(t)
print(len(new_tokens))

kept_ids = sorted(new_tokens)
new_size = len(kept_ids)
new_emb = torch.nn.Embedding(new_size, model.shared.embedding_dim)
new_head = torch.nn.Linear(in_features=model.lm_head.in_features, out_features=new_size, bias=False)
for new_id, old_id in enumerate(kept_ids):
    new_emb.weight.data[new_id] = model.shared.weight.data[old_id]
    new_head.weight.data[new_id] = model.lm_head.weight.data[old_id]
model.shared.weight = new_emb.weight
model.lm_head.weight = new_head.weight

model.config.__dict__['vocab_size'] = new_size
model.config.__dict__['_name_or_path'] = 'resources/your-t5-base'

import sentencepiece_model_pb2 as spmp
smp = tokenizer.sp_model.serialized_model_proto()
m = spmp.ModelProto()
m.ParseFromString(smp)

print('the loaded model has pieces:', len(m.pieces))
new_pieces = [m.pieces[idx] for idx in kept_ids]
print('the new pieces:', len(new_pieces))

# replace the content of the first 30K pieces
for i, p in enumerate(new_pieces):
    m.pieces[i].piece = p.piece
    m.pieces[i].score = p.score
    m.pieces[i].type = p.type

# drop the remaining pieces
n = len(new_pieces)
for i in trange(len(m.pieces) - n):
    m.pieces.pop(len(m.pieces) - 1)

print(len(m.pieces))
with open('resources/your_sp.model', 'wb') as f:
    f.write(m.SerializeToString())

new_tokenizer = T5Tokenizer('resources/your_sp.model', extra_ids=0)
new_tokenizer.save_pretrained('resources/your-t5-base')
model.save_pretrained('resources/your-t5-base')

#tf_model = TFMT5ForConditionalGeneration.from_pretrained('resources/keept5-base', from_pt=True)

#tf_model..save_pretrained('keep_saved_model', saved_model=True)
