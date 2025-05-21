import json
import torch
from model import BraLM, Vocab

with open("vocab_wiki_4k.json") as f:
     node_dict = json.load(f)
vocab = Vocab.from_node_dict(node_dict)

with open('word_frequency.json', 'r') as f:
    freq_dict = json.load(f)

zero_freq_edges = {}
for s in freq_dict:
    zero_freq_edges[s] = []
    for t in freq_dict[s]:
        if freq_dict[s][t] == 0:
            zero_freq_edges[s].append(t)

model = BraLM(hidden_size=32, zero_freq_edges=zero_freq_edges, vocab=vocab)
model.prepare_network(vocab)

state_dict = torch.load("model_zh.bin", weights_only=True)
model.load_state_dict(state_dict)
model.to_device("cuda:6")

head = "《罗马》描述了"
max_token = 32 - len(head)

start = [vocab((head[i]+ '->' +head[i+1])) for i in range(len(head)-1)]
ret = model.decode(start, vocab, max_token)
decode_tuple_list = [vocab.decode(p) for p in ret]
decode_sentence = decode_tuple_list[0][0] + "".join([p[-1] for p in decode_tuple_list])

print(decode_sentence)
