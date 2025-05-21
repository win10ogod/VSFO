import json
import torch
from model import BraLM, Vocab
from tokenizers import Tokenizer

bpe_tokenizer = Tokenizer.from_file("wiki_bpe_tokenizer_4000_bytelevel.json")

def decode_en_sentence(head, max_token=32, do_sample=False):
    bpe_tokens = bpe_tokenizer.encode(head).tokens
    if len(bpe_tokens) < 2:
        return head
    start = [vocab((bpe_tokens[i] + '->' + bpe_tokens[i+1])) for i in range(len(bpe_tokens)-1)]
    ret = model.decode(start, vocab, max_token, do_sample)
    decode_tuple_list = [vocab.decode(p).split('->') for p in ret]
    decode_sentence = decode_tuple_list[0][0] + "".join([p[-1] for p in decode_tuple_list])
    return decode_sentence


with open("./vocab_wiki_4k_en.json") as f:
     node_dict = json.load(f)
vocab = Vocab.from_node_dict(node_dict)

model = BraLM(hidden_size=32)
model.prepare_network(vocab)

state_dict = torch.load("model_en.bin", weights_only=True)
model.load_state_dict(state_dict)
model.to_device("cuda:6")

head = "In frogs, the hind legs are larger"
encoding = bpe_tokenizer.encode(head)
token_len = len(encoding.ids)
max_token = 32 - token_len
decode_sentence = decode_en_sentence(head, max_token).replace("Ġ", " ")

print(decode_sentence)
