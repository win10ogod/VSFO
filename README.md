# BriLLM: Brain-inspired Large Language Model

We release BriLLM-Chinese and BriLLM-English.

Our paper: https://arxiv.org/pdf/2503.11299

Our Github: https://github.com/brillm05/BriLLM0.5

Our huggingface: https://huggingface.co/BriLLM/BriLLM0.5


## Overview

BriLLM redefines the foundations of generative language modeling by departing from Transformer architectures, GPT frameworks, and traditional input-output constrained paradigms. Built on the Signal Fully-connected flowing (SiFu) mechanism—a directed graph-based neural network design—BriLLM enables full interpretability across all nodes, in contrast to conventional models limited to input-output interpretability.

In this framework, tokens are represented as graph nodes, with signal flows—either randomly initialized or user-defined—propagating along paths following a "least resistance" principle. The next token to be generated emerges as the target of this signal flow. Theoretically, BriLLM supports infinitely long n-gram modeling, with model size decoupled from input and prediction length. Its signal propagation dynamics mimic human-like cognitive patterns, enabling recall activation and inherent multi-modal compatibility.

![](./figs/tab1.png)


## SiFu Mechanism

![](./figs/fig1.png)

The SiFu (Signal Fully-connected Flowing) mechanism addresses fundamental limitations of current machine learning frameworks. Unlike traditional models that process discrete input streams through opaque computations, SiFu operates on a fully connected directed graph where:

- Each node represents an interpretable unit (token, concept, etc.)
- Signal tensors propagate through the graph following energy dynamics
- The next token is determined by maximizing signal energy
- All nodes can serve as both input and output interfaces

![](./figs/fig2.png)

Signal propagation follows the principle:
$v_i = \arg\max_{v'} \left\| r \oplus v_1 \otimes e_{12} \oplus v_2 \ldots \oplus v' \right\|$

where $\oplus$ and $\otimes$ denote tensor operations for node and edge interactions, and $\|\cdot\|$ represents signal energy.

Overall, SiFu's design as a directed fully connected graph with signal propagation confers two key advantages:
1. **Inherent full interpretability**: User-defined entities (concepts, tokens, or interpretable units) map directly to specific graph nodes;
2. **Unbounded contextual capacity**: Prediction is framed as signal propagation through node activations. Because signals propagate freely across nodes, sequence prediction naturally supports arbitrarily long contexts without increasing model size. 


## Architecture

![](./figs/fig3.png)

BriLLM implements the SiFu mechanism where each vocabulary token corresponds to a node defined by a GeLU-activated neuron layer with bias $b \in \mathbb{R}^{d_{node}}$. Edges between nodes are modeled as fully connected matrices $W_{u,v} \in \mathbb{R}^{d_{node} \times d_{node}}$, enabling bidirectional signaling.

Signal propagation begins with initial tensor $e_0 = [1, 1, \ldots, 1]^T \in \mathbb{R}^{d_{node}}$ and follows:

$e_{i+1} = \text{GeLU}(W_{u_i,u_{i+1}} e_i + b_{u_i,u_{i+1}} + PE_i)$

The final prediction maximizes the L2 norm: $v_{predict} = \arg\max_v \|E_{u,v}\|_2$


## Training Network

![](./figs/fig4.png)

Training BriLLM involves constructing a dedicated neural network for each sequence sample. The network connects input nodes sequentially, with all potential paths integrated into a final softmax layer that identifies the correct path via cross-entropy loss optimization.


## Implementation Details

BriLLM is implemented using PyTorch. It uses sinusoidal positional encoding, GeLU as the activation function, cross-entropy loss for next-token prediction, and an embedding size of $d_{model} = 32$. We used the AdamW optimizer with $\beta_1 = 0.9$, $\beta_2 = 0.999$ and $\epsilon = 10^{-8}$. The model size is about $512 + 4000 * 4000 * (32 * 32 + 32) \approx 16B$. We trained our models on one machine with 8 NVIDIA A800 GPUs for 1.5k steps.

![](./figs/fig5.png)


BriLLM leverages sparse token co-occurrence: most bigrams are low-frequency or absent, allowing shared parameters for inactive edges. Low-frequency bigrams use a fixed, non-updated matrix, reducing model size to 2B (Chinese) and 1B (English)—13.0\% and 5.7\% of the original size, respectively. This reduces parameters by ~90\% while accelerating training.

![](./figs/tab2.png)


## Case Study

### Chinese Examples
![](./figs/tab3.png)


### English Examples  
![](./figs/tab4.png)


## Comparison: Traditional LLMs vs BriLLM

![](./figs/tab5.png)


## Installation

```bash
pip install torch
```


## Model Checkpoints

[BriLLM0.5](https://huggingface.co/BriLLM/BriLLM0.5)


## Training

### BriLLM-Chinese
```bash
bash run_zh.sh
```

### BriLLM-English
```bash
bash run_en.sh
```


## Inference

### BriLLM-Chinese
```python
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
```

### BriLLM-English
```python
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
```