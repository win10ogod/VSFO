import argparse
import logging
import os
import random
import math
import copy
import json
import numpy as np
import torch
import torch.nn as nn
import glob
from tqdm.auto import tqdm, trange
from torch.autograd import Variable
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import InitProcessGroupKwargs
from torch.utils.data import IterableDataset, DataLoader, Dataset
import time
import torch.distributed as dist
import gc
from datetime import timedelta
from tokenizers import Tokenizer

import wandb

os.environ["WANDB_WATCH"] = "false"


class BraLM(nn.Module):
    def __init__(self, hidden_size, use_ds=False, zero_freq_edges=None, vocab=None):
        super().__init__()
        self.hidden_size = hidden_size
        self.activation = nn.GELU()
        self.positions = nn.Parameter(torch.ones(1, 512, 1))
        self.device = None
        
        # for fsdp
        self._tied_weights_keys = []

        self.use_ds = use_ds
        self.zero_freq_edges = zero_freq_edges
        self.vocab = vocab

    def prepare_network(self, vocab):
        # Create index mappings for the flattened structure
        self.weight_indices = {}  # Maps (s_idx, t_idx) to parameter index
        self.shared_param_idx = 0
        
        # Current index for new parameters
        current_idx = 1
        
        # Populate parameters and mappings
        for s_idx, s in enumerate(vocab.edge_dict):
            for t_idx, t in enumerate(vocab.edge_dict[s]):
                if self.zero_freq_edges is not None and t in self.zero_freq_edges[s]:
                    # Use shared parameters
                    self.weight_indices[(s_idx, t_idx)] = self.shared_param_idx
                else:
                    self.weight_indices[(s_idx, t_idx)] = current_idx
                    current_idx += 1

        # Create new parameters
        self.weights = nn.Parameter(torch.randn(current_idx, self.hidden_size, self.hidden_size).uniform_(-0.5, 0.5))
        self.biases = nn.Parameter(torch.randn(current_idx, 1, self.hidden_size).uniform_(-0.5, 0.5))

        self.node_bias = nn.Parameter(torch.randn(len(vocab.edge_dict), 1, self.hidden_size).uniform_(-0.5, 0.5))

    def to_device(self, device):
        self.weights.to(device)
        self.biases.to(device)
        self.positions.data = self.positions.data.to(device)
        self.device = device

    @staticmethod
    def _reshape12(x):
        return x.reshape(-1, x.size(-2), x.size(-1))
    
    def get_positional_encoding(self, seq_len, d_model):
        position = torch.arange(0, seq_len).reshape(-1, 1)
        div_term = 10000.0 ** (torch.arange(0, d_model, 2) / d_model)
        position_encoding = torch.zeros(seq_len, d_model)
        position_encoding[:, 0::2] = torch.sin(position * div_term)
        position_encoding[:, 1::2] = torch.cos(position * div_term)
        return position_encoding.unsqueeze(0).to(self.device)

    # def get_initial_tensor(self, batch_size, max_norm=1.0):
    #     # initialize energy_tensor
    #     energy_tensor = torch.zeros(batch_size, 1, self.hidden_size).normal_(0, 1).to(self.device)
    #     delta_norm = torch.norm(energy_tensor.view(energy_tensor.shape[0], -1), dim=-1, p="fro").detach()
    #     clip_mask = (delta_norm > max_norm).to(energy_tensor)
    #     clip_weights = max_norm / delta_norm * clip_mask + (1 - clip_mask)
    #     energy_tensor = (energy_tensor * clip_weights.view(-1, 1, 1)).detach()    #(bs, 1, hs)
    #     return energy_tensor
    
    def get_initial_tensor(self, batch_size, d, pe):
        # initialize energy_tensor
        energy_tensor = torch.ones(batch_size, 1, self.hidden_size) / self.hidden_size   #(bs, 1, hs)
        energy_tensor = energy_tensor.to(self.device)

        node_bias = self.node_bias[d[:, 0, 0]]
        energy_tensor = self.activation(energy_tensor + node_bias + Variable(pe[:,0], requires_grad=False))
        return energy_tensor


    def forward(self, neighbor_ids):
        # neighbor_ids: (bs, sen_len, 1+k, 2) ; k is the number of negative samples
        batch_size = neighbor_ids.size(0)
        loss = 0

        pe = self.get_positional_encoding(512, self.hidden_size)  #(1, 512, hs)

        for i in range(neighbor_ids.size(1)):
            d = neighbor_ids[:, i]  #(bs, 1+k, 2)
            
            if i == 0:
                # for the first token, initialize energy_tensor as an all-one tensor
                energy_tensor = self.get_initial_tensor(batch_size, d, pe)  #(bs, 1, hs) 
            else:
                energy_tensor = (energy_cache * self.positions[:, :i, :].softmax(1)).sum(1, keepdim=True)   #(bs, 1, hs) :fix dim bug

            # Vectorized parameter lookup
            src_idx = d[..., 0]  # (bs, 1+k)
            tgt_idx = d[..., 1]  # (bs, 1+k)
            param_indices = torch.tensor([self.weight_indices.get((s.item(), t.item()), self.shared_param_idx) 
                                        for s, t in zip(src_idx.reshape(-1), tgt_idx.reshape(-1))], 
                                       device=self.device).reshape(batch_size, -1)  # (bs, 1+k)
            
            # Batch gather operation
            w = self.weights[param_indices]  # (bs, 1+k, hidden_size, hidden_size)
            b = self.biases[param_indices]   # (bs, 1+k, 1, hidden_size)

            expand_energy_tensor = self._reshape12(energy_tensor.unsqueeze(1).repeat(1, w.size(1), 1, 1))  #(bs*(1+k), 1, hs)
            # for deepspeed fp16: expand_energy_tensor.half()
            if self.use_ds:
                expand_energy_tensor = expand_energy_tensor.half()
            nxt_energy_tensor = self.activation(expand_energy_tensor.bmm(self._reshape12(w))+self._reshape12(b)+Variable(pe[:,i+1], requires_grad=False))  #(bs*(1+k), 1, hs)
            output_tensor = nxt_energy_tensor.reshape(batch_size, -1, nxt_energy_tensor.size(-2), nxt_energy_tensor.size(-1))  #(bs, 1+k, 1, hs)

            if i == 0:
                energy_cache = output_tensor[:,0]   #(bs, 1, hs)
            else:
                energy_cache = torch.cat([energy_cache, output_tensor[:,0]], dim=1)  #(bs, i+1, hs)

            if 1:
                energy = output_tensor.norm(2, (-2, -1))
                label = torch.LongTensor([0 for _ in range(batch_size)]).to(self.device)
                loss += nn.CrossEntropyLoss()(energy, label)

        return loss / neighbor_ids.size(1)

    def decode(self, start, vocab, max_new_tokens=16, do_sample=False, temperature=1):
        ret = []
        pe = self.get_positional_encoding(512, self.hidden_size)
        
        for i, pair in enumerate(start):
            if i == 0:
                energy_tensor = self.get_initial_tensor(batch_size=1, d=torch.tensor([[pair]], device=self.device), pe=pe).squeeze(0)
            else:
                energy_tensor = (energy_cache * self.positions[:, :i, :].softmax(1)).sum(1, keepdim=True).squeeze(0)
            
            # Get parameter index for this edge
            param_idx = self.weight_indices.get((pair[0], pair[1]), self.shared_param_idx)
            
            # Get weights and biases using parameter index
            w = self.weights[param_idx].to(self.device)
            b = self.biases[param_idx].to(self.device)

            energy_tensor = self.activation(energy_tensor.mm(w) + b + pe.squeeze(0)[i])
            if i == 0:
                energy_cache = energy_tensor.unsqueeze(0)  # Add batch dimension
            else:
                energy_cache = torch.cat([energy_cache, energy_tensor.unsqueeze(0)], dim=1)
            ret += [pair]
        
        x = pair[1]
        prev_i = len(start)

        for i in range(max_new_tokens):
            candidates = vocab(vocab.get_neighbor_of_node(x, -1))
            
            # Get parameter indices for all candidates
            param_indices = torch.tensor([self.weight_indices.get((x, t[1]), self.shared_param_idx) 
                                        for t in candidates], device=self.device)
            
            # Get weights and biases for all candidates
            all_w = self.weights[param_indices].to(self.device)
            all_b = self.biases[param_indices].to(self.device)

            curr_i = prev_i + i
            energy_tensor = (energy_cache * self.positions[:, :curr_i, :].softmax(1)).sum(1, keepdim=True)
            expand_energy_tensor = energy_tensor.unsqueeze(1).repeat(1, all_w.size(0), 1, 1)
            expand_energy_tensor = self._reshape12(expand_energy_tensor)

            nxt_energy_tensor = self.activation(expand_energy_tensor.bmm(self._reshape12(all_w)) + self._reshape12(all_b) + pe[:,curr_i].unsqueeze(0))
            output_tensor = nxt_energy_tensor.reshape(1, -1, nxt_energy_tensor.size(-2), nxt_energy_tensor.size(-1))

            energy = output_tensor.norm(2, (-2,-1)).squeeze()

            probs = torch.softmax(energy, dim=-1)
            if temperature > 0:
                probs = probs / temperature
            if do_sample:
                index = torch.multinomial(probs, 1).item()
            else:
                index = probs.argmax(-1).item()

            y = candidates[index][-1]
            ret += [(x, y)]

            energy_tensor = output_tensor[0, index]
            x = y

            energy_cache = torch.cat([energy_cache, energy_tensor.unsqueeze(0)], dim=1)

        return ret


class Vocab:
    def __init__(self, node_dict, nodeindex_dict, edge_dict, edge_decode_dict):
        self.node_dict = node_dict              #{'node_p': index_p}    ----    size: num_nodes
        self.nodeindex_dict = nodeindex_dict    #{index_p: 'node_p'}    ----    size: num_nodes
        self.edge_dict = edge_dict              #{'node_p': {'node_q': (index_p, index_q), 'node_m': (index_p, index_m)},...}    ----    size: num_nodes
        self.edge_decode_dict = edge_decode_dict    #{(index_p, index_q): 'node_p->node_q'}    ----    size: num_nodes*num_nodes

    def __call__(self, x):
        if isinstance(x, list):
            return [self.__call__(_) for _ in x]
        else:
            return self.fetch(x)

    def fetch(self, x):
        s, t = x.split("->")
        return self.edge_dict[s][t] if s in self.edge_dict and t in self.edge_dict[s] else self.edge_dict[""][""]

    @classmethod
    def from_node_dict(cls, dictname):
        node_dict = dict()
        nodeindex_dict = dict()
        edge_dict = dict()
        edge_decode_dict = dict()
        for s in dictname:
            node_dict[s] = dictname[s]
            nodeindex_dict[dictname[s]] = s # nodeindex_dict: {index_p: 'node_p'}
            edge_dict[s] = {} # edge_dict: {'node_p': {'node_q': (index_p, index_q), 'node_m': (index_p, index_m)}}
            for t in dictname:
                edge_dict[s][t] = (dictname[s], dictname[t])
                edge_decode_dict[(dictname[s], dictname[t])] = "->".join([s, t])
        return cls(node_dict, nodeindex_dict, edge_dict, edge_decode_dict)

    @classmethod
    def from_edge(cls, filename):
        edge_dict = dict()
        edge_dict[""] = {}
        edge_dict[""][""] = (0, 0)
        edge_decode_dict = dict()
        with open(filename) as f:
            for line in f:
                # line: node_p->node_q
                s, t = line.strip().split("->")
                if s not in edge_dict:
                    i = len(edge_dict)
                    j = 0
                    edge_dict[s] = dict()
                else:
                    i = edge_dict[s][list(edge_dict[s].keys())[0]][0]
                    j = len(edge_dict[s])
                edge_dict[s][t] = (i, j)
                edge_decode_dict[(i, j)] = "->".join([s, t])
        return cls(None, edge_dict, edge_decode_dict)

    def get_neighbor_of_edge(self, key, k, frequency_dict=None):
        s, t = key.split("->") # s, t: node
        _s = s if s in self.edge_dict else ""
        
        # if s in self.edge_dict:
        #     ret = ["->".join([s, _t]) for _t in self.edge_dict[s].keys() if _t != t]
        # else:
        #     ret = ["->".join([s, _t]) for _t in self.edge_dict[""].keys() if _t != t]
        # ret = ["->".join([s, _t]) for _t in self.edge_dict[s].keys() if _t != t]
        
        # select by word_frequency
        if frequency_dict:
            frequency_lst = list(frequency_dict[_s].keys())
            # index = frequency_lst.index(t)
            # half = k // 2
            # if index <= k:
            #     t_lst = [x for i, x in enumerate(frequency_lst[:k+1]) if i != index]
            # else:
            #     t_lst = frequency_lst[:half] + frequency_lst[index-half:index]
            t_lst = [x for i, x in enumerate(frequency_lst[:k+1]) if x != t][:k]
            ret = ["->".join([_s, _t]) for _t in t_lst]
            random.shuffle(ret)
            return ret
        # randomly select k negative samples
        else:
            ret = ["->".join([_s, _t]) for _t in self.edge_dict[_s].keys() if _t != t]
            random.shuffle(ret)
            return ret[:k] if k != -1 else ret

    def get_neighbor_of_node(self, key, k):
        #key :index
        s = self.nodeindex_dict[key] #node
        #_t: node
        ret = ["->".join([s, _t]) for _t in self.edge_dict[s].keys() if _t != s]
        
        # randomly select k negative samples
        random.shuffle(ret) 
        return ret[:k] if k != -1 else ret
    
    def get_neighbor_of_edge_broadcast(self, key, edges, k=100):
        s, t = key.split("->")
        _ret = [_t for _t in self.edge_dict[s].keys() if _t != t] # all neighbors of s except t
        random.shuffle(_ret)
        ret = []
        for edge in edges:
            s, t = edge.split("->")
            ret += [["->".join([s, _t]) for _t in _ret[:k]]] 
        return ret

    @staticmethod
    def to_path(tokens):
        path = []
        for left, right in zip(tokens[:-1], tokens[1:]):
            path.append("->".join([left, right]))
        return path

    def get_edge_of_node(self, key):
        return list(self.edge_dict[key].values())

    def decode(self, x):
        return self.edge_decode_dict[x]


logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
                    datefmt="%m/%d/%Y %H:%M:%S",
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def stdf(string):
    def _h(char):
        inside_code = ord(char)
        if inside_code == 0x3000:
            inside_code = 0x0020
        else:
            inside_code -= 0xfee0
        if inside_code < 0x0020 or inside_code > 0x7e:
            return char
        return chr(inside_code)

    return "".join([_h(char) for char in string])


class WikiDataset(Dataset):
    """
    Processor for wiki data.
    """
    def __init__(self, filename, vocab, max_seq_length, num_neg_samples, seed, buffer_size=100000, shuffle=True, use_frequency=False, use_bpe=False, bpe_tokenizer=None):
        super().__init__()
        self.vocab = vocab
        self.max_seq_length = max_seq_length
        self.num_neg_samples = num_neg_samples
        self.generator = np.random.default_rng(seed=seed)
        self.use_bpe = use_bpe
        self.bpe_tokenizer = bpe_tokenizer
        
        self.data = self.read(filename)
        
        if use_frequency:
            freq_file = 'word_frequency_en.json' if use_bpe else 'word_frequency.json'
            with open(freq_file, 'r') as f:
                self.frequency_dict = json.load(f)
        else:
            self.frequency_dict = None

    def read(self, filename):
        lines = []
        with open(filename, "r", encoding="utf-8") as f:
            for line in f:
                if self.use_bpe:
                    lines.append(line.strip())
                else:
                    src = list(line.strip()[:self.max_seq_length])
                    lines.append(src)
        return lines

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        src = self.data[idx]
        return self.vectorize(src)

    def vectorize(self, src):
        if self.use_bpe:
            # For English with BPE
            bpe_tokens = self.bpe_tokenizer.encode(src).tokens
            # Truncate/pad
            pad_token = "[PAD]"
            if len(bpe_tokens) > self.max_seq_length:
                bpe_tokens = bpe_tokens[:self.max_seq_length]
            else:
                bpe_tokens.extend(pad_token for _ in range(self.max_seq_length - len(bpe_tokens)))
            tokens = bpe_tokens
        else:
            # For Chinese without BPE
            if len(src) > self.max_seq_length:
                src = src[:self.max_seq_length]
            else:
                src.extend("" for _ in range(self.max_seq_length-len(src)))
            tokens = src
            
        edges = self.vocab.to_path(tokens)
        edge_ids = self.vocab(edges)
        edge_ids = edge_ids[:self.max_seq_length]
        neighbor_ids = [self.vocab(self.vocab.get_neighbor_of_edge(e, self.num_neg_samples, self.frequency_dict)) for e in edges]

        new_neighbor_ids = []
        for i, e_ids in enumerate(edge_ids):
            new_neighbor_ids.append([e_ids] + neighbor_ids[i])
        return torch.LongTensor(new_neighbor_ids)


def main():
    parser = argparse.ArgumentParser()

    # Data config
    parser.add_argument("--data_dir", type=str, default="data/wiki",
                        help="Directory to contain the input data for all tasks.")
    parser.add_argument("--output_dir", type=str, default="model/",
                        help="Directory to output predictions and checkpoints.")
    parser.add_argument("--load_state_dict", type=str, default=None,
                        help="Trained model weights to load for evaluation if needed.")

    # Training config
    parser.add_argument("--do_train", action="store_true",
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true",
                        help="Whether to evaluate on the dev set.")
    parser.add_argument("--num_neg_samples", type=int, default=100,
                        help="Number of negative samples.")
    parser.add_argument("--max_seq_length", type=int, default=128,
                        help="Maximum total input sequence length after word-piece tokenization.")
    parser.add_argument("--train_batch_size", type=int, default=128,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size", type=int, default=128,
                        help="Total batch size for evaluation.")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                        help="Initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", type=float, default=3.0,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_train_steps", type=int, default=None,
                        help="Total number of training steps to perform. If provided, overrides training epochs.")
    parser.add_argument("--weight_decay", type=float, default=0.,
                        help="L2 weight decay for training.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward pass.")
    parser.add_argument("--no_cuda", action="store_true",
                        help="Whether not to use CUDA when available.")
    parser.add_argument("--fp16", action="store_true",
                        help="Whether to use mixed precision.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for initialization.")
    parser.add_argument("--save_steps", type=int, default=500,
                        help="How many steps to save the checkpoint once.")
    parser.add_argument("--hidden_size", type=int, default=32,
                        help="Mask rate for masked-fine-tuning.")
    parser.add_argument("--local_rank", type=int)
    parser.add_argument("--initial_file_number", type=int, default=0,
                        help="From which file to begin training.")
    parser.add_argument("--end_file_number", type=int, default=0,
                        help="End file number for training.")
    parser.add_argument("--wiki_sorted_size", type=int, default=70,
                        help="Total file numbers for sorted wikidata.")
    parser.add_argument("--run_name", type=str, default="plusb_pluspe_order",
                        help="Run name for wandb.")

    parser.add_argument("--use_frequency", action="store_true",
                        help="Whether to use word frequency.")
    parser.add_argument("--train_full", type=str, default=None,
                        help="Path to train on full text.")
    parser.add_argument("--checkpoint_save_step", type=int, default=0,
                        help="Interval to save checkpoint.(Only support when train_full is True)")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help="Path to checkpoint to resume training from")
    parser.add_argument("--num_workers", type=int, default=8,
                        help="Number of workers for data loading.")
    parser.add_argument("--vocab_path", type=str, default="vocab_wiki_4k.json",
                        help="Path to vocab file.")
    parser.add_argument("--use_ds", action="store_true",
                        help="Whether to use deepspeed.")
    parser.add_argument("--sparse", action="store_true",
                        help="Whether to use sparse.")
    parser.add_argument("--use_bpe", action="store_true",
                        help="Whether to use BPE tokenizer for English.")
    parser.add_argument("--bpe_tokenizer_path", type=str, default="wiki_bpe_tokenizer_4000_bytelevel.json",
                        help="Path to BPE tokenizer file.")

    args = parser.parse_args()
    
    
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    n_gpu = torch.cuda.device_count()
    logger.info("device: {}, n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, "-accelerate", args.fp16))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    # if not os.path.exists(args.output_dir):
    #     os.makedirs(args.output_dir)

    with open(args.vocab_path) as f:
        node_dict = json.load(f)
    vocab = Vocab.from_node_dict(node_dict)

    if args.sparse:
        with open('word_frequency.json', 'r') as f:
            freq_dict = json.load(f)

        zero_freq_edges = {}
        for s in freq_dict:
            zero_freq_edges[s] = []
            for t in freq_dict[s]:
                if freq_dict[s][t] == 0:
                    zero_freq_edges[s].append(t)
    else:
        zero_freq_edges = None
    

    def stat_cuda(epoch, cur_file_num, step, location):
        if accelerator.is_local_main_process:
            with open("cuda_stat.txt", "a") as f:
                if epoch is not None:
                    f.write('epoch: %d, cur_file_num: %d, step: %d\n' % (epoch, cur_file_num, step))
                f.write(f'--{location}\n')
                f.write('allocated: %dG, max allocated: %dG, cached: %dG, max cached: %dG\n' % (
                    torch.cuda.memory_allocated() / 1024 / 1024 / 1024,
                    torch.cuda.max_memory_allocated() / 1024 / 1024 / 1024,
                    torch.cuda.memory_reserved() / 1024 / 1024 / 1024,
                    torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024
            ))

    if args.do_train:
        # training arguments
        os.environ["NCCL_DEBUG"] = "WARN"
        os.environ["TORCH_NCCL_BLOCKING_WAIT"] = "1"
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        init_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=1080000))
        accelerator = Accelerator(kwargs_handlers=[ddp_kwargs, init_kwargs], cpu=args.no_cuda, mixed_precision="fp16" if args.fp16 else "no")
        device = accelerator.device

        # prepare model
        model = BraLM(args.hidden_size, args.use_ds, zero_freq_edges, vocab=vocab)
        model.prepare_network(vocab)
        # model.shared_weight.requires_grad = False
        # model.shared_bias.requires_grad = False

        # load model from checkpoint
        if args.load_state_dict:
            print(f"Loading model from checkpoint: {args.load_state_dict}")
            checkpoint = torch.load(args.load_state_dict, map_location="cpu")
            #model.load_state_dict(checkpoint["model_state_dict"])
            model.load_old(checkpoint["model_state_dict"])


        # Load checkpoint if specified
        wandb_id = None
        global_step = 0
        if args.resume_from_checkpoint:
            print(f"Resuming from checkpoint: {args.resume_from_checkpoint}")
            checkpoint = torch.load(args.resume_from_checkpoint, map_location="cpu")
            model.load_state_dict(checkpoint["model_state_dict"])
            #optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            start_epoch = checkpoint["epoch"] # + 1
            global_step = checkpoint.get("global_step", 0)  # Get saved global step
            wandb_id = checkpoint.get("wandb_id")
        else:
            start_epoch = 0

        # if accelerator.is_local_main_process:   
        #     for name, param in model.named_parameters():
        #         print(name)

        model.to_device(device)

        
        if accelerator.is_local_main_process:   
            print(f"start_epoch: {start_epoch}, global_step: {global_step}")
        
        # prepare optimizer
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": args.weight_decay
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0
            }
        ]
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

        if args.resume_from_checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        if accelerator.is_local_main_process:
            print(f"before prepare")
        #input('-' * 10)
        #stat_cuda(None, None, None, "before prepare")
        #print(f"{accelerator.device}, model: {model.weights.device}, tensor: {model.tensor.device}, pe: {model.positions.device}")
        if not args.use_ds:
            model, optimizer = accelerator.prepare(model, optimizer) # for deepspeed: # this line
        #stat_cuda(None, None, None, "after prepare")
        #print(f"{accelerator.device}, model: {model.module.weights.device}, tensor: {model.module.tensor.device}, pe: {model.module.positions.device}")
        if accelerator.is_local_main_process:
            print(f"after prepare")

    if args.do_train:

        if accelerator.is_local_main_process:
            # init wandb
            wandb.init(
                project="brain",
                name=args.run_name,
                id=wandb_id,  # 如果有之前的run_id，使用它；否则会创建新的
                resume="allow",  # "allow"表示如果有id就恢复，没有就创建新的
                config=vars(args)
            )
            wandb.define_metric("custom_step")
            wandb.define_metric("batch_*", step_metric="custom_step")
            wandb.define_metric("epoch")
            wandb.define_metric("epoch_*", step_metric="epoch")
            print(f"Started wandb run with id: {wandb.run.id}")
            print(f"View run at: {wandb.run.get_url()}")
        
        if args.train_full:
            cur_file_num = args.train_full
            cur_filename = f"{cur_file_num}.txt"
            if args.use_bpe:
                with open(args.bpe_tokenizer_path, 'r') as f:
                    bpe_tokenizer = json.load(f)
            else:
                bpe_tokenizer = None
            dataset = WikiDataset(
                os.path.join(args.data_dir, cur_filename), 
                vocab, 
                args.max_seq_length, 
                args.num_neg_samples, 
                seed=args.seed, 
                shuffle=True, 
                use_frequency=args.use_frequency,
                use_bpe=args.use_bpe,
                bpe_tokenizer=bpe_tokenizer
            )
            train_dataloader = DataLoader(dataset, batch_size=args.train_batch_size, num_workers=args.num_workers, pin_memory=True)
            train_dataloader = accelerator.prepare(train_dataloader)
        elif args.resume_from_checkpoint:
            cur_file_num = checkpoint["cur_file_num"]
            if isinstance(cur_file_num, int) or cur_file_num.isdigit():
                cur_file_num = int(cur_file_num) + 1
                #start_epoch = start_epoch - 1
        else:
            cur_file_num = args.initial_file_number

        if args.resume_from_checkpoint and global_step > 0:
            if args.train_full and global_step % len(train_dataloader) == 0:
                start_epoch = start_epoch + 1
            if not args.train_full and cur_file_num > args.end_file_number:
                start_epoch = start_epoch + 1
                cur_file_num = args.initial_file_number
        

        for epoch in trange(start_epoch, int(args.num_train_epochs), desc="Epoch"):
            # traverse all wiki files
            if epoch != start_epoch or args.train_full:
                cur_file_num = args.initial_file_number
            while cur_file_num <= args.wiki_sorted_size:
                if args.train_full:
                    cur_file_num = args.train_full
                logger.info("***** Running training for wiki = %s *****", cur_file_num)
                logger.info("  Batch size = %d", args.train_batch_size * accelerator.num_processes)
                
                # prepare data
                if not args.train_full:
                    cur_filename = f"{cur_file_num}.txt"
                    if args.use_bpe:
                        with open(args.bpe_tokenizer_path, 'r') as f:
                            bpe_tokenizer = json.load(f)
                    else:
                        bpe_tokenizer = None
                    dataset = WikiDataset(
                        os.path.join(args.data_dir, cur_filename), 
                        vocab, 
                        args.max_seq_length, 
                        args.num_neg_samples, 
                        seed=args.seed, 
                        shuffle=True, 
                        use_frequency=args.use_frequency,
                        use_bpe=args.use_bpe,
                        bpe_tokenizer=bpe_tokenizer
                    )
                    train_dataloader = DataLoader(dataset, batch_size=args.train_batch_size, num_workers=args.num_workers, pin_memory=True)
                    if not args.use_ds:
                        train_dataloader = accelerator.prepare(train_dataloader)
                    else:
                        model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, train_dataloader) # for deepspeed


                # training
                train_loss = 0
                num_train_examples = 0
                if accelerator.is_local_main_process:
                    progress_bar = tqdm(train_dataloader, desc="Iteration")
                    # start_time = time.time()
                
                #for _ in range(3):
                for step, batch in enumerate(train_dataloader, start=global_step % len(train_dataloader)):
                    # batch: (bs, sen_len, 1+k, 2)
                    batch_train_loss = 0
                    batch_num_train_examples = 0
                    #for ind in range(2, batch.size(1)):
                    for ind in range(batch.size(1) - 1, batch.size(1)): # fix: only use the sen_len-1
                        # ind: 2, 3, ..., sen_len-1
                        # if accelerator.is_local_main_process:
                        #     end_time = time.time()
                        #     step_time = end_time - start_time
                        #     logger.info(f"Step training time: {step_time:.2f} seconds")

                        model.train()
                        neighbor_ids = batch[:, :ind]   #(bs, ind, 1+k, 2)

                        #stat_cuda(epoch, cur_file_num, global_step, "before forward")   
                        outputs = model(neighbor_ids)
                        loss = outputs

                        # if n_gpu > 1:
                        #     loss = loss.mean()

                        if args.gradient_accumulation_steps > 1:
                            loss = loss / args.gradient_accumulation_steps
                        accelerator.backward(loss)

                        if n_gpu > 1:
                            dist.all_reduce(loss)
                            loss = loss / dist.get_world_size()

                        train_loss += loss.detach().item()
                        batch_train_loss += loss.detach().item()

                        num_train_examples += 1
                        batch_num_train_examples += 1
                        
                        del outputs
                        del loss
                        del neighbor_ids
                        gc.collect()
                        # if step % 5 == 0:
                        #    torch.cuda.empty_cache()

                        if (step + 1) % args.gradient_accumulation_steps == 0:
                            optimizer.step()
                            optimizer.zero_grad()

                            ## modified

                            ppl = math.exp(batch_train_loss / batch_num_train_examples)
                        
                            if accelerator.is_local_main_process:
                                progress_bar.update(1)
                                progress_bar.set_postfix(loss=batch_train_loss / batch_num_train_examples, perplexity=ppl)
                                
                                wandb.log({
                                    "batch_loss": batch_train_loss / batch_num_train_examples, 
                                    "batch_perplexity": math.exp(batch_train_loss / batch_num_train_examples),
                                    "batch_epoch": epoch,
                                    #"step": global_step,
                                    "custom_step": global_step
                                })#, step=global_step)

                        global_step += 1

                        # Save checkpoint every checkpoint_save_step steps at the end of each step
                        if accelerator.is_local_main_process and args.checkpoint_save_step > 0 and global_step % args.checkpoint_save_step == 0:
                            output_dir_f = f"{args.output_dir}/HS{args.hidden_size}/step_{global_step}/"
                            if not os.path.exists(output_dir_f):
                                os.makedirs(output_dir_f)
                            output_model_file = os.path.join(output_dir_f, f"checkpoint_{global_step}.bin")
                            
                            model_to_save = model.module if hasattr(model, "module") else model
                            checkpoint = {
                                "model_state_dict": model_to_save.state_dict(),
                                "optimizer_state_dict": optimizer.state_dict(), 
                                "epoch": epoch,
                                "global_step": global_step,
                                "args": vars(args),
                                "wandb_id": wandb.run.id
                            }
                            if not args.train_full:
                                checkpoint["cur_file_num"] = cur_file_num
                            print(f"Saving checkpoint to {output_model_file}")
                            torch.save(checkpoint, output_model_file)
                            print(f"Checkpoint saved to {output_model_file}")


                # save model for current training file
                if accelerator.is_local_main_process:
                    epoch_avg_loss = train_loss / num_train_examples
                    epoch_ppl = math.exp(epoch_avg_loss)
                    wandb.log({
                        "epoch_loss": epoch_avg_loss,
                        "epoch_perplexity": epoch_ppl,
                        "epoch": epoch,
                    })#, step=global_step)
                    
                    model_to_save = model.module if hasattr(model, "module") else model
                    output_dir_f = f"{args.output_dir}/HS{args.hidden_size}/EPOCH{epoch}/"
                    if not os.path.exists(output_dir_f):
                        os.makedirs(output_dir_f)
                    output_model_file = os.path.join(output_dir_f, "f{}_pytorch_model.bin".format(cur_file_num))
                    # only save the last model
                    if args.train_full or cur_file_num == args.end_file_number:
                        #torch.save(model_to_save.state_dict(), output_model_file)
                        checkpoint = {
                            "model_state_dict": model_to_save.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "epoch": epoch,
                            "global_step": global_step,  # Save global step
                            "args": vars(args),
                            "wandb_id": wandb.run.id  # 保存当前运行的wandb_id
                        }
                        if not args.train_full:
                            checkpoint["cur_file_num"] = cur_file_num
                        print(f"Saving model to {output_model_file}")
                        torch.save(checkpoint, output_model_file)
                        print(f"Model saved to {output_model_file}")

                if args.train_full:
                    break
                cur_file_num += 1
                if cur_file_num > args.end_file_number:
                    break



if __name__ == "__main__":
    main()
