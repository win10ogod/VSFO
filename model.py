import torch
import torch.nn as nn
import random
from torch.autograd import Variable

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
        self.node_bias.to(device)
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
    
    def get_initial_tensor(self, batch_size, d, pe):
        # initialize energy_tensor
        energy_tensor = torch.ones(batch_size, 1, self.hidden_size) / self.hidden_size   #(bs, 1, hs)
        energy_tensor = energy_tensor.to(self.device)
        
        # Ensure d is on the same device as node_bias
        d = d.to(self.device)
        node_bias = self.node_bias[d[:, 0, 0]]
        energy_tensor = self.activation(energy_tensor + node_bias + pe[:,0])
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
    

    