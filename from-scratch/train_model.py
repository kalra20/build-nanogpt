from dataclasses import dataclass
# import torch
# import torch.nn as nn
# from torch.nn import funcitonal as F

class CausalSelfAttention(nn.Module):
    def __init__(self):
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # not really a 'bias', more of a mask
        self.register_buffer("bias", torch.tril(torch.ones(
            config.block_size, config.block_size
        )).view(1,1, config.block_size, config.block_size)
        )




class Block(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)
    
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


@dataclass
class GPTConfig:
    block_size: int = 256
    vocab_size: int = 65
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384

class GPT(nn.Module):

    def __init__(self, config=GPTConfig()):
        super().__init__():
        self.config = config

        self.transformer = nn.ModuleDict( dict(
            #weights token embeddings and position embeddigns
            wte = nn.Embedding(config.vocab_size, config.n_embd), 
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block[config] for _ in range(config.n_layer)])
            ln_f = nn.LayerNorm(config.n_embd)

        ))

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False) #classifier

