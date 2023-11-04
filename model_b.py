#!/usr/bin/env bash
# Copyright 2023  Bofeng Huang

"""Reimplement model.py from Andrej Karpathy's nanoGPT repository from scratch."""

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class GPTConfig:
    n_layers: int = 12
    n_heads: int = 12
    emb_dim: int = 768
    dropout: float = 0.0
    max_position_embeddings: int = 1024
    vocab_size: int = 50304  # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    bias: bool = True  # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster


class LayerNorm(nn.Module):
    """LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False."""

    def __init__(self, ndim: int, bias: bool):
        super().__init__()

        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x: torch.Tensor):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()

        assert config.emb_dim % config.n_heads == 0

        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.emb_dim, config.emb_dim * 3, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.emb_dim, config.emb_dim, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.residual_dropout = nn.Dropout(config.dropout)

        self.head_dim = config.emb_dim // config.n_heads
        self.emb_dim = config.emb_dim
        self.dropout = config.dropout

        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer(
                "tril",
                torch.tril(
                    torch.ones(config.max_position_embeddings, config.max_position_embeddings).view(
                        1, 1, config.max_position_embeddings, config.max_position_embeddings
                    )
                ),
            )

    def forward(self, x: torch.Tensor):
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # specify chunk_size
        query_states, key_states, value_states = self.c_attn(x).split(self.emb_dim, dim=2)

        # B x n_heads x T x head_dim
        query_states = query_states.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # causal self-attention; Self-attend
        # B x n_heads x T x T
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            attn_output = torch.nn.functional.scaled_dot_product_attention(
                query_states,
                key_states,
                value_states,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0,
                is_causal=True,
            )
        else:
            # manual implementation of attention
            attn_weights = (query_states @ key_states.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
            attn_weights = attn_weights.masked_fill(self.tril[..., :T, :T] == 0, float("-inf"))
            attn_weights = F.softmax(attn_weights, dim=-1)
            attn_weights = self.attn_dropout(attn_weights)

            # B x n_heads x T x head_dim
            attn_output = attn_weights @ value_states

        # re-assemble all head outputs side by side
        # B x T x C
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, C)

        # output projection
        # B x T x C
        attn_output = self.c_proj(attn_output)
        attn_output = self.residual_dropout(attn_output)

        return attn_output


class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()

        self.c_fc = nn.Linear(config.emb_dim, config.emb_dim * 4, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(config.emb_dim * 4, config.emb_dim, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()

        self.ln_1 = LayerNorm(config.emb_dim, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.emb_dim, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor):
        x = self.attn(self.ln_1(x)) + x
        x = self.mlp(self.ln_2(x)) + x
        return x


class GPTLanguageModel(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()

        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.emb_dim),
                wpe=nn.Embedding(config.max_position_embeddings, config.emb_dim),
                dropout=nn.Dropout(config.dropout),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layers)]),
                ln_f=LayerNorm(config.emb_dim, bias=config.bias),
            )
        )

        # bias always False
        self.lm_head = nn.Linear(config.emb_dim, config.vocab_size, bias=False)
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        self.transformer.wte.weight = self.lm_head.weight  # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layers))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params() / 1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor, labels: Optional[torch.Tensor]):
        B, T = input_ids.size()
        assert (
            T <= self.config.max_position_embeddings
        ), f"Cannot forward sequence of length {T}, block size is only {self.config.max_position_embeddings}"

        # forward the GPT model itself
        # B x T x C
        token_embedding = self.transformer.wte(input_ids)
        # T x C
        position_embedding = self.transformer.wpe(torch.arange(T, dtype=torch.long, device=input_ids.device))
        # B x T x C
        x = self.transformer.dropout(token_embedding + position_embedding)

        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)

        if labels is not None:
            # if we are given some desired targets also calculate the loss
            # B x T x vocab_size
            logits = self.lm_head(x)
            loss_fct = nn.modules.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :])  # note: using list [-1] to preserve the time dim
            loss = None

        return dict(logits=logits, loss=loss)

    def crop_max_position_embeddings(self, max_position_embeddings):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert max_position_embeddings <= self.config.max_position_embeddings
        self.config.max_position_embeddings = max_position_embeddings
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:max_position_embeddings])
        for block in self.transformer.h:
            if hasattr(block.attn, "tril"):
                block.attn.tril = block.attn.tril[:, :, :max_position_embeddings, :max_position_embeddings]

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at max_position_embeddings
            input_ids_cond = (
                input_ids
                if input_ids.size(1) <= self.config.max_position_embeddings
                else input_ids[:, -self.config.max_position_embeddings :]
            )
            # forward the model to get the logits for the index in the sequence
            # B x T x vocab_size
            logits = self(input_ids_cond)["logits"]
            # pluck the logits at the final step and scale by desired temperature
            # B x vocab_size
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")
            # apply softmax to convert logits to (normalized) probabilities
            # B x vocab_size
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            # B
            next_token_ids = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            # B x (T + 1)
            input_ids = torch.cat([input_ids, next_token_ids], dim=1)

        return input_ids
