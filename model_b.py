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
    dropout_rate: float = 0.0
    max_position_embeddings: int = 1024
    vocab_size: int = 50304  # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    bias: bool = True  # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster


class LayerNorm(nn.Module):
    def __init__(self, ndim: int, bias: bool):
        super().__init__()

        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x: torch.Tensor):
        return F.layer_norm(x, x.shape[-1], self.weight, self.bias, 1e-6)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()

        self.head_dim = config.emb_dim // config.n_heads

        self.attn = nn.Linear(config.emb_dim, config.emb_dim * 3, bias=config.bias)
        self.o_proj = nn.Linear(config.emb_dim, config.emb_dim, bias=config.bias)
        self.dropout1 = nn.Dropout(config.dropout_rate)
        self.dropout2 = nn.Dropout(config.dropout_rate)

        self.register_buffer(
            "tril",
            torch.tril(
                torch.ones(config.max_position_embeddings, config.max_position_embeddings).view(
                    1, 1, config.max_position_embeddings, config.max_position_embeddings
                )
            ),
        )

    def forward(self, x: torch.Tensor):
        B, T, C = x.shape

        key_states, query_states, value_states = self.attn(x).split(3, dim=2)

        # B x n_heads x T x head_dim
        key_states = key_states.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        query_states = query_states.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # B x n_heads x T x T
        attn_weights = query_states @ key_states.transpose(-1, -2) / math.sqrt(self.head_dim)
        attn_weights = attn_weights.masked_fill(self.tril[..., :T, :T] == 0, float("-inf"))
        attn_scores = F.softmax(attn_weights, dim=-1)
        attn_scores = self.dropout1(attn_scores)

        # B x n_heads x T x head_dim
        attn_output = attn_scores @ value_states
        # B x T x C
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, C)

        # B x T x C
        attn_output = self.o_proj(attn_output)
        attn_output = self.dropout2(attn_output)

        return attn_output


class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(config.emb_dim, config.emb_dim * 4, bias=config.bias),
            nn.ReLU(),
            nn.Linear(config.emb_dim * 4, config.emb_dim, bias=config.bias),
            nn.Dropout(config.dropout_rate),
        )

    def forward(self, x: torch.Tensor):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()

        self.causal_self_attention = CausalSelfAttention(config)
        self.mlp = MLP(config)
        self.ln1 = LayerNorm(config.emb_dim, config.bias)
        self.ln2 = LayerNorm(config.emb_dim, config.bias)

    def forward(self, x: torch.Tensor):
        x = self.causal_self_attention(self.ln1(x)) + x
        x = self.mlp(self.ln2(x)) + x
        return x


class GPTLanguageModel(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()

        self.vocab_size = config.vocab_size

        self.embedding_layer = nn.Embedding(config.vocab_size, config.emb_dim)
        self.position_layer = nn.Embedding(config.max_position_embeddings, config.emb_dim)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layers)])
        self.final_ln = LayerNorm(config.emb_dim, config.bias)
        self.lm_head = nn.Linear(config.emb_dim, config.vocab_size, bias=config.bias)

    def forward(self, input_ids: torch.Tensor, labels: Optional[torch.Tensor]):
        # B x T x C
        embedding = self.embedding_layer(input_ids)
        position_embedding = self.position_layer(input_ids)
        x = embedding + position_embedding

        for block in self.blocks:
            x = block(x)

        x = self.final_ln(x)
        # B x T x vocab_size
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            loss_fct = nn.modules.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.vocab_size), labels.view(-1))

        return dict(logits=logits, loss=loss)

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int):
        for _ in range(max_new_tokens):
            # B x T x vocab_size
            logits = self(input_ids)
            scores = F.softmax(logits, dim=-1)
            # B x T
            scores = scores[..., -1]
            next_token_ids = torch.multinomial(scores)
            # B x (T + 1)
            input_ids = torch.cat([input_ids, next_token_ids], dim=-1)

        return input_ids
