from math import log2, floor
from typing import List

import numpy as onp
from jax import random, jit, nn, lax, numpy as np
from jax.numpy import einsum

from equinox import Module, static_field
from einops import rearrange, repeat

# rmsnorm

class RMSNorm(Module):
    gamma: np.ndarray
    scale: float = static_field()
    eps: float = static_field()

    def __init__(self, dim, eps = 1e-5):
        self.gamma = np.ones((dim,))
        self.eps = eps
        self.scale = dim ** 0.5

    def __call__(self, x):
        sum_of_squares = np.sum(np.square(x), axis = -1, keepdims = True)
        inv_norm = lax.rsqrt(sum_of_squares + self.eps)
        return inv_norm * x * self.gamma * self.scale

# AliBi

def get_alibi_slopes(heads):
    def get_slopes_power_of_2(n):
        start = (2 ** (-2 ** -(log2(n) - 3)))
        ratio = start
        return [start*ratio**i for i in range(n)]

    if log2(heads).is_integer():
        return get_slopes_power_of_2(heads)

    closest_power_of_2 = 2 ** floor(log2(heads))
    return get_slopes_power_of_2(closest_power_of_2) + get_slopes_power_of_2(2 * closest_power_of_2)[0::2][:heads-closest_power_of_2]

def calc_alibi_bias(seq_len, heads):
    slopes = get_alibi_slopes(heads)
    slopes = rearrange(onp.array(slopes), 'h -> h 1 1')
    bias = rearrange(onp.arange(seq_len), 'j -> 1 1 j')
    return slopes * bias

# feedforward
# SwiGLU variant

def swish(x):
    return x * nn.sigmoid(x)

class FeedForward(Module):
    norm: Module
    wi: np.ndarray
    wg: np.ndarray
    wo: np.ndarray

    def __init__(self, dim, key, mult = 4):
        inner_dim = int(mult * dim)
        self.norm = RMSNorm(dim = dim)

        self.wi = random.normal(key, (dim, inner_dim))
        self.wg = random.normal(key, (dim, inner_dim))
        self.wo = random.normal(key, (inner_dim, dim))

    def __call__(self, x):
        x = self.norm(x)
        x, gate = (x @ self.wi), (x @ self.wg)
        x *= swish(gate)
        return x @ self.wo

# attention
# multi-query, one-headed key / values variant

class Attention(Module):
    norm: Module
    wq: np.ndarray
    wk: np.ndarray
    wv: np.ndarray
    wo: np.ndarray

    heads: int = static_field()
    scale: float = static_field()    

    def __init__(
        self,
        dim,
        dim_head,
        heads,
        key
    ):
        inner_dim = dim_head * heads
        self.norm = RMSNorm(dim)

        self.wq = random.normal(key, (dim, inner_dim))
        self.wk = random.normal(key, (dim, dim_head))
        self.wv = random.normal(key, (dim, dim_head))
        self.wo = random.normal(key, (inner_dim, dim))

        self.heads = heads
        self.scale = dim_head ** -0.5

    def __call__(self, x, attn_bias):
        n = x.shape[-2]

        x = self.norm(x)

        q = x @ self.wq
        k = x @ self.wk
        v = x @ self.wv

        # split out heads

        q = rearrange(q, '... n (h d) -> ... h n d', h = self.heads)

        # scale

        q *= self.scale

        # sim

        sim = einsum('... h i d, ... j d -> ... h i j', q, k)

        # positional bias

        sim = sim + attn_bias

        # attention

        attn = nn.softmax(sim, axis = -1)

        # aggregate values

        out = einsum('... h i j, ... j d -> ... h i d', attn, v)

        # merge heads

        out = rearrange(out, '... h n d -> ... n (h d)')

        # combine heads out

        return out @ self.wo

# main class

class PaLM(Module):
    embedding: np.ndarray
    norm: Module
    layers: List[List[Module]]
    attn_bias: onp.ndarray

    def __init__(
        self,
        *,
        num_tokens,
        dim,
        dim_head,
        depth,
        heads,
        key,
        ff_mult = 4,
        max_seq_len = 2048,
        mask_value = -1e10
    ):
        self.embedding = random.normal(key, (num_tokens, dim)) * 0.02        

        causal_mask = onp.tril(onp.ones((max_seq_len, max_seq_len)))
        alibi_bias = calc_alibi_bias(max_seq_len, heads = heads)
        self.attn_bias = np.where(causal_mask, repeat(alibi_bias, 'h 1 j -> h i j', i = max_seq_len), mask_value)

        self.layers = []
        for _ in range(depth):
            attn = Attention(dim = dim, dim_head = dim_head, heads = heads, key = key)
            ff = FeedForward(dim = dim, mult = ff_mult, key = key)
            self.layers.append([attn, ff])

        self.norm = RMSNorm(dim)

    @jit
    def __call__(self, x):
        n = x.shape[-1]
        x = self.embedding[x]

        attn_bias = self.attn_bias[..., :n, :n]

        for attn, ff in self.layers:
            x = attn(x, attn_bias = attn_bias) + x
            x = ff(x) + x

        x = self.norm(x)
        return x @ self.embedding.transpose()
