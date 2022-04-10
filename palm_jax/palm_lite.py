from math import log2, floor
from typing import List

import numpy as onp
from jax import random, nn, lax, numpy as np
from jax.numpy import einsum

from equinox import Module, static_field
from einops import rearrange

# rmsnorm

class RMSNorm(Module):
    gamma: np.ndarray
    eps: float = static_field()

    def __init__(self, dim, eps = 1e-5):
        self.gamma = np.ones((dim,))
        self.eps = eps

    def __call__(self, x):
        mean_of_squares = np.mean(np.square(x), axis = -1, keepdims = True)
        inv_norm = lax.rsqrt(mean_of_squares + self.eps)
        return inv_norm * x * self.gamma

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
    mask_value: float = static_field()

    def __init__(
        self,
        dim,
        dim_head,
        heads,
        key,
        mask_value = 1e-10
    ):
        inner_dim = dim_head * heads
        self.norm = RMSNorm(dim)

        self.wq = random.normal(key, (dim, inner_dim))
        self.wk = random.normal(key, (dim, dim_head))
        self.wv = random.normal(key, (dim, dim_head))
        self.wo = random.normal(key, (inner_dim, dim))

        self.heads = heads
        self.scale = dim_head ** -0.5
        self.mask_value = mask_value

    def __call__(self, x, pos_bias, mask):
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

        sim = sim + pos_bias

        # causal mask

        sim = np.where(mask, sim, self.mask_value)

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
    alibi_bias: onp.ndarray
    causal_mask: onp.ndarray

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
        max_seq_len = 2048
    ):
        self.embedding = random.normal(key, (num_tokens, dim)) * 0.02        
        self.alibi_bias = calc_alibi_bias(max_seq_len, heads = heads)
        self.causal_mask = onp.tril(onp.ones((max_seq_len, max_seq_len)))

        self.layers = []
        for _ in range(depth):
            attn = Attention(dim = dim, dim_head = dim_head, heads = heads, key = key)
            ff = FeedForward(dim = dim, mult = ff_mult, key = key)
            self.layers.append([attn, ff])

        self.norm = RMSNorm(dim)

    def __call__(self, x):
        n = x.shape[-1]
        x = self.embedding[x]

        pos_bias = self.alibi_bias[..., :n]
        causal_mask = self.causal_mask[:n, :n]

        for attn, ff in self.layers:
            x = attn(x, pos_bias = pos_bias, mask = causal_mask) + x
            x = ff(x) + x

        x = self.norm(x)
        return x @ self.embedding.transpose()
