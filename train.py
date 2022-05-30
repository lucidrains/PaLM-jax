import os
from random import randrange
from functools import partial
import tqdm
import gzip
import numpy as np

import jax
import jax.numpy as jnp
from jax import nn

import equinox as eqx
from optax import adam, clip_by_global_norm, chain, apply_every

from palm_jax.palm_lite import PaLM
from palm_jax.utils import sample

# env

os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

# constants

NUM_BATCHES = int(1e5)
BATCH_SIZE = 4
GRADIENT_ACCUMULATE_EVERY = 4
LEARNING_RATE = 2e-4
MAX_GRAD_NORM = 0.5
VALIDATE_EVERY  = 100
SAMPLE_EVERY  = 500
SEQ_LEN = 1024

# helpers

def cycle(loader):
    while True:
        for data in loader:
            yield data

def decode_token(token):
    return str(chr(max(32, token)))

def decode_tokens(tokens):
    return ''.join(list(map(decode_token, tokens)))

# enwik8 data and data functions

with gzip.open('./data/enwik8.gz') as file:
    X = np.fromstring(file.read(int(95e6)), dtype=np.uint8)
    data_train, data_val = np.split(X, [int(90e6)])

def sample_seq_from_data(data, *, seq_len, batch_size):
    total_seq_len = data.shape[0]
    base_arange = np.arange(seq_len)
    start_indices = np.random.randint(0, total_seq_len - seq_len, (batch_size,))
    token_indices = start_indices[:, None] + base_arange
    return data[token_indices]

sample_seq_fn = partial(sample_seq_from_data, seq_len = SEQ_LEN, batch_size = BATCH_SIZE)

# setup model and params

key = jax.random.PRNGKey(0)

model = PaLM(
    num_tokens = 256,
    dim = 512,
    depth = 8,
    heads = 8,
    dim_head = 64,
    key = key
)

# cross entropy loss

def cross_entropy(logits, targets, axis = -1):
    logprobs = nn.log_softmax(logits, axis = axis)
    nll = jnp.take_along_axis(logprobs, jnp.expand_dims(targets, axis = axis), axis = axis)
    cross_entropy = -jnp.mean(nll)
    return cross_entropy

@eqx.filter_value_and_grad
def loss_fn(model, data):
    inp, labels = data[:, :-1], data[:, 1:]
    logits = model(inp)
    return cross_entropy(logits, labels, axis = -1)

# optimizer

optim = chain(
    clip_by_global_norm(MAX_GRAD_NORM),
    adam(LEARNING_RATE),
    apply_every(GRADIENT_ACCUMULATE_EVERY)
)

optim_state = optim.init(model)

# train step

@eqx.filter_jit(kwargs=dict(data=True))
def train_step(model, data, optim_state):
    loss, grads = loss_fn(model, data)
    updates, optim_state = optim.update(grads, optim_state)
    model = eqx.apply_updates(model, updates)
    return model, optim_state, loss

# training

for i in tqdm.tqdm(range(NUM_BATCHES), mininterval=10., desc='training'):
    for _ in range(GRADIENT_ACCUMULATE_EVERY):
        data = sample_seq_fn(data_train)
        model, optim_state, loss = train_step(model, data, optim_state)

    print(f'loss: {loss.item()}')

    if i % SAMPLE_EVERY == 0:
        valid_data = sample_seq_fn(data_val)
        prime = valid_data[0][:100]
        prime_str = decode_tokens(prime)
        print(prime_str, "\n", "*" * 40)

        sampled = sample(key, model, prime, SEQ_LEN, top_k = 25)
        sampled_str = decode_tokens(sampled[100:])
        print(sampled_str)
