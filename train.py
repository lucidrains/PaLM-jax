from random import randrange
import tqdm
import gzip
import numpy as np

from torch.utils.data import DataLoader, Dataset

import jax
import jax.numpy as jnp
from jax import nn
from optax import adam, clip_by_global_norm, chain, apply_every

from palm_jax import PaLM
import equinox as eqx
from utils import sample

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

# prepare enwik8 data

with gzip.open('../datasets/enwik8.gz') as file:
    X = np.fromstring(file.read(int(95e6)), dtype=np.uint8)
    data_train, data_val = np.split(X, [int(90e6)])

class TextSamplerDataset(Dataset):
    def __init__(self, data, seq_len):
        super().__init__()
        self.data = data
        self.seq_len = seq_len

    def __getitem__(self, index):
        rand_start = randrange(0, self.data.shape[0] - self.seq_len - 1)
        return self.data[rand_start: rand_start + self.seq_len + 1]

    def __len__(self):
        return self.data.shape[0] // self.seq_len

train_dataset = TextSamplerDataset(data_train, SEQ_LEN)
val_dataset   = TextSamplerDataset(data_val, SEQ_LEN)
train_loader  = cycle(DataLoader(train_dataset, batch_size = BATCH_SIZE))
val_loader    = cycle(DataLoader(val_dataset, batch_size = BATCH_SIZE))

# setup model and params

key = jax.random.PRNGKey(0)

model = PaLM(
    num_tokens = 256,
    dim = 512,
    depth=12,
    heads=8,
    dim_head=64,
    key = key
)

eval_model = PaLM(
    num_tokens = 256,
    dim = 512,
    depth=12,
    heads=8,
    dim_head=64,
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
    logits = jax.vmap(model)(inp)
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
    data = next(train_loader).numpy()
    model, optim_state, loss = train_step(model, data, optim_state)
    if i % GRADIENT_ACCUMULATE_EVERY == 0:
        print(f'loss: {loss.item()}')
    if i % SAMPLE_EVERY == 0:
        valid_data = next(val_loader).numpy()
        prime = valid_data[0][:100]
        prime_str = decode_tokens(prime)
        print(prime_str, "\n", "*" * 40)

        sampled = sample(key, eval_model, model, prime, SEQ_LEN, top_k = 25)
        sampled_str = decode_tokens(sampled[100:])
        print(sampled_str)
