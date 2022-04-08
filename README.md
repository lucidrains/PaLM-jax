## PaLM - Jax

Implementation of the specific Transformer architecture from <a href="https://ai.googleblog.com/2022/04/pathways-language-model-palm-scaling-to.html">PaLM - Scaling Language Modeling with Pathways</a> - in Jax using <a href="https://github.com/patrick-kidger/equinox">Equinox</a>

May as well start doing more Jax work, given Facebook (Meta's) uncertain future

<a href="https://github.com/lucidrains/PaLM-pytorch">Pytorch version</a>

## Install

```bash
$ pip install palm-jax
```

## Usage

```python
import jax
from palm_jax import PaLM

key = jax.random.PRNGKey(0)

model = PaLM(
    num_tokens = 20000,
    dim = 512,
    depth = 12,
    heads = 8,
    dim_head = 64,
    key = key
)

seq = jax.random.randint(key, (1, 2, 1024,), 0, 20000)

logits = model(seq) # (1, 1024, 20000)
```

The way the model is built doesn't require `vmap` at all. It can have any number of leading dimensions


## Todos

- [ ] bring in optax and setup a basic training on enwik8

## Citations

```bibtex
@inproceedings{Chowdhery2022PaLMSL,
    title   = {PaLM: Scaling Language Modeling with Pathways},
    author  = {Aakanksha Chowdhery and Sharan Narang and Jacob Devlin and Maarten Bosma and Gaurav Mishra and Adam Roberts and Paul Barham and Hyung Won Chung and Charles Sutton and Sebastian Gehrmann and Parker Schuh and Kensen Shi and Sasha Tsvyashchenko and Joshua Maynez and Abhishek Rao and Parker Barnes and Yi Tay and Noam M. Shazeer and Vinodkumar Prabhakaran and Emily Reif and Nan Du and Benton C. Hutchinson and Reiner Pope and James Bradbury and Jacob Austin and Michael Isard and Guy Gur-Ari and Pengcheng Yin and Toju Duke and Anselm Levskaya and Sanjay Ghemawat and Sunipa Dev and Henryk Michalewski and Xavier Garc{\'i}a and Vedant Misra and Kevin Robinson and Liam Fedus and Denny Zhou and Daphne Ippolito and David Luan and Hyeontaek Lim and Barret Zoph and Alexander Spiridonov and Ryan Sepassi and David Dohan and Shivani Agrawal and Mark Omernick and Andrew M. Dai and Thanumalayan Sankaranarayana Pillai and Marie Pellat and Aitor Lewkowycz and Erica Oliveira Moreira and Rewon Child and Oleksandr Polozov and Katherine Lee and Zongwei Zhou and Xuezhi Wang and Brennan Saeta and Mark Diaz and Orhan Firat and Michele Catasta and Jason Wei and Kathleen S. Meier-Hellstern and Douglas Eck and Jeff Dean and Slav Petrov and Noah Fiedel},
    year    = {2022}
}
```
