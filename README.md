<img src="./palm.gif" width="450px"></img>

## PaLM - Jax

Implementation of the specific Transformer architecture from <a href="https://ai.googleblog.com/2022/04/pathways-language-model-palm-scaling-to.html">PaLM - Scaling Language Modeling with Pathways</a> - in Jax using <a href="https://github.com/patrick-kidger/equinox">Equinox</a>

May as well start doing more Jax work, given Facebook (Meta's) uncertain future

<a href="https://github.com/lucidrains/PaLM-pytorch">Pytorch version</a>

## Install

```bash
$ pip install PaLM-jax
```

## Usage

The way the model is built doesn't require `vmap` at all. It can have any number of leading dimensions

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

seq = jax.random.randint(key, (1, 1024), 0, 20000)

logits = model(seq) # (1, 1024, 20000)
```

The 540B PaLM in the paper would be


```python

model = PaLM(
    num_tokens = 256000,
    dim = 18432,
    depth = 118,
    heads = 48,
    dim_head = 256,
    key = key
)

```

That's all it is. Attention (and scale) is all we need.

## Todos

- [ ] bring in optax and setup a basic training on enwik8
- [x] ALiBi positional encoding https://arxiv.org/abs/2108.12409 for PaLM-lite

## Citations

```bibtex
@inproceedings{Chowdhery2022PaLMSL,
    title   = {PaLM: Scaling Language Modeling with Pathways},
    author  = {Aakanksha Chowdhery and Sharan Narang and Jacob Devlin and Maarten Bosma and Gaurav Mishra and Adam Roberts and Paul Barham and Hyung Won Chung and Charles Sutton and Sebastian Gehrmann and Parker Schuh and Kensen Shi and Sasha Tsvyashchenko and Joshua Maynez and Abhishek Rao and Parker Barnes and Yi Tay and Noam M. Shazeer and Vinodkumar Prabhakaran and Emily Reif and Nan Du and Benton C. Hutchinson and Reiner Pope and James Bradbury and Jacob Austin and Michael Isard and Guy Gur-Ari and Pengcheng Yin and Toju Duke and Anselm Levskaya and Sanjay Ghemawat and Sunipa Dev and Henryk Michalewski and Xavier Garc{\'i}a and Vedant Misra and Kevin Robinson and Liam Fedus and Denny Zhou and Daphne Ippolito and David Luan and Hyeontaek Lim and Barret Zoph and Alexander Spiridonov and Ryan Sepassi and David Dohan and Shivani Agrawal and Mark Omernick and Andrew M. Dai and Thanumalayan Sankaranarayana Pillai and Marie Pellat and Aitor Lewkowycz and Erica Oliveira Moreira and Rewon Child and Oleksandr Polozov and Katherine Lee and Zongwei Zhou and Xuezhi Wang and Brennan Saeta and Mark Diaz and Orhan Firat and Michele Catasta and Jason Wei and Kathleen S. Meier-Hellstern and Douglas Eck and Jeff Dean and Slav Petrov and Noah Fiedel},
    year    = {2022}
}
```

```bibtex
@misc{press2021ALiBi,
    title   = {Train Short, Test Long: Attention with Linear Biases Enable Input Length Extrapolation},
    author  = {Ofir Press and Noah A. Smith and Mike Lewis},
    year    = {2021},
    url     = {https://ofir.io/train_short_test_long.pdf}
}
```

```bibtex
@article{Rae2021ScalingLM,
    title   = {Scaling Language Models: Methods, Analysis \& Insights from Training Gopher},
    author  = {Jack W. Rae and Sebastian Borgeaud and Trevor Cai and Katie Millican and Jordan Hoffmann and Francis Song and John Aslanides and Sarah Henderson and Roman Ring and Susannah Young and Eliza Rutherford and Tom Hennigan and Jacob Menick and Albin Cassirer and Richard Powell and George van den Driessche and Lisa Anne Hendricks and Maribeth Rauh and Po-Sen Huang and Amelia Glaese and Johannes Welbl and Sumanth Dathathri and Saffron Huang and Jonathan Uesato and John F. J. Mellor and Irina Higgins and Antonia Creswell and Nathan McAleese and Amy Wu and Erich Elsen and Siddhant M. Jayakumar and Elena Buchatskaya and David Budden and Esme Sutherland and Karen Simonyan and Michela Paganini and L. Sifre and Lena Martens and Xiang Lorraine Li and Adhiguna Kuncoro and Aida Nematzadeh and Elena Gribovskaya and Domenic Donato and Angeliki Lazaridou and Arthur Mensch and Jean-Baptiste Lespiau and Maria Tsimpoukelli and N. K. Grigorev and Doug Fritz and Thibault Sottiaux and Mantas Pajarskas and Tobias Pohlen and Zhitao Gong and Daniel Toyama and Cyprien de Masson d'Autume and Yujia Li and Tayfun Terzi and Vladimir Mikulik and Igor Babuschkin and Aidan Clark and Diego de Las Casas and Aurelia Guy and Chris Jones and James Bradbury and Matthew G. Johnson and Blake A. Hechtman and Laura Weidinger and Iason Gabriel and William S. Isaac and Edward Lockhart and Simon Osindero and Laura Rimell and Chris Dyer and Oriol Vinyals and Kareem W. Ayoub and Jeff Stanway and L. L. Bennett and Demis Hassabis and Koray Kavukcuoglu and Geoffrey Irving},
    journal = {ArXiv},
    year    = {2021},
    volume  = {abs/2112.11446}
}
```

```bibtex
@inproceedings{Zhang2019RootMS,
    title   = {Root Mean Square Layer Normalization},
    author  = {Biao Zhang and Rico Sennrich},
    booktitle = {NeurIPS},
    year    = {2019}
}
```
