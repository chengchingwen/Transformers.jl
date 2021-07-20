# Attention Is All You Need

This is an implementation of the seminal paper [Attention is all you need](https://arxiv.org/abs/1706.03762)
that introduced the Transformer architecture.
The architecture is explained [here](https://nlp.seas.harvard.edu/2018/04/03/attention.html).

## Running the example

The example expects a commandline argument, so it can be run like this:
```bash
$ julia --proj -i 1-model.jl [--gpu] <task>
```
where `<task>` is one of these: `{copy, wmt14, iwslt2016}`
