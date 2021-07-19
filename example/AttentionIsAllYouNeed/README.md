# Attention Is All You Need

This is an implementation of the seminal paper [Attention is all you need [annotated]](https://nlp.seas.harvard.edu/2018/04/03/attention.html)
that introduced the Transformer architecture.
The architecture is explained [here](https://medium.com/data-science-in-your-pocket/attention-is-all-you-need-understanding-with-example-c8d074c37767).

## Running the example

The example expects a commandline argument, so it can be run like this:
```bash
$ julia --proj -i 1-model.jl [--gpu] <task>
```
where `<task>` is one of these: `{copy, wmt14, iwslt2016}`
