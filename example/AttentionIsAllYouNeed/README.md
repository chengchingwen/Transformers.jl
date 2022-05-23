# Attention Is All You Need

This is a simple demo for using Transformers.jl to train a simple transfomer model (as the origin
paper [Attention is all you need](https://arxiv.org/abs/1706.03762) does)

The architecture explaination can be found [here](http://nlp.seas.harvard.edu/annotated-transformer/).

*disclaimer*: The code here is just for demostrating the API. All hyper-parameters are not tuned.

## Code structure

- `main.jl`: code for parse the argument and then including corresponding training file.
- `<task>/train.jl`: containing all the training argument, model definition, and training loop.
  + `train!()`: start training.
  + `translate(text::String)`: translate a given text with the model

## Running the example

The example expects a commandline argument, so it can be run like this:
```bash
$ julia --proj -i main.jl [--gpu] <task>
```
where `<task>` is one of these: `{copy, wmt14, iwslt2016}`

After getting that, you can call `train!()` that will start to train the model. When the training finished,
 call `translate("<some text here that I want to try with")` to test the model.
