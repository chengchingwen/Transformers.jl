# BERT

This is a simple demo for using Transformers.jl to finetune a simple bert model (as the origin
paper [Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805) does)

*disclaimer*: The code here is just for demostrating the API. All hyper-parameters are not tuned.

## Code structure

- `main.jl`: code for parse the argument and then including corresponding training file.
- `<task>/train.jl`: containing all the training argument, model definition, and training loop.
  + `train!()`: start training.
  + `test`: run the model on development set

## Running the example

The example expects a commandline argument, so it can be run like this:
```bash
$ julia --proj -i main.jl [--gpu] <task>
```
where `<task>` is one of these: `{cola, mnli, mrpc}`

After getting that, you can call `train!()` that will start to train the model.
