# HuggingFaceValidation

Codes for validate huggingface model. It would load the python model with `PyCall` and verify the Julia
 implementation. There are some limitation of this validation code, so be careful when interpreting the result.

## Validation steps

1. Check the based model loading:
   a. In order to load a model from huggingface hub, the model must be loadable for pytorch. So we would
	   load the model in python with PyCall. If this step failed, it probably means this model is unsupported.
   b. Usually there would be a based model and a task specific head. At this step we only test if the based
	   model work correctly. So we also load the based model in Julia with Transformers.HuggingFace. If
	    this step failed, it's probably:
			+ The model is not implemented
			+ The model architecture mapping is wrong
			+ The weight loading mechanism is wrong
   c. After loading model in both python and julia, we feed the same random sequence of indices (one with shift 1
	   for the 0/1-based indexing) to both model. and check the result is approximately the same. If the result is
	   not approximately equal, either the model implementation is probably wrong or the numeric error is too large.
2. Check the task specific head: Once the based model work correctly, we then test the task specific head.
    The strategy is similar to step 1.
3. Checking the tokenizer:
   a. Load the tokenizer in python. If this step failed, it's possibly that model doesn't have a tokenizer with it.
   b. Load the tokenizer in Julia. If this step failed, either the tokenizer is not supported or there is an issue
	   in the loading code.
   c. Given a corpus, apply the tokenizer on eachline of the corpus and examine the result.

## Limitation

+ We are using a implementation different from the huggingface one with different framework and computation
 kernel, so it's likely the model output is "slightly" different (and you would see some test error when running
 this code). Here we are checking the mean square error between two model output. Is possible that the model result
 in weird behavior even if all test is passed.
+ The tokenizer part is more tricky since feeding random string doesn't make sense. We need a proper corpus for testing.
 But even with a testing corpus, it doesn't mean the tokenizer is 100% the same. We can only know the tokenizer
 behave for some "common" texts.


# Run the validation

## Requirement

We are using PyCall.jl for the python part. To run the checker, you need the install the `torch` and `transformers`
 python package and make sure `pyimport` them work correctly.

## Command

```
some/path/to/this/HuggingFaceValidation$ julia --project main.jl --help
usage: main.jl [-s SUBJECT] [-n NUMBER] [-h] name [corpus]

positional arguments:
  name                  model name
  corpus                corpus for testing tokenizer

optional arguments:
  -s, --subject SUBJECT
                        a specific testing subject (default: "all")
  -n, --number NUMBER   the number of random sample for testing the
                        model (type: Int64, default: 100)
  -h, --help            show this help message and exit

```

+ `<model_name>` is the model (repo name on huggingface hub) that you want to validate.
+ `<subject>` is the specific stuff you want to test, can be `"based_model"`, `"task_head"`, or `"tokenizer"`
+ `<corpus>` is a file containing text to be tested with the tokenizer.
