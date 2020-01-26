using Flux: @functor

using ..Basic: AbstractTransformer, MultiheadAttention, PwFFN
using ..Stacks

struct Gpt2Transformer{A<:MultiheadAttention, L<:LayerNorm, F<:PwFFN} <: AbstractTransformer
    pre_norm::L
    attention::A
    attn_norm::L
    piecewise::F
end

@functor Gpt2Transformer

"""
    Gpt2Transformer(inputsize::Integer, num_heads::Integer,
                    ffhiddensize::Integer; look_ahead::Bool = false, activation = gelu)
    Gpt2Transformer(inputsize::Integer, num_heads::Integer, attnhiddensize::Integer,
                    ffhiddensize::Integer; look_ahead::Bool = false, activation = gelu)

Modified transformer layer for the [`Gpt2`](@ref) model.
Very similar to a standard [`Transformer`](@ref).

If `attnhiddensize` is not specified, use `div(size, head)` as the hidden size
of multi-head attention.
`activation` is the activation function of the positionwise feedforward layer.
When `look_ahead` is `false`, the k-th token can't see the j-th tokens where j > k.
"""
function Gpt2Transformer(inputsize::Integer, num_heads::Integer, ffhiddensize::Integer;
                         look_ahead::Bool=false, activation=gelu)
    if inputsize % num_heads != 0
        throw(ArgumentError("`inputsize` not divisible by `num_heads`"))
    end
    Gpt2Transformer(inputsize, num_heads, div(inputsize, num_heads), ffhiddensize;
                    look_ahead=look_ahead, activation=activation)
end

function Gpt2Transformer(inputsize::Integer, heads::Integer, attnhiddensize::Integer,
                         ffhiddensize::Integer; look_ahead::Bool=false, activation=gelu)
    Gpt2Transformer(
        LayerNorm(inputsize),
        MultiheadAttention(heads, inputsize, attnhiddensize, inputsize;
                           future=look_ahead, pdrop=0),
        LayerNorm(inputsize),
        PwFFN(inputsize, ffhiddensize, activation),
    )
end

function (b::Gpt2Transformer)(input::AbstractArray{T, N}, mask=nothing) where {T, N}
    normed_input = b.pre_norm(input)::typeof(input)
    attn = b.attention(normed_input, normed_input, normed_input; mask=mask)
    res_attn = input .+ attn
    if N == 3
        insize = size(res_attn)
        res_attn = reshape(res_attn, insize[1], :)
    end
    normed_res_attn = b.attn_norm(res_attn)::typeof(res_attn)
    pwffn = b.piecewise(normed_res_attn)
    res_pwffn = res_attn .+ pwffn
    if N == 3
        res_pwffn = reshape(res_pwffn, :, Base.tail(insize)...)
    end
    res_pwffn
end


struct Gpt2{S<:Stack, L<:LayerNorm} <: AbstractTransformer
    blocks::S
    final_norm::L
end

@functor Gpt2

"""
    Gpt2(inputsize::Integer, num_heads::Integer, ffhiddensize::Integer,
         num_layers::Integer; activation = gelu)
    Gpt2(inputsize::Integer, num_heads::Integer, hs::Integer, ffhiddensize::Integer,
         num_layers::Integer; activation = gelu)

Second version of the Generative Pre-trained Transformer model ([GPT-2](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)).

See also: [`Gpt`](@ref)
"""
function Gpt2(inputsize::Integer, num_heads::Integer, ffhiddensize::Integer,
              num_layers::Integer; activation=gelu)
    if inputsize % num_heads != 0
        throw(ArgumentError("`inputsize` not divisible by `num_heads`"))
    end
    Gpt2(inputsize, num_heads, div(inputsize, num_heads), ffhiddensize, num_layers;
         activation=activation)
end

function Gpt2(inputsize::Integer, num_heads::Integer, attnhiddensize::Integer,
              ffhiddensize::Integer, num_layers::Integer; activation=gelu)
    Gpt2(
        Stack(
            @nntopo_str("x':x => $num_layers"),
            [
                Gpt2Transformer(inputsize, num_heads, attnhiddensize, ffhiddensize;
                                look_ahead=false, activation=activation)
                for _ in 1:num_layers
            ]...
        ),
        LayerNorm(inputsize)
    )
end

"""
    (gpt::Gpt2)(input, mask=nothing; return_all_outputs::Bool=false)

Evaluate the GPT-2 model on the given `input`. If `mask` is given (of shape
`(1, seq_len, batch_size)`), mask the attention with it.
Set `return_all_outputs` to `true` to get all outputs of each internal transformer layer.
"""
function (gpt::Gpt2)(input, mask=nothing; return_all_outputs::Bool=false)
    output, all_outputs = gpt.blocks(input)
    output = gpt.final_norm(output)
    isnothing(mask) || (output .*= mask)
    if return_all_outputs
        return output, all_outputs
    else
        return output  # size(output) == (inputsize, seq_len, batch_len)
    end
end
