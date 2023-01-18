# Transformers.Layers

Layer building blocks of Transformers.jl. Most of the layers are designed to work with `NamedTuple`s. It would take a
 `NamedTuple` as input, finding correct names as its arguments for computation, ignoring extra fields in the
 `NamedTuple`, store the computation result in the input `NamedTuple` with correct names (conceptually, since
 `NamedTuple` is immutable) and return it.

These layer types are mostly compatible with Flux.

## API Reference

```@autodocs
Modules = [Transformers.Layers]
Order = [:type, :function]
```
