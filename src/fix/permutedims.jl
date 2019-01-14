using Flux
using Flux.Tracker: TrackedArray, track, data, @grad

using CuArrays

"permutedims_hack for gpu permutedims performance issue, move data back to cpu to permutedims and move back to gpu"
permutedims_hack(x, perm) = permutedims(x, perm)
function permutedims_hack(x::CuArray, perm)
    cpu_x = collect(x)
    cpu_px = permutedims(cpu_x, perm)
    cu(cpu_px)
end

permutedims_hack(x::TrackedArray, perm) = track(permutedims_hack, x, perm)
@grad permutedims_hack(x, perm) = permutedims_hack(data(x), perm), dt -> (permutedims_hack(dt, invperm(perm)) , nothing)

