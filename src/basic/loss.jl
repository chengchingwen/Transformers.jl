"compute the kl divergence with mask where p is already the log(p)"
logkldivergence(q::ThreeDimArray{T},
       logp::ThreeDimArray{T},
       mask) where T =
           sum(reshape(sum(sum(q .* (log.(q .+ eps(q[1])) .- logp); dims=1) .* mask; dims=2), :) ./ reshape(sum(mask; dims=2), :))

function logkldivergence(q, logp, mask)
    kld = (q .* (log.(q .+ eps(q[1])) .- logp)) #handle gpu broadcast error
    sum(kld .* mask) / sum(mask)
end

"compute the cross entropy with mask where p is already the log(p)"
logcrossentropy(q::ThreeDimArray{T},
                logp::ThreeDimArray{T},
                mask) where T =
                    sum(reshape(sum(-sum(q .* logp; dims=1) .* mask; dims=2), :) ./ reshape(sum(mask; dims=2), :))

function logcrossentropy(q, logp, mask)
    ce = q .* logp
    sum(ce .* mask) / sum(mask)
end
