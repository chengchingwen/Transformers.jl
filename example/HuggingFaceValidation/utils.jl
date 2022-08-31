macro tryrun(ex, msg = nothing)
    err_msg = isnothing(msg) ? nothing : :(@error $msg)
    return quote
        try
            $(esc(ex))
        catch e
            $err_msg
            rethrow(e)
        end
    end
end

rowmaj2colmaj(x) = permutedims(x, ndims(x):-1:1)
