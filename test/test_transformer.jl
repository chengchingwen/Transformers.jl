@testset "Transformer" begin
    t = Transformer(10, 3, 15, 20)
    td = TransformerDecoder(10, 3, 15, 20)
    x = randn(Float32, 10, 7, 3)
    y = randn(Float32, 10, 6, 3)

    m1 = reshape(Float32[1. 1. 1. 1. 0. 0. 0.; 1. 1. 1. 1. 1. 1. 0.; 1. 1. 1. 1. 1. 1. 1.]', 1, 7, 3)
    m2 = reshape(Float32[1. 1. 1. 0. 0. 0.; 1. 1. 1. 1. 1. 0.; 1. 1. 1. 1. 1. 1.]', 1, 6, 3)
    m = getmask(m1, m2)

    @test size(t(x)) == (10, 7, 3)
    @test size(t(x[:, :, 2])) == (10, 7)

    @test size(td(y, x)) == (10, 6, 3)
    @test size(td(y[:,:,2], x[:,:,2])) == (10, 6)

    @test_nowarn td(y, x, m)

    let STDOUT = stdout
        (outRead, outWrite) = redirect_stdout()
        print(outWrite, t)
        close(outWrite)

        output_string = String(readavailable(outRead))
        close(outRead)

        redirect_stdout(STDOUT)

        @test output_string == "Transformer(head=3, head_size=15, pwffn_size=20, size=10, dropout=0.1)"
    end

    let STDOUT = stdout
        (outRead, outWrite) = redirect_stdout()
        print(outWrite, t.mh)
        close(outWrite)

        output_string = String(readavailable(outRead))
        close(outRead)

        redirect_stdout(STDOUT)

        @test output_string == "MultiheadAttention(head=3, head_size=15, 10=>10, dropout=0.1)"
    end

    let STDOUT = stdout
        (outRead, outWrite) = redirect_stdout()
        print(outWrite, td)
        close(outWrite)

        output_string = String(readavailable(outRead))
        close(outRead)

        redirect_stdout(STDOUT)

        @test output_string == "TransformerDecoder(head=3, head_size=15, pwffn_size=20, size=10, dropout=0.1)"
    end

    Flux.testmode!(t, true)
    Flux.testmode!(td, true)

    let STDOUT = stdout
        (outRead, outWrite) = redirect_stdout()
        print(outWrite, t)
        close(outWrite)

        output_string = String(readavailable(outRead))
        close(outRead)

        redirect_stdout(STDOUT)

        @test output_string == "Transformer(head=3, head_size=15, pwffn_size=20, size=10)"
    end

    let STDOUT = stdout
        (outRead, outWrite) = redirect_stdout()
        print(outWrite, t.mh)
        close(outWrite)

        output_string = String(readavailable(outRead))
        close(outRead)

        redirect_stdout(STDOUT)

        @test output_string == "MultiheadAttention(head=3, head_size=15, 10=>10)"
    end

    let STDOUT = stdout
        (outRead, outWrite) = redirect_stdout()
        print(outWrite, td)
        close(outWrite)

        output_string = String(readavailable(outRead))
        close(outRead)

        redirect_stdout(STDOUT)

        @test output_string == "TransformerDecoder(head=3, head_size=15, pwffn_size=20, size=10)"
    end
end
