#handle scope
N = 3
f1(a, b) = a+b
g1(a, b) = (a*b, a-b)
h1(a, b) = a^b
k(a, b, c) = (a + b) / c

@testset "NNTopo" begin
    using Transformers.Stacks: print_topo
    f(x) = x+1
    g(x) = x+2
    h(x) = x+3

    topo1 = @nntopo x => a => b => y
    @test topo1((f,g,h), 10) == h(g(f(10)))

    topo2 = @nntopo x => 4 => y
    @test topo2((f,f,f,f, g), 10) == g(f(f(f(f(10)))))


    (x1, x2, x3, x4) = [2, 3, 7, 5]
    t = f1(x1, x2)
    z1, z2 = g1(t, x3)
    w = h1(x4, z1)
    y = k(x2, z2, w)

    topo3 = @nntopo (x1, x2, x3, x4):(x1, x2) => t:(t, x3) => (z1, z2):(x4, z1) => w:(x2, z2, w) => y
    @test topo3((f1, g1, h1, k), x1, x2, x3, x4) ≈ y


    let STDOUT = stdout
        (outRead, outWrite) = redirect_stdout()

        print_topo(outWrite, topo3; models=(f1, g1, h1, k))
        close(outWrite)

        topo_string = String(readavailable(outRead))
        close(outRead)

        redirect_stdout(STDOUT)

        @test topo_string == """topo_func(model, x1, x2, x3, x4)
	t = f1(x1, x2)
	(z1, z2) = g1(t, x3)
	w = h1(x4, z1)
	y = k(x2, z2, w)
	y
end
"""
    end

    topo = @nntopo((e, m, mask):e → pe:(e, pe) → t → (t:(t, m, mask) → t:(t, m, mask)) → $N:t → c)


    let STDOUT = stdout
        (outRead, outWrite) = redirect_stdout()

        print_topo(outWrite, topo)
        close(outWrite)

        topo_string = String(readavailable(outRead))
        close(outRead)

        redirect_stdout(STDOUT)

        @test topo_string == """topo_func(model, e, m, mask)
	pe = model[1](e)
	t = model[2](e, pe)
	t = model[3](t)
	t = model[4](t, m, mask)
	t = model[5](t, m, mask)
	t = model[6](t, m, mask)
	c = model[7](t)
	c
end
"""
    end

    localn = 3
    topo = @nntopo_str "(e, m, mask):e → pe:(e, pe) → t → (t:(t, m, mask) → t:(t, m, mask)) → $localn:t → c"


    let STDOUT = stdout
        (outRead, outWrite) = redirect_stdout()

        print_topo(outWrite, topo)
        close(outWrite)

        topo_string = String(readavailable(outRead))
        close(outRead)

        redirect_stdout(STDOUT)

        @test topo_string == """topo_func(model, e, m, mask)
	pe = model[1](e)
	t = model[2](e, pe)
	t = model[3](t)
	t = model[4](t, m, mask)
	t = model[5](t, m, mask)
	t = model[6](t, m, mask)
	c = model[7](t)
	c
end
"""
    end


    let STDOUT = stdout
        (outRead, outWrite) = redirect_stdout()


        topo = @nntopo x => ((y => z => t) => 3 => w) => 2
        print_topo(outWrite, topo)
        close(outWrite)

        topo_string = String(readavailable(outRead))
        close(outRead)

        redirect_stdout(STDOUT)

        @test topo_string == """topo_func(model, x)
	y = model[1](x)
	z = model[2](y)
	t = model[3](z)
	z = model[4](t)
	t = model[5](z)
	z = model[6](t)
	t = model[7](z)
	w = model[8](t)
	z = model[9](w)
	t = model[10](z)
	z = model[11](t)
	t = model[12](z)
	z = model[13](t)
	t = model[14](z)
	w = model[15](t)
	w
end
"""
    end

    let STDOUT = stdout
        (outRead, outWrite) = redirect_stdout()

        topo = @nntopo x => y' => 3 => z
        print_topo(outWrite, topo)
        close(outWrite)

        topo_string = String(readavailable(outRead))
        close(outRead)

        redirect_stdout(STDOUT)
        @test topo_string == """topo_func(model, x)
	y = model[1](x)
	%1 = y
	y = model[2](y)
	%2 = y
	y = model[3](y)
	%3 = y
	y = model[4](y)
	%4 = y
	z = model[5](y)
	(z, (%1, %2, %3, %4))
end
"""
    end


    let STDOUT = stdout
        (outRead, outWrite) = redirect_stdout()

        topo = @nntopo (x,y) => (a,b,c,d') => (w',r',y) => (m,n)' => z
        print_topo(outWrite, topo)
        close(outWrite)

        topo_string = String(readavailable(outRead))
        close(outRead)

        redirect_stdout(STDOUT)
        @test topo_string == """topo_func(model, x, y)
	(a, b, c, d) = model[1](x, y)
	%1 = d
	(w, r, y) = model[2](a, b, c, d)
	%2 = (w, r)
	(m, n) = model[3](w, r, y)
	%3 = (m, n)
	z = model[4](m, n)
	(z, (%1, %2, %3))
end
"""
    end
end
