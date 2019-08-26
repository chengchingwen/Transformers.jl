using Luxor
const JULIA_RED = (0.796, 0.235, 0.2)
const JULIA_PURPLE = (0.584, 0.345, 0.698)
const JULIA_BLUE = (0.251, 0.388, 0.847)
const JULIA_GREEN = (0.22, 0.598, 0.149)

const JULIA_LIGHT_RED = (0.828125, 0.40234375, 0.390625)
const JULIA_LIGHT_PURPLE = (0.66796875, 0.47265625, 0.7421875)
const JULIA_LIGHT_GREEN = (0.375, 0.671875, 0.31640625)

function halfring(w, h, c, bk = "white")
  @layer begin
    box(Point(-256, -256), Point(256, 1), :clip)
    sethue(JULIA_LIGHT_RED)
    ellipse(O, w, h, :fill)
    sethue(c)
    setline(5)
    ellipse(O, w, h, :stroke)
  end
  @layer begin
    box(Point(-256, -256), Point(256, 2), :clip)
    sethue(bk)
    ellipse(O, 0.8w, 0.8h, :fill)
    sethue(c)
    setline(3)
    ellipse(O, 0.8w, 0.8h, :stroke)
  end
end

function planet(w, h, deg, rc, pc, bk = "white")
  C = - Point(0.25h,0.35h)
  L = (0.9, 0.6, 1, 1)
  @layer (rotate(deg2rad(deg)); halfring(w, h, rc, bk))
  @layer begin
    sethue(JULIA_LIGHT_PURPLE)
    circle(O, h, :fill)
    sethue(pc)
    setline(5)
    circle(O, h, :stroke)
  end
  @layer (rotate(-π + deg2rad(deg)); halfring(w, h, rc, bk))
  @layer begin
    rotate(-π + deg2rad(deg))
    ellipse(O, 0.8w, 0.8h, :clip)
    @layer begin
      sethue(JULIA_LIGHT_PURPLE)
      circle(O, h, :fill)
      sethue(pc)
      setline(5)
      circle(O, h, :stroke)
    end
  end
  @layer totem(h, JULIA_PURPLE)
end

function logotext(h)
  @layer begin
    setline(0.5)
    fontsize(57)
    fontface("Tamil MN")
    textoutlines("Transformers.jl", Point(-2.2h, 1.8h), :path, halign=:left)
    fillpreserve()
  end
end

function totem(h, pc)
  @layer begin
    sethue(pc)
    setline(3)
    #left up
    C = O - 0.3h
    r = 0.2h
    circle(C, r, :stroke)
    line(C - Point(r, 0), C - Point(2r, 0), :stroke)
    sector(C, 2r, 2r, π/6, 1.2π, :stroke)
    line(C + 4r/3 + 1, C + 2r +r/3, :stroke)
    C = O + 0.4h + Point(0.2h, -0.2h)
  end
end

function moon(h, gc)
  @layer begin
    P = O + 0.3h + Point(0, 0.35h)
    l = 0.15h
    sethue(JULIA_LIGHT_GREEN)
    setline(3)
    circle(P, l, :fill)
    sethue(gc)
    circle(P, l, :stroke)
    setline(2)
    circle(P - Point(-0.04h, 0.03h), 0.03h, :stroke)
  end
end

function logo(h, r, rc, pc, gc, text)
  @layer planet(5h, h, r, rc, pc, "white")
  @layer moon(h, gc)
  text && @layer logotext(h)
end

const h = 100
const r = 30

@svg begin
  scale(1.15)
  translate(Point(0, -25))
  logo(h, r, JULIA_RED, JULIA_PURPLE, JULIA_GREEN, true)
end 512 400 "transformerslogo.svg"

@png begin
  scale(1.15)
  translate(Point(0, -25))
  logo(h, r, JULIA_RED, JULIA_PURPLE, JULIA_GREEN, true)
end 512 400 "transformerslogo.png"

@svg begin
  scale(1.15)
  logo(h, r, JULIA_RED, JULIA_PURPLE, JULIA_GREEN, false)
end 512 320 "logo.svg"

@png begin
  scale(1.15)
  logo(h, r, JULIA_RED, JULIA_PURPLE, JULIA_GREEN, false)
end 512 320 "logo.png"

@png begin
  scale(2.2)
  logo(h, r, JULIA_RED, JULIA_PURPLE, JULIA_GREEN, false)
end 1024 640 "logolarge.png"

