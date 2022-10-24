# frozen_string_literal: true

require 'narray'
include Math
$stdout = File.open('выходные данные.txt', 'w')

# начальные условия
def refresh_conditions
  @N = 10
  @a = 0.01
  @x0 = 0
  @x1 = 1
  @t0 = 0
  @T = 4
  @phi0 = 0
  @phi1 = 0
  @h = @x1.fdiv @N
  @tau = (@T - @t0).fdiv @N
  @sigma = (@a * @tau).fdiv (@h * @h).round(8)
  @u_arr = NMatrix.float(@N + 1, @N + 1)
  @ves = 0.5

  xi_fill
end

def xi_fill
  i = @x0
  while i <= @x1
    @u_arr[i * @N, 0] = xi(i).round(7)

    i = (i + @h).round(4)
  end

  @u_arr
end

def xi(x)
  sin(2.0 * PI * x)
end

def u(x, t)
  (E**(-4 * PI * PI * @a * t)) * sin(2.0 * PI * x)
end

def method1
  j = @t0
  while j <= @T

    i = @x0 + @h
    while i < @x1
      @u_arr[i * @N, j * @N / (@T - @t0) + 1] = @u_arr[i * @N, j * @N / (@T - @t0)] + \
                                                @sigma.round(7) * \
                                                (@u_arr[i * @N + 1, j * @N / (@T - @t0)] \
                                                - 2 * @u_arr[i * @N, j * @N / (@T - @t0)] \
                                                + @u_arr[i * @N - 1, j * @N / (@T - @t0)])

      i = (i + @h).round(3)
    end

    j += @tau
  end

  @u_arr
end

def method2
  j = @t0
  while j <= @T

    arr_a = Array.new(@N - 1, @sigma)
    arr_a[0] = 0
    arr_b = Array.new(@N - 1, -(1 + 2 * @sigma))
    arr_c = Array.new(@N - 1, @sigma)
    arr_c[@N - 2] = 0

    arr_d = [- (@u_arr[1, j * @N / (@T - @t0)] + @sigma * @phi0)]
    i = 2
    while i <= @N - 2
      arr_d[i - 1] = - @u_arr[i, j * @N / (@T - @t0)]
      i += 1
    end
    arr_d[@N - 2] = -(@u_arr[@N - 1, j * @N / (@T - @t0)] + @sigma * @phi1)

    answer = slau(arr_a, arr_b, arr_c, arr_d)

    (0...answer.size).each do |i|
      @u_arr[i + 1, j * @N / (@T - @t0) + 1] = answer[i]
    end

    j += @tau
  end
  @u_arr
end

def method3
  resh1 = method1
  refresh_conditions
  resh2 = method2

  (0..@N).each do |i|
    j = @t0
    while j <= @T
      @u_arr[i, j * @N / (@T - @t0)] = resh1[i, j * @N / (@T - @t0)] * @ves + resh2[i, j * @N / (@T - @t0)] * (1 - @ves)

      j += @tau
    end
  end

  @u_arr
end

def tochnoe_reshenie
  i = @x0
  while i <= @x1
    j = @t0
    while j <= @T
      @u_arr[i * @N, j * @N / (@T - @t0)] = u(i, j).round(7)
      j = (j + @tau).round(5)
    end

    i = (i + @h).round(3)
  end
  @u_arr
end

def output(arr)
  s = ''
  (0..@N).each do |i|
    (0..@N).each do |j|
      s += "#{arr[i, j].round(5)}; "
    end
    s += "\n"
  end
  s
end

def slau(arr_a, arr_b, arr_c, arr_d)
  (0...arr_a.size).each do |i|
    return if arr_b[i].abs < arr_a[i].abs + arr_c[i].abs
  end
  # прямой ход
  p = [- arr_c[0] / arr_b[0]]
  q = [arr_d[0] / arr_b[0]]

  (1...arr_a.length).each do |i|
    p.append(- arr_c[i] / (arr_b[i] + arr_a[i] * p[i - 1]))
    q.append((arr_d[i] - arr_a[i] * q[i - 1]) / (arr_b[i] + arr_a[i] * p[i - 1]))
  end

  # обратный ход
  x = [q[q.size - 1]]
  i = arr_a.size - 2
  while i >= 0
    x.unshift(p[i] * x.first + q[i])
    i -= 1
  end

  x
end

refresh_conditions
puts output(tochnoe_reshenie)
puts "\n"
refresh_conditions
puts output(method1)
puts "\n"
refresh_conditions
puts output(method2)
puts "\n"
refresh_conditions
puts output(method3)
