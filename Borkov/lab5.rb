require 'narray'
include Math
$stdout = File.open('выходные данные.txt', 'w')

# PI       # => 3.141592653589793
# E       # => 2.718281828459045
# sin(0.0) # => 0.0
# cos(0.0) # => 1.0

# начальные условия
def refresh_conditions
  @N = 10
  @a = 0.01
  @x0 = 0; @x1 = 1
  @t0 = 0; @T = 4
  @phi0 = 0; @phi1 = 0
  @h = @x1.fdiv @N
  @tau = (@T - @t0).fdiv @N
  @sigma = (@a * @tau).fdiv (@h * @h).round(8)
  @u_arr = NMatrix.float(@N + 1, @N + 1)

  xi_fill
end

def xi_fill
  i = @x0
  while i <= @x1 do
    @u_arr[i * @N, 0] = xi(i).round(7)

    i = (i + @h).round(1)
  end

  @u_arr
end

def xi(x)
  sin(2.0 * PI * x)
end

def u(x, t)
  (E ** (-4 * PI * PI * @a * t)) * sin(2.0 * PI * x)
end

def method1
  j = @t0
  while j <= @T do

    i = @x0 + @h
    while i < @x1 do
      @u_arr[i * @N, j * @N / (@T - @t0) + 1] = @u_arr[i * @N, j * @N / (@T - @t0)] + \
        @sigma.round(7) * \
          (@u_arr[i * @N + 1, j * @N / (@T - @t0)] \
          - 2 * @u_arr[i * @N, j * @N / (@T - @t0)] \
          + @u_arr[i * @N - 1, j * @N / (@T - @t0)])

      # puts i * @N
      # puts j * @N / (@T - @t0)
      # puts @u_arr[i * @N, j * @N / (@T - @t0)]

      i = (i + @h).round(1)
    end

    j += @tau
  end

  @u_arr
end

def method2
  j = @t0
  while j <= @T do

    arr_a = Array.new(@N - 2, @sigma)
    arr_a[0] = 0
    arr_b = Array.new(@N - 2, -(1 + 2 * @sigma))
    arr_c = Array.new(@N - 2, @sigma)
    arr_c[@N - 3] = 0
    arr_d = [- (@u_arr[0, j * @N / (@T - @t0)] + @sigma * @phi0)]
    arr_d[@N - 2] = -(@u_arr[@N - 1, j * @N / (@T - @t0)] + @sigma * @phi1)
    i = @x0 + @h
    while i < @x1 - (2 * @h) do
      arr_d[i * @N] = - @u_arr[i * @N, j * @N / (@T - @t0)]

      i = (i + @h).round(1)
    end
    # puts arr_a.to_s
    # puts arr_b.to_s
    # puts arr_c.to_s
    # puts arr_d.to_s
    # puts arr_d[@N - 2].to_s

    answer = slau(arr_a, arr_b, arr_c, arr_d)

    for i in 0...answer.size
      @u_arr[i + 1, j * @N / (@T - @t0) + 1] = answer[i]
    end

    j += @tau
  end
  @u_arr
end

def tochnoe_reshenie
  i = @x0
  while i <= @x1 do
    # puts i.to_s + ' !!!'
    # puts xi(i)
    j = @t0
    while j <= @T do
      # puts "#{i} and #{j}"
      @u_arr[i * @N, j * @N / (@T - @t0)] = u(i, j).round(7)
      # puts i * @N
      # puts j * @N / (@T - @t0)
      # puts @u_arr[i * @N, j * @N / (@T - @t0)]
      j = (j + @tau).round(5)
    end

    i = (i + @h).round(1)
  end
  @u_arr
end

def output(arr)
  s = ''
  for i in 0..@N
    for j in 0..@N
      s += arr[i, j].to_s + '; '
    end
    s += "\n"
  end
  s
end

def slau(arr_a, arr_b, arr_c, arr_d)
  for i in 0...arr_a.size
    return if arr_b[i].abs < arr_a[i].abs + arr_c[i].abs
  end
  #прямой ход
  p = [- arr_c[0] / arr_b[0]]
  q = [- arr_d[0] / arr_b[0]]

  for i in 1...arr_a.length
    p.append(- arr_c[i] / (arr_b[i] + arr_a[i] * p[i - 1]))
    q.append((arr_d[i] - arr_a[i] * q[i - 1]) / (arr_b[i] + arr_a[i] * p[i - 1]))
  end

  #обратный ход
  x = [q[-1]]
  for i in 1...arr_a.count
    x.append(p[arr_a.length - i - 1] * x[i - 1] + q[arr_a.size - i - 1])
  end

  puts x.reverse.to_s
  return x.reverse
end

# refresh_conditions
# puts output(tochnoe_reshenie)
# puts "\n"
# refresh_conditions
# puts output(method1)
# puts "\n"
refresh_conditions
puts output(method2)
