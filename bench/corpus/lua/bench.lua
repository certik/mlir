-- Small CPU benchmark for the Lua interpreter: recursion, loops, tables,
-- strings, closures. Deterministic output (checked against a reference).
local function fib(n) if n < 2 then return n end return fib(n-1) + fib(n-2) end

local N = tonumber(arg and arg[1]) or 1
local t = {}
for r = 1, N do
  -- recursion
  t.fib = fib(27)
  -- integer loop
  local s = 0
  for i = 1, 3000000 do s = s + i % 7 end
  t.loop = s
  -- tables: sieve
  local n, sieve, count = 200000, {}, 0
  for i = 2, n do sieve[i] = true end
  for i = 2, n do
    if sieve[i] then count = count + 1; for j = i*i, n, i do sieve[j] = false end end
  end
  t.primes = count
  -- strings
  local parts = {}
  for i = 1, 20000 do parts[#parts+1] = string.format("%d:%s", i, string.rep("x", i % 5)) end
  local str = table.concat(parts, ",")
  t.strlen = #str
  local hits = 0
  for w in string.gmatch(str, "%d+:xx,") do hits = hits + 1 end
  t.gmatch = hits
  -- sort + closures
  local arr = {}
  local seed = 42
  for i = 1, 50000 do seed = (seed * 1103515245 + 12345) % 2147483648; arr[i] = seed end
  table.sort(arr, function(a, b) return a > b end)
  t.sorted = arr[1] .. " " .. arr[50000]
  -- floats
  local x = 0.0
  for i = 1, 200000 do x = x + 1.0 / i end
  t.harmonic = string.format("%.10f", x)
end
print("fib", t.fib)
print("loop", t.loop)
print("primes", t.primes)
print("strlen", t.strlen, "gmatch", t.gmatch)
print("sorted", t.sorted)
print("harmonic", t.harmonic)
print("math", math.floor(3.7), math.max(3, 9, 2), 7 // 2, 7.5 // 2, 2^10, 10 / 4)
