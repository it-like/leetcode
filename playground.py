from tqdm import tqdm
import time

# Recursion 
def fib_rec(n: int) -> int:
    if n == 0 or n == 1:
        return n
    return fib_rec(n - 1) + fib_rec(n - 2)

# Memoized Recursion
def fib_mem(n : int, memo : dict=None):
    if memo is None:
        memo = {}
    if n in memo:
        return memo[n]
    if n == 0 or n == 1:
        return n
    memo[n] = fib_mem(n - 1,memo) + fib_mem(n - 2,memo)
    #print(list(memo.values())[-1])
    return memo[n]


# Bottom-Up 
def fib_bot_up(n:int)->int:
    if n ==0 or n ==1 or n ==2:
        return n
    btmup = [0] * (n+1)
    btmup[0] = 1
    btmup[1] = 1
    for i in range(2,n):
        btmup[i] = btmup[i-1] + btmup[i-2]
    return btmup[n-1]
'''
TESTING DP PROBLEMS
n = 10
iterations = 500000

print("Running fib_rec (recursion):")
start_time = time.time()
for _ in tqdm(range(iterations), desc="fib_mem rec"):
    result_mem = fib_rec(n)
end_time = time.time()

print(f"Result: {result_mem}")
print(f"Total time for {iterations} iterations of fib_mem: {end_time - start_time:.4f} seconds")
print(f"Average time per call: {(end_time - start_time) / iterations:.8f} seconds")

print("Running fib_mem (memoized recursion):")
start_time = time.time()
for _ in tqdm(range(iterations), desc="fib_mem iterations"):
    result_mem = fib_mem(n)
end_time = time.time()

print(f"Result: {result_mem}")
print(f"Total time for {iterations} iterations of fib_mem: {end_time - start_time:.4f} seconds")
print(f"Average time per call: {(end_time - start_time) / iterations:.8f} seconds")

print("Running btmup (Bottom-Up approach):")
start_time = time.time()
for _ in tqdm(range(iterations), desc="btmup iterations"):
    result_rec = fib_bot_up(n)
end_time = time.time()
print(f"Result: {result_rec}")
print(f"Total time for {iterations} iterations of btmup: {end_time - start_time:.4f} seconds")
print(f"Average time per call: {(end_time - start_time) / iterations:.8f} seconds\n")
'''


# Euler approximation
#e = 0
#print(e*2)
#import math
#for n in range(20):
#    e += (2**n)/math.factorial(n)
#print(e) 





