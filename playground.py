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



# Find most divisible number up to n, return the number and it's divisibility

def find_max_div_sieve(n):
    # Create a list to hold divisor counts for numbers 0 to n
    div_counts = [0] * (n + 1)
    
    # For each number i, add 1 to the count of all multiples of i
    for i in range(1, n + 1):
        for j in range(i, n + 1, i):
            div_counts[j] += 1
            
    # Identify the number with the maximum divisor count
    max_number = 0
    max_div = 0
    for i in range(1, n + 1):
        if div_counts[i] > max_div:
            max_div = div_counts[i]
            max_number = i
            
    return max_number, max_div

#print(find_max_div_sieve(100_000))
def compute_divisors_sieve(n):
    # Array to store the smallest prime factor for every number
    spf = list(range(n + 1))
    
    # Sieve to compute smallest prime factors
    for i in range(2, int(n**0.5) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i * i, n + 1, i):
                if spf[j] == j:
                    spf[j] = i
                    
    # Array to store the number of divisors for every number
    divisors = [1] * (n + 1)
    divisors[0] = 0  # 0 is not used
    divisors[1] = 1  # 1 has one divisor

    # Compute divisor counts using prime factorization:
    # For a number i, if i = p^a * m, then d(i) = d(m) * (a+1)
    for i in range(2, n + 1):
        p = spf[i]
        exp = 0
        temp = i
        while temp % p == 0:
            temp //= p
            exp += 1
        divisors[i] = divisors[temp] * (exp + 1)
        
    return divisors

def find_max_divisor(n):
    divisors = compute_divisors_sieve(n)
    max_div_count = 0
    max_number = 0
    for i in range(1, n + 1):
        if divisors[i] > max_div_count:
            max_div_count = divisors[i]
            max_number = i
    return max_number, max_div_count

import numba as nb
@nb.njit
def find_max_divisor_numba(n):
    # Allocate a list for divisor counts
    divisors = [0] * (n + 1)
    # Count divisors using a sieve‐like method
    for i in range(1, n + 1):
        for j in range(i, n + 1, i):
            divisors[j] += 1
    max_div = 0
    max_num = 0
    # Find the number with the maximum divisor count
    for i in range(1, n + 1):
        if divisors[i] > max_div:
            max_div = divisors[i]
            max_num = i
    return max_num, max_div

print(find_max_divisor_numba(2_200_000))







'''
# Find most divisible number up to n, return the number and it's divisibility
def find_div(value):
    cur = 0
    for i in range(1,value+1):
        if value % i == 0:
            cur += 1
    
    return cur


def find_max_div(value):
    m = 0
    for i in range(1,value+1):
        curm = find_div(i)   
        if curm > m:
            ind = i
            m = curm
    return ind, m

print(find_max_div(100_000))
'''