#medium.py
import numpy as np

'''209. Minimum Size Subarray Sum'''
def minSubArrayLen(target: int, nums) -> int:
        smallest = float('inf')
        left = 0
        current_sum = 0
        for right in range(len(nums)):
            current_sum += nums[right]
            while current_sum >= target:
                smallest = min(smallest, right - left + 1)
                current_sum -= nums[left]
                left += 1
        return smallest if smallest != float('inf') else 0
#print(minSubArrayLen(7, nums =[2,3,1,2,4,3]))



## testing sets
#x = {f"{x + 1}" for x in range(9)}
#x.add(".")
#y = {f'{x+1}' for x in range(9) if x % 2 ==0}
#print(x)
#print(y)
#print(y.issubset(x))
#

matrix = [[col+1 for col in range(9)] for row in range(9)]
print(matrix[4:6])

pi =f'{np.pi}'
print(pi)



'''2364. Count Number of Bad Pairs'''
from collections import defaultdict
def countBadPairs(nums) -> int:
    ## Check complement? All possible bad pairs - good pairs
    # should be (n-1) + (n-2) + ... + 1  possible pairs?
    good = 0
    total = 0
    count = defaultdict(int)
    for i in range(len(nums)):
        total += i
        good += count[nums[i] - i]
        count[nums[i] - i] += 1
    return total - good
'''Brute force'''
'''
def countBadPairs(nums) -> int:
     bad = 0
     for i in range(len(nums)-1):
         for j in range(i+1, len(nums)):
             if j-i != nums[j] - nums[i]:
                 bad += 1
     return bad
'''