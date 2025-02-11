'''medium problems'''
#import numpy as np

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

#matrix = [[col+1 for col in range(9)] for row in range(9)]
#print(matrix[4:6])
#
#pi =f'{np.pi}'
#print(pi)



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



'''1910. Remove All occurrences of a Substring'''
def removeOccurrences(s: str, part: str) -> str:
    
    i =0
    part = 'abc'
    while i <= len(s) - len(part):
        if s[i:i + len(part)] == part:
           print(s)
           s = s[:i] + s[i+len(part):]
           i = 0
        i += 1
    return s
#print(removeOccurrences(s = 'daabcbaabcbc',part='abc'))

# expected "dab"

s = '0123456789'
print(s[0:9])

'''Find Largest trapped region of water, solved wrong question LOL'''
def trap(height) -> int:
    max_water = 0
    left = 0
    right = len(height) - 1
    while right > left -1:
        bottom = right - left
        tot_area = bottom * min(height[left], height[right])
        # Now remove the occupied water
        water = tot_area - sum(height[left: right]) 
        max_water = max(max_water, water)
        if height[left] > height[right]:
            right -= 1
        else:
            left += 1
    return max_water