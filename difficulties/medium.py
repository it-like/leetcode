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

#s = '0123456789'
#print(s[0:9])


def minOperations(nums, k: int) -> int:
    # First sort list,
    # then add together the operand on the bottom
    def weird_operand(bottom):
        x,y = bottom
        return [min(x,y) * 2 + max(x,y)]
    
    def quicksort(nums):
        if len(nums) <= 1:
            return nums
        pivot = nums[-1]
        greater = []
        lower = []
        for element in nums[:-1]: # exclude pivot
            if element >= pivot:
                greater.append(element)
            else:
                lower.append(element)
        return quicksort(lower) + [pivot] + quicksort(greater)
    

    nums = quicksort(nums)
    operands = 0
    while nums[0] < k:
        operands += 1
        nums = nums[2:] + weird_operand(nums[:2]) 
    return operands



#import heapq
#values = [9,98,52,8]
#values2 = [9,98,52,8]
#print(values)
#heapq.heapify(values)
#heapq.heapify(values2[:])
#print(values)
#print(values2)
#print(minOperations([9,98,52,8], 98))
#values = [1,2,3,4,5,6,7,8,9]
#k = 4
#s = 1
#for i in range(k):
#    s *= values[-(i+1)]
#    print(s)


import itertools

def punishmentNumberFake( n: int) -> int:
    # start with brute-fore
    # Find punishment number
    def isPunishment(fish : int) -> int:
        sfish = list(map(int, str(fish**2)))
        for L in range(fish +1):
            for subset in itertools.combinations(sfish, L):
                if sum(subset) == fish:
                    print(subset)
                    return int(''.join(map(str, subset)))
        return 0
    run = 0
    for i in range(1,n):
        run += isPunishment(i)
        
    return run

def punishmentNumber( n: int) -> int:
    res = 0

    def isPunishment(i: int, cur: int, target: int, sub: str):
        if cur>target: # Early break
            return False
    
        if i == len(sub) and cur == target:
            return True
        
        for j in range(i ,len(sub)):
            if isPunishment(j+1, cur + int(sub[i:j+1]), target, sub):
                return True
        return False
    

    for i in range(1,n+1):
        if isPunishment(0,0, i, str(i*i)):
            res += i**2
    return res
            
#print(punishmentNumber(10))






class Solution:
    def constructDistancedSequence(self, n: int):
        # The length of the sequence is 2*n - 1
        trial = [0] * ((2 * n) - 1)
        
        def backtrack(num: int) -> bool:
            # Base case: 
            if num == 1:
                for i in range(len(trial)):
                    if trial[i] == 0:
                        trial[i] = 1
                        return True
                    
            for i in range(len(trial) - num):
                if trial[i] == 0 and trial[i + num ] == 0:
                    
                    trial[i] = num
                    trial[i + num] = num
                    # Recursively place the next smaller number
                    if backtrack(num - 1):
                        return True
                    trial[i] = 0
                    trial[i + num ] = 0
            
            return False

        backtrack(n)
        return trial


#print(Solution().constructDistancedSequence(5))
print(max(x**2 for x in range(4)))


'''78. subsets'''
def subsets(nums):
    # Backtracking problem  
    ret = []

    def back(start, path):

        ret.append(path[:])

        for i in range(start, len(nums)):

            path.append(nums[i])
            back((i+1), path)

            path.pop() # Remove old prev

    back(0, [])
    return ret

#print(subsets([1,2,3]))


'''39. Combination Sum'''
def combinationSum(candidates, target)-> list:
    # For element in 
    ret = []
    def dfs(i, cur, val):
        if val == target:
            ret.append(cur.copy())
            return
        if val > target or i>= len(candidates):
            return

        
        cur.append(candidates[i])
        # Case 1
        dfs(i, cur, val + candidates[i])
        # Case 2
        cur.pop()
        dfs(i+1, cur, val)


    dfs(0,[],0)
    return ret

print(combinationSum([1,2,3],5))

    
    