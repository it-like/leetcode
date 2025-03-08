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
#print(max(x**2 for x in range(4)))


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

#print(combinationSum([1,2,3],5))

    
    

def lenLongestFibSubseq(arr) -> int:
    # Bruteforce
    def dp(i,j,k,cur):
        if k >= len(arr):
            return
        if arr[i] + arr[j] == arr[k]:
            cur += 1
            dp(j,k,k+1,cur)
        # Either move forward k alone or j and k!
        # Move k, value currently too small
        if arr[i] + arr[j] > arr[k]:
            dp(i,j,k+1,cur)
        if arr[i] + arr[j] < arr[k]: # Move forward
            dp(i,j+1,k+1,cur)
        
        return cur
            
    cmax = 0
    vals = [0] * len(arr)
    i,j,k = 0,1,2
    while i != len(arr)-1:
        h=i
        cur = dp(i,j,k,vals[h])
        cmax = max(cmax,cur)
        i += 1
    
    return cmax 
#a = [1,2,3,4,5,6,7,8]
#b= [1,3,7,11,12,14,18]
#print(lenLongestFibSubseq(a))



# New record on implementation! Less than 3 minutes
def pivotArray(nums, pivot: int):
    # First thought, quicksort approach
    # Just one pass of quicksort should do it
    lower = []
    middle = []
    upper = []
    for num in nums:
        if num < pivot:
            lower.append(num)
        elif num > pivot:
            upper.append(num)
        else:
            middle.append(num)
    return lower + middle + upper



def checkPowersOfThree(n: int) -> bool:
    # DP problem, can be 
    # seen as take-skip 
    # as values are distinct
    powers = [3**i for i in range(16)] # Largest value
    
    r = len(powers) - 1
    # find upper bound
    while r >= 0 and powers[r] > n:
        r -= 1

    def dfs(i, target):
        if target == 0:
            return True
        if i < 0 or target < 0:
            return False
        # Take
        if dfs(i - 1, target - powers[i]):
            return True
        # Skip
        if dfs(i - 1, target):
            return True
        
        return False
    
    return dfs(r, n)    


'''2579. Count Total Number of Colored Cells'''
def coloredCells(n: int) -> int:
    if n == 1:
        return n
    bricks = n * 2 - 1
    for i in range(1,n):
        bricks += (i * 2 - 1) * 2
    return bricks




'''2523. Closest Prime Numbers in Range'''
# Was 25% faster than all but very memory efficient
def closestPrimes( left: int, right: int):
    prime = []
    pair = [-1,-1]
    cmin = float('inf')
    for i in range(left, right+1):
        done = False
        if i % 2 == 0 and i > 2: # do not even consider even if not 2
            continue
        upper = i // 2 -1 if  (i // 2 )%2 == 0 else i //2
        for j in range(2,upper,1):
            if i % j == 0: 
                done = True
                break # found divisible term, leave
        if not done:
            if i == 1: # corner case, if ever found will be at beginning
                continue
            else:
                prime.append(i)
            if len(prime)>1:
                curmin = prime[-1] - prime[-2] # Two furthest back 
                if curmin < cmin:
                    pair = [prime[-2], prime[-1]]
                    cmin = curmin
                    if cmin <=2:
                        return pair
    return pair
print(closestPrimes(901000,1000000))