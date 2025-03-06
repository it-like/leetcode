'''228. Summary range'''
#nums = [0,1,2,4,5,7]
#print(nums[1:])



'''1752. Check if Array is Sorted and Rotated'''
def find_violation(nums):
    violations = 0
    left = nums[0]
    for right in range(1,len(nums)):
        if left > nums[right]:
            violations += 1
        left = nums[right]
    if violations == 0 or (violations == 1 and nums[-1] <= nums[0]):
        return True
    else:
        return False    
#nums = [6, 10 ,6]
#print(find_violation(nums))



'''1800. Maximum Ascending Subarray Sum'''
def maxAscendingSum(nums):
    max_sum = nums[0]
    current_sum = nums[0]
    for i in range(1, len(nums)):
        if nums[i-1] < nums[i]:
            current_sum += nums[i]
        else:
            max_sum = max(max_sum, current_sum)
            current_sum = nums[i]
    return max(current_sum, max_sum)

        
#nums=[12,17,15,13,10,11,12] 
#print(maxAscendingSum(nums))




'''17. Letter Combinations of a Phone Number'''

def letterCombinations( digits: str):
    res = []
    digit_to_letter = {
                    '2' : 'abc',
                    '3' : 'def',
                    '4' : 'ghi',
                    '5' : 'jkl',
                    '6' : 'mno',
                    '7' : 'pqrs',
                    '8' : 'tuv',
                    '9' : 'xyzw'
                    }
    
    def itsGoingDown(i,substr):
        if len(substr) == len(digits):
            res.append(substr)
            return
        
        for char in digit_to_letter[digits[i]]:
            itsGoingDown(i + 1, substr + char)

    if digits:
        itsGoingDown(0,'')

    return res
#print(letterCombinations('23'))



'''3160. Find the Number of Distinct Colors Among the Balls'''
#d = {'1': 'blue'}
#d['2'] = 'orange'
#for k,v in d.items():
#    print(k)





'''3427. Sum of Variable Length Subarrays'''
def subarraySum(nums) -> int:
    res = 0
    for i in range(len(nums)):
        start = max(0, i - nums[i])
        res += sum(nums[start:i+1])
    return res
            



'''83. Removed Duplicate from Sorted List'''
# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def deleteDuplicates(head):
    left = head
    res = ListNode(0,head)
    right = head.next
   
    while right.next:
        while left.val == right.val:
            if right.next is None:
                break
            right = right.next
        left.next = right
        left = right
    return res.next
            
#list = []
#print(deleteDuplicates([1,1,2]))


#for combination in reversed(range(2**2)):
#        s_b_value = str(bin(combination))[2:]
#        if len(s_b_value) < 2:
#            print('here')
#            s_b_value = s_b_value.ljust(2,'0')
#        print(s_b_value)







def mergeArrays(nums1,nums2):
    # Problem is that indexes will not be on same, but is okay
    # as we just add their sum
    ret = []
    # Two pointer approach
    one,two = 0,0
    while one < len(nums1)  and two < len(nums2): 
        id1 = nums1[one][0]
        id2 = nums2[two][0]
        val1 = nums1[one][1]
        val2 = nums2[two][1]

        if id1 == id2:
            ret.append([id1, val1 + val2])
            one += 1
            two += 1
        elif id1 < id2: 
            ret.append([id1, val1])
            one += 1
        else: 
            ret.append([id2, val2])
            two += 1
    # add last part, both can be here since one will be empty
    if one < len(nums1):
        for i in range(one,len(nums1)):
            ret.append([nums1[i][0],nums1[i][1]])
            one += 1
            
    if two < len(nums2):
        for i in range(two,len(nums2)):
            ret.append([nums2[i][0],nums2[i][1]])
            two += 1
    return ret

#a =  [[2,4],[3,6],[5,5]]
#b =  [[1,3],[4,3]]
#a = [[1,2],[2,3],[4,5]]
#b = [[1,4],[3,2],[4,1]]
#
#print(mergeArrays(a,b))



'''2965. Find Missing and Repeated Values'''
def findMissingAndRepeatedValues(grid):
    size = len(grid)
    freq = {i+1: 0 for i in range(size**2)}

    for row in grid:
        for num in row:
            freq[num] += 1
            if freq[num] == 2:
                repeated = num

    missing = next(key for key, count in freq.items() if count == 0)
    return [repeated, missing]

#print(findMissingAndRepeatedValues([[9,1,7],[8,9,2],[3,4,6]]))