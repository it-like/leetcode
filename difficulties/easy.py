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