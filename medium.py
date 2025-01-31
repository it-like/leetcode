#medium.py


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

print(minSubArrayLen(7, nums =[2,3,1,2,4,3]))