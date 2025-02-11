'''Find Largest trapped region of water, solved wrong question LOL  '''
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