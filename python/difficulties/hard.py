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



'''1092. Shortest common Supersequence'''

large = 'abac'
small = 'cab'
max_len = len(small) + len(large)
l = 0
r = 0
for i in range(0,max_len):
    clarge = large[:i+1]
    if i <= len(small):  
        csmall = small[-i:]
    else:
        csmall = small[:max_len-i]
        clarge = large[i-len(large)+1:i]
    print(csmall)
    print(clarge)
    if csmall == clarge:
       print("Truuu")
# bruh