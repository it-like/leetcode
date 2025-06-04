


def pro1():
    '''Quicksort'''
    def quick_sort(array : list) -> list:
        if len(array) <= 1:
            return array
        else:
            pivot = array[-1]
            lower = [x for x in array[:-1] if x < pivot]
            greater = [x for x in array[:-1] if x > pivot]
            return quick_sort(lower) + [pivot] + quick_sort(greater)

    #array = [64, 34, 25, 12, 22, 11, 90]
    #sorted_array = quick_sort(array)
    #print(sorted_array)
 


'''List to hastable -> {index, value}'''
def pro2():
    def list_to_dict(arr)-> dict:
        table = {i: arr[i] for i in range(len(arr))}
        return table
    array = [1,2]
    dictio = list_to_dict(array)
    print(dictio.keys())





def pro3():
    def threeSum(nums):       
        # Make some type of hashtable to look up the values, sort list with quicksort.

        if len(nums) <= 2:
            return []

        solutions = set()
        def quick_sort(arr):
        
            if len(arr) <= 1:
                return arr
            else:
                pivot = arr[0]
                lower = [x for x in arr[1:] if x < pivot]  
                greater = [x for x in arr[1:] if x >= pivot]
                return quick_sort(lower) + [pivot] + quick_sort(greater)  

        nums = quick_sort(nums)

        for i in range(len(nums)):
            left = i + 1
            right = len(nums) - 1
            while left < right:
                sum = nums[i] + nums[left] + nums[right]
                if sum == 0:
                    solutions.add((nums[i], nums[left], nums[right]))
                    left += 1
                    right -= 1
                elif sum < 0:
                    left += 1
                else:
                    right -= 1
        return list(solutions)  

    print(threeSum([-1,0,1,2,-1,-4]))  





def quicksort2(nums): 
    if len(nums) <= 1:
        return nums
    else:
        bottom = []
        top = []
        pivot = nums[-1]
        for element in nums[:-1]:
            if element < pivot:  
                bottom.append(element)
            else:
                top.append(element)
        return quicksort2(bottom) + [pivot] + quicksort2(top)
    
ad =  [64, 34, 25, 12, 22, 11, 90]
print(ad[:2])
#print(quicksort2( [64, 34, 25, 12, 22, 11, 90]))