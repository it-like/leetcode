namespace Easy
{
    public class Solution
    {
        public int[] TwoSum(int[] nums, int target)
        {
            var seen = new Dictionary<int, int>();
            for (int i = 0; i < nums.Length; i++)
            {

                int complement = target - nums[i];

                if (seen.ContainsKey(complement))
                {
                    return new int[] { seen[complement], i };
                }
                if (!seen.ContainsKey(complement))
                {
                    seen[nums[i]] = i;
                }
            }

            throw new ArgumentException("No two sum solution");
        }
    }
}