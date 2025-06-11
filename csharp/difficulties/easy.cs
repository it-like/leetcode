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





        public int NumberOfEmployeesWhoMetTarget(int[] hours, int target)
        {
            int ret = 0;
            foreach (int h in hours)
            {
                if (h >= target)
                {
                    ret++;
                }
            }
            return ret;
        }




        public IList<int> FindWordsContaining(string[] words, char x)
        {
            var ret = new List<int>();
            for (int i = 0; i < words.Length; i++)
            {
                for (int j = 0; j < words[i].Length; j++)
                {
                    if (words[i][j] == x)
                    {
                        ret.Add(i);
                        break; // No duplicate alllowed
                    }
                }
            }
            return ret;
        }
    }
}