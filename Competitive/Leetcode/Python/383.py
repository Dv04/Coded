from collections import Counter


class Solution:
    def canConstruct(self, ransomNote: str, magazine: str) -> bool:
        dict1 = Counter(ransomNote)
        dict2 = Counter(magazine)

        check = True
        
        for key in dict1.keys():
            if dict1[key] > dict2[key]:
                check = False
                break
        if check == True:
            return True
        else:
            return False


print(Solution().canConstruct("aab", "baa"))
