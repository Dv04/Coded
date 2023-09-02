# Given a list of words, list of  single letters (might be repeating) and score of every character.

# Return the maximum score of any valid set of words formed by using the given letters (words[i] cannot be used two or more times).

# It is not necessary to use all characters in letters and each letter can only be used once. Score of letters 'a', 'b', 'c', ... ,'z' is given by score[0], score[1], ... , score[25] respectively.

# Example 1:

# Input:
# words = ["dog","cat","dad","good"]
# letters = ["a","a","c","d","d","d","g","o","o"]
# score = [1,0,9,5,0,0,3,0,0,0,0,0,0,0,2,0,0,0,0,0,0,0,0,0,0,0]
# Output: 23
# Explanation:
# Score  a=1, c=9, d=5, g=3, o=2
# Given letters, we can form the words "dad" (5+1+5) and "good" (3+2+2+5) with a score of 23.
# Words "dad" and "dog" only get a score of 21.

# Example 2:

# Input:
# words = ["xxxz","ax","bx","cx"]
# letters = ["z","a","b","c","x","x","x"]
# score = [4,4,4,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,5,0,10]
# Output: 27
# Explanation:
# Score  a=4, b=4, c=4, x=5, z=10
# Given letters, we can form the words "ax" (4+5), "bx" (4+5) and "cx" (4+5) with a score of 27.
# Word "xxxz" only get a score of 25.

# Example 3:

# Input: words = ["leetcode"]
# letters = ["l","e","t","c","o","d"]
# score = [0,0,1,1,1,0,0,0,0,0,0,1,0,0,1,0,0,0,0,1,0,0,0,0,0,0]
# Output: 0
# Explanation:
# Letter "e" can only be used once.

import collections


class Solution:
    def maxScoreWords(self, words: list[str], letters: list[str], score: list[int]) -> int:
        def dfs(i, score):
            if i == len(words):
                return score

            # if we don't use the word
            ans = dfs(i + 1, score)

            # if we use the word
            for c in words[i]:
                count[c] -= 1
                if count[c] < 0:
                    return ans
            ans = max(ans, dfs(i + 1, score + word_score[i]))
            for c in words[i]:
                count[c] += 1
            return ans

        word_score = [0] * len(words)
        count = collections.Counter(letters)
        for i, word in enumerate(words):
            for c in word:
                word_score[i] += score[ord(c) - ord('a')]
        return dfs(0, 0)


words = ["azb", "ax", "awb", "ayb", "bpppp"]
letters = ["z", "a", "w", "x", "y", "b", "p", "p", "p"]
score = [10, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 3, 2, 3, 3]

print(Solution().maxScoreWords(words, letters, score))
