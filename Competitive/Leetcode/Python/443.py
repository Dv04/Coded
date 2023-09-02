class Solution:
    def compress(self, chars: list[str]) -> int:
        n = len(chars)
        if n == 1:
            return 1
        
        i = 0
        j = 0
        
        while i < n:
            count = 1
            while i < n - 1 and chars[i] == chars[i+1]:
                count += 1
                i += 1
            
            chars[j] = chars[i]
            j += 1
            
            if count > 1:
                for c in str(count):
                    chars[j] = c
                    j += 1
            
            i += 1
        return j

            
        #     if (chars[i] != chars[i-1]) | (i == len(chars)-1):
        #         if count == 1:
        #             chars[iter] = str(chars[i-1])
        #             iter += 1
        #             if chars[i] == chars[i-1]:
        #                 chars[iter] = str(2)
        #                 iter += 1
        #                 continue
        #             if (i == len(chars)-1):
        #                 chars[iter] = str(chars[i])
        #                 iter += 1
# 
        #         else:
        #             if (i == len(chars)-1) and not (chars[i] != chars[i-1]):
        #                 count += 1
        #             if count > 10:
        #                 chars[iter] = str(chars[i-1])
        #                 iter += 1
        #                 t = 0
        #                 while count > 0:
        #                     digit = count % 10
        #                     chars.insert(iter, str(digit))
        #                     count //= 10
        #                     t += 1
        #                 iter += t
# 
        #             else:
        #                 chars[iter] = str(chars[i-1])
        #                 iter += 1
        #                 chars[iter] = str(count)
        #                 iter += 1
                    # 
        #             if (i == len(chars)-1) and (chars[i] != chars[i-1]):
        #                 chars[iter] = str(chars[i])
        #                 iter += 1
        #         count = 1
        #     elif (chars[i] == chars[i-1]):
        #         count += 1
        # chars = chars[:iter]
        # return chars


print(Solution().compress(["a","a","a","a","a","b"]))
