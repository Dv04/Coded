class Solution:
    def canPlaceFlowers(self, flowerbed: list[int], n: int) -> bool:

        if n == 0:
            return True

        if len(flowerbed) == 1:
            if flowerbed[0] == 0:
                return True
            else:
                return False

        if flowerbed[0] == 0 and flowerbed[1] == 0:
            flowerbed[0] = 1
            n -= 1

        if flowerbed[-1] == 0 and flowerbed[-2] == 0:
            flowerbed[-1] = 1
            n -= 1

        for i in range(1, len(flowerbed)-1):
            if flowerbed[i] == 0 and flowerbed[i-1] == 0 and flowerbed[i+1] == 0:
                flowerbed[i] = 1
                n -= 1

        if n <= 0:
            return True
        else:
            return False

flowerbed = [1,0,0,0,0,1]
n = 2

print(Solution().canPlaceFlowers(flowerbed,n))