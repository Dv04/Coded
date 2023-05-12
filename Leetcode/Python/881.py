# class Solution:
#     def numRescueBoats(self, people: list[int], limit: int) -> int:
#         count, check, ind, i = 0, limit, -1, 0
#         people.sort(reverse=True)
#         while(i<len(people) and i-ind<=len(people)):
#             if people[i] == check:
#                 count += 1
#                 i+=1
#             elif (people[i] + people[ind]) <= limit:
#                 ind -= 1
#                 count += 1
#                 i+=1
#             elif (people[i] + people[ind]) > limit:
#                 check = people[i]

#         return count

class Solution:
    def numRescueBoats(self, people: list[int], limit: int) -> int:

        people.sort()
        boats = 0

        while people:
            if len(people) > 1 and people[-1] + people[0] <= limit:
                people.pop(0)
            people.pop(-1)
            boats += 1
        
        return boats

print(Solution().numRescueBoats(people = [3,2,2,1], limit = 3))
