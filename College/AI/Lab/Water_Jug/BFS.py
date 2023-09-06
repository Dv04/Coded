# Problem Statement: You are given an m liter jug and a n liter jug. Both the jugs are initially empty. The jugs don’t have markings to allow measuring smaller quantities. You have to use the jugs to measure d liters of water where d is less than n. 

# Code:

from collections import deque

def BFS(m, n, k):

 set_of_jug = {}
 isSolvable = False
 path = []
 q = deque()
 q.append((0, 0))
 while (len(q) > 0):
  u = q.popleft()
  if ((u[0], u[1]) in set_of_jug):
   continue
  if (u[0] > m or u[1] > n or u[0] < 0 or u[1] < 0):
   continue
  path.append([u[0], u[1]])
  set_of_jug[(u[0], u[1])] = 1
  if (u[0] == k or u[1] == k):
   isSolvable = True
   if (u[0] == k):
    if (u[1] != 0):      
     path.append([u[0], 0])
   else:
    if (u[0] != 0):      
     path.append([0, u[1]])
   sz = len(path)
   for i in range(sz):
    print("The",i,"th step is (", path[i][0], ",", path[i][1], ")")
   break
  q.append([u[0], n]) 
  q.append([m, u[1]]) 
  for ap in range(max(m, n) + 1):
   c = u[0] + ap
   d = u[1] - ap
   if (c == m or (d == 0 and d >= 0)):
    q.append([c, d])
   c = u[0] - ap
   d = u[1] + ap
   if ((c == 0 and c >= 0) or d == n):
    q.append([c, d])
  q.append([m, 0])
  q.append([0, n])
 if (not isSolvable):
  print("No solution")

if __name__ == '__main__':
 m, n, k = map(int,input("PLease Enter the sizes of two jugs and the final limit you want to reach: ").split())
 print("Path from initial state to solution state ::")
 BFS(m, n, k)