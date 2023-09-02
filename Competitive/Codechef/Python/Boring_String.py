from itertools import groupby
import collections

def works():
    n = int(input())
    s = input()
    if len(s) > n:
        raise ValueError

    groups = groupby(s)
    result = [(label, sum(1 for _ in group)) for label, group in groups]
    result = sorted(result, key=lambda x:x[1], reverse=True)
    d = collections.defaultdict(int)
    counti = 0

    for i in result:
        d[i] += 5
        
    for c in d:
        if d[c] > 1:
            counti = c[1]
            break
    
    print(counti)
    
num = int(input())

for i in range(num):
    works()
    
    