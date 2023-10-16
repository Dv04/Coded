t = int(input())
for i in range(t):
    str1,str2 = "codeforces", input()
    count = sum(1 for a, b in zip(str1, str2) if a != b) 
    print (count)