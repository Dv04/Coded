a = int(input())

print(int(a/365),"years")
print(int((a%365)/30),"months")
print(int((a%365)%30),"days")