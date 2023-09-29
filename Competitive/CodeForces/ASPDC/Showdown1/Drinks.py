a = int(input())
b = [float(x) / 100 for x in input().split()]

print(sum(b) / len(b) * 100)
